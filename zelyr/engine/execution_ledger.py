"""Authoritative Turn Execution Ledger for Zelyr Runtime.

Maintains strict, evidence-based machine state authority across agent turns.
Tracks tool call lifecycle from proposal through validation, dispatch,
execution observation, and deterministic verification.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
import json
import os
from pathlib import Path
import re
import time
from typing import Any, Dict, List, Optional, Set, Tuple


class InvalidStateTransitionError(ValueError):
    """Raised when an invalid action lifecycle state transition is attempted."""


def _redact_sensitive_text(text: Optional[str]) -> Optional[str]:
    """Scrub common credentials/tokens before storing error strings in the ledger."""
    if text is None:
        return None
    raw = str(text)
    if not raw:
        return raw
    scrubbed = re.sub(r"(?i)(https?://)([^:/@\s]+):([^/@\s]+)@", r"\1\2:[REDACTED]@", raw)
    scrubbed = re.sub(r"(?i)\b(bearer\s+)[a-z0-9._\-+=]+", r"\1[REDACTED]", scrubbed)
    scrubbed = re.sub(r"\bsk-[A-Za-z0-9_\-]{8,}", "[REDACTED]", scrubbed)
    scrubbed = re.sub(
        r"(?i)\b(api[_-]?key|token|secret|password)\s*([=:])\s*['\"]?([^\s,'\"&;]+)['\"]?",
        r"\1\2[REDACTED]",
        scrubbed,
    )
    return scrubbed


class ActionState(str, Enum):
    """Lifecycle states of an agent action."""

    PROPOSED = "proposed"
    VALIDATED = "validated"
    AUTHORIZED = "authorized"
    DISPATCHED = "dispatched"
    EXECUTED = "executed"
    SUCCESS = "success"
    FAILED = "failed"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    SUPERSEDED = "superseded"
    ABANDONED = "abandoned"
    REPLANNED = "replanned"
    PARTIALLY_COMPLETED = "partially_completed"
    OBSERVED = "observed"
    VERIFIED = "verified"
    UNVERIFIED = "unverified"


@dataclass(frozen=True)
class ActionRecord:
    """Immutable record of an individual action and its verified outcome."""

    call_id: str
    tool_name: str
    arguments: Dict[str, Any]
    state: ActionState
    exit_code: Optional[int] = None
    error: Optional[str] = None
    verified: bool = False
    verification_detail: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    terminal: bool = False
    recovery_attempts: int = 0
    turn: Optional[int] = None
    raw_name: Optional[str] = None
    rejection_reason: Optional[str] = None
    verification_failed: bool = False
    failed_targets: List[str] = field(default_factory=list)
    verified_targets: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert record to a serializable dictionary."""
        d = asdict(self)
        d["state"] = self.state.value if isinstance(self.state, ActionState) else str(self.state)
        d["terminal"] = self.terminal
        d["recovery_attempts"] = self.recovery_attempts
        d["verification_failed"] = self.verification_failed
        d["failed_targets"] = list(self.failed_targets)
        d["verified_targets"] = list(self.verified_targets)
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ActionRecord:
        """Reconstruct an ActionRecord from a serialized dictionary (supporting legacy records)."""
        if not isinstance(data, dict):
            raise TypeError("ActionRecord.from_dict expects a dict")
        raw_state = data.get("state", ActionState.PROPOSED)
        if isinstance(raw_state, ActionState):
            state = raw_state
        else:
            try:
                state = ActionState(str(raw_state).lower())
            except ValueError:
                state = ActionState.PROPOSED

        is_failed_state = state in (
            ActionState.FAILED,
            ActionState.CANCELLED,
            ActionState.REJECTED,
            ActionState.ABANDONED,
            ActionState.SUPERSEDED,
        )
        raw_ver = data.get("verified", False)
        verified = bool(isinstance(raw_ver, bool) and raw_ver is True and not is_failed_state)

        raw_vf = data.get("verification_failed", False)
        verification_failed = bool(
            isinstance(raw_vf, bool)
            and raw_vf is True
            and not is_failed_state
            and not verified
        )
        if verified and state == ActionState.SUCCESS:
            state = ActionState.VERIFIED

        raw_ft = data.get("failed_targets")
        failed_targets = (
            [str(t).strip() for t in raw_ft if t and isinstance(t, str) and str(t).strip()]
            if isinstance(raw_ft, list) and verification_failed
            else []
        )
        raw_vt = data.get("verified_targets")
        verified_targets = (
            [str(t).strip() for t in raw_vt if t and isinstance(t, str) and str(t).strip()]
            if isinstance(raw_vt, list) and verified
            else []
        )

        return cls(
            call_id=str(data.get("call_id", "")),
            tool_name=str(data.get("tool_name", "")),
            arguments=dict(data.get("arguments") or {}),
            state=state,
            exit_code=data.get("exit_code"),
            error=_redact_sensitive_text(data.get("error")),
            verified=verified,
            verification_detail=data.get("verification_detail"),
            timestamp=float(data.get("timestamp", time.time())),
            terminal=bool(data.get("terminal", False)),
            recovery_attempts=int(data.get("recovery_attempts", 0)),
            turn=data.get("turn"),
            raw_name=data.get("raw_name"),
            rejection_reason=data.get("rejection_reason"),
            verification_failed=verification_failed,
            failed_targets=failed_targets,
            verified_targets=verified_targets,
        )


def _canonicalize_args(args: Any) -> str:
    """Deterministic JSON string serialization for argument dictionaries."""
    if args is None:
        return "{}"
    raw = args
    if isinstance(raw, str):
        trimmed = raw.strip()
        if not trimmed:
            return "{}"
        if trimmed.startswith(("{", "[")):
            try:
                raw = json.loads(trimmed)
            except Exception:
                pass

    def _norm(obj: Any) -> Any:
        if isinstance(obj, bool):
            return obj
        if isinstance(obj, float) and obj.is_integer():
            return int(obj)
        if isinstance(obj, dict):
            return {str(k): _norm(v) for k, v in sorted(obj.items(), key=lambda item: str(item[0]))}
        if isinstance(obj, list):
            return [_norm(item) for item in obj]
        return obj

    try:
        return json.dumps(_norm(raw), sort_keys=True)
    except Exception:
        return str(raw)


def _is_same_capability(tool_a: str, tool_b: str) -> bool:
    """Check if tool_b provides equivalent or resolving capability for tool_a."""
    if not tool_a or not tool_b:
        return False
    if tool_a == tool_b:
        return True
    # Verification tool confirms overall system/filesystem outcome
    if tool_b == "verify_outcome":
        return True
    # Foundational actuators and verification can resolve verification failures
    if tool_a == "verify_outcome" and tool_b in ("execute_command", "manage_workspace_file", "verify_outcome"):
        return True
    # Foundational actuators can resolve each other
    if tool_a in ("execute_command", "manage_workspace_file") and tool_b in ("execute_command", "manage_workspace_file"):
        return True
    # Browser action equivalence
    browser_tools = {
        "interact_with_browser",
        "interact_with_ui",
        "get_page_snapshot",
        "browser_cdp",
        "browser_navigate",
    }
    if tool_a in browser_tools and tool_b in browser_tools:
        return True
    return False


def canonicalize_target(target: str, check_type: Optional[str] = None) -> str:
    """Canonicalize a target resource identifier based on its check type or format."""
    if not target:
        return ""
    t_str = str(target).strip()
    if not t_str:
        return ""

    # Non-filesystem check types: preserve as type-tagged identity
    if check_type in ("port_listening", "port", "listen"):
        return f"port:{t_str}"
    if check_type in ("process_running", "process", "proc"):
        return f"process:{t_str.lower()}"
    if check_type in ("command_output", "command", "cmd"):
        return f"cmd:{t_str}"

    # Strip any trailing punctuation that might leak from sentence splitting
    t_clean = re.sub(r"[.,:;!?]+$", "", t_str).strip()

    # Windows drive paths (e.g. C:\path or C:/path)
    if re.match(r"^[a-zA-Z]:[\\/]", t_clean):
        return t_clean.replace("/", "\\").lower()

    try:
        p = Path(os.path.expanduser(t_clean))
        return str(p.resolve())
    except Exception:
        return os.path.abspath(os.path.expanduser(t_clean))


def _is_explicit_abs_or_home(raw: str) -> bool:
    """Return True if raw path is explicitly absolute, home-relative, or Windows drive-rooted."""
    if not raw:
        return False
    s = str(raw).strip()
    return os.path.isabs(s) or s.startswith("~") or bool(re.match(r"^[a-zA-Z]:[\\/]", s))


def targets_match(
    candidate_raw: str,
    known_canon: str,
    all_known_canons: Optional[Any] = None,
    allow_dir_containment: bool = False,
    check_type: Optional[str] = None,
) -> bool:
    """Compare a candidate target reference against a canonical target without false basename collisions.

    - Non-filesystem targets (port:, process:, cmd:) require exact canonical identity.
    - Explicit absolute/home/drive paths are resolved (collapsing '..' and symlinks) and require
      exact canonical equality (or directory containment when allow_dir_containment=True).
      Two distinct absolute paths with the same basename NEVER match.
    - Relative paths are first checked via canonicalize_target (cwd resolution). If cwd differs from
      the workspace of known_canon, a relative path may suffix-match known_canon ONLY when the entire
      normalized relative subpath matches the trailing path components of known_canon AND is
      unambiguous across all_known_canons.
    """
    if not candidate_raw or not known_canon:
        return False

    non_fs_prefixes = ("port:", "process:", "cmd:")
    if known_canon.startswith(non_fs_prefixes) or check_type in (
        "port_listening", "port", "listen",
        "process_running", "process", "proc",
        "command_output", "command", "cmd",
    ):
        cand_clean = str(candidate_raw).strip()
        if cand_clean.startswith("process:"):
            cand_norm = f"process:{cand_clean[8:].lower()}"
        elif cand_clean.startswith(non_fs_prefixes):
            cand_norm = cand_clean
        else:
            cand_norm = canonicalize_target(candidate_raw, check_type=check_type)
        return cand_norm == known_canon

    raw_clean = re.sub(r"[.,:;!?]+$", "", str(candidate_raw).strip()).strip()
    if not raw_clean or raw_clean.startswith(non_fs_prefixes):
        return False

    cand_canon = canonicalize_target(raw_clean)
    if cand_canon == known_canon:
        return True

    if allow_dir_containment:
        sep = "\\" if "\\" in known_canon else "/"
        kc_prefix = known_canon.rstrip("/\\") + sep
        if cand_canon.startswith(kc_prefix):
            return True

    # Explicit absolute paths that resolved to a different canonical path never match by suffix/basename.
    if _is_explicit_abs_or_home(raw_clean):
        return False

    rel_norm = os.path.normpath(raw_clean).replace("\\", "/")
    if rel_norm in ("", ".", "..") or rel_norm.startswith("../") or rel_norm.startswith("/"):
        return False

    known_norm = known_canon.replace("\\", "/").rstrip("/")
    fs_knowns = [
        str(k).replace("\\", "/").rstrip("/")
        for k in (all_known_canons or [])
        if k and not str(k).startswith(non_fs_prefixes)
    ]

    if known_norm.endswith("/" + rel_norm):
        if "/" not in rel_norm and fs_knowns:
            same_name = {k for k in fs_knowns if Path(k).name == rel_norm}
            if len(same_name) > 1:
                return False
        if fs_knowns:
            suffix_matches = {k for k in fs_knowns if k.endswith("/" + rel_norm)}
            if len(suffix_matches) > 1:
                return False
        return True

    if allow_dir_containment and "/" in rel_norm:
        first_seg = rel_norm.split("/", 1)[0]
        if known_norm.endswith("/" + first_seg):
            if fs_knowns:
                dir_matches = {k for k in fs_knowns if k.endswith("/" + first_seg)}
                if len(dir_matches) > 1:
                    return False
            return True

    return False


def criterion_applies_to_target(
    criterion: Dict[str, Any],
    candidate_target: str,
    all_known_canons: Optional[Any] = None,
) -> bool:
    """Return True if an acceptance criterion applies to candidate_target directly or via referenced file path."""
    if not criterion or not candidate_target:
        return False
    c_target = str(criterion.get("target") or "").strip()
    c_check = str(criterion.get("check") or "").strip()
    if not c_target:
        return False
    if targets_match(candidate_target, c_target, all_known_canons, check_type=c_check):
        return True
    if c_check.lower() in ("command_output", "command", "cmd"):
        for ft in extract_file_targets_from_text(c_target):
            if targets_match(candidate_target, ft, all_known_canons):
                return True
    return False


_REPETITION_ONLY_TAIL_RE = re.compile(
    r"^\s*(?:afterwards?|again|twice|thrice|more\s+than\s+once|a\s+second\s+time|\d+\s+times)\b",
    flags=re.IGNORECASE,
)

_UNCONDITIONAL_PROHIBIT_VERBS = {
    "create", "creating", "make", "making", "delete", "deleting",
    "remove", "removing", "unlink", "unlinking", "rename", "renaming",
    "move", "moving", "truncate", "truncating", "touch", "touching",
    "write", "writing", "write to", "writing to",
}

_POST_CREATE_MODIFY_VERBS = {
    "modify", "modifying", "edit", "editing", "update", "updating",
    "overwrite", "overwriting", "append", "appending", "append to", "appending to",
    "alter", "altering", "change", "changing", "patch", "patching",
    "replace", "replacing",
}


def _looks_like_path_token(tok: str) -> bool:
    """Return True if token looks like a file or directory path rather than an ordinary English word."""
    if not tok:
        return False
    if tok.lower() in {
        "it", "them", "this", "that", "these", "those", "anything", "everything",
        "nothing", "something", "file", "files", "directory", "directories", "dir",
        "folder", "folders", "path", "paths", "content", "contents", "data", "code",
        "changes", "modifications", "operations", "actions", "existing", "other",
        "any", "all", "new", "old", "the", "a", "an",
    }:
        return False
    return "/" in tok or "\\" in tok or "." in tok


def extract_prohibited_targets(prompt: str, ledger: Optional[Any] = None) -> List[str]:
    """Conservatively extract canonical file/directory targets the user forbade mutating.

    Policy:
    - Strips fenced code blocks so literal code samples do not manufacture prohibitions.
    - Matches explicit negative directives ('do not', 'don't', 'never', 'must not', 'should not',
      'shall not', 'cannot', 'can't', 'may not', 'avoid', 'refrain from', 'without') followed by
      a creation, write, modification, move/rename, or deletion verb, as well as
      'leave/keep <path> untouched/alone/unchanged/intact/unmodified'.
    - Skips repetition-only qualifiers ('again', 'twice', 'afterward', 'more than once').
    - Contradictory instructions (e.g. 'Create X' and 'Do not create X') fail closed: X is prohibited.
    - When the user asks to create X AND says 'do not modify/overwrite/edit X' (without banning
      creation), X is permitted to be created once and becomes prohibited against any subsequent
      mutation as soon as it has been created or verified.
    """
    if not prompt:
        return []

    # Strip fenced markdown code blocks
    cleaned = re.sub(r"```.*?```", " ", prompt, flags=re.DOTALL)

    all_verbs = sorted(_UNCONDITIONAL_PROHIBIT_VERBS | _POST_CREATE_MODIFY_VERBS, key=len, reverse=True)
    verbs_alt = "|".join(re.sub(r"\s+", r"\\s+", re.escape(v)) for v in all_verbs)

    pat_neg_verb = re.compile(
        rf"\b(?:do\s+not|don'?t|never|must\s+not|should\s+not|shall\s+not|cannot|can'?t|may\s+not|avoid|refrain\s+from|without)\s+"
        rf"(?:ever\s+)?(?:any\s+)?(?P<verb>{verbs_alt})"
        rf"(?:(?:\s+(?:anything|any\s+files?|files?))?\s+(?:in|into|inside|under|within|to|from|on))?\s+"
        rf"(?:the\s+)?(?:(?:existing|other)\s+)?(?:(?:file|directory|dir|folder|path)s?\s+)?"
        rf"(?P<targets>[\"'`]?[^\s\"'`,;:!?]+[\"'`]?(?:\s*(?:,|or|and|nor)\s+(?:the\s+)?(?:(?:file|directory|dir|folder|path)s?\s+)?[\"'`]?[^\s\"'`,;:!?]+[\"'`]?)*)",
        flags=re.IGNORECASE,
    )

    pat_leave = re.compile(
        r"\b(?:leave|keep)\s+(?:the\s+)?(?:(?:existing|other)\s+)?(?:(?:file|directory|dir|folder|path)s?\s+)?"
        r"[\"'`]?(?P<target>[^\s\"'`,;:!?]+)[\"'`]?\s+"
        r"(?:untouched|alone|unchanged|intact|unmodified|as[\s\-]is)\b",
        flags=re.IGNORECASE,
    )

    unconditional_raw: List[str] = []
    post_create_raw: List[str] = []

    for m in pat_neg_verb.finditer(cleaned):
        # Skip if inside a quoted string or immediately preceded by meta-quoting
        prefix = cleaned[max(0, m.start() - 30):m.start()]
        if (
            (m.start() > 0 and cleaned[m.start() - 1] in ("\"", "'", "`"))
            or re.search(
                r"\b(?:said|says|message|error|warning|output|prints?|what|why|whether|means?)\s*[:\"'`]+\s*$",
                prefix,
                flags=re.IGNORECASE,
            )
        ):
            continue

        tail = cleaned[m.end():m.end() + 35].lower()
        if _REPETITION_ONLY_TAIL_RE.match(tail):
            continue

        verb_norm = re.sub(r"\s+", " ", m.group("verb").strip().lower())
        targets_blob = m.group("targets")
        raw_parts = re.split(r"\s*(?:,|or|and|nor)\s+", targets_blob, flags=re.IGNORECASE)

        for part in raw_parts:
            part_clean = re.sub(
                r"^(?:the\s+)?(?:(?:existing|other)\s+)?(?:(?:file|directory|dir|folder|path)s?\s+)?",
                "",
                part.strip(),
                flags=re.IGNORECASE,
            )
            tok = part_clean.strip("\"'`").rstrip(".)")
            if not _looks_like_path_token(tok):
                continue
            if verb_norm in _POST_CREATE_MODIFY_VERBS:
                post_create_raw.append(tok)
            else:
                unconditional_raw.append(tok)

    for m in pat_leave.finditer(cleaned):
        tok = m.group("target").strip("\"'`").rstrip(".)")
        if _looks_like_path_token(tok):
            unconditional_raw.append(tok)

    found: List[str] = []
    for tok in unconditional_raw:
        canon = canonicalize_target(tok)
        if canon and canon not in found:
            found.append(canon)

    for tok in post_create_raw:
        canon = canonicalize_target(tok)
        if not canon or canon in found:
            continue
        tok_name = Path(canon).name
        # Check if prompt contains an affirmative non-negated creation request for this target
        has_affirmative_create = False
        for cm in re.finditer(
            rf"\b(?:create|make|generate)\s+(?:(?:a|an|the)\s+)?(?:new\s+)?(?:file\s+)?[\"'`]?([^\s\"'`,;:!?]+)",
            cleaned,
            flags=re.IGNORECASE,
        ):
            pre = cleaned[max(0, cm.start() - 25):cm.start()]
            if re.search(
                r"\b(?:do\s+not|don'?t|never|must\s+not|should\s+not|shall\s+not|cannot|can'?t|may\s+not|avoid|refrain\s+from|without)\s+(?:ever\s+)?$",
                pre,
                flags=re.IGNORECASE,
            ):
                continue
            c_tok = cm.group(1).strip("\"'`").rstrip(".)")
            if _looks_like_path_token(c_tok):
                if canonicalize_target(c_tok) == canon or Path(c_tok).name == tok_name:
                    has_affirmative_create = True
                    break

        if not has_affirmative_create:
            found.append(canon)
        elif ledger is not None:
            # Target was asked to be created AND forbidden from subsequent modification:
            # prohibit as soon as it has been created or verified once (without unresolved failures).
            has_unresolved_on_canon = bool(
                hasattr(ledger, "unresolved_failures")
                and any(
                    _extract_action_target(u.tool_name, u.arguments) is not None
                    and targets_match(_extract_action_target(u.tool_name, u.arguments) or "", canon)
                    for u in ledger.unresolved_failures()
                )
            )
            already_created = not has_unresolved_on_canon and bool(
                (hasattr(ledger, "is_target_verified") and ledger.is_target_verified(canon))
                or (
                    hasattr(ledger, "successful_actions")
                    and any(
                        not getattr(r, "verification_failed", False)
                        and not _is_inspection_action(r.tool_name, r.arguments)
                        and _extract_action_target(r.tool_name, r.arguments) is not None
                        and targets_match(
                            str(
                                r.arguments.get("path")
                                or r.arguments.get("filepath")
                                or r.arguments.get("file")
                                or r.arguments.get("target")
                                or _extract_action_target(r.tool_name, r.arguments)
                            ),
                            canon,
                        )
                        for r in ledger.successful_actions()
                    )
                )
            )
            if already_created:
                found.append(canon)

    return found


def extract_file_targets_from_text(text: str) -> List[str]:
    """Extract file path candidates mentioned in prompt or text."""
    if not text:
        return []

    # 1. Match absolute paths
    raw_abs = re.findall(r"(?:/(?:[\w\.\-]+))+", text)
    cleaned_abs = set()
    for p in raw_abs:
        p_clean = re.sub(r"[.,:;!?]+$", "", p).strip()
        if p_clean and not p_clean.startswith(("/dev/", "/bin/", "/usr/bin/", "/proc/", "/sys/")):
            cleaned_abs.add(canonicalize_target(p_clean))

    # 2. Remove absolute paths from text before matching standalone relative paths
    text_no_abs = re.sub(r"(?:/(?:[\w\.\-]+))+", " ", text)
    rel_candidates = re.findall(
        r"\b[\w\-]+\.(?:txt|py|sh|json|md|yaml|yml|log|csv|toml|html|css|js|rs|go|cpp|c|h)\b",
        text_no_abs,
        re.IGNORECASE,
    )
    cleaned_rel = set()
    for rp in rel_candidates:
        rp_clean = re.sub(r"[.,:;!?]+$", "", rp).strip()
        if rp_clean:
            cleaned_rel.add(canonicalize_target(rp_clean))

    return sorted(list(cleaned_abs | cleaned_rel))


def _is_inspection_action(tool_name: str, arguments: Dict[str, Any]) -> bool:
    """Check if an action is purely an inspection/read operation."""
    t_lower = (tool_name or "").lower()
    if t_lower == "manage_workspace_file":
        act = str(arguments.get("action", "read")).lower()
        return act in ("read", "view", "list", "list_tree", "stat", "exists", "inspect", "check")
    if t_lower in ("read_file", "view_file", "list_dir", "list_tree", "get_state", "inspect_state"):
        return True
    if t_lower.startswith(("read_", "get_", "list_", "check_", "inspect_", "stat_")):
        return True
    if t_lower in ("execute_command", "bash", "sh"):
        cmd = str(arguments.get("command", "") or arguments.get("cmd", "")).strip()
        cmd_parts = cmd.split()
        first_word = cmd_parts[0].lower() if cmd_parts else ""
        if first_word in (
            "ls", "cat", "head", "tail", "grep", "find", "pwd", "whoami", "uname", "stat", "file",
            "wc", "du", "df", "ps", "id", "env", "printenv", "which", "type", "readlink", "realpath",
            "md5sum", "sha256sum", "diff", "test", "[",
            "uptime", "ss", "netstat", "free", "hostname", "date", "lsof", "lsblk", "who", "w",
        ):
            if not any(op in cmd for op in (">", ">>", "| tee", "|tee")):
                return True
        if first_word == "git" and len(cmd_parts) >= 2:
            git_sub = cmd_parts[1].lower()
            if git_sub in ("status", "diff", "log", "show", "branch", "rev-parse", "describe", "tag", "check-ignore", "version", "--version"):
                if not any(op in cmd for op in (">", ">>", "| tee", "|tee")):
                    return True
    if t_lower == "manage_memory":
        act = str(arguments.get("action", "recall")).lower()
        return act in ("recall", "search", "get", "list")
    return False


def _extract_action_target(tool_name: str, arguments: Dict[str, Any]) -> Optional[str]:
    """Extract and canonicalize the target resource of an action."""
    if not isinstance(arguments, dict):
        return None
    t_lower = (tool_name or "").lower()

    if t_lower == "manage_workspace_file":
        for k in ("path", "filepath", "file", "target", "filename", "dst", "destination"):
            val = arguments.get(k)
            if val and isinstance(val, str) and val.strip():
                return canonicalize_target(val.strip())
        return None

    if t_lower == "verify_outcome":
        val = arguments.get("target")
        if val and isinstance(val, str) and val.strip():
            return canonicalize_target(val.strip(), check_type=arguments.get("check"))
        return None

    if t_lower in ("execute_command", "bash", "sh"):
        cmd = str(arguments.get("command", "") or arguments.get("cmd", "")).strip()
        extracted = extract_file_targets_from_text(cmd)
        if extracted:
            return canonicalize_target(extracted[0])
        for k in ("target", "file", "path"):
            val = arguments.get(k)
            if val and isinstance(val, str) and val.strip():
                return canonicalize_target(val.strip())
        return None

    for k in ("target", "path", "file", "url", "id", "name"):
        val = arguments.get(k)
        if val and isinstance(val, str) and val.strip():
            return str(val).strip()
    return None


def _extract_action_operation(tool_name: str, arguments: Dict[str, Any]) -> str:
    """Extract normalized operation type for an action."""
    t_lower = (tool_name or "").lower()
    if t_lower == "manage_workspace_file":
        return str(arguments.get("action", "write")).lower()
    if t_lower == "verify_outcome":
        return f"verify:{str(arguments.get('check', '')).lower()}"
    if t_lower in ("execute_command", "bash", "sh"):
        cmd = str(arguments.get("command", "") or arguments.get("cmd", "")).strip()
        first_word = cmd.split()[0].lower() if cmd.split() else ""
        return f"cmd:{first_word}"
    return t_lower


def _normalize_target_list(targets: Any) -> List[str]:
    """Canonicalize unique non-empty target strings while preserving typed prefixes."""
    if not isinstance(targets, (list, tuple, set)):
        return []
    out: List[str] = []
    for raw in targets:
        if not raw or not isinstance(raw, str):
            continue
        val = raw.strip()
        if not val:
            continue
        if not val.startswith(("cmd:", "port:", "process:")):
            val = canonicalize_target(val)
        if val and val not in out:
            out.append(val)
    return out


def _collect_later_resolution_targets(later: ActionRecord, later_is_inspect: bool) -> List[str]:
    """Collect canonical targets resolved or verified by a later successful action."""
    targets: List[str] = []
    if later.verified or later.state == ActionState.VERIFIED:
        targets = _normalize_target_list(getattr(later, "verified_targets", []))
    target_later = _extract_action_target(later.tool_name, later.arguments)
    if target_later and (not later_is_inspect or later.tool_name == "verify_outcome"):
        norm_tl = canonicalize_target(target_later) if not target_later.startswith(("cmd:", "port:", "process:")) else target_later
        if norm_tl and norm_tl not in targets:
            targets.append(norm_tl)
    return targets


def _is_action_resolved_by_later(failed: ActionRecord, later: ActionRecord) -> bool:
    """Check if later successful action genuinely resolves the earlier failed action."""
    if later.state not in (ActionState.SUCCESS, ActionState.VERIFIED) or getattr(later, "verification_failed", False):
        return False

    failed_is_inspect = _is_inspection_action(failed.tool_name, failed.arguments)
    later_is_inspect = _is_inspection_action(later.tool_name, later.arguments)

    # 1. An inspection must never resolve a mutation or execution failure
    if not failed_is_inspect and later_is_inspect:
        return False

    target_failed = _extract_action_target(failed.tool_name, failed.arguments)
    target_later = _extract_action_target(later.tool_name, later.arguments)

    op_failed = _extract_action_operation(failed.tool_name, failed.arguments)
    op_later = _extract_action_operation(later.tool_name, later.arguments)

    failed_ver_targets = (
        _normalize_target_list(getattr(failed, "failed_targets", []))
        if getattr(failed, "verification_failed", False)
        else []
    )
    later_targets = _collect_later_resolution_targets(later, later_is_inspect)

    # 2. When verification confirms the target resource, it resolves the target's failure
    if later.tool_name == "verify_outcome" and (later.verified or later.state == ActionState.VERIFIED):
        if failed_ver_targets:
            return bool(later_targets) and all(
                any(
                    targets_match(ft, lt)
                    or criterion_applies_to_target(
                        {"check": later.arguments.get("check"), "target": lt},
                        ft,
                    )
                    for lt in later_targets
                )
                for ft in failed_ver_targets
            )
        if target_failed and later_targets:
            return any(
                targets_match(target_failed, lt)
                or criterion_applies_to_target(
                    {"check": later.arguments.get("check"), "target": later.arguments.get("target")},
                    target_failed,
                )
                for lt in later_targets
            )
        if not target_failed:
            return True
        return False

    # 2.5 Structured verification_failed targets: resolved only when all failed targets are verified/acted upon
    if failed_ver_targets:
        if later_is_inspect or not later_targets:
            return False
        return all(
            any(targets_match(ft, lt) for lt in later_targets)
            for ft in failed_ver_targets
        )

    # 3. If failed action had a specific target:
    if target_failed is not None:
        if not later_targets or not any(targets_match(target_failed, lt) for lt in later_targets):
            # Different targets never resolve each other
            return False
        # Same target: operations must be compatible (e.g. write resolves write, append resolves append,
        # or any non-inspection corrective mutation/verification resolves a verification_failed state)
        if op_failed == op_later:
            return True
        if op_failed in ("write", "overwrite") and op_later in ("write", "overwrite"):
            return True
        if (getattr(failed, "verification_failed", False) or later.verified or later.state == ActionState.VERIFIED) and not later_is_inspect:
            return True
        if failed.tool_name == "manage_workspace_file" and later.tool_name in ("execute_command", "bash", "sh"):
            return True
        if failed.tool_name in ("execute_command", "bash", "sh") and later.tool_name in ("execute_command", "bash", "sh", "manage_workspace_file"):
            return True
        return False

    # 4. If failed action had no specific target (e.g. schema error, missing arguments, or untargeted shell command):
    if target_failed is None:
        if failed.tool_name == later.tool_name:
            if op_failed == op_later:
                return True
            if failed.tool_name in ("execute_command", "bash", "sh"):
                return True
            if failed_is_inspect and later_is_inspect:
                return True
            if not failed_is_inspect and not later_is_inspect:
                return True
        if failed.tool_name in ("execute_command", "bash", "sh") and later.tool_name in ("execute_command", "bash", "sh"):
            return True
        return False

    return False


class TurnExecutionLedger:
    """Authoritative ledger recording all proposed, dispatched, and observed actions."""

    def __init__(self) -> None:
        self._records: Dict[str, ActionRecord] = {}
        self._order: List[str] = []
        self._verified_targets: Set[str] = set()
        self._pending_call_ids: List[str] = []

    def record_target_verified(
        self,
        target: str,
        detail: Optional[str] = None,
        check_type: Optional[str] = None,
    ) -> None:
        """Record a target resource as independently verified."""
        if target:
            norm = canonicalize_target(target, check_type=check_type)
            if any(targets_match(target, vt, self._verified_targets, check_type=check_type) for vt in self._verified_targets):
                return
            self._verified_targets.add(norm)

    def invalidate_target_verification(
        self,
        target: str,
        check_type: Optional[str] = None,
    ) -> None:
        """Remove a target from the verified set when subsequent mutations or failed checks invalidate it."""
        if not target:
            return
        to_remove = [
            vt for vt in list(self._verified_targets)
            if targets_match(target, vt, self._verified_targets, check_type=check_type)
        ]
        for vt in to_remove:
            self._verified_targets.discard(vt)
        for cid in self._order:
            rec = self._records.get(cid)
            if not rec or rec.state != ActionState.VERIFIED:
                continue
            rec_target = _extract_action_target(rec.tool_name, rec.arguments)
            if rec_target and targets_match(target, rec_target, check_type=check_type):
                self._records[cid] = ActionRecord(
                    call_id=rec.call_id,
                    tool_name=rec.tool_name,
                    arguments=rec.arguments,
                    state=ActionState.UNVERIFIED,
                    exit_code=rec.exit_code,
                    error=rec.error,
                    verified=False,
                    verification_detail=rec.verification_detail,
                    timestamp=rec.timestamp,
                    terminal=rec.terminal,
                    recovery_attempts=rec.recovery_attempts,
                    turn=rec.turn,
                    raw_name=rec.raw_name,
                    rejection_reason=rec.rejection_reason,
                    verification_failed=rec.verification_failed,
                )

    def is_target_verified(self, target: str, check_type: Optional[str] = None) -> bool:
        """Check if a target resource has been verified without colliding distinct directory paths."""
        if not target:
            return False
        norm = canonicalize_target(target, check_type=check_type)
        if norm in self._verified_targets:
            return True
        for vt in self._verified_targets:
            if targets_match(target, vt, self._verified_targets, check_type=check_type):
                return True
        return False

    def get_verified_targets(self) -> List[str]:
        """Return list of verified targets."""
        return sorted(list(self._verified_targets))

    def _validate_transition(
        self,
        call_id: str,
        target_state: ActionState,
        allowed_from: Tuple[ActionState, ...],
        *,
        require_existing: bool = True,
    ) -> Optional[ActionRecord]:
        old = self._records.get(call_id)
        if old is None:
            if require_existing:
                raise InvalidStateTransitionError(
                    f"Cannot transition unknown action '{call_id}' to '{target_state.value}'."
                )
            return None
        if old.state not in allowed_from:
            raise InvalidStateTransitionError(
                f"Invalid lifecycle transition for '{call_id}': '{old.state.value}' -> '{target_state.value}'."
            )
        return old

    def record_proposed(
        self,
        call_id: str,
        tool_name: str,
        arguments: Dict[str, Any],
        turn: Optional[int] = None,
        raw_name: Optional[str] = None,
    ) -> None:
        """Record an action proposal."""
        if call_id in self._records and self._records[call_id].state != ActionState.PROPOSED:
            raise InvalidStateTransitionError(
                f"Cannot re-propose action '{call_id}' already in state '{self._records[call_id].state.value}'."
            )
        rec = ActionRecord(
            call_id=call_id,
            tool_name=tool_name,
            arguments=dict(arguments) if isinstance(arguments, dict) else {},
            state=ActionState.PROPOSED,
            turn=turn,
            raw_name=raw_name or tool_name,
        )
        self._records[call_id] = rec
        if call_id not in self._order:
            self._order.append(call_id)
        if call_id not in self._pending_call_ids:
            self._pending_call_ids.append(call_id)

    def get_pending_proposals(self) -> List[ActionRecord]:
        """Return list of proposed actions awaiting validation and dispatch."""
        return [
            self._records[cid]
            for cid in self._pending_call_ids
            if cid in self._records and self._records[cid].state == ActionState.PROPOSED
        ]

    def has_pending_proposals(self) -> bool:
        """Check if any actions remain in pending proposal state."""
        return len(self.get_pending_proposals()) > 0

    def pop_next_pending_proposal(self) -> Optional[ActionRecord]:
        """Pop and return the next pending action proposal."""
        while self._pending_call_ids:
            cid = self._pending_call_ids.pop(0)
            if cid in self._records and self._records[cid].state == ActionState.PROPOSED:
                return self._records[cid]
        return None

    def peek_next_pending_proposal(self) -> Optional[ActionRecord]:
        """Return the next pending action proposal without popping."""
        for cid in self._pending_call_ids:
            if cid in self._records and self._records[cid].state == ActionState.PROPOSED:
                return self._records[cid]
        return None

    def discard_pending_proposals(self, reason: str = "task_terminated") -> List[str]:
        """Cancel and clear all remaining pending proposals with an explicit reason."""
        discarded: List[str] = []
        pending_cids = list(self._pending_call_ids)
        self._pending_call_ids.clear()
        for cid in pending_cids:
            if cid in self._records and self._records[cid].state == ActionState.PROPOSED:
                self.record_cancelled(cid, reason=f"Pending call discarded: {reason}")
                discarded.append(cid)
        return discarded

    def cancel_pending_proposal(self, call_id: str, reason: str = "cancelled") -> None:
        """Cancel a specific pending proposal."""
        if call_id in self._pending_call_ids:
            self._pending_call_ids.remove(call_id)
        if call_id in self._records and self._records[call_id].state == ActionState.PROPOSED:
            self.record_cancelled(call_id, reason=reason)

    def record_validated(self, call_id: str) -> None:
        """Record that a proposed action passed parameter and permission validation."""
        old = self._validate_transition(
            call_id,
            ActionState.VALIDATED,
            (ActionState.PROPOSED, ActionState.VALIDATED),
        )
        if old is not None:
            self._records[call_id] = ActionRecord(
                call_id=old.call_id,
                tool_name=old.tool_name,
                arguments=old.arguments,
                state=ActionState.VALIDATED,
                exit_code=old.exit_code,
                error=old.error,
                verified=old.verified,
                verification_detail=old.verification_detail,
                timestamp=old.timestamp,
                turn=old.turn,
                raw_name=old.raw_name,
                rejection_reason=old.rejection_reason,
                verification_failed=old.verification_failed,
            )

    def record_dispatched(self, call_id: str) -> None:
        """Record that an action was dispatched for host execution."""
        old = self._validate_transition(
            call_id,
            ActionState.DISPATCHED,
            (ActionState.PROPOSED, ActionState.VALIDATED, ActionState.AUTHORIZED, ActionState.DISPATCHED),
            require_existing=False,
        )
        if call_id in self._pending_call_ids:
            self._pending_call_ids.remove(call_id)
        if old is not None:
            self._records[call_id] = ActionRecord(
                call_id=old.call_id,
                tool_name=old.tool_name,
                arguments=old.arguments,
                state=ActionState.DISPATCHED,
                exit_code=old.exit_code,
                error=old.error,
                verified=old.verified,
                verification_detail=old.verification_detail,
                timestamp=old.timestamp,
                turn=old.turn,
                raw_name=old.raw_name,
                rejection_reason=old.rejection_reason,
                verification_failed=old.verification_failed,
            )
        else:
            rec = ActionRecord(
                call_id=call_id,
                tool_name="",
                arguments={},
                state=ActionState.DISPATCHED,
            )
            self._records[call_id] = rec
            self._order.append(call_id)

    def record_authorized(self, call_id: str) -> None:
        """Record that an action was authorized for host execution."""
        old = self._validate_transition(
            call_id,
            ActionState.AUTHORIZED,
            (ActionState.PROPOSED, ActionState.VALIDATED, ActionState.AUTHORIZED),
        )
        if old is not None:
            self._records[call_id] = ActionRecord(
                call_id=old.call_id,
                tool_name=old.tool_name,
                arguments=old.arguments,
                state=ActionState.AUTHORIZED,
                exit_code=old.exit_code,
                error=old.error,
                verified=old.verified,
                verification_detail=old.verification_detail,
                timestamp=old.timestamp,
                turn=old.turn,
                raw_name=old.raw_name,
                rejection_reason=old.rejection_reason,
                verification_failed=old.verification_failed,
            )

    def record_executed(self, call_id: str) -> None:
        """Record that an action finished host execution before outcome verification."""
        old = self._validate_transition(
            call_id,
            ActionState.EXECUTED,
            (
                ActionState.PROPOSED,
                ActionState.VALIDATED,
                ActionState.AUTHORIZED,
                ActionState.DISPATCHED,
                ActionState.EXECUTED,
            ),
        )
        if old is not None:
            self._records[call_id] = ActionRecord(
                call_id=old.call_id,
                tool_name=old.tool_name,
                arguments=old.arguments,
                state=ActionState.EXECUTED,
                exit_code=old.exit_code,
                error=old.error,
                verified=old.verified,
                verification_detail=old.verification_detail,
                timestamp=old.timestamp,
                turn=old.turn,
                raw_name=old.raw_name,
                rejection_reason=old.rejection_reason,
                verification_failed=old.verification_failed,
            )

    def record_rejected(self, call_id: str, reason: str = "rejected") -> None:
        """Record that an action proposal was rejected."""
        if call_id in self._pending_call_ids:
            self._pending_call_ids.remove(call_id)
        old = self._records.get(call_id)
        if old is not None and old.state == ActionState.VERIFIED:
            raise InvalidStateTransitionError(
                f"Invalid lifecycle transition for '{call_id}': cannot reject already verified action."
            )
        tool_name = old.tool_name if old else ""
        arguments = old.arguments if old else {}
        orig_ts = old.timestamp if old else time.time()
        safe_reason = _redact_sensitive_text(reason) or "rejected"
        self._records[call_id] = ActionRecord(
            call_id=call_id,
            tool_name=tool_name,
            arguments=arguments,
            state=ActionState.REJECTED,
            exit_code=old.exit_code if old else None,
            error=safe_reason,
            verified=False,
            verification_detail=None,
            timestamp=orig_ts,
            turn=old.turn if old else None,
            raw_name=old.raw_name if old else None,
            rejection_reason=safe_reason,
        )
        if call_id not in self._order:
            self._order.append(call_id)

    def record_cancelled(self, call_id: str, reason: Optional[str] = None) -> None:
        """Record that an action was cancelled or aborted before or during execution."""
        if call_id in self._pending_call_ids:
            self._pending_call_ids.remove(call_id)
        old = self._records.get(call_id)
        if old is not None and old.state == ActionState.VERIFIED:
            raise InvalidStateTransitionError(
                f"Invalid lifecycle transition for '{call_id}': cannot cancel already verified action."
            )
        tool_name = old.tool_name if old else ""
        arguments = old.arguments if old else {}
        orig_ts = old.timestamp if old else time.time()
        safe_reason = _redact_sensitive_text(reason) if reason is not None else None

        self._records[call_id] = ActionRecord(
            call_id=call_id,
            tool_name=tool_name,
            arguments=arguments,
            state=ActionState.CANCELLED,
            exit_code=old.exit_code if old else None,
            error=safe_reason or (old.error if old else "Action cancelled"),
            verified=False,
            verification_detail=None,
            timestamp=orig_ts,
            turn=old.turn if old else None,
            raw_name=old.raw_name if old else None,
            rejection_reason=safe_reason or (old.rejection_reason if old else None),
        )
        if call_id not in self._order:
            self._order.append(call_id)

    def record_partially_completed(
        self,
        call_id: str,
        output: Any = None,
        error: Optional[str] = None,
        exit_code: Optional[int] = None,
    ) -> None:
        """Record that an action executed partially or was truncated, remaining distinct from full success."""
        old = self._validate_transition(
            call_id,
            ActionState.PARTIALLY_COMPLETED,
            (
                ActionState.PROPOSED,
                ActionState.VALIDATED,
                ActionState.AUTHORIZED,
                ActionState.DISPATCHED,
                ActionState.EXECUTED,
                ActionState.PARTIALLY_COMPLETED,
            ),
            require_existing=False,
        )
        tool_name = old.tool_name if old else ""
        arguments = old.arguments if old else {}
        orig_ts = old.timestamp if old else time.time()
        safe_err = _redact_sensitive_text(error) if error is not None else (old.error if old else None)

        self._records[call_id] = ActionRecord(
            call_id=call_id,
            tool_name=tool_name,
            arguments=arguments,
            state=ActionState.PARTIALLY_COMPLETED,
            exit_code=exit_code if exit_code is not None else (old.exit_code if old else None),
            error=safe_err,
            verified=False,
            verification_detail="Partially completed: operation did not finish to full verified state",
            timestamp=orig_ts,
        )
        if call_id not in self._order:
            self._order.append(call_id)

    def record_observed(
        self,
        call_id: str,
        output: Any = None,
        exit_code: Optional[int] = 0,
    ) -> None:
        """Record tool observation where output was received but lacks independent verification."""
        old = self._validate_transition(
            call_id,
            ActionState.OBSERVED,
            (
                ActionState.PROPOSED,
                ActionState.VALIDATED,
                ActionState.AUTHORIZED,
                ActionState.DISPATCHED,
                ActionState.EXECUTED,
                ActionState.OBSERVED,
                ActionState.SUCCESS,
            ),
        )
        if old is None:
            return
        self._records[call_id] = ActionRecord(
            call_id=call_id,
            tool_name=old.tool_name,
            arguments=old.arguments,
            state=ActionState.OBSERVED,
            exit_code=exit_code,
            verified=False,
            verification_detail="Observed output without independent outcome verification",
            timestamp=old.timestamp,
            turn=old.turn,
            raw_name=old.raw_name,
        )

    def record_unverified(
        self,
        call_id: str,
        reason: Optional[str] = "Lacks independent verification",
    ) -> None:
        """Record that an action completed without independent verification evidence."""
        old = self._validate_transition(
            call_id,
            ActionState.UNVERIFIED,
            (
                ActionState.DISPATCHED,
                ActionState.EXECUTED,
                ActionState.OBSERVED,
                ActionState.SUCCESS,
                ActionState.UNVERIFIED,
                ActionState.PARTIALLY_COMPLETED,
            ),
        )
        if old is None:
            return
        self._records[call_id] = ActionRecord(
            call_id=call_id,
            tool_name=old.tool_name,
            arguments=old.arguments,
            state=ActionState.UNVERIFIED,
            exit_code=old.exit_code if old.exit_code is not None else 0,
            verified=False,
            verification_detail=reason,
            timestamp=old.timestamp,
            turn=old.turn,
            raw_name=old.raw_name,
        )

    def record_verified(
        self,
        call_id: str,
        detail: Optional[str] = None,
        verified_targets: Optional[List[str]] = None,
    ) -> None:
        """Promote an executed/observed action to VERIFIED after independent verification passes."""
        old = self._validate_transition(
            call_id,
            ActionState.VERIFIED,
            (
                ActionState.DISPATCHED,
                ActionState.EXECUTED,
                ActionState.OBSERVED,
                ActionState.SUCCESS,
                ActionState.UNVERIFIED,
                ActionState.VERIFIED,
            ),
        )
        if old is None:
            return
        if old.verification_failed or (old.exit_code is not None and old.exit_code != 0):
            raise InvalidStateTransitionError(
                f"Cannot verify action '{call_id}' with failed verification or non-zero exit code."
            )
        merged_targets = list(old.verified_targets)
        for t in _normalize_target_list(verified_targets):
            if t not in merged_targets:
                merged_targets.append(t)
        self._records[call_id] = ActionRecord(
            call_id=call_id,
            tool_name=old.tool_name,
            arguments=old.arguments,
            state=ActionState.VERIFIED,
            exit_code=old.exit_code,
            error=None,
            verified=True,
            verification_detail=detail or old.verification_detail or "Independently verified",
            timestamp=old.timestamp,
            turn=old.turn,
            raw_name=old.raw_name,
            rejection_reason=old.rejection_reason,
            verification_failed=False,
            failed_targets=[],
            verified_targets=merged_targets,
        )

    def record_observation(
        self,
        call_id: str,
        status: str,
        output: Any,
        error: Optional[str] = None,
        exit_code: Optional[int] = None,
        verified: bool = False,
        verification_detail: Optional[str] = None,
        verification_failed: bool = False,
        failed_targets: Optional[List[str]] = None,
        verified_targets: Optional[List[str]] = None,
        rejection_reason: Optional[str] = None,
    ) -> None:
        """Record host observation and final state for an action."""
        old = self._records.get(call_id)
        if old and old.state == ActionState.CANCELLED:
            # Cancelled actions cannot be promoted to success
            return

        effective_error = error
        effective_exit_code = exit_code
        effective_verified = isinstance(verified, bool) and verified is True
        effective_detail = verification_detail
        effective_verification_failed = isinstance(verification_failed, bool) and verification_failed is True
        eff_failed_targets = _normalize_target_list(failed_targets)
        eff_verified_targets = _normalize_target_list(verified_targets)
        effective_rejection_reason = rejection_reason or (old.rejection_reason if old else None)

        parsed_output: Optional[Dict[str, Any]] = None
        if isinstance(output, dict):
            parsed_output = output
        elif isinstance(output, str):
            trimmed = output.strip()
            if trimmed.startswith("{") and trimmed.endswith("}"):
                try:
                    parsed = json.loads(trimmed)
                    if isinstance(parsed, dict):
                        parsed_output = parsed
                except Exception:
                    pass
            elif trimmed.startswith("[TOOL OBSERVATION:"):
                if "[Outcome Verification Failed]:" in trimmed or re.search(r"^- Verified:\s*FALSE\b", trimmed, re.MULTILINE):
                    effective_verification_failed = True

        if parsed_output is not None:
            if effective_exit_code is None and "exit_code" in parsed_output and not isinstance(parsed_output["exit_code"], bool):
                try:
                    effective_exit_code = int(parsed_output["exit_code"])
                except Exception:
                    pass
            if not effective_error:
                effective_error = parsed_output.get("error") or parsed_output.get("detail")
            raw_p_ver = parsed_output.get("verified")
            if not effective_verified and isinstance(raw_p_ver, bool) and raw_p_ver is True:
                effective_verified = True
            raw_p_vf = parsed_output.get("verification_failed")
            if (isinstance(raw_p_vf, bool) and raw_p_vf is True) or (
                isinstance(raw_p_ver, bool)
                and raw_p_ver is False
                and any(k in parsed_output for k in ("failed_criterion", "verifier_reason", "required_correction"))
            ):
                effective_verification_failed = True
            if not eff_failed_targets and isinstance(parsed_output.get("failed_targets"), list):
                eff_failed_targets = _normalize_target_list(parsed_output.get("failed_targets"))
            if not eff_verified_targets and isinstance(parsed_output.get("verified_targets"), list):
                eff_verified_targets = _normalize_target_list(parsed_output.get("verified_targets"))
            if not effective_detail and parsed_output.get("verification_detail"):
                effective_detail = str(parsed_output.get("verification_detail"))
            if parsed_output.get("status") in ("error", "failed", "rejected", "tool_execution_failed"):
                status = "error"
            if not effective_rejection_reason and parsed_output.get("status") == "rejected" and parsed_output.get("reason"):
                effective_rejection_reason = str(parsed_output.get("reason"))
            if parsed_output.get("success") is False:
                status = "error"

        is_failure = (
            status in ("error", "failed", "rejected", "tool_execution_failed")
            or (effective_exit_code is not None and effective_exit_code != 0)
            or bool(effective_error and status != "success")
        )

        if is_failure:
            final_state = ActionState.FAILED
            effective_verified = False  # NEVER allow verified to be True on failure
            effective_verification_failed = False
            eff_failed_targets = []
            eff_verified_targets = []
        elif effective_verified and not effective_verification_failed:
            final_state = ActionState.VERIFIED
            effective_verification_failed = False
            eff_failed_targets = []
        else:
            final_state = ActionState.SUCCESS
            effective_verified = False
            eff_verified_targets = []
            if not effective_verification_failed:
                eff_failed_targets = []

        if call_id in self._pending_call_ids:
            self._pending_call_ids.remove(call_id)

        tool_name = old.tool_name if old else ""
        arguments = old.arguments if old else {}
        orig_ts = old.timestamp if old else time.time()

        self._records[call_id] = ActionRecord(
            call_id=call_id,
            tool_name=tool_name,
            arguments=arguments,
            state=final_state,
            exit_code=effective_exit_code,
            error=_redact_sensitive_text(effective_error),
            verified=effective_verified,
            verification_detail=effective_detail,
            timestamp=orig_ts,
            turn=old.turn if old else None,
            raw_name=old.raw_name if old else None,
            rejection_reason=effective_rejection_reason,
            verification_failed=effective_verification_failed,
            failed_targets=eff_failed_targets,
            verified_targets=eff_verified_targets,
        )
        if call_id not in self._order:
            self._order.append(call_id)

    def get_record(self, call_id: str) -> Optional[ActionRecord]:
        """Return an ActionRecord by call identity, or None if not recorded."""
        return self._records.get(call_id)

    def actions_for_target(self, target: str) -> List[ActionRecord]:
        """Query ordered ActionRecords matching a target resource path/name."""
        if not target or not isinstance(target, str):
            return []
        canon_target = canonicalize_target(target)
        matched: List[ActionRecord] = []
        for cid in self._order:
            rec = self._records[cid]
            rec_target = _extract_action_target(rec.tool_name, rec.arguments)
            candidates = list(rec.verified_targets) + list(rec.failed_targets)
            if rec_target:
                candidates.append(rec_target)
            if any(c == canon_target or targets_match(target, c) for c in candidates if c):
                matched.append(rec)
        return matched

    def total_actions(self) -> int:
        """Count of dispatched or finished tool calls."""
        return sum(
            1
            for cid in self._order
            if self._records[cid].state
            in (
                ActionState.DISPATCHED,
                ActionState.SUCCESS,
                ActionState.FAILED,
                ActionState.VERIFIED,
                ActionState.CANCELLED,
                ActionState.PARTIALLY_COMPLETED,
                ActionState.OBSERVED,
                ActionState.UNVERIFIED,
            )
        )

    def successful_actions(self) -> List[ActionRecord]:
        """List of actions that terminated in success, verified, or observed state."""
        return [
            self._records[cid]
            for cid in self._order
            if self._records[cid].state in (ActionState.SUCCESS, ActionState.VERIFIED, ActionState.OBSERVED)
        ]

    def failed_actions(self) -> List[ActionRecord]:
        """List of actions that terminated in failed state."""
        return [
            self._records[cid]
            for cid in self._order
            if self._records[cid].state == ActionState.FAILED
        ]

    def cancelled_actions(self) -> List[ActionRecord]:
        """List of actions that were cancelled."""
        return [
            self._records[cid]
            for cid in self._order
            if self._records[cid].state == ActionState.CANCELLED
        ]

    def partially_completed_actions(self) -> List[ActionRecord]:
        """List of actions that terminated in partially completed state."""
        return [
            self._records[cid]
            for cid in self._order
            if self._records[cid].state == ActionState.PARTIALLY_COMPLETED
        ]

    def verified_actions(self) -> List[ActionRecord]:
        """List of actions that were independently verified."""
        return [
            self._records[cid]
            for cid in self._order
            if self._records[cid].state == ActionState.VERIFIED or self._records[cid].verified
        ]

    def unverified_actions(self) -> List[ActionRecord]:
        """List of actions that executed without independent verification."""
        return [
            self._records[cid]
            for cid in self._order
            if self._records[cid].state in (ActionState.UNVERIFIED, ActionState.OBSERVED, ActionState.SUCCESS)
            and not self._records[cid].verified
        ]

    def has_identical_successful_action(
        self, tool_name: str, arguments: Any
    ) -> bool:
        """Check if an action with identical tool name and canonical arguments has already succeeded."""
        target_canon = _canonicalize_args(arguments)
        for cid in self._order:
            rec = self._records[cid]
            if (
                rec.tool_name == tool_name
                and rec.state in (ActionState.SUCCESS, ActionState.VERIFIED)
                and not getattr(rec, "verification_failed", False)
            ):
                if _canonicalize_args(rec.arguments) == target_canon:
                    return True
        return False

    def unresolved_failures(self) -> List[ActionRecord]:
        """Actions that failed, were cancelled, partially completed, or failed outcome verification and were not subsequently resolved."""
        unresolved: List[ActionRecord] = []
        for i, cid in enumerate(self._order):
            rec = self._records[cid]
            if (
                rec.state not in (ActionState.FAILED, ActionState.CANCELLED, ActionState.PARTIALLY_COMPLETED)
                and not rec.verification_failed
            ):
                continue

            # If the action was rejected because the outcome was already verified, duplicate suppressed,
            # or was an unexecuted pending proposal discarded/superseded/abandoned,
            # this does not leave the host machine state in an execution failure.
            if rec.error and (
                "outcome_already_verified" in str(rec.error)
                or "prohibited_by_user" in str(rec.error)
                or "already satisfies the verified acceptance criteria" in str(rec.error)
                or "identical parameters has already succeeded" in str(rec.error)
                or "Pending call discarded" in str(rec.error)
                or "unconfirmed_by_model" in str(rec.error)
                or "superseded_by_model" in str(rec.error)
                or "abandoned_by_model" in str(rec.error)
            ):
                continue

            # Note: A prior successful action MUST NOT hide a distinct later failed action.
            # A failure at index i can only be resolved by matching corrective evidence from a subsequent action at j > i.
            resolved = False
            later_records = [self._records[self._order[j]] for j in range(i + 1, len(self._order))]
            for later in later_records:
                if _is_action_resolved_by_later(rec, later):
                    resolved = True
                    break

            if not resolved and getattr(rec, "verification_failed", False):
                rec_failed_targets = _normalize_target_list(getattr(rec, "failed_targets", []))
                if rec_failed_targets:
                    all_later_targets: List[str] = []
                    for later in later_records:
                        if later.state not in (ActionState.SUCCESS, ActionState.VERIFIED) or getattr(later, "verification_failed", False):
                            continue
                        later_is_inspect = _is_inspection_action(later.tool_name, later.arguments)
                        if not later_is_inspect or later.tool_name == "verify_outcome":
                            all_later_targets.extend(_collect_later_resolution_targets(later, later_is_inspect))
                    if all_later_targets and all(
                        any(targets_match(ft, lt) for lt in all_later_targets)
                        for ft in rec_failed_targets
                    ):
                        resolved = True

            if not resolved:
                unresolved.append(rec)
        return unresolved

    def has_unresolved_failures(self) -> bool:
        """Whether any failed action remains unresolved."""
        return len(self.unresolved_failures()) > 0

    def increment_recovery_attempts(self, call_id: Optional[str] = None) -> int:
        """Increment and return the recovery attempts counter for a target action."""
        if not call_id:
            unresolved = self.unresolved_failures()
            if not unresolved:
                return 0
            call_id = unresolved[-1].call_id

        if call_id in self._records:
            old = self._records[call_id]
            new_attempts = old.recovery_attempts + 1
            self._records[call_id] = ActionRecord(
                call_id=old.call_id,
                tool_name=old.tool_name,
                arguments=old.arguments,
                state=old.state,
                exit_code=old.exit_code,
                error=old.error,
                verified=old.verified,
                verification_detail=old.verification_detail,
                timestamp=old.timestamp,
                terminal=old.terminal,
                recovery_attempts=new_attempts,
                turn=old.turn,
                raw_name=old.raw_name,
                rejection_reason=old.rejection_reason,
                verification_failed=old.verification_failed,
            )
            return new_attempts
        return 0

    def get_recovery_attempts(self, call_id: Optional[str] = None) -> int:
        """Get recovery attempts count for a specific action or the most recent unresolved failure."""
        if not call_id:
            unresolved = self.unresolved_failures()
            if not unresolved:
                return 0
            call_id = unresolved[-1].call_id
        if call_id in self._records:
            return self._records[call_id].recovery_attempts
        return 0

    def mark_terminal(self, call_id: Optional[str] = None) -> None:
        """Mark an action as a terminal blocker."""
        if not call_id:
            unresolved = self.unresolved_failures()
            if not unresolved:
                return
            call_id = unresolved[-1].call_id

        if call_id in self._records:
            old = self._records[call_id]
            self._records[call_id] = ActionRecord(
                call_id=old.call_id,
                tool_name=old.tool_name,
                arguments=old.arguments,
                state=ActionState.FAILED if not old.verification_failed else old.state,
                exit_code=old.exit_code,
                error=old.error,
                verified=old.verified,
                verification_detail=old.verification_detail,
                timestamp=old.timestamp,
                terminal=True,
                recovery_attempts=old.recovery_attempts,
                turn=old.turn,
                raw_name=old.raw_name,
                rejection_reason=old.rejection_reason,
                verification_failed=old.verification_failed,
            )

    def is_terminal(self, call_id: Optional[str] = None) -> bool:
        """Check if an action is marked as a terminal blocker."""
        if not call_id:
            unresolved = self.unresolved_failures()
            if not unresolved:
                return False
            call_id = unresolved[-1].call_id
        if call_id in self._records:
            return self._records[call_id].terminal
        return False

    def hydrate_from_messages(self, messages: List[Dict[str, Any]]) -> None:
        """Hydrate ledger from existing message history if unpopulated."""
        if not messages:
            return
        from zelyr.engine.observation_framing import ObservationFramingEngine

        for i, m in enumerate(messages):
            role = m.get("role")
            if role == "assistant" and m.get("tool_calls"):
                for tc in m["tool_calls"]:
                    cid = tc.get("id") or f"msg_tc_{i}"
                    fn = tc.get("function", {})
                    fname = fn.get("name", "") or tc.get("name", "")
                    fargs = fn.get("arguments", "{}") if "function" in tc else tc.get("arguments", "{}")
                    try:
                        fargs_dict = json.loads(fargs) if isinstance(fargs, str) else dict(fargs)
                    except Exception:
                        fargs_dict = {}
                    if cid not in self._records:
                        self.record_proposed(cid, fname, fargs_dict)
                        self.record_dispatched(cid)
            elif role == "tool":
                c = str(m.get("content", ""))
                parsed = ObservationFramingEngine.parse_observation(c)
                cid = m.get("tool_call_id") or ""
                if not cid and i > 0 and messages[i - 1].get("role") == "assistant":
                    prev_tcs = messages[i - 1].get("tool_calls", [])
                    if prev_tcs:
                        cid = prev_tcs[0].get("id") or ""
                if not cid:
                    cid = f"tool_msg_{i}"

                tname = m.get("name") or parsed.get("tool") or parsed.get("tool_name") or "tool_action"
                if cid not in self._records:
                    self.record_proposed(cid, tname, {})
                    self.record_dispatched(cid)
                elif not self._records[cid].tool_name and tname:
                    self.record_proposed(cid, tname, self._records[cid].arguments)

                status = "error" if (
                    parsed.get("status") == "error"
                    or m.get("status") == "error"
                    or (parsed.get("exit_code") is not None and parsed.get("exit_code") != 0)
                ) else "success"
                err = parsed.get("error") or m.get("error")
                ecode = parsed.get("exit_code") if parsed.get("exit_code") is not None else m.get("exit_code")
                ver_val = parsed.get("verified") if parsed.get("verified") is not None else m.get("verified")
                verified = isinstance(ver_val, bool) and ver_val is True
                verification_failed = bool(
                    status == "success"
                    and (
                        parsed.get("verification_failed") is True
                        or m.get("verification_failed") is True
                        or (
                            "verification_failed" not in parsed
                            and "verification_failed" not in m
                            and isinstance(ver_val, bool)
                            and ver_val is False
                        )
                    )
                )
                v_detail = parsed.get("verification_detail") or m.get("verification_detail")
                self.record_observation(
                    cid,
                    status=status,
                    output=c,
                    error=err,
                    exit_code=ecode,
                    verified=verified,
                    verification_detail=v_detail,
                    verification_failed=verification_failed,
                )
                rec = self._records.get(cid)
                if rec is not None:
                    tgt = _extract_action_target(rec.tool_name, rec.arguments) or parsed.get("target")
                    if tgt and isinstance(tgt, str):
                        if rec.state == ActionState.VERIFIED and rec.verified and not rec.verification_failed:
                            self.record_target_verified(tgt, detail=v_detail)
                        elif not _is_inspection_action(rec.tool_name, rec.arguments):
                            self.invalidate_target_verification(tgt)

    def to_dict(self) -> Dict[str, Any]:
        """Structured summary for CLI/headless reporting."""
        records = [self._records[cid] for cid in self._order]
        return {
            "total_actions": self.total_actions(),
            "successful_actions": [a.to_dict() for a in self.successful_actions()],
            "failed_actions": [a.to_dict() for a in self.failed_actions()],
            "unresolved_failures": [a.to_dict() for a in self.unresolved_failures()],
            "has_unresolved_failures": self.has_unresolved_failures(),
            "pending_proposals": [a.to_dict() for a in self.get_pending_proposals()],
            "records": [r.to_dict() for r in records],
        }


# False-success and non-actuation detection rules

FALSE_SUCCESS_PATTERNS = [
    r"\bsuccessfully\b",
    r"\bsuccessful\b",
    r"\bcompleted\b",
    r"\ball set\b",
    r"\bfixed the issue\b",
    r"\bhas been resolved\b",
    r"\bhave resolved\b",
    r"\bfinished\b",
    r"\bit is done\b",
    r"\bi have done\b",
    r"\beverything is ready\b",
    r"\bworking now\b",
]

TRUTHFUL_FAILURE_PATTERNS = [
    r"\b(failed|failure)\b",
    r"\b(unable to|could not|cannot|can't)\b",
    r"\berror\b",
    r"\bnot found\b",
    r"\bdenied\b",
    r"\bblocked\b",
    r"\bdid not (succeed|work|finish)\b",
    r"\bfailed with\b",
    r"\bnon-zero exit code\b",
]

APOLOGY_OR_DELEGATION_PATTERNS = [
    r"\b(apologize|apologies|apology|sorry)\b",
    r"\bplease\s+(?:manually\s+)?(?:inspect|verify|check|run|install|try)\b",
    r"\byou\s+should\s+(?:manually\s+)?(?:inspect|verify|check|run|install|try)\b",
]

CONSULTANT_DEFERRAL_PATTERNS = [
    r"\bplease\s+(?:check|verify|ensure|provide|inspect|try|run)\b",
    r"\byou\s+(?:can|could|should|might)\s+(?:try|check|verify|ensure|run)\b",
    r"\blet\s+me\s+know\b",
    r"\b(i\s+apologize|i\s+am\s+sorry|i['’]m\s+sorry)\b",
    r"\bmanual\s+intervention\b",
    r"\bplease\s+(?:manually|feel\s+free\s+to)\b",
    r"\byou\s+may\s+want\s+to\b",
    r"\bcheck\s+the\s+(?:file|path|spelling|permissions|syntax)\b",
]


def detect_consultant_deferral(text: str) -> bool:
    """Detect if model text offers conversational deferrals, apologies, or asks user intervention instead of acting."""
    if not text:
        return False
    lower = text.lower()
    return any(re.search(pat, lower) for pat in CONSULTANT_DEFERRAL_PATTERNS)


def is_unresolvable_terminal_blocker(
    error_type: Optional[str] = None,
    error_message: str = "",
    tool_name: str = "",
) -> bool:
    """Determine if a tool failure represents an unresolvable external environment boundary."""
    err_lower = (error_message or "").lower()
    etype_upper = (error_type or "").upper()
    if etype_upper == "PERMISSION_DENIED" or "permission denied" in err_lower or "operation not permitted" in err_lower:
        return True
    if any(k in err_lower for k in ("read-only file system", "hardware error", "no space left on device", "read-only")):
        return True
    return False

IMPERATIVE_ACTION_DIRECTIVES = [
    r"\b(execute|run|delete|remove|modify|install|touch|restart|kill|build)\b",
    r"\bcreate\s+(?:a\s+)?(?:new\s+)?(?:file|directory|folder|script|repo|branch|socket|service)\b",
    r"\bwrite\s+(?:to\s+)?(?:the\s+)?(?:file|disk)\b",
    r"\b(mkdir|rm\s+-|chmod|chown)\b",
]

INACTION_JUSTIFICATION_PATTERNS = [
    r"\b(unsafe|harmful|security policy|not authorized|permission denied)\b",
    r"\b(please specify|please provide|could you clarify|which file|what path)\b",
    r"\b(do not have access|tool is not available|cannot run in headless|no tool exists)\b",
    r"\b(i cannot|i am unable to)\s+(?:directly\s+)?(?:access|execute|interact)\b",
]


def is_truthful_failure_report(text: str) -> bool:
    """Check if model text truthfully acknowledges a failure or inability to proceed without evasion."""
    lower = text.lower()
    has_failure_indicator = any(re.search(kw, lower) for kw in TRUTHFUL_FAILURE_PATTERNS)
    if not has_failure_indicator:
        return False

    if any(re.search(p, lower) for p in APOLOGY_OR_DELEGATION_PATTERNS) or detect_consultant_deferral(text):
        return False

    cleaned_lower = re.sub(
        r"\b(not|cannot|could not|unable to|failed to)(?:\s+be)?\s+(?:successfully\s+)?(completed|finished|done|resolved|run|execute)\b",
        " failure_negated ",
        lower,
    )
    claims_success = any(re.search(s_kw, cleaned_lower) for s_kw in FALSE_SUCCESS_PATTERNS)
    return not claims_success


def detect_false_success(text: str, has_unresolved: bool) -> bool:
    """Detect if model claims false success in terminal text when unresolved failures exist."""
    if not has_unresolved:
        return False
    lower = text.lower()
    if is_truthful_failure_report(text):
        return False
    for kw in FALSE_SUCCESS_PATTERNS:
        if re.search(kw, lower):
            return True
    return False


def has_imperative_action_intent(prompt: str) -> bool:
    """Check if user prompt explicitly requests machine host actuation."""
    lower = prompt.lower()
    for pattern in IMPERATIVE_ACTION_DIRECTIVES:
        if re.search(pattern, lower):
            return True
    return False


def has_valid_inaction_justification(text: str) -> bool:
    """Check if model text provides a valid safety, clarification, or limitation reason for inaction."""
    lower = text.lower()
    for pattern in INACTION_JUSTIFICATION_PATTERNS:
        if re.search(pattern, lower):
            return True
    return False


KNOWN_VERIFICATION_CHECKS = {
    "file_exists", "exists",
    "file_contains", "contains",
    "file_checksum", "checksum", "sha256",
    "file_json", "json",
    "process_running", "process", "proc",
    "port_listening", "port", "listen",
    "command_output", "command", "cmd",
}


def _strip_matching_quotes(val: Optional[str]) -> Optional[str]:
    """Strip outermost matching single or double quotes from a string value."""
    if val is None:
        return None
    s = val.strip()
    if len(s) >= 2 and ((s[0] == '"' and s[-1] == '"') or (s[0] == "'" and s[-1] == "'")):
        return s[1:-1]
    return s


def _split_criteria_records(text: str) -> List[str]:
    """Split a composite criteria string into individual specification records.

    Splits on newlines, semicolons, and commas when followed by a criterion check name.
    Preserves delimiters inside quotes or escape sequences.
    """
    records: List[str] = []
    current: List[str] = []
    in_quote: Optional[str] = None
    escape = False
    i = 0
    n = len(text)

    while i < n:
        ch = text[i]
        if escape:
            current.append(ch)
            escape = False
            i += 1
            continue

        if ch == "\\":
            escape = True
            current.append(ch)
            i += 1
            continue

        if in_quote:
            if ch == in_quote:
                in_quote = None
            current.append(ch)
            i += 1
            continue

        if ch in ('"', "'"):
            in_quote = ch
            current.append(ch)
            i += 1
            continue

        if ch in (",", ";", "\n", "\r"):
            tail = text[i + 1:]
            m = re.match(r"^\s*([a-zA-Z0-9_\-]+)\s*:", tail)
            if m and (ch in ("\n", "\r") or m.group(1).lower() in KNOWN_VERIFICATION_CHECKS):
                rec = "".join(current).strip()
                if rec:
                    records.append(rec)
                current = []
                i += 1
                continue

        current.append(ch)
        i += 1

    rec = "".join(current).strip()
    if rec:
        records.append(rec)
    return records


def _parse_single_criterion_string(spec: str) -> Dict[str, Any]:
    """Parse a single criterion record string into check, target, and expected."""
    trimmed = spec.strip()
    if ":" not in trimmed:
        return {
            "check": trimmed,
            "target": "",
            "expected": None,
        }

    check, rest = trimmed.split(":", 1)
    check = check.strip()
    rest = rest.strip()

    # Check if target is a Windows drive path (e.g. C:\path\to\file or C:/path)
    win_drive_match = re.match(r"^([a-zA-Z]:[\\/][^:]*)(?::(.*))?$", rest)
    if win_drive_match:
        target = win_drive_match.group(1).strip()
        expected = win_drive_match.group(2) if win_drive_match.group(2) is not None else None
    elif rest.startswith(('"', "'")):
        # Target is quoted: find the matching closing quote
        quote_char = rest[0]
        end_quote = rest.find(quote_char, 1)
        if end_quote != -1:
            target = rest[1:end_quote]
            remainder = rest[end_quote + 1:].strip()
            if remainder.startswith(":"):
                expected = remainder[1:].strip()
            else:
                expected = None
        else:
            parts = rest.split(":", 1)
            target = parts[0].strip()
            expected = parts[1].strip() if len(parts) > 1 else None
    else:
        parts = rest.split(":", 1)
        target = parts[0].strip()
        expected = parts[1].strip() if len(parts) > 1 else None

    clean_target = _strip_matching_quotes(target) or ""
    clean_expected = _strip_matching_quotes(expected)

    return {
        "check": check,
        "target": clean_target,
        "expected": clean_expected,
    }


def parse_acceptance_criteria(criteria: Any) -> List[Dict[str, Any]]:
    """Parse acceptance criteria from str, dict, list, or JSON into standard check records."""
    if not criteria:
        return []

    # If JSON string representation of list or dict
    if isinstance(criteria, str) and criteria.strip().startswith(("[", "{")):
        try:
            decoded = json.loads(criteria.strip())
            if isinstance(decoded, (list, dict)):
                return parse_acceptance_criteria(decoded)
        except Exception:
            pass

    items = criteria if isinstance(criteria, list) else [criteria]
    parsed_list: List[Dict[str, Any]] = []

    for item in items:
        if isinstance(item, dict):
            parsed_list.append({
                "check": str(item.get("check", "")).strip(),
                "target": str(item.get("target", "")).strip(),
                "expected": item.get("expected"),
            })
        elif isinstance(item, str) and item.strip():
            raw_specs = _split_criteria_records(item)
            for spec in raw_specs:
                rec = _parse_single_criterion_string(spec)
                if rec:
                    parsed_list.append(rec)

    return parsed_list


_SEQ_CMD_VERBS_RE = (
    r"(?:run|execute|launch|start|invoke|build|test|validate|verify|cat|read|display|print|"
    r"restart|reload|stop|check|curl|fetch|ping|inspect|grep|head|tail|chmod|compile)"
)


def _has_subsequent_command_requirement(user_prompt: str) -> bool:
    """Return True only when the prompt expresses a genuine sequential follow-up command requirement.

    Does NOT match initial instructions ('Read the file', 'Run the script', 'Display the result',
    'Print the contents'), conditional fallbacks ('If it does not exist, then ...'), historical
    recollections ('Previously I had to ...'), quoted strings, fenced code blocks, or negated
    clauses ('then do not run ...').
    """
    if not user_prompt or not isinstance(user_prompt, str):
        return False

    # 1. Strip fenced code blocks and multi-word quoted literals
    cleaned = re.sub(r"```.*?```", " ", user_prompt, flags=re.DOTALL)
    cleaned = re.sub(
        r'"[^"\n]*\s[^"\n]*"|\'[^\'\n]*\s[^\'\n]*\'|`[^`\n]*\s[^`\n]*`',
        " ",
        cleaned,
    )

    # 2. Strip historical / past-tense reporting clauses (stopping at sentence boundaries, not dots in filenames)
    cleaned = re.sub(
        r"\b(?:previously|earlier|yesterday|in\s+the\s+past|last\s+time|before\s+this|originally|when\s+i\s+tried)\b.*?(?=\.(?:\s|$)|[!?\n]|$)",
        " ",
        cleaned,
        flags=re.IGNORECASE,
    )

    # 3. Strip negated clauses ('do not ...', 'never ...', 'without ...')
    cleaned = re.sub(
        r"\b(?:do\s+not|don'?t|never|must\s+not|should\s+not|cannot|can'?t|avoid|refrain\s+from|without|instead\s+of)\s+.*?(?=\.(?:\s|$)|[,;!?:\n]|$)",
        " ",
        cleaned,
        flags=re.IGNORECASE,
    )

    # 4. Pattern A: Explicit sequential ordering transition followed by a command verb
    pat_transition = re.compile(
        rf"\b(?P<trans>and\s+then|then|afterwards?|subsequently|next|followed\s+by|secondly|step\s*2|as\s+a\s+second\s+step|finally)"
        rf"(?:[,:]\s*|\s+)"
        rf"(?:(?:please|also|immediately|now|you\s+must|you\s+should|use|using)\s+){{0,2}}"
        rf"(?P<verb>{_SEQ_CMD_VERBS_RE})\b",
        flags=re.IGNORECASE,
    )
    for m in pat_transition.finditer(cleaned):
        prefix = cleaned[:m.start()].strip()
        trans_lower = re.sub(r"\s+", " ", m.group("trans").strip().lower())
        if trans_lower == "then":
            if not prefix:
                continue
            clause_before = re.split(r"(?:\.(?=\s|$)|[!?\n])+", prefix)[-1]
            # Exclude 'if/unless/when <condition>, then <verb>' when no prior creation/mutation was requested
            if re.search(r"\b(?:if|unless|whether|when)\b", clause_before, flags=re.IGNORECASE) and not re.search(
                r"\b(?:create|write|save|append|generate|make|touch|edit|modify|update|patch|build|install)\b",
                clause_before,
                flags=re.IGNORECASE,
            ):
                continue
        return True

    # 5. Pattern B: Temporal completion clause ('Once the file exists, execute it', 'After creating the file, run it')
    pat_temporal = (
        r"\b(?:once|after|as\s+soon\s+as)\s+"
        r"(?:(?:the|this|that|a|an|your)\s+)?"
        r"(?:[a-zA-Z0-9_.\-/\"']+\s+){0,4}"
        r"(?:exists?|is\s+(?:created|written|saved|generated|ready|complete|completed|done|built)|"
        r"has\s+been\s+(?:created|written|saved|generated|built)|"
        r"creating|writing|saving|generating)\s*,\s*"
        r"(?:(?:please|then|now|immediately)\s+){0,2}"
        rf"{_SEQ_CMD_VERBS_RE}\b"
    )
    if re.search(pat_temporal, cleaned, flags=re.IGNORECASE):
        return True

    # 6. Pattern C: Coordinated creation/write clause followed by ', and <cmd_verb> it / the file / the script'
    pat_coord = (
        r"\b(?:create|write|save|generate|make|touch|output|produce)\b"
        r"(?:(?!\.(?:\s|$)|[!?\n]).){1,120}?"
        r"(?:,\s*and|\band)\s+"
        r"(?:(?:then|afterwards?|subsequently|next|please)\s+)*"
        r"(?:execute|run|launch|cat|validate|verify|test)\s+"
        r"(?:it|the\s+(?:script|command|file|configuration|config|output|result))\b"
    )
    if re.search(pat_coord, cleaned, flags=re.IGNORECASE):
        return True

    return False


_NON_TASK_CONTEXT_CLAUSE_RE = re.compile(
    r"\b(?:"
    r"for\s+context(?:\s+only)?|"
    r"context\s+only|"
    r"not\s+part\s+of\s+this\s+task|"
    r"handled\s+(?:later\s+)?by\s+(?:another|a\s+different)\s+team|"
    r"out\s+of\s+scope|"
    r"for\s+reference\s+only|"
    r"unrelated\s+to\s+this\s+task"
    r")\b",
    re.IGNORECASE,
)


def _is_requested_mutation_target(user_prompt: str, target_path: str) -> bool:
    """Return True when target_path is requested for creation/mutation inside an active task clause."""
    if not user_prompt or not target_path:
        return False
    ft_name = Path(target_path).name
    if not ft_name:
        return False

    cleaned = re.sub(r"```.*?```", " ", user_prompt, flags=re.DOTALL)
    clauses = re.split(
        r"(?:\.(?=\s|$)|[;!?()\[\]]|\n(?=\s*(?:[-*]|\d+[.)]|Step\s+\d+:)))+",
        cleaned,
        flags=re.IGNORECASE,
    )
    mut_pattern = (
        rf"(?<!\bdo\snot\s)(?<!\bdon\'t\s)(?<!\bnever\s)(?<!\bmust\snot\s)(?<!\bshould\snot\s)"
        rf"(?<!\bcannot\s)(?<!\bcan\'t\s)(?<!\bavoid\s)(?<!\bwithout\s)(?<!\bhad\sto\s)(?<!\btried\sto\s)"
        rf"\b(?:create|write|save|append|generate|output|modify|touch|produce|store|record|edit|update|patch)\b"
        rf"(?:(?!\b(?:read|inspect|view|check|do\s+not|don\'t|never|avoid|refrain|without)\b).){{1,320}}?[\'\"]?{re.escape(ft_name)}\b"
    )
    for clause in clauses:
        clause_stripped = clause.strip()
        if not clause_stripped:
            continue
        if _NON_TASK_CONTEXT_CLAUSE_RE.search(clause_stripped):
            continue
        if re.search(mut_pattern, clause_stripped, flags=re.IGNORECASE):
            return True
    return False


def has_remaining_explicit_steps(
    user_prompt: str,
    ledger: TurnExecutionLedger,
    current_target: Optional[str] = None,
) -> bool:
    """Determine whether unfulfilled distinct steps remain in the user prompt."""
    if not user_prompt:
        return False

    # 1. Check if an unresolved failure exists needing recovery
    if ledger.has_unresolved_failures():
        return True

    # 2. Check for unfulfilled file targets mentioned in prompt
    file_targets = extract_file_targets_from_text(user_prompt)
    prohibited_targets = extract_prohibited_targets(user_prompt, ledger=ledger)
    norm_curr = canonicalize_target(current_target) if current_target else None

    # Collect all targets that have been mutated by a successful action (excluding unverified criteria failures,
    # pure read/inspection actions, and targets that currently have unresolved failures).
    unresolved_targets = {
        _extract_action_target(u.tool_name, u.arguments)
        for u in ledger.unresolved_failures()
        if _extract_action_target(u.tool_name, u.arguments) is not None
    }
    successful_targets = set()
    for rec in ledger.successful_actions():
        if getattr(rec, "verification_failed", False):
            continue
        if _is_inspection_action(rec.tool_name, rec.arguments):
            continue
        for k in ("path", "filepath", "file", "target", "filename", "dst", "destination"):
            val = rec.arguments.get(k)
            if val and isinstance(val, str) and val.strip():
                canon_val = canonicalize_target(val.strip())
                if not any(targets_match(canon_val, ut) for ut in unresolved_targets if ut):
                    successful_targets.add(canon_val)
        cmd = str(rec.arguments.get("command", "") or rec.arguments.get("cmd", ""))
        if cmd:
            for ft in file_targets:
                ft_name = Path(ft).name
                if ft in cmd or (ft_name and re.search(rf"\b{re.escape(ft_name)}\b", cmd)):
                    if not any(targets_match(ft, ut) for ut in unresolved_targets if ut):
                        successful_targets.add(ft)

    for ft in file_targets:
        if norm_curr and targets_match(ft, norm_curr, file_targets):
            continue
        if any(targets_match(ft, pt, prohibited_targets, allow_dir_containment=True) for pt in prohibited_targets):
            continue
        if not _is_requested_mutation_target(user_prompt, ft):
            continue
        if not ledger.is_target_verified(ft) and not any(targets_match(ft, st, successful_targets) for st in successful_targets):
            return True

    # 3. Check for sequential commands/executions (e.g., "then run it", "afterward validate it", "once the file exists, execute it")
    has_subsequent_command = _has_subsequent_command_requirement(user_prompt)
    if has_subsequent_command:
        has_executed_cmd = any(
            rec.tool_name in ("execute_command", "bash", "sh")
            and rec.state in (ActionState.SUCCESS, ActionState.VERIFIED)
            and not getattr(rec, "verification_failed", False)
            for rec in ledger.successful_actions()
        )
        if not has_executed_cmd:
            return True

    # 4. Check for explicit repetition count (e.g. "twice", "3 times")
    repeat_count_match = re.search(
        r"(?<!\bdo\snot\s)(?<!\bdon\'t\s)(?<!\bnever\s)\b(?:run|do|execute|write|append|perform)?\s*(?:(\d+)\s+times|twice|thrice)\b",
        user_prompt,
        flags=re.IGNORECASE,
    )
    if repeat_count_match:
        match_str = repeat_count_match.group(0).lower()
        if "twice" in match_str:
            target_count = 2
        elif "thrice" in match_str:
            target_count = 3
        elif repeat_count_match.group(1):
            target_count = int(repeat_count_match.group(1))
        else:
            target_count = 1

        if target_count > 1:
            count = sum(
                1 for rec in ledger.successful_actions()
                if not getattr(rec, "verification_failed", False)
                and norm_curr
                and any(
                    targets_match(str(rec.arguments.get(k, "")), norm_curr)
                    for k in ("path", "filepath", "file", "target")
                    if rec.arguments.get(k)
                )
            )
            if count < target_count:
                return True

    return False


def find_unfulfilled_task_requirements(
    user_prompt: str,
    ledger: TurnExecutionLedger,
    acceptance_spec: Optional[Any] = None,
) -> Tuple[List[str], List[str]]:
    """Identify verified outcomes and remaining unfulfilled requirements for goal-aware replanning.

    Returns:
        (verified_outcomes, unfulfilled_outcomes)
    """
    verified_outcomes: List[str] = []
    unfulfilled_outcomes: List[str] = []

    # 1. Evaluate explicit acceptance criteria if provided
    parsed_criteria = parse_acceptance_criteria(acceptance_spec)
    if parsed_criteria:
        from zelyr.engine.verifier import OutcomeVerifier

        # Group criteria results by canonical target so multi-criterion targets are only marked
        # verified when ALL criteria for that target succeed.
        target_results: Dict[str, List[Tuple[Dict[str, Any], Any]]] = {}
        for c in parsed_criteria:
            check_name = c["check"]
            target = c["target"]
            expected = c.get("expected")
            vr = OutcomeVerifier.verify(check=check_name, target=target, expected=expected)
            norm_t = canonicalize_target(target, check_type=check_name) if target else ""
            if norm_t:
                target_results.setdefault(norm_t, []).append((c, vr))
            if check_name.lower() in ("command_output", "command", "cmd") and target:
                for ref_ft in extract_file_targets_from_text(target):
                    norm_ft = canonicalize_target(ref_ft)
                    if norm_ft:
                        target_results.setdefault(norm_ft, []).append(
                            ({"check": "file_exists", "target": ref_ft, "expected": expected}, vr)
                        )
            if vr.success:
                desc = f"Target '{target}' verified ({check_name}" + (f": '{expected}'" if expected is not None else "") + ")"
                if desc not in verified_outcomes:
                    verified_outcomes.append(desc)
            else:
                desc = f"Target '{target}' unfulfilled: {check_name}" + (f" expected '{expected}'" if expected is not None else "")
                if desc not in unfulfilled_outcomes:
                    unfulfilled_outcomes.append(desc)

        for norm_t, results in target_results.items():
            all_passed = all(vr.success for _, vr in results)
            first_c, _ = results[0]
            raw_target = first_c["target"]
            check_name = first_c["check"]
            if all_passed:
                if raw_target and hasattr(ledger, "record_target_verified"):
                    ledger.record_target_verified(
                        raw_target,
                        detail=f"Target '{raw_target}' verified ({check_name})",
                        check_type=check_name,
                    )
            else:
                if raw_target and hasattr(ledger, "invalidate_target_verification"):
                    ledger.invalidate_target_verification(raw_target, check_type=check_name)

    # 2. Extract file targets from user prompt
    file_targets = extract_file_targets_from_text(user_prompt)
    prohibited_targets = extract_prohibited_targets(user_prompt, ledger=ledger)
    unresolved_targets = {
        _extract_action_target(u.tool_name, u.arguments)
        for u in ledger.unresolved_failures()
        if _extract_action_target(u.tool_name, u.arguments) is not None
    }
    successful_targets = set()
    for rec in ledger.successful_actions():
        if getattr(rec, "verification_failed", False):
            continue
        if _is_inspection_action(rec.tool_name, rec.arguments):
            continue
        for k in ("path", "filepath", "file", "target", "filename", "dst", "destination"):
            val = rec.arguments.get(k)
            if val and isinstance(val, str) and val.strip():
                canon_val = canonicalize_target(val.strip())
                if not any(targets_match(canon_val, ut) for ut in unresolved_targets if ut):
                    successful_targets.add(canon_val)
        cmd = str(rec.arguments.get("command", "") or rec.arguments.get("cmd", ""))
        if cmd:
            for ft in file_targets:
                ft_name = Path(ft).name
                if ft in cmd or (ft_name and re.search(rf"\b{re.escape(ft_name)}\b", cmd)):
                    if not any(targets_match(ft, ut) for ut in unresolved_targets if ut):
                        successful_targets.add(ft)

    for ft in file_targets:
        if any(criterion_applies_to_target(c, ft, file_targets) for c in parsed_criteria if c.get("target")):
            continue
        if any(targets_match(ft, pt, prohibited_targets, allow_dir_containment=True) for pt in prohibited_targets):
            continue

        if not _is_requested_mutation_target(user_prompt, ft):
            # File is mentioned for inspection/reading, prohibition, or non-task context, not as a creation outcome
            continue

        ft_name = Path(ft).name
        if ledger.is_target_verified(ft) or any(targets_match(ft, st, successful_targets) for st in successful_targets):
            ft_desc = f"Target '{ft}' verified or created"
            if ft_desc not in verified_outcomes:
                verified_outcomes.append(ft_desc)
        else:
            expected_match = re.search(
                rf"{re.escape(ft_name)}[\'\"]?\s+(?:containing|with(?:\s+content)?)\s+[\'\"]([^\'\"]+)[\'\"]",
                user_prompt,
                flags=re.IGNORECASE,
            ) or re.search(
                rf"(?:containing|with(?:\s+content)?)\s+[\'\"]([^\'\"]+)[\'\"]\s+(?:in|into|to|for)?\s+[\'\"]?{re.escape(ft_name)}",
                user_prompt,
                flags=re.IGNORECASE,
            )
            content_hint = f" with content '{expected_match.group(1)}'" if expected_match else ""
            ft_desc = f"Target '{ft}' not yet created/verified{content_hint}"
            if ft_desc not in unfulfilled_outcomes:
                unfulfilled_outcomes.append(ft_desc)

    # 3. Check for sequential commands / executions requested in prompt
    has_subsequent_command = _has_subsequent_command_requirement(user_prompt)
    if has_subsequent_command:
        has_executed_cmd = any(
            rec.tool_name in ("execute_command", "bash", "sh")
            and rec.state in (ActionState.SUCCESS, ActionState.VERIFIED)
            and not getattr(rec, "verification_failed", False)
            for rec in ledger.successful_actions()
        )
        if not has_executed_cmd:
            seq_desc = "Requested sequential command execution not yet performed"
            if seq_desc not in unfulfilled_outcomes:
                unfulfilled_outcomes.append(seq_desc)

    # 4. Check explicit repetition count
    repeat_count_match = re.search(
        r"\b(?:run|do|execute|write|append|perform)?\s*(?:(\d+)\s+times|twice|thrice)\b",
        user_prompt,
        flags=re.IGNORECASE,
    )
    if repeat_count_match:
        m_str = repeat_count_match.group(0).lower()
        t_count = 2 if "twice" in m_str else (3 if "thrice" in m_str else (int(repeat_count_match.group(1)) if repeat_count_match.group(1) else 1))
        if t_count > 1:
            total_succ = sum(1 for r in ledger.successful_actions() if not getattr(r, "verification_failed", False))
            if total_succ < t_count:
                rep_desc = f"Repetition incomplete: {total_succ}/{t_count} executions performed"
                if rep_desc not in unfulfilled_outcomes:
                    unfulfilled_outcomes.append(rep_desc)

    # 5. Check unresolved failures
    if ledger.has_unresolved_failures():
        for f in ledger.unresolved_failures():
            if getattr(f, "verification_failed", False):
                fail_desc = f"Unverified outcome for '{f.tool_name}': {f.verification_detail or f.error or 'acceptance criteria not satisfied'}"
            else:
                fail_desc = f"Unresolved failure for '{f.tool_name}': {f.error or 'action failed'}"
            if fail_desc not in unfulfilled_outcomes:
                unfulfilled_outcomes.append(fail_desc)

    return verified_outcomes, unfulfilled_outcomes
