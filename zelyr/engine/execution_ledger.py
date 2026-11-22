"""Authoritative Turn Execution Ledger for Zelyr Runtime.

Maintains strict, evidence-based machine state authority across agent turns.
Tracks tool call lifecycle from proposal through validation, dispatch,
execution observation, and deterministic verification.
"""

from __future__ import annotations

from contextlib import contextmanager
import contextvars
from dataclasses import asdict, dataclass, field
from enum import Enum
import json
import logging
import os
from pathlib import Path
import re
import sqlite3
import threading
import time
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple, Union
import uuid


logger = logging.getLogger("zelyr.execution_ledger")


class InvalidStateTransitionError(ValueError):
    """Raised when an invalid action lifecycle state transition is attempted."""


class LedgerPersistenceError(RuntimeError):
    """Raised when persistent execution ledger storage fails."""


class ExecutionIsolationError(PermissionError):
    """Raised when an operation attempts to cross task execution boundaries."""


MAX_LEDGER_TEXT_BYTES = 1024
MAX_LEDGER_METADATA_BYTES = 4096
MAX_LEDGER_COLLECTION_ITEMS = 24
MAX_LEDGER_DEPTH = 4

_SENSITIVE_KEY_SUBSTRINGS = (
    "api_key",
    "apikey",
    "token",
    "secret",
    "password",
    "passwd",
    "authorization",
    "auth_header",
    "cookie",
    "private_key",
    "credential",
    "x_api_key",
    "x-api-key",
    "bearer",
)

_LARGE_PAYLOAD_KEYS = (
    "image_b64",
    "_vision_b64",
    "screenshot_b64",
    "raw_bytes",
    "binary_data",
    "screenshot",
    "audio_bytes",
    "video_bytes",
)

_NON_SECRET_ENV_NAMES = frozenset({
    "HOME",
    "PATH",
    "PWD",
    "OLDPWD",
    "SHELL",
    "TERM",
    "TMPDIR",
    "TEMP",
    "TMP",
    "USER",
    "USERNAME",
    "LOGNAME",
    "LANG",
    "LC_ALL",
    "XDG_CONFIG_HOME",
    "XDG_DATA_HOME",
    "XDG_CACHE_HOME",
    "XDG_RUNTIME_DIR",
    "ZELYR_CONFIG_DIR",
    "ZELYR_DATA_DIR",
    "ZELYR_MEMORY_DB",
    "ZELYR_LEDGER_DB",
    "PYTHONPATH",
    "VIRTUAL_ENV",
    "DISPLAY",
    "WAYLAND_DISPLAY",
})


def _collect_sensitive_env_values() -> List[str]:
    """Collect non-trivial secret environment values for dynamic scrubbing."""
    secrets: List[str] = []
    for k, v in os.environ.items():
        if not v or not isinstance(v, str) or k in _NON_SECRET_ENV_NAMES:
            continue
        k_upper = k.upper()
        if any(tok in k_upper for tok in ("KEY", "TOKEN", "SECRET", "PASS", "CREDENTIAL", "AUTH", "PRIVATE")):
            stripped = v.strip()
            if len(stripped) >= 8 and not stripped.startswith(("/", "~")):
                secrets.append(stripped)
    secrets.sort(key=len, reverse=True)
    return secrets


def _redact_sensitive_text(text: Optional[str]) -> Optional[str]:
    """Scrub common credentials/tokens and sensitive env values before storing strings in the ledger."""
    if text is None:
        return None
    raw = str(text)
    if not raw:
        return raw
    scrubbed = re.sub(r"(?i)(https?://)([^:/@\s]+):([^/@\s]+)@", r"\1\2:[REDACTED]@", raw)
    scrubbed = re.sub(r"(?i)\b(bearer\s+)[a-z0-9._\-+=]+", r"\1[REDACTED]", scrubbed)
    scrubbed = re.sub(r"\bsk-[A-Za-z0-9_\-]{8,}", "[REDACTED]", scrubbed)
    scrubbed = re.sub(r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{12,}", "[REDACTED]", scrubbed)
    scrubbed = re.sub(r"\bgithub_pat_[A-Za-z0-9_]{16,}", "[REDACTED]", scrubbed)
    scrubbed = re.sub(r"\bxox[baprs]-[A-Za-z0-9\-]{10,}", "[REDACTED]", scrubbed)
    scrubbed = re.sub(r"\bAKIA[0-9A-Z]{16}\b", "[REDACTED]", scrubbed)
    scrubbed = re.sub(
        r"(?i)\b(api[_-]?key|token|secret|password|passwd|authorization|cookie)\s*([=:])\s*['\"]?([^\s,'\"&;]+)['\"]?",
        r"\1\2[REDACTED]",
        scrubbed,
    )
    for env_secret in _collect_sensitive_env_values():
        if env_secret in scrubbed:
            scrubbed = scrubbed.replace(env_secret, "[REDACTED]")
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

    def __init__(
        self,
        *,
        execution_id: Optional[str] = None,
        central_ledger: Optional[CentralExecutionLedger] = None,
        caller_ring: Optional[int] = None,
        task_local_scoped: bool = False,
        source_component: Optional[str] = None,
    ) -> None:
        self._records: Dict[str, ActionRecord] = {}
        self._order: List[str] = []
        self._verified_targets: Set[str] = set()
        self._pending_call_ids: List[str] = []
        self._execution_id: Optional[str] = execution_id
        self._central_ledger: Optional[CentralExecutionLedger] = central_ledger
        self._caller_ring: Optional[int] = caller_ring
        self._task_local_scoped: bool = bool(task_local_scoped)
        self._source_component: Optional[str] = source_component
        self._call_source_meta: Dict[str, Dict[str, Any]] = {}
        self._verification_status_override: Optional[str] = None

    def attach_central_ledger(
        self,
        central_ledger: Optional[CentralExecutionLedger],
        execution_id: str,
        *,
        caller_ring: Optional[int] = None,
        task_local_scoped: Optional[bool] = None,
        source_component: Optional[str] = None,
    ) -> None:
        """Bind a persistent CentralExecutionLedger instance to this turn ledger."""
        self._central_ledger = central_ledger
        self._execution_id = execution_id
        if caller_ring is not None:
            self._caller_ring = caller_ring
        if task_local_scoped is not None:
            self._task_local_scoped = bool(task_local_scoped)
        if source_component is not None:
            self._source_component = source_component

    def set_call_source_metadata(self, call_id: str, **meta: Any) -> None:
        """Attach trusted source/telemetry metadata (e.g., external MCP or VLA info) for a call."""
        if not call_id:
            return
        entry = self._call_source_meta.setdefault(call_id, {})
        for k, v in meta.items():
            if v is not None:
                entry[k] = v

    def set_verification_status(self, verification_status: Union[VerificationStatus, str]) -> None:
        """Record an explicit overall verification status override for reporting."""
        self._verification_status_override = (
            verification_status.value
            if isinstance(verification_status, VerificationStatus)
            else str(verification_status)
        )

    def _emit_central_event(
        self,
        event_type: LedgerEventType,
        *,
        call_id: Optional[str] = None,
        tool_id: Optional[str] = None,
        status: str = "recorded",
        metadata: Optional[Dict[str, Any]] = None,
        idempotency_suffix: Optional[str] = None,
    ) -> None:
        if self._central_ledger is None or not self._execution_id:
            return
        src_meta = self._call_source_meta.get(call_id or "", {})
        eff_tool = tool_id or (self._records[call_id].tool_name if call_id and call_id in self._records else None)
        merged_meta = dict(src_meta.get("extra", {})) if isinstance(src_meta.get("extra"), dict) else {}
        if metadata:
            merged_meta.update(metadata)
        idem_key = (
            f"{self._execution_id}:{call_id}:{idempotency_suffix}"
            if call_id and idempotency_suffix
            else (f"{self._execution_id}:{idempotency_suffix}" if idempotency_suffix else None)
        )
        try:
            self._central_ledger.record_event(
                self._execution_id,
                event_type,
                tool_id=eff_tool,
                source_id=src_meta.get("source_id"),
                source_type=src_meta.get("source_type"),
                is_external=bool(src_meta.get("is_external", False)),
                status=status,
                correlation_id=call_id,
                caller_ring=self._caller_ring,
                task_local_scoped=self._task_local_scoped,
                metadata=merged_meta,
                idempotency_key=idem_key,
            )
        except Exception as exc:
            logger.debug("Non-fatal central event hook error (%s): %s", event_type.value, exc)

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
            self._emit_central_event(
                LedgerEventType.VERIFICATION_RECORDED,
                status=VerificationStatus.VERIFIED.value,
                metadata={"target": norm, "check_type": check_type, "detail": detail or "verified"},
            )

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
        invalidated_any = bool(to_remove)
        for cid in self._order:
            rec = self._records.get(cid)
            if not rec or rec.state != ActionState.VERIFIED:
                continue
            rec_target = _extract_action_target(rec.tool_name, rec.arguments)
            if rec_target and targets_match(target, rec_target, check_type=check_type):
                invalidated_any = True
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
        if invalidated_any:
            self._emit_central_event(
                LedgerEventType.VERIFICATION_RECORDED,
                status="verification_invalidated",
                metadata={"target": target, "check_type": check_type},
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
        self._emit_central_event(
            LedgerEventType.TOOL_SELECTED,
            call_id=call_id,
            tool_id=tool_name,
            status="proposed",
            metadata={"turn": turn, "raw_name": raw_name or tool_name, "arguments": rec.arguments},
            idempotency_suffix="selected",
        )

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
            self._emit_central_event(
                LedgerEventType.ARGUMENT_VALIDATED,
                call_id=call_id,
                tool_id=old.tool_name,
                status="valid",
                metadata={"arguments": old.arguments},
                idempotency_suffix="validated",
            )

    def record_dispatched(self, call_id: str, *, emit_event: bool = True) -> None:
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
            if emit_event:
                self._emit_central_event(
                    LedgerEventType.TOOL_EXECUTION_START,
                    call_id=call_id,
                    tool_id=old.tool_name,
                    status="started",
                    metadata={"arguments": old.arguments},
                    idempotency_suffix="started",
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
            if emit_event:
                self._emit_central_event(
                    LedgerEventType.TOOL_EXECUTION_START,
                    call_id=call_id,
                    status="started",
                    idempotency_suffix="started",
                )

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
            self._emit_central_event(
                LedgerEventType.AUTHORIZATION_DECISION,
                call_id=call_id,
                tool_id=old.tool_name,
                status="allowed",
                idempotency_suffix="auth",
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
        self._emit_central_event(
            LedgerEventType.AUTHORIZATION_DECISION,
            call_id=call_id,
            tool_id=tool_name or None,
            status="denied",
            metadata={"reason": safe_reason},
            idempotency_suffix="rejected",
        )

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
        self._emit_central_event(
            LedgerEventType.TOOL_EXECUTION_RESULT,
            call_id=call_id,
            tool_id=tool_name or None,
            status="cancelled",
            metadata={"reason": safe_reason or "Action cancelled"},
            idempotency_suffix="cancelled",
        )

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
        self._emit_central_event(
            LedgerEventType.TOOL_EXECUTION_RESULT,
            call_id=call_id,
            tool_id=tool_name or None,
            status="partially_completed",
            metadata={"error": safe_err, "exit_code": exit_code},
            idempotency_suffix="partial",
        )

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
        self._emit_central_event(
            LedgerEventType.OBSERVATION_UPDATED,
            call_id=call_id,
            tool_id=old.tool_name,
            status="observed",
            metadata={"exit_code": exit_code, "output_preview": str(output)[:512] if output is not None else None},
            idempotency_suffix="observed",
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
        self._emit_central_event(
            LedgerEventType.VERIFICATION_RECORDED,
            call_id=call_id,
            tool_id=old.tool_name,
            status=VerificationStatus.NOT_VERIFIED.value,
            metadata={"detail": reason},
            idempotency_suffix="unverified",
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
        self._emit_central_event(
            LedgerEventType.VERIFICATION_RECORDED,
            call_id=call_id,
            tool_id=old.tool_name,
            status=VerificationStatus.VERIFIED.value,
            metadata={"detail": detail or "Independently verified", "verified_targets": merged_targets},
            idempotency_suffix="verified",
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
        if status == "timed_out" or (
            isinstance(effective_error, str) and "timed out" in effective_error.lower() and is_failure
        ):
            res_status = "timed_out"
        elif final_state == ActionState.FAILED and (
            status == "rejected" or (effective_rejection_reason is not None)
        ):
            res_status = "rejected"
        elif is_failure:
            res_status = "failed"
        else:
            res_status = "succeeded"
        self._emit_central_event(
            LedgerEventType.TOOL_EXECUTION_RESULT,
            call_id=call_id,
            tool_id=tool_name or None,
            status=res_status,
            metadata={
                "exit_code": effective_exit_code,
                "error": effective_error,
                "rejection_reason": effective_rejection_reason,
                "verified": effective_verified,
            },
            idempotency_suffix="result",
        )
        self._emit_central_event(
            LedgerEventType.OBSERVATION_UPDATED,
            call_id=call_id,
            tool_id=tool_name or None,
            status=final_state.value,
            metadata={
                "output_preview": str(output)[:512] if output is not None else None,
                "verification_detail": effective_detail,
                "verification_failed": effective_verification_failed,
                "verified_targets": eff_verified_targets,
                "failed_targets": eff_failed_targets,
            },
            idempotency_suffix="observed",
        )
        if effective_verified:
            self._emit_central_event(
                LedgerEventType.VERIFICATION_RECORDED,
                call_id=call_id,
                tool_id=tool_name or None,
                status=VerificationStatus.VERIFIED.value,
                metadata={"detail": effective_detail, "verified_targets": eff_verified_targets},
                idempotency_suffix="verified",
            )
        elif effective_verification_failed:
            self._emit_central_event(
                LedgerEventType.VERIFICATION_RECORDED,
                call_id=call_id,
                tool_id=tool_name or None,
                status=VerificationStatus.VERIFICATION_FAILED.value,
                metadata={"detail": effective_detail, "failed_targets": eff_failed_targets},
                idempotency_suffix="verification_failed",
            )

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
            self._emit_central_event(
                LedgerEventType.RECOVERY_ATTEMPT,
                call_id=call_id,
                tool_id=old.tool_name,
                status="retry_requested",
                metadata={"attempt": new_attempts, "error": old.error},
                idempotency_suffix=f"recovery:{new_attempts}",
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
            self._emit_central_event(
                LedgerEventType.RECOVERY_ATTEMPT,
                call_id=call_id,
                tool_id=old.tool_name,
                status="terminal_blocker",
                metadata={"recovery_attempts": old.recovery_attempts, "error": old.error},
                idempotency_suffix="terminal",
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
        out: Dict[str, Any] = {
            "total_actions": self.total_actions(),
            "successful_actions": [a.to_dict() for a in self.successful_actions()],
            "failed_actions": [a.to_dict() for a in self.failed_actions()],
            "unresolved_failures": [a.to_dict() for a in self.unresolved_failures()],
            "has_unresolved_failures": self.has_unresolved_failures(),
            "pending_proposals": [a.to_dict() for a in self.get_pending_proposals()],
            "records": [r.to_dict() for r in records],
        }
        if self._execution_id is not None or self._central_ledger is not None:
            out["execution_id"] = self._execution_id
            if self._verification_status_override is not None:
                out["verification_status"] = self._verification_status_override
            elif self.verified_actions():
                out["verification_status"] = VerificationStatus.VERIFIED.value
            elif any(getattr(r, "verification_failed", False) for r in records):
                out["verification_status"] = VerificationStatus.VERIFICATION_FAILED.value
            else:
                out["verification_status"] = VerificationStatus.NOT_VERIFIED.value
            out["persistence_errors"] = (
                self._central_ledger.get_persistence_errors()
                if self._central_ledger is not None
                else []
            )
        return out


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


# ---------------------------------------------------------------------------
# Centralized Persistent Execution Ledger
# ---------------------------------------------------------------------------


class ExecutionStatus(str, Enum):
    """Top-level task or standalone execution status."""

    INITIALIZED = "initialized"
    RUNNING = "running"
    COMPLETED = "completed"
    COMPLETED_UNVERIFIED = "completed_unverified"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMED_OUT = "timed_out"


class VerificationStatus(str, Enum):
    """Outcome verification status distinct from model completion claims."""

    NOT_VERIFIED = "not_verified"
    VERIFIED = "verified"
    VERIFICATION_FAILED = "verification_failed"
    CLAIMED_UNVERIFIED = "claimed_unverified"
    INTERRUPTED = "interrupted"


class LedgerEventType(str, Enum):
    """Structured event types recorded in chronological execution order."""

    TASK_INITIALIZED = "task_initialized"
    PLANNING_DECISION = "planning_decision"
    TOOL_SELECTED = "tool_selected"
    AUTHORIZATION_DECISION = "authorization_decision"
    ARGUMENT_VALIDATED = "argument_validated"
    TOOL_EXECUTION_START = "tool_execution_start"
    TOOL_EXECUTION_RESULT = "tool_execution_result"
    RECOVERY_ATTEMPT = "recovery_attempt"
    OBSERVATION_UPDATED = "observation_updated"
    VERIFICATION_RECORDED = "verification_recorded"
    TASK_TERMINATED = "task_terminated"


_PROMPT_INJECTION_PATTERNS = (
    re.compile(r"(?i)\[\s*(?:system|developer|assistant|admin|override|instruction)[^\]]*\]"),
    re.compile(r"(?i)<\|\s*(?:im_start|im_end|system|assistant|user)\s*\|>"),
    re.compile(r"(?i)\bignore\s+(?:all\s+)?(?:previous|prior|above)\s+instructions?\b"),
    re.compile(r"(?i)\byou\s+are\s+now\b"),
)


def _strip_control_and_bidi(raw: str) -> str:
    """Remove non-printable control characters, ANSI escapes, and bidi overrides."""
    cleaned = re.sub(r"\x1b\[[0-9;?]*[ -/]*[@-~]", "", raw)
    cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f\u200b-\u200f\u202a-\u202e\u2066-\u2069\ufeff]", "", cleaned)
    return cleaned


def sanitize_ledger_text(
    text: Optional[str],
    *,
    max_bytes: int = MAX_LEDGER_TEXT_BYTES,
    is_external: bool = False,
    strip_injection: bool = True,
) -> Tuple[Optional[str], bool, bool]:
    """Sanitize, redact secrets, neutralize injections, and bound ledger text.

    Returns:
        Tuple of ``(sanitized_text, redacted, truncated)``.
    """
    if text is None:
        return None, False, False

    raw = str(text)
    if not raw:
        return "", False, False

    redacted = False
    truncated = False

    scrubbed = _redact_sensitive_text(raw) or ""
    if scrubbed != raw:
        redacted = True

    cleaned = _strip_control_and_bidi(scrubbed)
    if strip_injection or is_external:
        try:
            from zelyr.tools.external import sanitize_external_text

            cleaned = sanitize_external_text(cleaned, max_len=max(max_bytes, 128) * 2)
        except Exception:
            pass
        for pat in _PROMPT_INJECTION_PATTERNS:
            if pat.search(cleaned):
                cleaned = pat.sub("[FILTERED_INSTRUCTION]", cleaned)
                redacted = True

    encoded = cleaned.encode("utf-8", errors="replace")
    if len(encoded) > max_bytes:
        truncated = True
        suffix = b"... [TRUNCATED]"
        budget = max(0, max_bytes - len(suffix))
        clipped = encoded[:budget].decode("utf-8", errors="ignore")
        cleaned = clipped + "... [TRUNCATED]"

    return cleaned, redacted, truncated


def _is_sensitive_key_name(key: str) -> bool:
    low = key.lower().strip()
    if any(sub in low for sub in _SENSITIVE_KEY_SUBSTRINGS):
        return True
    upper = key.upper().strip()
    if upper not in _NON_SECRET_ENV_NAMES and any(
        tok in upper for tok in ("_KEY", "_TOKEN", "_SECRET", "_PASS", "CREDENTIAL", "AUTH")
    ):
        return True
    return False


def _sanitize_value(
    value: Any,
    *,
    depth: int,
    max_text_bytes: int,
    is_external: bool,
    key_hint: str = "",
) -> Tuple[Any, bool, bool]:
    """Recursively sanitize, redact, and bound arbitrary Python values for storage."""
    if key_hint:
        low_key = key_hint.lower()
        if _is_sensitive_key_name(key_hint):
            return "[REDACTED]", True, False
        if any(lk in low_key for lk in _LARGE_PAYLOAD_KEYS):
            size_hint = len(value) if isinstance(value, (str, bytes, bytearray, list, dict)) else None
            summary = {"omitted": True, "reason": "large_payload"}
            if size_hint is not None:
                summary["size"] = size_hint
            return summary, False, True

    if value is None or isinstance(value, (bool, int)):
        return value, False, False

    if isinstance(value, float):
        return value, False, False

    if isinstance(value, (bytes, bytearray)):
        return {"omitted": True, "reason": "binary_payload", "size": len(value)}, False, True

    if isinstance(value, str):
        safe_s, was_redacted, was_trunc = sanitize_ledger_text(
            value,
            max_bytes=max_text_bytes,
            is_external=is_external,
            strip_injection=True,
        )
        return safe_s, was_redacted, was_trunc

    if depth >= MAX_LEDGER_DEPTH:
        safe_s, was_redacted, _ = sanitize_ledger_text(
            str(value),
            max_bytes=min(128, max_text_bytes),
            is_external=is_external,
            strip_injection=True,
        )
        return f"{safe_s} [TRUNCATED_DEPTH]", was_redacted, True

    redacted = False
    truncated = False

    if isinstance(value, dict):
        out: Dict[str, Any] = {}
        items = list(value.items())
        if len(items) > MAX_LEDGER_COLLECTION_ITEMS:
            items = items[:MAX_LEDGER_COLLECTION_ITEMS]
            truncated = True
        for k, v in items:
            safe_k, k_red, k_trunc = sanitize_ledger_text(
                str(k),
                max_bytes=128,
                is_external=is_external,
                strip_injection=True,
            )
            redacted = redacted or k_red
            truncated = truncated or k_trunc
            safe_v, v_red, v_trunc = _sanitize_value(
                v,
                depth=depth + 1,
                max_text_bytes=max_text_bytes,
                is_external=is_external,
                key_hint=str(k),
            )
            redacted = redacted or v_red
            truncated = truncated or v_trunc
            out[safe_k or "_"] = safe_v
        if len(value) > MAX_LEDGER_COLLECTION_ITEMS:
            out["_omitted_keys"] = len(value) - MAX_LEDGER_COLLECTION_ITEMS
        return out, redacted, truncated

    if isinstance(value, (list, tuple, set)):
        seq = list(value)
        if len(seq) > MAX_LEDGER_COLLECTION_ITEMS:
            seq = seq[:MAX_LEDGER_COLLECTION_ITEMS]
            truncated = True
        out_list: List[Any] = []
        for item in seq:
            safe_item, i_red, i_trunc = _sanitize_value(
                item,
                depth=depth + 1,
                max_text_bytes=max_text_bytes,
                is_external=is_external,
            )
            redacted = redacted or i_red
            truncated = truncated or i_trunc
            out_list.append(safe_item)
        if len(value) > MAX_LEDGER_COLLECTION_ITEMS:
            out_list.append({"_omitted_items": len(value) - MAX_LEDGER_COLLECTION_ITEMS})
        return out_list, redacted, truncated

    safe_s, was_red, was_trunc = sanitize_ledger_text(
        str(value),
        max_bytes=min(256, max_text_bytes),
        is_external=is_external,
        strip_injection=True,
    )
    return safe_s, was_red, was_trunc


def sanitize_ledger_payload(
    data: Optional[Dict[str, Any]],
    *,
    is_external: bool = False,
    max_bytes: int = MAX_LEDGER_METADATA_BYTES,
) -> Tuple[Dict[str, Any], bool, bool]:
    """Sanitize, redact, and size-bound metadata dictionary before SQLite persistence."""
    if not data:
        return {}, False, False

    raw_dict = data if isinstance(data, dict) else {"value": data}
    sanitized, redacted, truncated = _sanitize_value(
        raw_dict,
        depth=0,
        max_text_bytes=min(MAX_LEDGER_TEXT_BYTES, max_bytes // 2 if max_bytes >= 512 else max_bytes),
        is_external=is_external,
    )
    if not isinstance(sanitized, dict):
        sanitized = {"value": sanitized}

    encoded = json.dumps(sanitized, ensure_ascii=False, separators=(",", ":"), default=str).encode("utf-8")
    if len(encoded) > max_bytes:
        truncated = True
        # Progressively shrink bulky preview fields first, then keep compact summary
        trimmed = dict(sanitized)
        for drop_key in ("raw_output", "output", "output_preview", "arguments", "arguments_preview", "detail", "summary", "error"):
            if len(json.dumps(trimmed, ensure_ascii=False, separators=(",", ":"), default=str).encode("utf-8")) <= max_bytes:
                break
            if drop_key in trimmed and isinstance(trimmed[drop_key], str):
                clipped, _, _ = sanitize_ledger_text(trimmed[drop_key], max_bytes=128, is_external=is_external)
                trimmed[drop_key] = clipped
            elif drop_key in trimmed:
                trimmed[drop_key] = "[TRUNCATED]"
        while (
            len(trimmed) > 1
            and len(json.dumps(trimmed, ensure_ascii=False, separators=(",", ":"), default=str).encode("utf-8")) > max_bytes
        ):
            longest_key = max(trimmed.keys(), key=lambda k: len(str(trimmed[k])))
            trimmed.pop(longest_key, None)
        trimmed["truncated"] = True
        sanitized = trimmed

    return sanitized, redacted, truncated


@dataclass(frozen=True)
class ExecutionEvent:
    """Single immutable ordered event in an execution timeline."""

    event_id: str
    execution_id: str
    sequence_number: int
    timestamp: float
    event_type: str
    tool_id: Optional[str] = None
    source_id: Optional[str] = None
    source_type: Optional[str] = None
    is_external: bool = False
    status: str = "recorded"
    correlation_id: Optional[str] = None
    caller_ring: Optional[int] = None
    task_local_scoped: bool = False
    redacted: bool = False
    truncated: bool = False
    idempotency_key: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "execution_id": self.execution_id,
            "sequence_number": self.sequence_number,
            "timestamp": self.timestamp,
            "event_type": self.event_type,
            "tool_id": self.tool_id,
            "source_id": self.source_id,
            "source_type": self.source_type,
            "is_external": bool(self.is_external),
            "status": self.status,
            "correlation_id": self.correlation_id,
            "caller_ring": self.caller_ring,
            "task_local_scoped": bool(self.task_local_scoped),
            "redacted": bool(self.redacted),
            "truncated": bool(self.truncated),
            "idempotency_key": self.idempotency_key,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_row(cls, row: Union[sqlite3.Row, Dict[str, Any]]) -> ExecutionEvent:
        d = dict(row)
        raw_meta = d.get("metadata_json") if "metadata_json" in d else d.get("metadata", {})
        if isinstance(raw_meta, str):
            try:
                meta = json.loads(raw_meta) if raw_meta else {}
            except Exception:
                meta = {}
        elif isinstance(raw_meta, dict):
            meta = dict(raw_meta)
        else:
            meta = {}
        return cls(
            event_id=str(d["event_id"]),
            execution_id=str(d["execution_id"]),
            sequence_number=int(d["sequence_number"]),
            timestamp=float(d["timestamp"]),
            event_type=str(d["event_type"]),
            tool_id=d.get("tool_id"),
            source_id=d.get("source_id"),
            source_type=d.get("source_type"),
            is_external=bool(d.get("is_external", False)),
            status=str(d.get("status", "recorded")),
            correlation_id=d.get("correlation_id"),
            caller_ring=d.get("caller_ring"),
            task_local_scoped=bool(d.get("task_local_scoped", False)),
            redacted=bool(d.get("redacted", False)),
            truncated=bool(d.get("truncated", False)),
            idempotency_key=d.get("idempotency_key"),
            metadata=meta,
        )


@dataclass(frozen=True)
class ExecutionRecord:
    """Top-level persistent execution record."""

    execution_id: str
    task_prompt: str
    status: str = ExecutionStatus.INITIALIZED.value
    verification_status: str = VerificationStatus.NOT_VERIFIED.value
    source_component: str = "runtime"
    parent_execution_id: Optional[str] = None
    caller_ring: Optional[int] = None
    task_local_scoped: bool = False
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None
    final_summary: Optional[str] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    redacted: bool = False
    truncated: bool = False
    event_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "execution_id": self.execution_id,
            "parent_execution_id": self.parent_execution_id,
            "task_prompt": self.task_prompt,
            "status": self.status,
            "verification_status": self.verification_status,
            "source_component": self.source_component,
            "caller_ring": self.caller_ring,
            "task_local_scoped": bool(self.task_local_scoped),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "completed_at": self.completed_at,
            "final_summary": self.final_summary,
            "error": self.error,
            "metadata": dict(self.metadata),
            "redacted": bool(self.redacted),
            "truncated": bool(self.truncated),
            "event_count": self.event_count,
        }

    @classmethod
    def from_row(cls, row: Union[sqlite3.Row, Dict[str, Any]], event_count: Optional[int] = None) -> ExecutionRecord:
        d = dict(row)
        raw_meta = d.get("metadata_json") if "metadata_json" in d else d.get("metadata", {})
        if isinstance(raw_meta, str):
            try:
                meta = json.loads(raw_meta) if raw_meta else {}
            except Exception:
                meta = {}
        elif isinstance(raw_meta, dict):
            meta = dict(raw_meta)
        else:
            meta = {}
        eff_count = event_count if event_count is not None else int(d.get("event_count") or max(int(d.get("next_seq", 1)) - 1, 0))
        return cls(
            execution_id=str(d["execution_id"]),
            parent_execution_id=d.get("parent_execution_id"),
            task_prompt=str(d.get("task_prompt", "")),
            status=str(d.get("status", ExecutionStatus.INITIALIZED.value)),
            verification_status=str(d.get("verification_status", VerificationStatus.NOT_VERIFIED.value)),
            source_component=str(d.get("source_component", "runtime")),
            caller_ring=d.get("caller_ring"),
            task_local_scoped=bool(d.get("task_local_scoped", False)),
            created_at=float(d.get("created_at", 0.0)),
            updated_at=float(d.get("updated_at", 0.0)),
            completed_at=float(d["completed_at"]) if d.get("completed_at") is not None else None,
            final_summary=d.get("final_summary"),
            error=d.get("error"),
            metadata=meta,
            redacted=bool(d.get("redacted", False)),
            truncated=bool(d.get("truncated", False)),
            event_count=eff_count,
        )


@dataclass
class ExecutionContextToken:
    """Trusted runtime scope describing the active task execution."""

    execution_id: str
    ledger: "CentralExecutionLedger"
    caller_ring: Optional[int] = None
    task_local_scoped: bool = False
    source_component: str = "runtime"
    correlation_id: Optional[str] = None
    owned_by_outer: bool = False
    telemetry_bag: Dict[str, Any] = field(default_factory=dict)


_LEDGER_EXECUTION_CONTEXT: contextvars.ContextVar[Optional[ExecutionContextToken]] = contextvars.ContextVar(
    "zelyr_ledger_exec_ctx",
    default=None,
)


def get_active_execution_context() -> Optional[ExecutionContextToken]:
    """Return the active trusted ledger execution context, if any."""
    return _LEDGER_EXECUTION_CONTEXT.get()


def attach_call_telemetry(metadata: Dict[str, Any]) -> None:
    """Attach sanitized runtime telemetry (e.g., VLA backend, external source) to active call context."""
    ctx = _LEDGER_EXECUTION_CONTEXT.get()
    if ctx is None or not isinstance(metadata, dict):
        return
    clean_meta, _, _ = sanitize_ledger_payload(metadata, max_bytes=1024)
    ctx.telemetry_bag.update(clean_meta)


@contextmanager
def execution_scope(
    ledger: "CentralExecutionLedger",
    execution_id: str,
    *,
    caller_ring: Optional[int] = None,
    task_local_scoped: bool = False,
    source_component: str = "runtime",
    correlation_id: Optional[str] = None,
    owned_by_outer: bool = False,
) -> Iterator[ExecutionContextToken]:
    """Bind an isolated trusted execution context across sync/async execution steps."""
    parent = _LEDGER_EXECUTION_CONTEXT.get()
    eff_ring = caller_ring if caller_ring is not None else (parent.caller_ring if parent else None)
    eff_scoped = bool(task_local_scoped or (parent.task_local_scoped if parent else False))
    token_obj = ExecutionContextToken(
        execution_id=str(execution_id),
        ledger=ledger,
        caller_ring=eff_ring,
        task_local_scoped=eff_scoped,
        source_component=source_component,
        correlation_id=correlation_id,
        owned_by_outer=owned_by_outer,
        telemetry_bag={},
    )
    var_token = _LEDGER_EXECUTION_CONTEXT.set(token_obj)
    try:
        yield token_obj
    finally:
        _LEDGER_EXECUTION_CONTEXT.reset(var_token)


def resolve_execution_ledger_db_path(db_path: Optional[Union[str, Path]] = None) -> Path:
    """Resolve persistent execution ledger SQLite path without colliding with memory.db."""
    if db_path is not None:
        raw_str = str(db_path)
        if raw_str == ":memory:":
            return Path(":memory:")
        return Path(raw_str).expanduser()

    env_ledger = os.environ.get("ZELYR_LEDGER_DB", "").strip()
    if env_ledger:
        if env_ledger == ":memory:":
            return Path(":memory:")
        return Path(env_ledger).expanduser()

    try:
        from zelyr.config import get_config_dir

        return get_config_dir() / "execution_ledger.db"
    except Exception:
        cfg_override = os.environ.get("ZELYR_CONFIG_DIR") or os.environ.get("XDG_CONFIG_HOME")
        if cfg_override:
            base = Path(cfg_override).expanduser()
            if os.environ.get("ZELYR_CONFIG_DIR"):
                return base / "execution_ledger.db"
            return base / "zelyr" / "execution_ledger.db"
        return Path.home() / ".config" / "zelyr" / "execution_ledger.db"


_EXECUTION_LEDGER_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS executions (
    execution_id TEXT PRIMARY KEY,
    parent_execution_id TEXT,
    task_prompt TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'initialized',
    verification_status TEXT NOT NULL DEFAULT 'not_verified',
    source_component TEXT NOT NULL DEFAULT 'runtime',
    caller_ring INTEGER,
    task_local_scoped INTEGER NOT NULL DEFAULT 0,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    completed_at REAL,
    final_summary TEXT,
    error TEXT,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    redacted INTEGER NOT NULL DEFAULT 0,
    truncated INTEGER NOT NULL DEFAULT 0,
    next_seq INTEGER NOT NULL DEFAULT 1
);

CREATE TABLE IF NOT EXISTS execution_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id TEXT NOT NULL,
    execution_id TEXT NOT NULL,
    sequence_number INTEGER NOT NULL,
    timestamp REAL NOT NULL,
    event_type TEXT NOT NULL,
    tool_id TEXT,
    source_id TEXT,
    source_type TEXT,
    is_external INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'recorded',
    correlation_id TEXT,
    caller_ring INTEGER,
    task_local_scoped INTEGER NOT NULL DEFAULT 0,
    redacted INTEGER NOT NULL DEFAULT 0,
    truncated INTEGER NOT NULL DEFAULT 0,
    idempotency_key TEXT,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    FOREIGN KEY (execution_id) REFERENCES executions(execution_id) ON DELETE CASCADE,
    UNIQUE (execution_id, sequence_number),
    UNIQUE (execution_id, event_id),
    UNIQUE (execution_id, idempotency_key)
);

CREATE INDEX IF NOT EXISTS idx_exec_created ON executions(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_exec_status ON executions(status, updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_events_exec_seq ON execution_events(execution_id, sequence_number ASC);
CREATE INDEX IF NOT EXISTS idx_events_tool ON execution_events(tool_id);
CREATE INDEX IF NOT EXISTS idx_events_type ON execution_events(event_type);
"""


class CentralExecutionLedger:
    """Thread-safe, process-persistent centralized execution ledger backed by SQLite WAL."""

    def __init__(
        self,
        db_path: Optional[Union[str, Path]] = None,
        *,
        raise_on_persistence_error: bool = False,
        timeout: float = 5.0,
    ) -> None:
        self.db_path: Path = resolve_execution_ledger_db_path(db_path)
        self.raise_on_persistence_error: bool = bool(raise_on_persistence_error)
        self.timeout: float = float(timeout)
        self._lock = threading.RLock()
        self._conn: Optional[sqlite3.Connection] = None
        self._thread_conns: Dict[int, sqlite3.Connection] = {}
        self._initialized: bool = False
        self._persistence_errors: List[Dict[str, Any]] = []
        self._memory_executions: Dict[str, ExecutionRecord] = {}
        self._memory_events: Dict[str, List[ExecutionEvent]] = {}
        self._ensure_connection(raise_on_error=self.raise_on_persistence_error)

    def _record_persistence_failure(
        self,
        operation: str,
        exc: Exception,
        *,
        execution_id: Optional[str] = None,
        strict: Optional[bool] = None,
    ) -> None:
        safe_err, _, _ = sanitize_ledger_text(str(exc), max_bytes=512)
        entry = {
            "timestamp": time.time(),
            "operation": operation,
            "execution_id": execution_id,
            "exception_type": type(exc).__name__,
            "error": safe_err or type(exc).__name__,
        }
        self._persistence_errors.append(entry)
        logger.error(
            "Execution ledger persistence failure during %s (execution_id=%s): %s",
            operation,
            execution_id or "-",
            entry["error"],
        )
        should_raise = self.raise_on_persistence_error if strict is None else strict
        if should_raise:
            raise LedgerPersistenceError(
                f"Execution ledger persistence failed during '{operation}': {entry['error']}"
            ) from exc

    def _open_connection(self) -> sqlite3.Connection:
        if str(self.db_path) == ":memory:":
            conn = sqlite3.connect(":memory:", timeout=self.timeout, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA foreign_keys = ON;")
            return conn
        from zelyr.db.connection import get_db_connection

        return get_db_connection(self.db_path, timeout=self.timeout, row_factory=True)

    def _ensure_connection(self, *, raise_on_error: bool = False, execution_id: Optional[str] = None) -> Optional[sqlite3.Connection]:
        with self._lock:
            is_memory = str(self.db_path) == ":memory:"
            tid = 0 if is_memory else threading.get_ident()
            existing = self._conn if is_memory else self._thread_conns.get(tid)
            if existing is not None and self._initialized:
                return existing
            try:
                conn = existing if existing is not None else self._open_connection()
                if is_memory:
                    self._conn = conn
                else:
                    self._thread_conns[tid] = conn
                    if self._conn is None:
                        self._conn = conn
                if not self._initialized:
                    with conn:
                        conn.executescript(_EXECUTION_LEDGER_SCHEMA_SQL)
                    self._initialized = True
                return conn
            except Exception as exc:
                if not is_memory:
                    self._thread_conns.pop(tid, None)
                self._conn = None
                self._initialized = False
                self._record_persistence_failure(
                    "initialize",
                    exc,
                    execution_id=execution_id,
                    strict=raise_on_error,
                )
                return None

    def close(self) -> None:
        """Checkpoint WAL and close all thread-local SQLite ledger connections cleanly."""
        with self._lock:
            conns: List[sqlite3.Connection] = []
            seen_ids: Set[int] = set()
            for conn in list(self._thread_conns.values()) + ([self._conn] if self._conn is not None else []):
                if conn is not None and id(conn) not in seen_ids:
                    seen_ids.add(id(conn))
                    conns.append(conn)
            self._thread_conns.clear()
            self._conn = None
            self._initialized = False
            for idx, conn in enumerate(conns):
                try:
                    if str(self.db_path) == ":memory:":
                        conn.close()
                    elif idx == len(conns) - 1:
                        from zelyr.db.connection import checkpoint_and_close

                        checkpoint_and_close(conn)
                    else:
                        conn.close()
                except Exception:
                    try:
                        conn.close()
                    except Exception:
                        pass

    def get_persistence_errors(self) -> List[Dict[str, Any]]:
        """Return structured diagnostics for any ledger persistence errors encountered."""
        with self._lock:
            return [dict(item) for item in self._persistence_errors]

    def has_persistence_errors(self) -> bool:
        with self._lock:
            return len(self._persistence_errors) > 0

    def _enforce_scope_isolation(self, target_execution_id: str, *, enforce_active_scope: bool) -> None:
        if not enforce_active_scope:
            return
        ctx = get_active_execution_context()
        if ctx is not None and ctx.execution_id != target_execution_id:
            raise ExecutionIsolationError(
                f"Execution scope isolation violation: active execution '{ctx.execution_id}' cannot mutate '{target_execution_id}'."
            )

    def create_execution(
        self,
        task_prompt: str = "",
        *,
        execution_id: Optional[str] = None,
        parent_execution_id: Optional[str] = None,
        source_component: str = "runtime",
        caller_ring: Optional[int] = None,
        task_local_scoped: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
        status: Union[ExecutionStatus, str] = ExecutionStatus.INITIALIZED,
        emit_init_event: bool = True,
        strict: Optional[bool] = None,
    ) -> ExecutionRecord:
        """Create and persist a new execution record."""
        eff_exec_id = str(execution_id or f"exec_{uuid.uuid4().hex[:16]}")
        now = time.time()
        safe_prompt, prompt_red, prompt_trunc = sanitize_ledger_text(
            task_prompt or "",
            max_bytes=MAX_LEDGER_TEXT_BYTES,
            strip_injection=False,
        )
        safe_meta, meta_red, meta_trunc = sanitize_ledger_payload(metadata)
        was_red = bool(prompt_red or meta_red)
        was_trunc = bool(prompt_trunc or meta_trunc)
        status_str = status.value if isinstance(status, ExecutionStatus) else str(status)
        if parent_execution_id is None:
            active_ctx = get_active_execution_context()
            if active_ctx is not None and active_ctx.execution_id != eff_exec_id:
                parent_execution_id = active_ctx.execution_id

        with self._lock:
            conn = self._ensure_connection(
                raise_on_error=(self.raise_on_persistence_error if strict is None else strict),
                execution_id=eff_exec_id,
            )
            init_event_count = 0
            if conn is not None:
                try:
                    from zelyr.db.connection import managed_transaction

                    with managed_transaction(conn):
                        existing = conn.execute(
                            "SELECT * FROM executions WHERE execution_id = ?",
                            (eff_exec_id,),
                        ).fetchone()
                        if existing is not None:
                            return ExecutionRecord.from_row(existing)
                        conn.execute(
                            """
                            INSERT INTO executions (
                                execution_id, parent_execution_id, task_prompt, status,
                                verification_status, source_component, caller_ring,
                                task_local_scoped, created_at, updated_at, metadata_json,
                                redacted, truncated, next_seq
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """,
                            (
                                eff_exec_id,
                                parent_execution_id,
                                safe_prompt or "",
                                status_str,
                                VerificationStatus.NOT_VERIFIED.value,
                                str(source_component or "runtime"),
                                caller_ring,
                                1 if task_local_scoped else 0,
                                now,
                                now,
                                json.dumps(safe_meta, ensure_ascii=False, separators=(",", ":"), default=str),
                                1 if was_red else 0,
                                1 if was_trunc else 0,
                                1,
                            ),
                        )
                        if emit_init_event:
                            ev_id = f"evt_{uuid.uuid4().hex[:12]}"
                            ev_meta, ev_red, ev_trunc = sanitize_ledger_payload(
                                {"task_prompt": safe_prompt or "", **(safe_meta or {})},
                            )
                            conn.execute(
                                """
                                INSERT INTO execution_events (
                                    event_id, execution_id, sequence_number, timestamp, event_type,
                                    status, caller_ring, task_local_scoped, redacted, truncated,
                                    idempotency_key, metadata_json
                                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                                """,
                                (
                                    ev_id,
                                    eff_exec_id,
                                    1,
                                    now,
                                    LedgerEventType.TASK_INITIALIZED.value,
                                    status_str,
                                    caller_ring,
                                    1 if task_local_scoped else 0,
                                    1 if (ev_red or was_red) else 0,
                                    1 if (ev_trunc or was_trunc) else 0,
                                    f"{eff_exec_id}:task_initialized",
                                    json.dumps(ev_meta, ensure_ascii=False, separators=(",", ":"), default=str),
                                ),
                            )
                            conn.execute(
                                "UPDATE executions SET next_seq = 2, updated_at = ? WHERE execution_id = ?",
                                (now, eff_exec_id),
                            )
                            init_event_count = 1
                except Exception as exc:
                    self._record_persistence_failure(
                        "create_execution",
                        exc,
                        execution_id=eff_exec_id,
                        strict=strict,
                    )

            record = ExecutionRecord(
                execution_id=eff_exec_id,
                parent_execution_id=parent_execution_id,
                task_prompt=safe_prompt or "",
                status=status_str,
                verification_status=VerificationStatus.NOT_VERIFIED.value,
                source_component=str(source_component or "runtime"),
                caller_ring=caller_ring,
                task_local_scoped=bool(task_local_scoped),
                created_at=now,
                updated_at=now,
                metadata=safe_meta,
                redacted=was_red,
                truncated=was_trunc,
                event_count=init_event_count,
            )
            self._memory_executions[eff_exec_id] = record
            self._memory_events.setdefault(eff_exec_id, [])
            if emit_init_event and not self._memory_events[eff_exec_id]:
                self._memory_events[eff_exec_id].append(
                    ExecutionEvent(
                        event_id=f"evt_{uuid.uuid4().hex[:12]}",
                        execution_id=eff_exec_id,
                        sequence_number=1,
                        timestamp=now,
                        event_type=LedgerEventType.TASK_INITIALIZED.value,
                        status=status_str,
                        caller_ring=caller_ring,
                        task_local_scoped=bool(task_local_scoped),
                        redacted=was_red,
                        truncated=was_trunc,
                        idempotency_key=f"{eff_exec_id}:task_initialized",
                        metadata=safe_meta,
                    )
                )
            return record

    def record_event(
        self,
        execution_id: str,
        event_type: Union[LedgerEventType, str],
        *,
        tool_id: Optional[str] = None,
        source_id: Optional[str] = None,
        source_type: Optional[str] = None,
        is_external: bool = False,
        status: str = "recorded",
        correlation_id: Optional[str] = None,
        caller_ring: Optional[int] = None,
        task_local_scoped: Optional[bool] = None,
        metadata: Optional[Dict[str, Any]] = None,
        idempotency_key: Optional[str] = None,
        event_id: Optional[str] = None,
        enforce_active_scope: bool = False,
        execution_status: Optional[Union[ExecutionStatus, str]] = None,
        verification_status: Optional[Union[VerificationStatus, str]] = None,
        strict: Optional[bool] = None,
    ) -> Optional[ExecutionEvent]:
        """Append an ordered, redacted, idempotent event to an execution record."""
        if not execution_id:
            raise ValueError("execution_id must be a non-empty string")
        self._enforce_scope_isolation(execution_id, enforce_active_scope=enforce_active_scope)

        ev_type = event_type.value if isinstance(event_type, LedgerEventType) else str(event_type)
        now = time.time()
        safe_tool = sanitize_ledger_text(tool_id, max_bytes=256, is_external=is_external)[0] if tool_id else None
        safe_source = sanitize_ledger_text(source_id, max_bytes=256, is_external=is_external)[0] if source_id else None
        safe_source_type = sanitize_ledger_text(source_type, max_bytes=64, is_external=is_external)[0] if source_type else None
        safe_meta, meta_red, meta_trunc = sanitize_ledger_payload(metadata, is_external=is_external)
        eff_event_id = str(event_id or f"evt_{uuid.uuid4().hex[:16]}")

        ctx = get_active_execution_context()
        eff_ring = caller_ring if caller_ring is not None else (ctx.caller_ring if ctx and ctx.execution_id == execution_id else None)
        eff_scoped = (
            bool(task_local_scoped)
            if task_local_scoped is not None
            else (bool(ctx.task_local_scoped) if ctx and ctx.execution_id == execution_id else False)
        )

        with self._lock:
            conn = self._ensure_connection(
                raise_on_error=(self.raise_on_persistence_error if strict is None else strict),
                execution_id=execution_id,
            )
            if conn is not None:
                try:
                    from zelyr.db.connection import managed_transaction

                    with managed_transaction(conn):
                        exec_row = conn.execute(
                            "SELECT next_seq, status, verification_status FROM executions WHERE execution_id = ?",
                            (execution_id,),
                        ).fetchone()
                        if exec_row is None:
                            raise ValueError(f"Unknown execution_id '{execution_id}'")

                        if idempotency_key:
                            dup = conn.execute(
                                "SELECT * FROM execution_events WHERE execution_id = ? AND idempotency_key = ?",
                                (execution_id, idempotency_key),
                            ).fetchone()
                            if dup is not None:
                                return ExecutionEvent.from_row(dup)

                        seq = int(exec_row["next_seq"])
                        conn.execute(
                            """
                            INSERT INTO execution_events (
                                event_id, execution_id, sequence_number, timestamp, event_type,
                                tool_id, source_id, source_type, is_external, status,
                                correlation_id, caller_ring, task_local_scoped,
                                redacted, truncated, idempotency_key, metadata_json
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """,
                            (
                                eff_event_id,
                                execution_id,
                                seq,
                                now,
                                ev_type,
                                safe_tool,
                                safe_source,
                                safe_source_type,
                                1 if is_external else 0,
                                str(status or "recorded"),
                                correlation_id,
                                eff_ring,
                                1 if eff_scoped else 0,
                                1 if meta_red else 0,
                                1 if meta_trunc else 0,
                                idempotency_key,
                                json.dumps(safe_meta, ensure_ascii=False, separators=(",", ":"), default=str),
                            ),
                        )

                        new_exec_status = (
                            (execution_status.value if isinstance(execution_status, ExecutionStatus) else str(execution_status))
                            if execution_status is not None
                            else (
                                ExecutionStatus.RUNNING.value
                                if exec_row["status"] == ExecutionStatus.INITIALIZED.value
                                and ev_type != LedgerEventType.TASK_INITIALIZED.value
                                else str(exec_row["status"])
                            )
                        )
                        new_ver_status = (
                            (verification_status.value if isinstance(verification_status, VerificationStatus) else str(verification_status))
                            if verification_status is not None
                            else str(exec_row["verification_status"])
                        )
                        conn.execute(
                            """
                            UPDATE executions
                            SET next_seq = ?, updated_at = ?, status = ?, verification_status = ?,
                                redacted = MAX(redacted, ?), truncated = MAX(truncated, ?)
                            WHERE execution_id = ?
                            """,
                            (
                                seq + 1,
                                now,
                                new_exec_status,
                                new_ver_status,
                                1 if meta_red else 0,
                                1 if meta_trunc else 0,
                                execution_id,
                            ),
                        )
                        return ExecutionEvent(
                            event_id=eff_event_id,
                            execution_id=execution_id,
                            sequence_number=seq,
                            timestamp=now,
                            event_type=ev_type,
                            tool_id=safe_tool,
                            source_id=safe_source,
                            source_type=safe_source_type,
                            is_external=bool(is_external),
                            status=str(status or "recorded"),
                            correlation_id=correlation_id,
                            caller_ring=eff_ring,
                            task_local_scoped=bool(eff_scoped),
                            redacted=bool(meta_red),
                            truncated=bool(meta_trunc),
                            idempotency_key=idempotency_key,
                            metadata=safe_meta,
                        )
                except ValueError:
                    raise
                except Exception as exc:
                    self._record_persistence_failure(
                        "record_event",
                        exc,
                        execution_id=execution_id,
                        strict=strict,
                    )

            mem_events = self._memory_events.setdefault(execution_id, [])
            if idempotency_key:
                for existing_ev in mem_events:
                    if existing_ev.idempotency_key == idempotency_key:
                        return existing_ev
            seq_num = len(mem_events) + 1
            mem_ev = ExecutionEvent(
                event_id=eff_event_id,
                execution_id=execution_id,
                sequence_number=seq_num,
                timestamp=now,
                event_type=ev_type,
                tool_id=safe_tool,
                source_id=safe_source,
                source_type=safe_source_type,
                is_external=bool(is_external),
                status=str(status or "recorded"),
                correlation_id=correlation_id,
                caller_ring=eff_ring,
                task_local_scoped=bool(eff_scoped),
                redacted=bool(meta_red),
                truncated=bool(meta_trunc),
                idempotency_key=idempotency_key,
                metadata=safe_meta,
            )
            mem_events.append(mem_ev)
            return mem_ev

    def finalize_execution(
        self,
        execution_id: str,
        *,
        status: Union[ExecutionStatus, str],
        verification_status: Optional[Union[VerificationStatus, str]] = None,
        final_summary: Optional[str] = None,
        error: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        emit_terminal_event: bool = True,
        strict: Optional[bool] = None,
    ) -> Optional[ExecutionRecord]:
        """Finalize an execution record and emit its terminal lifecycle event."""
        now = time.time()
        status_str = status.value if isinstance(status, ExecutionStatus) else str(status)
        ver_str = (
            (verification_status.value if isinstance(verification_status, VerificationStatus) else str(verification_status))
            if verification_status is not None
            else None
        )
        safe_summary, sum_red, sum_trunc = sanitize_ledger_text(final_summary, max_bytes=MAX_LEDGER_TEXT_BYTES)
        safe_error, err_red, err_trunc = sanitize_ledger_text(error, max_bytes=MAX_LEDGER_TEXT_BYTES)
        safe_meta, meta_red, meta_trunc = sanitize_ledger_payload(metadata)
        any_red = bool(sum_red or err_red or meta_red)
        any_trunc = bool(sum_trunc or err_trunc or meta_trunc)

        if emit_terminal_event:
            term_meta = dict(safe_meta)
            if safe_summary:
                term_meta["summary"] = safe_summary
            if safe_error:
                term_meta["error"] = safe_error
            if ver_str:
                term_meta["verification_status"] = ver_str
            try:
                self.record_event(
                    execution_id,
                    LedgerEventType.TASK_TERMINATED,
                    status=status_str,
                    metadata=term_meta,
                    idempotency_key=f"{execution_id}:terminated:{status_str}",
                    execution_status=status_str,
                    verification_status=ver_str,
                    strict=strict,
                )
            except ValueError:
                return None

        with self._lock:
            conn = self._ensure_connection(
                raise_on_error=(self.raise_on_persistence_error if strict is None else strict),
                execution_id=execution_id,
            )
            if conn is not None:
                try:
                    from zelyr.db.connection import managed_transaction

                    with managed_transaction(conn):
                        existing = conn.execute(
                            "SELECT * FROM executions WHERE execution_id = ?",
                            (execution_id,),
                        ).fetchone()
                        if existing is None:
                            return None
                        merged_meta = {}
                        try:
                            merged_meta = json.loads(existing["metadata_json"] or "{}")
                        except Exception:
                            merged_meta = {}
                        merged_meta.update(safe_meta)
                        eff_ver = ver_str or existing["verification_status"]
                        conn.execute(
                            """
                            UPDATE executions
                            SET status = ?, verification_status = ?, updated_at = ?, completed_at = ?,
                                final_summary = COALESCE(?, final_summary),
                                error = COALESCE(?, error),
                                metadata_json = ?,
                                redacted = MAX(redacted, ?),
                                truncated = MAX(truncated, ?)
                            WHERE execution_id = ?
                            """,
                            (
                                status_str,
                                eff_ver,
                                now,
                                now,
                                safe_summary,
                                safe_error,
                                json.dumps(merged_meta, ensure_ascii=False, separators=(",", ":"), default=str),
                                1 if any_red else 0,
                                1 if any_trunc else 0,
                                execution_id,
                            ),
                        )
                        updated = conn.execute(
                            "SELECT * FROM executions WHERE execution_id = ?",
                            (execution_id,),
                        ).fetchone()
                        if updated is not None:
                            return ExecutionRecord.from_row(updated)
                except Exception as exc:
                    self._record_persistence_failure(
                        "finalize_execution",
                        exc,
                        execution_id=execution_id,
                        strict=strict,
                    )

            mem_rec = self._memory_executions.get(execution_id)
            if mem_rec is None:
                return None
            updated_rec = ExecutionRecord(
                execution_id=mem_rec.execution_id,
                parent_execution_id=mem_rec.parent_execution_id,
                task_prompt=mem_rec.task_prompt,
                status=status_str,
                verification_status=ver_str or mem_rec.verification_status,
                source_component=mem_rec.source_component,
                caller_ring=mem_rec.caller_ring,
                task_local_scoped=mem_rec.task_local_scoped,
                created_at=mem_rec.created_at,
                updated_at=now,
                completed_at=now,
                final_summary=safe_summary or mem_rec.final_summary,
                error=safe_error or mem_rec.error,
                metadata={**mem_rec.metadata, **safe_meta},
                redacted=mem_rec.redacted or any_red,
                truncated=mem_rec.truncated or any_trunc,
                event_count=len(self._memory_events.get(execution_id, [])),
            )
            self._memory_executions[execution_id] = updated_rec
            return updated_rec

    def get_execution(self, execution_id: str) -> Optional[ExecutionRecord]:
        """Fetch a single execution record by ID."""
        if not execution_id:
            return None
        with self._lock:
            conn = self._ensure_connection(raise_on_error=False, execution_id=execution_id)
            if conn is not None:
                try:
                    row = conn.execute(
                        "SELECT * FROM executions WHERE execution_id = ?",
                        (execution_id,),
                    ).fetchone()
                    if row is not None:
                        return ExecutionRecord.from_row(row)
                except Exception as exc:
                    self._record_persistence_failure("get_execution", exc, execution_id=execution_id, strict=False)
            return self._memory_executions.get(execution_id)

    def find_execution_by_prefix(self, prefix: str) -> Tuple[Optional[ExecutionRecord], List[str]]:
        """Resolve an execution ID or unique prefix match."""
        clean = (prefix or "").strip()
        if not clean:
            return None, []
        exact = self.get_execution(clean)
        if exact is not None:
            return exact, [exact.execution_id]
        with self._lock:
            conn = self._ensure_connection(raise_on_error=False)
            if conn is not None:
                try:
                    rows = conn.execute(
                        "SELECT * FROM executions WHERE execution_id LIKE ? ORDER BY created_at DESC LIMIT 20",
                        (f"{clean}%",),
                    ).fetchall()
                    ids = [str(r["execution_id"]) for r in rows]
                    if len(rows) == 1:
                        return ExecutionRecord.from_row(rows[0]), ids
                    return None, ids
                except Exception as exc:
                    self._record_persistence_failure("find_execution_by_prefix", exc, strict=False)
        return None, []

    def get_events(
        self,
        execution_id: str,
        *,
        event_type: Optional[str] = None,
        tool_id: Optional[str] = None,
        correlation_id: Optional[str] = None,
    ) -> List[ExecutionEvent]:
        """Retrieve deterministically ordered events for an execution."""
        if not execution_id:
            return []
        with self._lock:
            conn = self._ensure_connection(raise_on_error=False, execution_id=execution_id)
            if conn is not None:
                try:
                    clauses = ["execution_id = ?"]
                    params: List[Any] = [execution_id]
                    if event_type:
                        clauses.append("event_type = ?")
                        params.append(str(event_type))
                    if tool_id:
                        clauses.append("tool_id = ?")
                        params.append(str(tool_id))
                    if correlation_id:
                        clauses.append("correlation_id = ?")
                        params.append(str(correlation_id))
                    where_sql = " AND ".join(clauses)
                    rows = conn.execute(
                        f"SELECT * FROM execution_events WHERE {where_sql} ORDER BY sequence_number ASC",
                        params,
                    ).fetchall()
                    return [ExecutionEvent.from_row(r) for r in rows]
                except Exception as exc:
                    self._record_persistence_failure("get_events", exc, execution_id=execution_id, strict=False)
            evs = list(self._memory_events.get(execution_id, []))
            if event_type:
                evs = [e for e in evs if e.event_type == str(event_type)]
            if tool_id:
                evs = [e for e in evs if e.tool_id == str(tool_id)]
            if correlation_id:
                evs = [e for e in evs if e.correlation_id == str(correlation_id)]
            return sorted(evs, key=lambda e: e.sequence_number)

    def list_executions(
        self,
        *,
        limit: int = 20,
        status: Optional[str] = None,
        tool_id: Optional[str] = None,
        event_type: Optional[str] = None,
        verification_status: Optional[str] = None,
    ) -> List[ExecutionRecord]:
        """List recent executions ordered newest-first with optional filters."""
        eff_limit = max(1, min(int(limit or 20), 500))
        with self._lock:
            conn = self._ensure_connection(raise_on_error=False)
            if conn is not None:
                try:
                    clauses: List[str] = []
                    params: List[Any] = []
                    if status:
                        clauses.append("e.status = ?")
                        params.append(str(status))
                    if verification_status:
                        clauses.append("e.verification_status = ?")
                        params.append(str(verification_status))
                    if tool_id:
                        clauses.append(
                            "EXISTS (SELECT 1 FROM execution_events ev WHERE ev.execution_id = e.execution_id AND ev.tool_id = ?)"
                        )
                        params.append(str(tool_id))
                    if event_type:
                        clauses.append(
                            "EXISTS (SELECT 1 FROM execution_events ev WHERE ev.execution_id = e.execution_id AND ev.event_type = ?)"
                        )
                        params.append(str(event_type))
                    where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
                    rows = conn.execute(
                        f"SELECT e.* FROM executions e {where_sql} ORDER BY e.created_at DESC, e.rowid DESC LIMIT ?",
                        (*params, eff_limit),
                    ).fetchall()
                    return [ExecutionRecord.from_row(r) for r in rows]
                except Exception as exc:
                    self._record_persistence_failure("list_executions", exc, strict=False)
            recs = list(self._memory_executions.values())
            if status:
                recs = [r for r in recs if r.status == str(status)]
            if verification_status:
                recs = [r for r in recs if r.verification_status == str(verification_status)]
            recs.sort(key=lambda r: r.created_at, reverse=True)
            return recs[:eff_limit]

    def get_execution_summary(
        self,
        execution_id: str,
        *,
        event_type: Optional[str] = None,
        tool_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Assemble a complete structured diagnostic summary for an execution."""
        record = self.get_execution(execution_id)
        if record is None:
            return None
        all_events = self.get_events(execution_id)
        filtered_events = (
            [
                ev
                for ev in all_events
                if (not event_type or ev.event_type == event_type) and (not tool_id or ev.tool_id == tool_id)
            ]
            if (event_type or tool_id)
            else all_events
        )
        tools_used: List[str] = []
        external_tools: List[Dict[str, Any]] = []
        seen_ext: Set[Tuple[Optional[str], Optional[str]]] = set()
        auth_decisions: List[Dict[str, Any]] = []
        recovery_attempts: int = 0
        verifications: List[Dict[str, Any]] = []

        for ev in all_events:
            if ev.tool_id and ev.tool_id not in tools_used:
                tools_used.append(ev.tool_id)
            if ev.is_external and (ev.tool_id, ev.source_id) not in seen_ext:
                seen_ext.add((ev.tool_id, ev.source_id))
                external_tools.append({
                    "tool_id": ev.tool_id,
                    "source_id": ev.source_id,
                    "source_type": ev.source_type,
                })
            if ev.event_type == LedgerEventType.AUTHORIZATION_DECISION.value:
                auth_decisions.append({
                    "sequence_number": ev.sequence_number,
                    "tool_id": ev.tool_id,
                    "status": ev.status,
                    "caller_ring": ev.caller_ring,
                    "task_local_scoped": ev.task_local_scoped,
                    "reason": ev.metadata.get("reason"),
                })
            elif ev.event_type == LedgerEventType.RECOVERY_ATTEMPT.value:
                recovery_attempts += 1
            elif ev.event_type == LedgerEventType.VERIFICATION_RECORDED.value:
                verifications.append({
                    "sequence_number": ev.sequence_number,
                    "status": ev.status,
                    "tool_id": ev.tool_id,
                    "detail": ev.metadata.get("detail"),
                    "target": ev.metadata.get("target"),
                })

        exec_persistence_errors = [
            err for err in self.get_persistence_errors()
            if err.get("execution_id") in (execution_id, None)
        ]

        return {
            "execution": record.to_dict(),
            "event_count": len(all_events),
            "tools_used": tools_used,
            "external_tools": external_tools,
            "authorization_decisions": auth_decisions,
            "recovery_attempts": recovery_attempts,
            "verifications": verifications,
            "persistence_errors": exec_persistence_errors,
            "events": [ev.to_dict() for ev in filtered_events],
        }


_DEFAULT_LEDGER_LOCK = threading.RLock()
_DEFAULT_LEDGER_INSTANCES: Dict[Path, CentralExecutionLedger] = {}


def get_default_central_ledger(db_path: Optional[Union[str, Path]] = None) -> CentralExecutionLedger:
    """Return a cached thread-safe CentralExecutionLedger instance for the resolved path."""
    resolved = resolve_execution_ledger_db_path(db_path)
    with _DEFAULT_LEDGER_LOCK:
        existing = _DEFAULT_LEDGER_INSTANCES.get(resolved)
        if existing is not None:
            return existing
        instance = CentralExecutionLedger(db_path=resolved, raise_on_persistence_error=False)
        _DEFAULT_LEDGER_INSTANCES[resolved] = instance
        return instance


def reset_default_central_ledger() -> None:
    """Close and clear all cached default CentralExecutionLedger instances (test helper)."""
    with _DEFAULT_LEDGER_LOCK:
        for inst in _DEFAULT_LEDGER_INSTANCES.values():
            inst.close()
        _DEFAULT_LEDGER_INSTANCES.clear()
