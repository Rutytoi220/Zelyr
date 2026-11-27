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
import functools
import hashlib
import json
import logging
import os
from pathlib import Path
import re
import sqlite3
import threading
import time
from typing import Any, Dict, Iterator, List, Optional, Sequence, Set, Tuple, Union
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


_SENSITIVE_ENV_CACHE_LOCK = threading.RLock()
_SENSITIVE_ENV_CACHE_DATA: Optional[Dict[Any, Any]] = None
_SENSITIVE_ENV_CACHE_VALUES: Tuple[str, ...] = ()


def _collect_sensitive_env_values() -> Tuple[str, ...]:
    """Collect non-trivial secret environment values for dynamic scrubbing."""
    global _SENSITIVE_ENV_CACHE_DATA, _SENSITIVE_ENV_CACHE_VALUES
    env_data = getattr(os.environ, "_data", None)
    if isinstance(env_data, dict):
        with _SENSITIVE_ENV_CACHE_LOCK:
            if _SENSITIVE_ENV_CACHE_DATA is not None and env_data == _SENSITIVE_ENV_CACHE_DATA:
                return _SENSITIVE_ENV_CACHE_VALUES
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
    result = tuple(secrets)
    if isinstance(env_data, dict):
        with _SENSITIVE_ENV_CACHE_LOCK:
            _SENSITIVE_ENV_CACHE_DATA = env_data.copy()
            _SENSITIVE_ENV_CACHE_VALUES = result
    return result


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


_GLOBAL_NO_FILE_MOD_PATTERNS = (
    re.compile(
        r"\b(?:do\s+not|don'?t|never|must\s+not|should\s+not|shall\s+not|cannot|can'?t|may\s+not|avoid|refrain\s+from|without)\s+"
        r"(?:ever\s+)?(?:any\s+)?(?:modify|modifying|edit|editing|change|changing|alter|altering|write(?:\s+to)?|writing(?:\s+to)?|"
        r"create|creating|delete|deleting|remove|removing|touch|touching|patch|patching|update|updating|overwrite|overwriting|mutate|mutating)\s+"
        r"(?:anything|any\s+(?:files?|directories?|dirs?|folders?|paths?|code)|files?|the\s+(?:repo|repository|project|workspace|codebase))\b",
        flags=re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:leave|keep)\s+(?:all\s+files|everything|the\s+(?:repo|repository|project|workspace|codebase))\s+"
        r"(?:untouched|alone|unchanged|intact|unmodified|as[\s\-]is)\b",
        flags=re.IGNORECASE,
    ),
    re.compile(r"\bread[\s\-]?only\s+(?:inspection|mode|analysis)\b", flags=re.IGNORECASE),
)

_GLOBAL_NO_MUTATING_CMD_PATTERNS = (
    re.compile(
        r"\b(?:do\s+not|don'?t|never|must\s+not|should\s+not|cannot|can'?t|avoid|refrain\s+from|without)\b[^.;\n]{0,80}"
        r"\b(?:run|running|execute|executing|use|using)\s+"
        r"(?:any\s+)?mutating\s+(?:shell\s+)?(?:commands?|tools?|actions?)\b",
        flags=re.IGNORECASE,
    ),
)

_DIAGNOSTIC_TEST_INTENT_RE = re.compile(
    r"\b(?:run|execute)\s+(?:the\s+)?(?:existing\s+)?(?:test\s+suite|tests|pytest|unittest)\b|"
    r"\bwhether\s+(?:the\s+)?tests\s+pass\s+or\s+fail\b",
    flags=re.IGNORECASE,
)

_MUTATING_SHELL_OP_PATTERNS = (
    re.compile(r"(?:^|[^0-9])>{1,2}|<<"),
    re.compile(r"\|\s*tee\b", flags=re.IGNORECASE),
    re.compile(r"\b(?:rm|unlink|mv|cp|touch|mkdir|rmdir|truncate|chmod|chown|chgrp|ln|dd|install)\b"),
    re.compile(r"\b(?:sed|perl)\s+-[^\s]*i"),
    re.compile(r"\b(?:pip|pip3|uv\s+pip)\s+(?:install|uninstall)\b"),
    re.compile(r"\buv\s+(?:add|remove|sync)\b"),
    re.compile(r"\bpython(?:3(?:\.\d+)?)?\s+-m\s+venv\b|\bvirtualenv\b"),
    re.compile(r"\bgit\s+(?:add|commit|reset|checkout|clean|rm|mv|switch|restore|apply|cherry-pick|rebase|merge|tag\s+-)\b"),
    re.compile(r"\bpython(?:3(?:\.\d+)?)?\s+-c\b[^\n]*(?:open\s*\(|write_text|write_bytes|mkdir|unlink|rmdir|rmtree)"),
)

_DIAGNOSTIC_TEST_CMD_RE = re.compile(
    r"(?:^|(?:&&|\|\||;)\s*)(?:PYTHONPATH=[^\s]+\s+)?(?:\S*/)?(?:pytest|py\.test|python(?:3(?:\.\d+)?)?\s+-m\s+(?:pytest|unittest)|ruff\s+check|mypy)\b",
    flags=re.IGNORECASE,
)


def _strip_fenced_and_quoted(prompt: str) -> str:
    """Remove fenced code blocks and backtick/quoted literals to inspect imperative prompt instructions."""
    if not prompt:
        return ""
    cleaned = re.sub(r"```.*?```", " ", prompt, flags=re.DOTALL)
    cleaned = re.sub(r"`[^`]*`", " ", cleaned)
    return cleaned


def extract_global_mutation_prohibition(prompt: str) -> Dict[str, bool]:
    """Detect workspace-wide read-only / no-mutation directives without false positives on code blocks."""
    cleaned = _strip_fenced_and_quoted(prompt)
    if not cleaned:
        return {"no_file_modifications": False, "no_mutating_commands": False, "is_test_diagnostic": False}

    no_file_mod = any(pat.search(cleaned) for pat in _GLOBAL_NO_FILE_MOD_PATTERNS)
    no_mut_cmd = any(pat.search(cleaned) for pat in _GLOBAL_NO_MUTATING_CMD_PATTERNS)
    is_test_diag = bool(no_file_mod and _DIAGNOSTIC_TEST_INTENT_RE.search(cleaned))
    return {
        "no_file_modifications": no_file_mod,
        "no_mutating_commands": no_mut_cmd,
        "is_test_diagnostic": is_test_diag,
    }


_DESKTOP_INTERACT_PROHIBIT_RE = re.compile(
    r"\b(?:do\s+not|don'?t|never|must\s+not|should\s+not|shall\s+not|cannot|can'?t|may\s+not|avoid|refrain\s+from|without)\b"
    r"[^.;\n]{0,120}\b(?:click(?:ing)?|double[\s\-]?click(?:ing)?|right[\s\-]?click(?:ing)?|type|typing|press(?:ing)?\s+keys?|"
    r"send(?:ing)?\s+(?:input|keys?|clicks?)|move\s+(?:the\s+)?mouse|drag(?:ging)?|scroll(?:ing)?|"
    r"interact(?:ing)?\s+with|modify(?:ing)?|alter(?:ing)?|change|changing|trigger(?:ing)?|mutate|mutating)\b"
    r"[^.;\n]{0,80}\b(?:fixture|window|ui|desktop|screen|button|textbox|text\s+field|input\s+field|controls?)\b|"
    r"\b(?:leave|keep)\s+(?:the\s+)?(?:fixture|desktop\s+window|window|ui)\s+"
    r"(?:untouched|alone|unchanged|intact|unmodified|as[\s\-]is)\b",
    flags=re.IGNORECASE,
)

_DESKTOP_MUTATION_SHELL_RE = re.compile(
    r"\b(?:wtype|ydotool|xdotool|dotool)\b|\bhyprctl\s+dispatch\b",
    flags=re.IGNORECASE,
)


def extract_desktop_interaction_prohibition(prompt: str) -> bool:
    """Return True when the prompt explicitly forbids clicking, typing, or mutating desktop UI/windows."""
    cleaned = _strip_fenced_and_quoted(prompt)
    if not cleaned:
        return False
    if re.search(
        r"\b(?:do\s+not|don'?t|never|must\s+not|without|avoid)\s+(?:click|type|press\s+keys?|interact|modify|change|trigger)\b"
        r"[^.;\n]{0,80}\b(?:read[\s\-]?only|inspect|without\s+(?:clicking|typing|modifying))\b",
        cleaned,
        flags=re.IGNORECASE,
    ):
        return True
    return bool(_DESKTOP_INTERACT_PROHIBIT_RE.search(cleaned))


def is_diagnostic_test_command(tool_name: str, arguments: Dict[str, Any]) -> bool:
    """Return True if tool call runs a non-mutating test/lint command (e.g. pytest, unittest, ruff check)."""
    if (tool_name or "").lower() not in ("execute_command", "bash", "sh"):
        return False
    cmd = str((arguments or {}).get("command") or (arguments or {}).get("cmd") or "").strip()
    if not cmd:
        return False
    if any(pat.search(cmd) for pat in _MUTATING_SHELL_OP_PATTERNS):
        return False
    return bool(_DIAGNOSTIC_TEST_CMD_RE.search(cmd))


def is_mutating_tool_call(
    tool_name: str,
    arguments: Dict[str, Any],
    *,
    allow_read_only_tests: bool = False,
) -> Tuple[bool, str]:
    """Return (is_mutating, reason) for enforcing workspace-wide read-only prohibitions."""
    t_lower = (tool_name or "").lower().strip()
    args = arguments if isinstance(arguments, dict) else {}

    if t_lower == "manage_workspace_file":
        act = str(args.get("action") or "read").strip().lower()
        if act in ("read", "view", "list", "list_tree", "stat", "exists", "inspect", "check"):
            return False, ""
        return True, f"manage_workspace_file action '{act}' mutates workspace state"

    if t_lower in ("file_write", "write_file", "file_edit", "edit_file", "craft_tool", "create_tool", "manage_tools"):
        return True, f"tool '{t_lower}' mutates files or plugins"

    if t_lower in ("execute_command", "bash", "sh"):
        cmd = str(args.get("command") or args.get("cmd") or "").strip()
        if not cmd:
            return False
        if any(pat.search(cmd) for pat in _MUTATING_SHELL_OP_PATTERNS) or _DESKTOP_MUTATION_SHELL_RE.search(cmd):
            return True, f"shell command performs write/mutation: {cmd[:80]}"
        if _is_inspection_action(t_lower, args):
            return False, ""
        if allow_read_only_tests and is_diagnostic_test_command(t_lower, args):
            return False, ""
        return True, f"shell command '{cmd[:80]}' is not in the read-only inspection set"

    if _is_inspection_action(t_lower, args):
        return False, ""
    if t_lower in ("remember_fact", "verify_outcome", "query_system_journal", "inspect_network", "web_search", "fetch_web_page"):
        return False, ""
    if t_lower == "manage_system_clipboard":
        return str(args.get("action", "read")).lower() != "read", "clipboard write"
    return True, f"tool '{t_lower}' may mutate host state"


def extract_file_targets_from_text(text: str) -> List[str]:
    """Extract file path candidates mentioned in prompt or text."""
    if not text:
        return []

    # Strip HTTP/WS/FTP URLs so URL route segments (e.g. /api/records/REC-204)
    # are never misclassified as local filesystem paths.
    scrubbed = re.sub(r"\b(?:https?|wss?|ftp)://[^\s\)\]\'\"<>]+", " ", text, flags=re.IGNORECASE)

    # 1. Match absolute paths
    raw_abs = re.findall(r"(?:/(?:[\w\.\-]+))+", scrubbed)
    cleaned_abs = set()
    for p in raw_abs:
        p_clean = re.sub(r"[.,:;!?]+$", "", p).strip()
        if p_clean and not p_clean.startswith(("/dev/", "/bin/", "/usr/bin/", "/proc/", "/sys/")):
            cleaned_abs.add(canonicalize_target(p_clean))

    # 2. Remove absolute paths from text before matching standalone relative paths
    text_no_abs = re.sub(r"(?:/(?:[\w\.\-]+))+", " ", scrubbed)
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
    if t_lower in (
        "read_file",
        "view_file",
        "list_dir",
        "list_tree",
        "get_state",
        "inspect_state",
        "capture_screen",
        "screen_capture",
        "capture_som_screen",
        "gui_inspect",
    ):
        return True
    if t_lower in ("desktop_control", "desktop_automation"):
        act = str(arguments.get("action") or "get_screen_size").strip().lower()
        return act in (
            "get_screen_size",
            "observe",
            "screenshot",
            "inspect_windows",
            "get_active_window",
            "get_cursor_position",
        )
    if t_lower == "manage_desktop_window":
        act = str(arguments.get("action") or "list").strip().lower()
        return act in ("list", "get_active", "monitors", "geometry", "inspect")
    if t_lower in ("interact_with_browser", "browser_cdp"):
        act = str(arguments.get("action") or "get_page_snapshot").strip().lower()
        return act in (
            "get_page_snapshot",
            "extract_page_content",
            "list_tabs",
            "get_active_tab",
            "get_dom",
            "scroll_page",
            "get_cookies",
        )
    if t_lower in ("get_page_snapshot", "fetch_web_page", "web_search"):
        return True
    if t_lower.startswith(("read_", "get_", "list_", "check_", "inspect_", "stat_")):
        return True
    if t_lower in ("execute_command", "bash", "sh"):
        cmd = str(arguments.get("command", "") or arguments.get("cmd", "")).strip()
        cmd_parts = cmd.split()
        first_word = cmd_parts[0].lower() if cmd_parts else ""
        if _DESKTOP_MUTATION_SHELL_RE.search(cmd):
            return False
        if first_word in (
            "ls", "cat", "head", "tail", "grep", "find", "pwd", "whoami", "uname", "stat", "file",
            "wc", "du", "df", "ps", "pgrep", "id", "env", "printenv", "which", "type", "readlink", "realpath",
            "md5sum", "sha256sum", "diff", "test", "[",
            "uptime", "ss", "netstat", "free", "hostname", "date", "lsof", "lsblk", "who", "w",
        ):
            if not any(op in cmd for op in (">", ">>", "| tee", "|tee")):
                return True
        if first_word == "curl":
            has_shell_mut = any(op in cmd for op in (">", ">>", "| tee", "|tee", ";", "&&", "||", "`", "$("))
            if not has_shell_mut:
                mut_curl = False
                skip_next_method = False
                for tok in cmd_parts[1:]:
                    if skip_next_method:
                        skip_next_method = False
                        if tok.upper() not in ("GET", "HEAD"):
                            mut_curl = True
                            break
                        continue
                    if tok in ("-X", "--request"):
                        skip_next_method = True
                        continue
                    if tok.startswith("-X") and tok[2:].upper() not in ("GET", "HEAD"):
                        mut_curl = True
                        break
                    if tok.startswith("--request=") and tok.split("=", 1)[1].upper() not in ("GET", "HEAD"):
                        mut_curl = True
                        break
                    if (
                        tok in ("-d", "-F", "-T", "-o", "-O")
                        or tok.startswith(("-d", "-F", "-T", "-o"))
                        or tok.startswith(("--data", "--form", "--upload-file", "--json", "--output", "--remote-name"))
                    ):
                        mut_curl = True
                        break
                if not mut_curl:
                    return True
        if first_word == "wget":
            has_shell_mut = any(op in cmd for op in (">", ">>", "| tee", "|tee", ";", "&&", "||", "`", "$("))
            if not has_shell_mut:
                stdout_wget = any(tok in ("-O-", "-qO-") for tok in cmd_parts[1:]) or any(
                    i + 1 < len(cmd_parts) and cmd_parts[i] in ("-O", "--output-document") and cmd_parts[i + 1] == "-"
                    for i in range(1, len(cmd_parts))
                )
                mut_wget = any(
                    tok.startswith(("--post-data", "--post-file", "--body-data", "--body-file"))
                    or (tok.startswith("--method=") and tok.split("=", 1)[1].upper() not in ("GET", "HEAD"))
                    for tok in cmd_parts[1:]
                )
                if stdout_wget and not mut_wget:
                    return True
        if first_word == "hyprctl" and len(cmd_parts) >= 2:
            h_sub = cmd_parts[1].lower()
            if h_sub in ("monitors", "clients", "activewindow", "activeworkspace", "workspaces", "cursorpos", "version"):
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

    if t_lower in (
        "capture_screen",
        "screen_capture",
        "desktop_control",
        "desktop_automation",
        "manage_desktop_window",
    ):
        for k in ("window", "expected_window"):
            val = arguments.get(k)
            if val and isinstance(val, str) and val.strip():
                return val.strip()
        raw_tgt = arguments.get("target")
        if raw_tgt and isinstance(raw_tgt, str) and raw_tgt.strip():
            raw_clean = raw_tgt.strip()
            if t_lower in ("capture_screen", "screen_capture") and raw_clean.lower() in (
                "active_monitor",
                "active_window",
                "window",
                "monitor",
                "region",
                "full",
            ):
                return None
            return raw_clean
        for k in ("monitor", "name", "id"):
            val = arguments.get(k)
            if val and isinstance(val, str) and val.strip():
                return val.strip()
        return None

    if t_lower in ("interact_with_browser", "browser_cdp"):
        act = str(arguments.get("action") or "").strip().lower()
        if act in ("navigate_url", "open_tab", "fetch_web_page"):
            raw_url = arguments.get("url")
            if raw_url and isinstance(raw_url, str) and raw_url.strip():
                return raw_url.strip().rstrip("/")
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
    if t_lower in ("interact_with_browser", "browser_cdp"):
        return f"browser:{str(arguments.get('action', '')).strip().lower()}"
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


def _extract_http_read_url(tool_name: str, arguments: Any) -> Optional[str]:
    """Extract normalized HTTP/HTTPS URL targeted by an HTTP fetch/browser/curl action."""
    if not isinstance(arguments, dict):
        return None
    t_lower = (tool_name or "").lower()
    if t_lower == "fetch_web_page":
        raw = arguments.get("url")
        return str(raw).strip().rstrip("/") if isinstance(raw, str) and raw.strip() else None
    if t_lower in ("interact_with_browser", "browser_cdp"):
        act = str(arguments.get("action") or "").strip().lower()
        if act in ("navigate_url", "open_tab", "get_page_snapshot", "extract_page_content", "get_dom"):
            raw = arguments.get("url")
            return str(raw).strip().rstrip("/") if isinstance(raw, str) and raw.strip() else None
        return None
    if t_lower in ("execute_command", "bash", "sh"):
        cmd = str(arguments.get("command") or arguments.get("cmd") or "").strip()
        if cmd.split()[:1] in (["curl"], ["wget"]):
            m = re.search(r"\bhttps?://[^\s\'\"<>`]+", cmd, flags=re.IGNORECASE)
            if m:
                return m.group(0).rstrip(".,;:!?\"'").rstrip("/")
    return None


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

    # Cross-surface recovery: a failed browser switch_tab lookup for a desktop window
    # resolves when Zelyr transitions to the matching OS desktop window surface.
    if (
        (failed.tool_name or "").lower() in ("interact_with_browser", "browser_cdp")
        and isinstance(failed.arguments, dict)
        and str(failed.arguments.get("action") or "").strip().lower() == "switch_tab"
        and "no open tab" in str(failed.error or "").lower()
        and (later.tool_name or "").lower() in ("capture_screen", "screen_capture", "manage_desktop_window", "desktop_control")
    ):
        q = str(failed.arguments.get("query") or failed.arguments.get("url") or "").strip().lower()
        t_later = _extract_action_target(later.tool_name, later.arguments)
        if not q or not t_later or q in t_later.lower() or t_later.lower() in q:
            return True

    # 1. An inspection must never resolve a mutation or execution failure
    if not failed_is_inspect and later_is_inspect:
        return False

    target_failed = _extract_action_target(failed.tool_name, failed.arguments)
    target_later = _extract_action_target(later.tool_name, later.arguments)

    # 1b. A later successful read-only observation with valid args resolves an earlier
    # failed observation attempt on the same tool & matching/superset target,
    # or an equivalent read-only HTTP inspection (e.g. curl GET resolving SSRF-blocked fetch_web_page).
    if failed_is_inspect and later_is_inspect:
        url_failed = _extract_http_read_url(failed.tool_name, failed.arguments)
        url_later = _extract_http_read_url(later.tool_name, later.arguments)
        if url_failed and url_later and url_failed == url_later:
            return True
        if (
            failed.tool_name
            and failed.tool_name == later.tool_name
            and failed.tool_name.lower() in ("capture_screen", "screen_capture", "manage_desktop_window")
        ):
            if target_failed is None or target_later is None:
                return True
            if targets_match(canonicalize_target(target_failed), canonicalize_target(target_later)):
                return True

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
        if (
            (failed.tool_name or "").lower() in ("interact_with_browser", "browser_cdp")
            and (later.tool_name or "").lower() in ("interact_with_browser", "browser_cdp")
        ):
            nav_ops = ("browser:navigate_url", "browser:open_tab", "browser:reload_tab")
            if op_failed in nav_ops and op_later in nav_ops:
                if op_later == "browser:reload_tab" or not target_later:
                    return True
                if target_later == target_failed or targets_match(target_failed, target_later):
                    return True
        if not later_targets or not any(targets_match(target_failed, lt) for lt in later_targets):
            # Different targets never resolve each other
            return False
        # Same target: operations must be compatible (e.g. write resolves write, append resolves append,
        # write/patch/edit/update on the same file resolve each other,
        # or any non-inspection corrective mutation/verification resolves a verification_failed state)
        # A window-management action (e.g. refocusing a window) prepares focus state
        # but never substitutes for executing a failed desktop_control input actuation.
        if (failed.tool_name or "").lower() in ("desktop_control", "desktop_automation") and op_later == "manage_desktop_window":
            return False
        if op_failed == op_later:
            return True
        edit_write_ops = ("write", "overwrite", "patch", "edit", "update", "replace", "create")
        if op_failed in edit_write_ops and op_later in edit_write_ops:
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


_LEDGER_INIT_LOCK = threading.Lock()


def _synchronized(fn: Any) -> Any:
    """Serialize access to TurnExecutionLedger state using its re-entrant lock."""

    @functools.wraps(fn)
    def wrapper(self: "TurnExecutionLedger", *args: Any, **kwargs: Any) -> Any:
        with self._sync_lock():
            return fn(self, *args, **kwargs)

    return wrapper


class TurnExecutionLedger:
    """Authoritative ledger recording all proposed, dispatched, and observed actions."""

    def __new__(cls, *args: Any, **kwargs: Any) -> "TurnExecutionLedger":
        instance = super().__new__(cls)
        instance._lock = threading.RLock()
        return instance

    def _sync_lock(self) -> threading.RLock:
        lock = getattr(self, "_lock", None)
        if lock is None:
            with _LEDGER_INIT_LOCK:
                lock = getattr(self, "_lock", None)
                if lock is None:
                    lock = threading.RLock()
                    self._lock = lock
        return lock

    def __init__(
        self,
        *,
        execution_id: Optional[str] = None,
        central_ledger: Optional[CentralExecutionLedger] = None,
        caller_ring: Optional[int] = None,
        task_local_scoped: bool = False,
        source_component: Optional[str] = None,
    ) -> None:
        self._lock = getattr(self, "_lock", None) or threading.RLock()
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
        self._structured_observations: Dict[str, Any] = {}
        self._verification_status_override: Optional[str] = None
        self._proposal_meta: Dict[str, Dict[str, Any]] = {}
        self._interrupted_proposal_snapshots: List[Dict[str, Any]] = []
        self._next_proposal_order: int = 0
        self._task_graph: Optional[Any] = None
        self._model_request_count: int = 0
        self._model_request_duration_ms: float = 0.0
        self._orchestration_step_count: int = 0
        self._cached_desktop_vision_b64: Optional[str] = None

    @_synchronized
    def record_model_request(self, duration_ms: float) -> None:
        """Record a measured monotonic model-request count and duration in memory."""
        self._model_request_count += 1
        try:
            self._model_request_duration_ms += max(0.0, float(duration_ms))
        except (TypeError, ValueError):
            pass

    @_synchronized
    def record_orchestration_step(self, count: int = 1) -> None:
        """Increment measured orchestration step counter cleanly in memory."""
        self._orchestration_step_count += max(0, int(count or 0))

    @_synchronized
    def get_observability_counters(self) -> Dict[str, Any]:
        """Return measured in-memory orchestration & model request counters."""
        return {
            "model_request_count": int(getattr(self, "_model_request_count", 0)),
            "model_request_duration_ms": round(float(getattr(self, "_model_request_duration_ms", 0.0)), 3),
            "orchestration_step_count": int(getattr(self, "_orchestration_step_count", 0)),
        }

    def flush_observability_metrics(self) -> Optional[ExecutionEvent]:
        """Persist measured in-memory orchestration/model metrics once per execution without hot-path I/O."""
        counters = self.get_observability_counters()
        if counters["model_request_count"] == 0 and counters["orchestration_step_count"] == 0:
            return None
        return self._emit_central_event(
            LedgerEventType.PLANNING_DECISION,
            tool_id=None,
            status="metrics_snapshot",
            metadata={
                "metrics_snapshot": True,
                "measurement_kind": "measured",
                **counters,
            },
            idempotency_suffix=f"metrics:{counters['orchestration_step_count']}:{counters['model_request_count']}",
        )

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

    def attach_task_graph(self, graph: Optional[Any]) -> None:
        """Bind a TaskGraph instance to this turn ledger."""
        self._task_graph = graph

    def get_task_graph(self) -> Optional[Any]:
        """Return the TaskGraph attached to this turn ledger, if present."""
        return self._task_graph

    def record_graph_mutation(
        self,
        mutation_type: str,
        *,
        graph_id: Optional[str] = None,
        node_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[ExecutionEvent]:
        """Record an ordered graph mutation decision in the centralized execution ledger."""
        meta = dict(metadata or {})
        meta["mutation_type"] = mutation_type
        if graph_id:
            meta["graph_id"] = graph_id
        if node_id:
            meta["node_id"] = node_id
        return self._emit_central_event(
            LedgerEventType.PLANNING_DECISION,
            tool_id=None,
            status="graph_mutated",
            metadata=meta,
        )

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
    ) -> Optional[ExecutionEvent]:
        if self._central_ledger is None or not self._execution_id:
            return None
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
            return self._central_ledger.record_event(
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
            return None

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

    def _update_proposal_lifecycle(
        self,
        call_id: str,
        status: str,
        *,
        dispatch_attempted: Optional[bool] = None,
        has_result: Optional[bool] = None,
        blocked_reason: Optional[str] = None,
    ) -> None:
        meta = self._proposal_meta.get(call_id)
        if meta is None:
            return
        meta["status"] = status
        meta["updated_at"] = time.time()
        if dispatch_attempted is not None:
            meta["dispatch_attempted"] = bool(dispatch_attempted)
        if has_result is not None:
            meta["has_result"] = bool(has_result)
        if blocked_reason is not None:
            meta["blocked_reason"] = blocked_reason
        elif status in ("validated", "dispatched", "completed"):
            meta["blocked_reason"] = None

    def get_proposal_metadata(self, call_id: str) -> Optional[Dict[str, Any]]:
        """Return durable proposal metadata tracked for call_id, if present."""
        meta = self._proposal_meta.get(call_id)
        return dict(meta) if isinstance(meta, dict) else None

    def get_tracked_proposals(self) -> List[Dict[str, Any]]:
        """Return ordered proposal metadata snapshots synchronized with current ledger states."""
        state_map = {
            ActionState.PROPOSED: "pending",
            ActionState.VALIDATED: "validated",
            ActionState.AUTHORIZED: "validated",
            ActionState.DISPATCHED: "dispatched",
            ActionState.EXECUTED: "dispatched",
            ActionState.PARTIALLY_COMPLETED: "dispatched",
            ActionState.VERIFIED: "completed",
            ActionState.SUCCESS: "completed",
            ActionState.OBSERVED: "completed",
            ActionState.UNVERIFIED: "completed",
            ActionState.FAILED: "failed",
            ActionState.REJECTED: "blocked",
            ActionState.CANCELLED: "cancelled",
        }
        items: List[Dict[str, Any]] = []
        seen_cids: Set[str] = set()
        for cid, raw_meta in self._proposal_meta.items():
            seen_cids.add(cid)
            entry = dict(raw_meta)
            rec = self._records.get(cid)
            if rec is not None:
                entry["tool_id"] = rec.tool_name or entry.get("tool_id") or ""
                entry["normalized_arguments"] = dict(rec.arguments)
                if entry.get("status") not in ("expired", "blocked"):
                    entry["status"] = state_map.get(rec.state, entry.get("status") or "pending")
                elif entry.get("status") == "blocked" and rec.state in (
                    ActionState.DISPATCHED,
                    ActionState.VERIFIED,
                    ActionState.SUCCESS,
                    ActionState.FAILED,
                ):
                    entry["status"] = state_map.get(rec.state, entry["status"])
                if rec.state in (
                    ActionState.DISPATCHED,
                    ActionState.EXECUTED,
                    ActionState.PARTIALLY_COMPLETED,
                    ActionState.VERIFIED,
                    ActionState.SUCCESS,
                    ActionState.OBSERVED,
                    ActionState.UNVERIFIED,
                    ActionState.FAILED,
                ):
                    entry["dispatch_attempted"] = True
                if rec.state in (
                    ActionState.VERIFIED,
                    ActionState.SUCCESS,
                    ActionState.OBSERVED,
                    ActionState.UNVERIFIED,
                    ActionState.FAILED,
                    ActionState.REJECTED,
                ):
                    entry["has_result"] = True
                if rec.state == ActionState.REJECTED and rec.rejection_reason:
                    entry["blocked_reason"] = rec.rejection_reason
                elif rec.state == ActionState.CANCELLED and rec.error and not entry.get("blocked_reason"):
                    entry["blocked_reason"] = rec.error
            items.append(entry)
        # Deduplicate reconfirmed placeholders when a newer execution entry shares the same proposal_id
        dedup_by_pid: Dict[str, Dict[str, Any]] = {}
        for item in items:
            pid = str(item.get("proposal_id") or "")
            if not pid:
                continue
            prev = dedup_by_pid.get(pid)
            if prev is None:
                dedup_by_pid[pid] = item
                continue
            prev_reconf = prev.get("status") == "cancelled" and "reconfirmed_and_dispatched" in str(prev.get("blocked_reason") or "")
            cur_reconf = item.get("status") == "cancelled" and "reconfirmed_and_dispatched" in str(item.get("blocked_reason") or "")
            if prev_reconf and not cur_reconf:
                dedup_by_pid[pid] = item
        result_items = list(dedup_by_pid.values()) + [i for i in items if not i.get("proposal_id")]
        result_items.sort(key=lambda x: (int(x.get("proposal_order", 0)), float(x.get("created_at", 0.0))))
        return result_items

    def record_proposed(
        self,
        call_id: str,
        tool_name: str,
        arguments: Dict[str, Any],
        turn: Optional[int] = None,
        raw_name: Optional[str] = None,
        *,
        proposal_id: Optional[str] = None,
        proposal_order: Optional[int] = None,
        depends_on: Optional[Sequence[str]] = None,
        expires_at: Optional[float] = None,
        external_source: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record an action proposal."""
        if call_id in self._records and self._records[call_id].state != ActionState.PROPOSED:
            raise InvalidStateTransitionError(
                f"Cannot re-propose action '{call_id}' already in state '{self._records[call_id].state.value}'."
            )
        if isinstance(external_source, dict) and external_source:
            self.set_call_source_metadata(
                call_id,
                is_external=bool(external_source.get("is_external", True)),
                source_id=external_source.get("source_id"),
                source_type=external_source.get("source_type"),
                extra=(
                    {"external_tool_name": external_source.get("external_tool_name")}
                    if external_source.get("external_tool_name")
                    else None
                ),
            )
        now = time.time()
        clean_args = dict(arguments) if isinstance(arguments, dict) else {}
        rec = ActionRecord(
            call_id=call_id,
            tool_name=tool_name,
            arguments=clean_args,
            state=ActionState.PROPOSED,
            turn=turn,
            raw_name=raw_name or tool_name,
        )
        self._records[call_id] = rec
        if call_id not in self._order:
            self._order.append(call_id)
        if call_id not in self._pending_call_ids:
            self._pending_call_ids.append(call_id)

        existing_meta = self._proposal_meta.get(call_id)
        eff_order = (
            int(proposal_order)
            if proposal_order is not None
            else (int(existing_meta["proposal_order"]) if isinstance(existing_meta, dict) and "proposal_order" in existing_meta else self._next_proposal_order)
        )
        if eff_order >= self._next_proposal_order:
            self._next_proposal_order = eff_order + 1
        eff_prop_id = str(
            proposal_id
            or (existing_meta.get("proposal_id") if isinstance(existing_meta, dict) else None)
            or f"prop_{call_id}"
        )
        eff_depends = [str(d).strip() for d in (depends_on if depends_on is not None else (existing_meta.get("depends_on") if isinstance(existing_meta, dict) else []) or []) if str(d).strip()]
        src_meta = self._call_source_meta.get(call_id, {})
        eff_ext_src = dict(external_source) if isinstance(external_source, dict) and external_source else (
            {
                "is_external": True,
                "source_id": src_meta.get("source_id") or "external",
                "source_type": src_meta.get("source_type") or "external",
                "external_tool_name": (src_meta.get("extra") or {}).get("external_tool_name") or tool_name,
            }
            if src_meta.get("is_external")
            else None
        )

        ev_meta: Dict[str, Any] = {
            "turn": turn,
            "raw_name": raw_name or tool_name,
            "arguments": rec.arguments,
            "proposal_id": eff_prop_id,
            "proposal_order": eff_order,
        }
        if eff_depends:
            ev_meta["depends_on"] = eff_depends
        if expires_at is not None:
            ev_meta["expires_at"] = float(expires_at)
        ev = self._emit_central_event(
            LedgerEventType.TOOL_SELECTED,
            call_id=call_id,
            tool_id=tool_name,
            status="proposed",
            metadata=ev_meta,
            idempotency_suffix="selected",
        )
        self._proposal_meta[call_id] = {
            "proposal_id": eff_prop_id,
            "execution_id": self._execution_id or "",
            "original_call_id": call_id,
            "tool_id": tool_name,
            "raw_name": raw_name or tool_name,
            "normalized_arguments": clean_args,
            "proposal_order": eff_order,
            "creation_event_id": (ev.event_id if ev is not None else (existing_meta.get("creation_event_id") if isinstance(existing_meta, dict) else None)),
            "creation_seq": (int(ev.sequence_number) if ev is not None else int(existing_meta.get("creation_seq", 0)) if isinstance(existing_meta, dict) else 0),
            "status": "pending",
            "depends_on": eff_depends,
            "external_source": eff_ext_src,
            "turn": turn,
            "dispatch_attempted": False,
            "has_result": False,
            "expires_at": float(expires_at) if expires_at is not None else (existing_meta.get("expires_at") if isinstance(existing_meta, dict) else None),
            "blocked_reason": None,
            "created_at": existing_meta.get("created_at", now) if isinstance(existing_meta, dict) else now,
            "updated_at": now,
        }

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
                meta = self._proposal_meta.get(cid)
                if isinstance(meta, dict):
                    snap = dict(meta)
                    snap["status"] = "cancelled"
                    snap["blocked_reason"] = reason
                    snap["updated_at"] = time.time()
                    self._interrupted_proposal_snapshots.append(snap)
                self._update_proposal_lifecycle(cid, "cancelled", blocked_reason=reason)
                self.record_cancelled(cid, reason=f"Pending call discarded: {reason}")
                discarded.append(cid)
        return discarded

    def cancel_pending_proposal(self, call_id: str, reason: str = "cancelled") -> None:
        """Cancel a specific pending proposal."""
        if call_id in self._pending_call_ids:
            self._pending_call_ids.remove(call_id)
        if call_id in self._records and self._records[call_id].state == ActionState.PROPOSED:
            self._update_proposal_lifecycle(call_id, "cancelled", blocked_reason=reason)
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
            self._update_proposal_lifecycle(call_id, "validated")
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
        self._update_proposal_lifecycle(call_id, "dispatched", dispatch_attempted=True)
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
        self._update_proposal_lifecycle(call_id, "blocked", has_result=True, blocked_reason=safe_reason)
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
        self._update_proposal_lifecycle(call_id, "cancelled", blocked_reason=safe_reason or "Action cancelled")
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

    def get_structured_observation(self, call_id: str) -> Optional[Any]:
        """Return the StructuredObservation recorded for call_id, if present."""
        store = getattr(self, "_structured_observations", None)
        return store.get(call_id) if isinstance(store, dict) else None

    def get_structured_observations(self) -> List[Any]:
        """Return ordered StructuredObservation instances recorded during this turn."""
        store = getattr(self, "_structured_observations", None)
        if not isinstance(store, dict):
            return []
        return [store[cid] for cid in self._order if cid in store]

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
        structured_observation: Optional[Any] = None,
        duration_ms: Optional[float] = None,
        verification_duration_ms: Optional[float] = None,
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
        if final_state == ActionState.FAILED:
            self._update_proposal_lifecycle(
                call_id,
                "blocked" if effective_rejection_reason else "failed",
                dispatch_attempted=False if (status == "rejected" or effective_rejection_reason) else True,
                has_result=True,
                blocked_reason=effective_rejection_reason or _redact_sensitive_text(effective_error),
            )
        else:
            self._update_proposal_lifecycle(call_id, "completed", dispatch_attempted=True, has_result=True)
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
        result_meta: Dict[str, Any] = {
            "exit_code": effective_exit_code,
            "error": effective_error,
            "rejection_reason": effective_rejection_reason,
            "verified": effective_verified,
        }
        if duration_ms is not None:
            result_meta["duration_ms"] = round(max(0.0, float(duration_ms)), 3)
        if verification_duration_ms is not None:
            result_meta["verification_duration_ms"] = round(max(0.0, float(verification_duration_ms)), 3)
        self._emit_central_event(
            LedgerEventType.TOOL_EXECUTION_RESULT,
            call_id=call_id,
            tool_id=tool_name or None,
            status=res_status,
            metadata=result_meta,
            idempotency_suffix="result",
        )
        if structured_observation is not None:
            store = getattr(self, "_structured_observations", None)
            if not isinstance(store, dict):
                store = {}
                self._structured_observations = store
            store[call_id] = structured_observation

        if (
            structured_observation is not None
            and getattr(self, "_central_ledger", None) is not None
            and getattr(self, "_execution_id", None)
        ):
            self._central_ledger.record_observation_event(
                self._execution_id,
                structured_observation,
                correlation_id=call_id,
            )
        else:
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
            ver_ev_meta: Dict[str, Any] = {"detail": effective_detail, "verified_targets": eff_verified_targets}
            if verification_duration_ms is not None:
                ver_ev_meta["duration_ms"] = round(max(0.0, float(verification_duration_ms)), 3)
            self._emit_central_event(
                LedgerEventType.VERIFICATION_RECORDED,
                call_id=call_id,
                tool_id=tool_name or None,
                status=VerificationStatus.VERIFIED.value,
                metadata=ver_ev_meta,
                idempotency_suffix="verified",
            )
        elif effective_verification_failed:
            ver_ev_meta = {"detail": effective_detail, "failed_targets": eff_failed_targets}
            if verification_duration_ms is not None:
                ver_ev_meta["duration_ms"] = round(max(0.0, float(verification_duration_ms)), 3)
            self._emit_central_event(
                LedgerEventType.VERIFICATION_RECORDED,
                call_id=call_id,
                tool_id=tool_name or None,
                status=VerificationStatus.VERIFICATION_FAILED.value,
                metadata=ver_ev_meta,
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

    def has_identical_failed_verification_action(
        self, tool_name: str, arguments: Any
    ) -> bool:
        """Check if identical mutating action failed verification with no later unrelated state change or verification."""
        target_canon = _canonicalize_args(arguments)
        failed_match = False
        match_targets: List[str] = []
        for cid in self._order:
            rec = self._records[cid]
            is_same_call = rec.tool_name == tool_name and _canonicalize_args(rec.arguments) == target_canon
            if is_same_call:
                if getattr(rec, "verification_failed", False):
                    failed_match = True
                    match_targets = _collect_later_resolution_targets(rec, False)
                elif rec.state == ActionState.VERIFIED or (rec.state == ActionState.SUCCESS and rec.verified):
                    failed_match = False
            elif failed_match and rec.state in (ActionState.SUCCESS, ActionState.VERIFIED):
                is_inspect = _is_inspection_action(rec.tool_name, rec.arguments)
                if not is_inspect and not is_diagnostic_test_command(rec.tool_name, rec.arguments):
                    later_targets = _collect_later_resolution_targets(rec, False)
                    if not match_targets or not later_targets or any(
                        not any(targets_match(mt, lt) for mt in match_targets)
                        for lt in later_targets
                    ):
                        failed_match = False
        return failed_match

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
                or "identical parameters already ran and failed verification" in str(rec.error)
                or "Pending call discarded" in str(rec.error)
                or "unconfirmed_by_model" in str(rec.error)
                or "superseded_by_model" in str(rec.error)
                or "abandoned_by_model" in str(rec.error)
                or "reconfirmed_and_dispatched" in str(rec.error)
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

    @_synchronized
    def record_node_reservation_event(
        self,
        event_kind: str,
        *,
        node_id: str,
        owner_id: str,
        reservation_id: Optional[str] = None,
        call_id: Optional[str] = None,
        tool_id: Optional[str] = None,
        resources: Optional[Sequence[str]] = None,
        status: Optional[str] = None,
        detail: Optional[str] = None,
        idempotency_suffix: Optional[str] = None,
    ) -> Optional[ExecutionEvent]:
        """Record an unambiguous node reservation, release, or conflict event in the ledger."""
        meta: Dict[str, Any] = {
            "node_id": node_id,
            "owner_id": owner_id,
            "reservation_event": event_kind,
        }
        if reservation_id:
            meta["reservation_id"] = reservation_id
        if resources:
            meta["resources"] = list(resources)
        if detail:
            meta["detail"] = _redact_sensitive_text(detail)
        return self._emit_central_event(
            LedgerEventType.PLANNING_DECISION,
            call_id=call_id,
            tool_id=tool_id,
            status=status or event_kind,
            metadata=meta,
            idempotency_suffix=idempotency_suffix,
        )


for _meth_name, _meth_obj in list(vars(TurnExecutionLedger).items()):
    if _meth_name.startswith("__") or _meth_name in ("_sync_lock", "record_node_reservation_event"):
        continue
    if callable(_meth_obj) and not isinstance(_meth_obj, (staticmethod, classmethod, property, type)):
        setattr(TurnExecutionLedger, _meth_name, _synchronized(_meth_obj))
del _meth_name, _meth_obj


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
    r"\b(fail|fails|failing|failed|failure)\b",
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


_PROSPECTIVE_ACTION_PROSE_PATTERNS = (
    re.compile(
        r"\b(?:let['’]?s|let\s+us|we['’]?ll|i['’]?ll|i\s+will|we\s+will|now,?\s+let['’]?s|next,?\s+let['’]?s)\s+"
        r"(?:fix|write|update|patch|edit|modify|change|implement|create|run|re-?run|test|verify|start\s+by)\b",
        flags=re.IGNORECASE,
    ),
    re.compile(
        r"\bhere['’]?s\s+(?:a|the)\s+(?:revised|updated|fixed|corrected|patched)\b",
        flags=re.IGNORECASE,
    ),
)


def has_prospective_action_prose(text: str) -> bool:
    """Return True when text narrates planned/unexecuted code changes or next tool steps instead of terminal reporting."""
    if not text:
        return False
    return any(pat.search(text) for pat in _PROSPECTIVE_ACTION_PROSE_PATTERNS)


def is_truthful_failure_report(text: str) -> bool:
    """Check if model text truthfully acknowledges a failure or inability to proceed without evasion."""
    lower = text.lower()
    has_failure_indicator = any(re.search(kw, lower) for kw in TRUTHFUL_FAILURE_PATTERNS)
    if not has_failure_indicator:
        return False

    if any(re.search(p, lower) for p in APOLOGY_OR_DELEGATION_PATTERNS) or detect_consultant_deferral(text):
        return False

    if has_prospective_action_prose(text):
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


def is_read_only_diagnostic_failures_only(
    prompt: str,
    ledger: Optional[Any],
    final_answer: str = "",
) -> bool:
    """Return True when all unresolved failures are expected read-only test/inspection diagnostics."""
    if ledger is None or not hasattr(ledger, "unresolved_failures"):
        return False
    unresolved_recs = list(ledger.unresolved_failures())
    if not unresolved_recs:
        return False
    global_prohib = extract_global_mutation_prohibition(prompt)
    if not (global_prohib.get("no_file_modifications") and global_prohib.get("is_test_diagnostic")):
        return False
    cleaned_answer = re.sub(r"<think>.*?</think>", "", str(final_answer or ""), flags=re.DOTALL)
    cleaned_answer = re.sub(r"<think>.*$", "", cleaned_answer, flags=re.DOTALL).strip()
    if not cleaned_answer or detect_false_success(cleaned_answer, has_unresolved=True):
        return False

    succ_read_names = set()
    for sr in (ledger.successful_actions() if hasattr(ledger, "successful_actions") else []):
        if _is_inspection_action(sr.tool_name, sr.arguments):
            t_s = _extract_action_target(sr.tool_name, sr.arguments)
            if t_s:
                succ_read_names.add(Path(t_s).name)

    for rec in unresolved_recs:
        if is_diagnostic_test_command(rec.tool_name, rec.arguments):
            continue
        if _is_inspection_action(rec.tool_name, rec.arguments):
            t_f = _extract_action_target(rec.tool_name, rec.arguments)
            if t_f and Path(t_f).name in succ_read_names:
                continue
        return False
    return True


def compute_policy_compliance(
    prompt: str,
    ledger: Optional[Any],
) -> Dict[str, Any]:
    """Summarize whether an execution respected explicit user prohibitions."""
    clean_prompt = str(prompt or "")
    prohibited_targets = extract_prohibited_targets(clean_prompt, ledger=ledger)
    global_prohib = extract_global_mutation_prohibition(clean_prompt)
    desktop_prohib = extract_desktop_interaction_prohibition(clean_prompt)
    has_prohibitions = bool(
        prohibited_targets
        or global_prohib.get("no_file_modifications")
        or global_prohib.get("no_mutating_commands")
        or desktop_prohib
    )
    if not has_prohibitions:
        return {
            "has_prohibitions": False,
            "compliant": True,
            "prohibited_attempts_blocked": 0,
            "prohibited_violations": 0,
            "status": "not_applicable",
        }

    records: List[ActionRecord] = []
    if ledger is not None:
        raw_order = getattr(ledger, "_order", None)
        raw_records = getattr(ledger, "_records", None)
        if isinstance(raw_order, list) and isinstance(raw_records, dict):
            records = [raw_records[cid] for cid in raw_order if cid in raw_records]

    allow_tests = bool(
        global_prohib.get("is_test_diagnostic")
        and not global_prohib.get("no_mutating_commands")
    )
    blocked = 0
    violations = 0

    for rec in records:
        rej = str(getattr(rec, "rejection_reason", None) or "")
        err = str(getattr(rec, "error", None) or "")
        if rec.state in (ActionState.FAILED, ActionState.REJECTED) and (
            "prohibited_by_user" in rej or "prohibited_by_user" in err
        ):
            blocked += 1
            continue

        if rec.state not in (
            ActionState.SUCCESS,
            ActionState.VERIFIED,
            ActionState.OBSERVED,
            ActionState.PARTIALLY_COMPLETED,
        ):
            continue

        args = rec.arguments if isinstance(rec.arguments, dict) else {}
        if global_prohib.get("no_file_modifications") or global_prohib.get("no_mutating_commands"):
            is_mut, _ = is_mutating_tool_call(
                rec.tool_name,
                args,
                allow_read_only_tests=allow_tests,
            )
            if is_mut:
                violations += 1
                continue

        if desktop_prohib and not _is_inspection_action(rec.tool_name, args):
            t_low = (rec.tool_name or "").lower().strip()
            is_desktop_mut = t_low in (
                "desktop_control",
                "desktop_automation",
                "interact_with_ui",
                "gui_actuate",
                "click_tag",
                "type_text",
                "press_key",
                "move_mouse",
                "click_mouse",
                "manage_desktop_window",
            ) or (
                t_low in ("execute_command", "bash", "sh")
                and bool(_DESKTOP_MUTATION_SHELL_RE.search(str(args.get("command") or args.get("cmd") or "")))
            )
            if is_desktop_mut:
                violations += 1
                continue

        if prohibited_targets and not _is_inspection_action(rec.tool_name, args):
            cand_targets: List[str] = []
            direct_t = _extract_action_target(rec.tool_name, args)
            if direct_t:
                cand_targets.append(direct_t)
            for k in ("path", "filepath", "file", "target", "filename", "dst", "destination", "src", "source"):
                val = args.get(k)
                if val and isinstance(val, str) and val.strip():
                    cand_targets.append(canonicalize_target(val.strip()))
            cmd_txt = str(args.get("command") or args.get("cmd") or "")
            if cmd_txt:
                is_mut_cmd, _ = is_mutating_tool_call(
                    rec.tool_name,
                    args,
                    allow_read_only_tests=allow_tests,
                )
                if is_mut_cmd:
                    cand_targets.extend(extract_file_targets_from_text(cmd_txt))
            if any(
                targets_match(ct, pt, prohibited_targets, allow_dir_containment=True)
                for ct in cand_targets
                for pt in prohibited_targets
            ):
                violations += 1

    compliant = violations == 0
    return {
        "has_prohibitions": True,
        "compliant": compliant,
        "prohibited_attempts_blocked": blocked,
        "prohibited_violations": violations,
        "status": "compliant" if compliant else "violated",
    }


@dataclass(frozen=True)
class RootOutcomeResolution:
    """Authoritative final execution, verification, policy, and CLI exit status decision."""

    execution_status: ExecutionStatus
    verification_status: VerificationStatus
    error: Optional[str] = None
    interruption_reason: Optional[str] = None
    diagnostic_failures_only: bool = False
    policy_compliance: Dict[str, Any] = field(default_factory=dict)
    v_results: List[Tuple[str, str, Any]] = field(default_factory=list)
    all_v_ok: bool = False
    has_unresolved: bool = False
    verified_targets: List[str] = field(default_factory=list)
    exit_code: int = 0


def _assess_desktop_verification_state(ledger: Optional[Any]) -> Dict[str, Any]:
    """Inspect chronological desktop mutations vs visual/outcome observations in ledger."""
    state: Dict[str, Any] = {
        "last_desktop_mut_idx": -1,
        "last_desktop_mut_rec": None,
        "last_desktop_obs_idx": -1,
        "last_click_idx": -1,
        "last_type_idx": -1,
        "has_verified_desktop_mut": False,
        "has_only_pre_mutation_or_window_verified": False,
        "trailing_unobserved_mutation": False,
    }
    if ledger is None:
        return state

    if hasattr(ledger, "_order") and hasattr(ledger, "_records"):
        records = [ledger._records[cid] for cid in ledger._order if cid in ledger._records]
    elif hasattr(ledger, "records") and callable(ledger.records):
        records = list(ledger.records())
    else:
        return state
    desktop_mut_tools = {"desktop_control", "desktop_automation", "interact_with_ui"}
    desktop_obs_tools = {"capture_screen", "screen_capture", "capture_som_screen", "verify_outcome"}
    verified_indices: List[Tuple[int, ActionRecord]] = []

    for idx, rec in enumerate(records):
        if rec.state not in (ActionState.SUCCESS, ActionState.VERIFIED) or getattr(rec, "verification_failed", False):
            continue
        t_low = (rec.tool_name or "").strip().lower()
        args = rec.arguments if isinstance(rec.arguments, dict) else {}

        if rec.state == ActionState.VERIFIED:
            verified_indices.append((idx, rec))

        if t_low in desktop_mut_tools and not _is_inspection_action(rec.tool_name, args):
            state["last_desktop_mut_idx"] = idx
            state["last_desktop_mut_rec"] = rec
            act_low = str(args.get("action") or "").strip().lower()
            if "click" in act_low:
                state["last_click_idx"] = idx
            if act_low in ("keyboard_type", "type", "type_text"):
                state["last_type_idx"] = idx
            if rec.state == ActionState.VERIFIED:
                state["has_verified_desktop_mut"] = True
        elif t_low in desktop_obs_tools or (
            t_low in ("desktop_control", "desktop_automation")
            and str(args.get("action") or "").strip().lower() in ("observe", "screenshot")
        ):
            state["last_desktop_obs_idx"] = idx

    last_mut_idx = state["last_desktop_mut_idx"]
    last_obs_idx = state["last_desktop_obs_idx"]
    trailing_unobserved = (
        last_mut_idx >= 0
        and last_mut_idx > last_obs_idx
        and not state["has_verified_desktop_mut"]
    )
    state["trailing_unobserved_mutation"] = trailing_unobserved
    if trailing_unobserved and verified_indices:
        state["has_only_pre_mutation_or_window_verified"] = all(
            ((r.tool_name or "").strip().lower() == "manage_desktop_window" or v_idx < last_mut_idx)
            and (r.tool_name or "").strip().lower() != "verify_outcome"
            for v_idx, r in verified_indices
        )
    return state


def resolve_root_execution_outcome(
    *,
    prompt: str,
    ledger: Optional[Any],
    final_text: str = "",
    acceptance_spec: Optional[Any] = None,
    v_results: Optional[List[Tuple[str, str, Any]]] = None,
    was_cancelled: bool = False,
    was_timed_out: bool = False,
    caught_exc: Optional[BaseException] = None,
    provider_error: Optional[str] = None,
    is_truncated: bool = False,
    model_error: Optional[str] = None,
    reached_step_limit: bool = False,
    invalid_verify_spec_error: Optional[str] = None,
) -> RootOutcomeResolution:
    """Resolve one consistent outcome across orchestrator, CLI exit code, and CentralExecutionLedger."""
    import asyncio

    total_acts = int(ledger.total_actions()) if (ledger is not None and hasattr(ledger, "total_actions")) else 0
    unresolved_fn = getattr(ledger, "unresolved_failures", None) if ledger is not None else None
    unresolved_recs: List[ActionRecord] = list(unresolved_fn()) if callable(unresolved_fn) else []
    if callable(unresolved_fn):
        has_unresolved = bool(unresolved_recs)
    elif ledger is not None and hasattr(ledger, "has_unresolved_failures"):
        has_unresolved = bool(ledger.has_unresolved_failures())
    else:
        has_unresolved = False

    verified_targets = (
        list(ledger.get_verified_targets())
        if (ledger is not None and hasattr(ledger, "get_verified_targets"))
        else []
    )
    desktop_ver_state = _assess_desktop_verification_state(ledger)
    has_verified = bool(
        (ledger is not None and hasattr(ledger, "verified_actions") and ledger.verified_actions())
        or verified_targets
    )
    policy_compliance = compute_policy_compliance(prompt, ledger)
    has_ver_fail_unresolved = any(getattr(r, "verification_failed", False) for r in unresolved_recs)

    eff_v_results: List[Tuple[str, str, Any]] = list(v_results) if v_results is not None else []
    if (
        has_verified
        and not eff_v_results
        and not verified_targets
        and desktop_ver_state["trailing_unobserved_mutation"]
        and desktop_ver_state["has_only_pre_mutation_or_window_verified"]
    ):
        has_verified = False
    eff_invalid_spec = invalid_verify_spec_error
    if v_results is None and acceptance_spec is not None and not eff_invalid_spec:
        if isinstance(acceptance_spec, str) and not acceptance_spec.strip():
            eff_invalid_spec = "Empty verification specification provided."
        else:
            parsed_specs = parse_acceptance_criteria(acceptance_spec)
            if parsed_specs:
                from zelyr.engine.verifier import OutcomeVerifier

                for item in parsed_specs:
                    chk = item.get("check", "")
                    tgt = item.get("target", "")
                    vr = OutcomeVerifier.verify(
                        check=chk,
                        target=tgt,
                        expected=item.get("expected"),
                    )
                    eff_v_results.append((chk, tgt, vr))
            else:
                eff_invalid_spec = f"Unable to parse verification specification: '{acceptance_spec}'"

    all_v_ok = bool(eff_v_results) and all(bool(getattr(vr, "success", False)) for _, _, vr in eff_v_results)
    for _, vt, vr in eff_v_results:
        if getattr(vr, "success", False) and vt and vt not in verified_targets:
            verified_targets.append(vt)

    # If a full post-run verification suite just ran and all criteria passed, prior intermediate
    # draft verification_failed annotations on otherwise-succeeded actions are superseded.
    if eff_v_results and all_v_ok and unresolved_recs:
        unresolved_recs = [
            r
            for r in unresolved_recs
            if (
                r.state in (ActionState.FAILED, ActionState.CANCELLED, ActionState.PARTIALLY_COMPLETED)
                or not getattr(r, "verification_failed", False)
            )
        ]
        has_unresolved = bool(unresolved_recs)
        has_ver_fail_unresolved = any(getattr(r, "verification_failed", False) for r in unresolved_recs)

    diagnostic_failures_only = False
    if (
        has_unresolved
        and acceptance_spec is None
        and not eff_v_results
        and not eff_invalid_spec
        and not reached_step_limit
        and not is_truncated
        and not model_error
        and not was_cancelled
        and not was_timed_out
        and not provider_error
        and caught_exc is None
        and unresolved_recs
        and is_read_only_diagnostic_failures_only(prompt, ledger, final_text)
    ):
        diagnostic_failures_only = True
        has_unresolved = False

    if was_cancelled or isinstance(caught_exc, asyncio.CancelledError):
        return RootOutcomeResolution(
            execution_status=ExecutionStatus.CANCELLED,
            verification_status=VerificationStatus.INTERRUPTED,
            error="Task cancelled",
            interruption_reason="user_cancelled",
            diagnostic_failures_only=False,
            policy_compliance=policy_compliance,
            v_results=eff_v_results,
            all_v_ok=all_v_ok,
            has_unresolved=has_unresolved,
            verified_targets=verified_targets,
            exit_code=1,
        )

    if was_timed_out or isinstance(caught_exc, (TimeoutError, asyncio.TimeoutError)):
        is_task_to = isinstance(caught_exc, (TimeoutError, asyncio.TimeoutError))
        return RootOutcomeResolution(
            execution_status=ExecutionStatus.TIMED_OUT,
            verification_status=VerificationStatus.INTERRUPTED,
            error=(str(caught_exc) if is_task_to else "Tool execution timed out"),
            interruption_reason=("task_timeout" if is_task_to else "tool_timeout"),
            diagnostic_failures_only=False,
            policy_compliance=policy_compliance,
            v_results=eff_v_results,
            all_v_ok=all_v_ok,
            has_unresolved=has_unresolved,
            verified_targets=verified_targets,
            exit_code=1,
        )

    if provider_error:
        return RootOutcomeResolution(
            execution_status=ExecutionStatus.INTERRUPTED,
            verification_status=VerificationStatus.INTERRUPTED,
            error=str(provider_error),
            interruption_reason="provider_failure",
            diagnostic_failures_only=False,
            policy_compliance=policy_compliance,
            v_results=eff_v_results,
            all_v_ok=all_v_ok,
            has_unresolved=has_unresolved,
            verified_targets=verified_targets,
            exit_code=1,
        )

    if caught_exc is not None:
        return RootOutcomeResolution(
            execution_status=ExecutionStatus.INTERRUPTED,
            verification_status=VerificationStatus.INTERRUPTED,
            error=str(caught_exc) or type(caught_exc).__name__,
            interruption_reason=f"exception:{type(caught_exc).__name__}",
            diagnostic_failures_only=False,
            policy_compliance=policy_compliance,
            v_results=eff_v_results,
            all_v_ok=all_v_ok,
            has_unresolved=has_unresolved,
            verified_targets=verified_targets,
            exit_code=1,
        )

    if eff_invalid_spec:
        return RootOutcomeResolution(
            execution_status=ExecutionStatus.FAILED,
            verification_status=VerificationStatus.VERIFICATION_FAILED,
            error=eff_invalid_spec,
            interruption_reason="invalid_verification_spec",
            diagnostic_failures_only=False,
            policy_compliance=policy_compliance,
            v_results=eff_v_results,
            all_v_ok=False,
            has_unresolved=has_unresolved,
            verified_targets=verified_targets,
            exit_code=2,
        )

    if model_error:
        is_prov = model_error.startswith(("⚠️ [Connection Error]", "⚠️ [API Error]", "⚠️ [Request Error]"))
        is_trunc_err = "truncated" in model_error.lower()
        return RootOutcomeResolution(
            execution_status=(ExecutionStatus.INTERRUPTED if is_prov else ExecutionStatus.FAILED),
            verification_status=(VerificationStatus.INTERRUPTED if (is_prov or is_trunc_err) else VerificationStatus.NOT_VERIFIED),
            error=model_error,
            interruption_reason=("provider_failure" if is_prov else "model_error"),
            diagnostic_failures_only=False,
            policy_compliance=policy_compliance,
            v_results=eff_v_results,
            all_v_ok=all_v_ok,
            has_unresolved=has_unresolved,
            verified_targets=verified_targets,
            exit_code=1,
        )

    if is_truncated:
        return RootOutcomeResolution(
            execution_status=ExecutionStatus.INTERRUPTED,
            verification_status=VerificationStatus.INTERRUPTED,
            error="Model generation reached token length limit before completion.",
            interruption_reason="output_truncated",
            diagnostic_failures_only=False,
            policy_compliance=policy_compliance,
            v_results=eff_v_results,
            all_v_ok=all_v_ok,
            has_unresolved=has_unresolved,
            verified_targets=verified_targets,
            exit_code=1,
        )

    if eff_v_results and not all_v_ok:
        failed_descs = [
            f"{chk}:{tgt}: {getattr(vr, 'error', None) or 'failed'}"
            for chk, tgt, vr in eff_v_results
            if not getattr(vr, "success", False)
        ]
        return RootOutcomeResolution(
            execution_status=ExecutionStatus.FAILED,
            verification_status=VerificationStatus.VERIFICATION_FAILED,
            error="; ".join(failed_descs)[:512] or "Post-run outcome verification failed",
            interruption_reason="verification_failed",
            diagnostic_failures_only=False,
            policy_compliance=policy_compliance,
            v_results=eff_v_results,
            all_v_ok=False,
            has_unresolved=has_unresolved,
            verified_targets=verified_targets,
            exit_code=2,
        )

    if reached_step_limit:
        if all_v_ok and not has_unresolved:
            step_ver = VerificationStatus.VERIFIED
        elif has_ver_fail_unresolved:
            step_ver = VerificationStatus.VERIFICATION_FAILED
        elif detect_false_success(final_text, has_unresolved=has_unresolved):
            step_ver = VerificationStatus.CLAIMED_UNVERIFIED
        else:
            step_ver = VerificationStatus.NOT_VERIFIED
        return RootOutcomeResolution(
            execution_status=ExecutionStatus.FAILED,
            verification_status=step_ver,
            error="Reached maximum autonomous ReAct steps before task completion.",
            interruption_reason="step_limit_exceeded",
            diagnostic_failures_only=False,
            policy_compliance=policy_compliance,
            v_results=eff_v_results,
            all_v_ok=all_v_ok,
            has_unresolved=has_unresolved,
            verified_targets=verified_targets,
            exit_code=2,
        )

    if policy_compliance.get("has_prohibitions") and not policy_compliance.get("compliant"):
        return RootOutcomeResolution(
            execution_status=ExecutionStatus.FAILED,
            verification_status=(VerificationStatus.VERIFIED if all_v_ok else VerificationStatus.NOT_VERIFIED),
            error="Executed action violated explicit user prohibition.",
            interruption_reason="policy_violation",
            diagnostic_failures_only=False,
            policy_compliance=policy_compliance,
            v_results=eff_v_results,
            all_v_ok=all_v_ok,
            has_unresolved=has_unresolved,
            verified_targets=verified_targets,
            exit_code=2,
        )

    if has_ver_fail_unresolved:
        return RootOutcomeResolution(
            execution_status=ExecutionStatus.FAILED,
            verification_status=VerificationStatus.VERIFICATION_FAILED,
            error="Outcome verification failed",
            interruption_reason="verification_failed",
            diagnostic_failures_only=False,
            policy_compliance=policy_compliance,
            v_results=eff_v_results,
            all_v_ok=all_v_ok,
            has_unresolved=True,
            verified_targets=verified_targets,
            exit_code=2,
        )

    if has_unresolved:
        err_rec = unresolved_recs[-1].error if unresolved_recs else "Unresolved action failure"
        if eff_v_results and all_v_ok:
            ver_st = VerificationStatus.VERIFIED
        elif detect_false_success(final_text, has_unresolved=True):
            ver_st = VerificationStatus.CLAIMED_UNVERIFIED
        else:
            ver_st = VerificationStatus.NOT_VERIFIED
        return RootOutcomeResolution(
            execution_status=ExecutionStatus.FAILED,
            verification_status=ver_st,
            error=err_rec,
            interruption_reason="unresolved_failure",
            diagnostic_failures_only=False,
            policy_compliance=policy_compliance,
            v_results=eff_v_results,
            all_v_ok=all_v_ok,
            has_unresolved=True,
            verified_targets=verified_targets,
            exit_code=2,
        )

    if eff_v_results and all_v_ok:
        return RootOutcomeResolution(
            execution_status=ExecutionStatus.COMPLETED,
            verification_status=VerificationStatus.VERIFIED,
            error=None,
            interruption_reason=None,
            diagnostic_failures_only=False,
            policy_compliance=policy_compliance,
            v_results=eff_v_results,
            all_v_ok=True,
            has_unresolved=False,
            verified_targets=verified_targets,
            exit_code=0,
        )

    if has_verified:
        return RootOutcomeResolution(
            execution_status=ExecutionStatus.COMPLETED,
            verification_status=VerificationStatus.VERIFIED,
            error=None,
            interruption_reason=None,
            diagnostic_failures_only=False,
            policy_compliance=policy_compliance,
            v_results=eff_v_results,
            all_v_ok=all_v_ok,
            has_unresolved=False,
            verified_targets=verified_targets,
            exit_code=0,
        )

    if total_acts > 0:
        ver_st = (
            VerificationStatus.CLAIMED_UNVERIFIED
            if (
                (acceptance_spec and detect_false_success(final_text, has_unresolved=False))
                or (desktop_ver_state["trailing_unobserved_mutation"] and detect_false_success(final_text, has_unresolved=True))
            )
            else VerificationStatus.NOT_VERIFIED
        )
        return RootOutcomeResolution(
            execution_status=ExecutionStatus.COMPLETED_UNVERIFIED,
            verification_status=ver_st,
            error=None,
            interruption_reason=None,
            diagnostic_failures_only=diagnostic_failures_only,
            policy_compliance=policy_compliance,
            v_results=eff_v_results,
            all_v_ok=all_v_ok,
            has_unresolved=False,
            verified_targets=verified_targets,
            exit_code=0,
        )

    if acceptance_spec:
        return RootOutcomeResolution(
            execution_status=ExecutionStatus.FAILED,
            verification_status=VerificationStatus.NOT_VERIFIED,
            error="Task ended without executing or verifying required acceptance criteria",
            interruption_reason="unfulfilled_criteria",
            diagnostic_failures_only=False,
            policy_compliance=policy_compliance,
            v_results=eff_v_results,
            all_v_ok=False,
            has_unresolved=False,
            verified_targets=verified_targets,
            exit_code=2,
        )

    return RootOutcomeResolution(
        execution_status=ExecutionStatus.COMPLETED,
        verification_status=VerificationStatus.NOT_VERIFIED,
        error=None,
        interruption_reason=None,
        diagnostic_failures_only=False,
        policy_compliance=policy_compliance,
        v_results=eff_v_results,
        all_v_ok=False,
        has_unresolved=False,
        verified_targets=verified_targets,
        exit_code=0,
    )


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
    if extract_global_mutation_prohibition(user_prompt).get("no_file_modifications"):
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

    # 4b. Check trailing unobserved desktop GUI mutations when visual verification was established or requested
    desktop_ver_state = _assess_desktop_verification_state(ledger)
    if desktop_ver_state["trailing_unobserved_mutation"]:
        prompt_low = (user_prompt or "").lower()
        has_visual_verify_request = bool(
            re.search(r"\b(?:capture|screenshot|screen\s*shot|verify|confirm|inspect|check)\b", prompt_low)
            and re.search(r"\b(?:window|screen|ui|desktop|button|input|field|counter|applied|fixture)\b", prompt_low)
        )
        prior_obs_taken = desktop_ver_state["last_desktop_obs_idx"] >= 0
        type_before_reclick = (
            desktop_ver_state["last_click_idx"] >= 0
            and desktop_ver_state["last_type_idx"] >= 0
            and desktop_ver_state["last_click_idx"] > desktop_ver_state["last_type_idx"]
            and bool(re.search(r"\b(?:type|enter|input)\b", prompt_low))
        )
        if prior_obs_taken or has_visual_verify_request or type_before_reclick:
            last_mut_rec = desktop_ver_state["last_desktop_mut_rec"]
            mut_act = ""
            if last_mut_rec is not None and isinstance(last_mut_rec.arguments, dict):
                mut_act = str(last_mut_rec.arguments.get("action") or "").strip()
            mut_desc = (
                f"{last_mut_rec.tool_name}:{mut_act}"
                if (last_mut_rec is not None and mut_act)
                else (last_mut_rec.tool_name if last_mut_rec is not None else "desktop_control")
            )
            dt_desc = (
                f"Trailing desktop GUI action ('{mut_desc}') executed after the last 'capture_screen' "
                "observation; complete any remaining requested UI input/actions and call 'capture_screen' "
                "to visually observe and verify the current UI state before concluding."
            )
            if dt_desc not in unfulfilled_outcomes:
                unfulfilled_outcomes.append(dt_desc)

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
    INTERRUPTED = "interrupted"
    COMPLETED = "completed"
    COMPLETED_UNVERIFIED = "completed_unverified"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMED_OUT = "timed_out"
    ABANDONED = "abandoned"


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
    TASK_CHECKPOINTED = "task_checkpointed"
    TASK_RESUMED = "task_resumed"
    TASK_ABANDONED = "task_abandoned"
    TASK_TERMINATED = "task_terminated"


class StaleCheckpointError(LedgerPersistenceError):
    """Raised when a concurrent operation tries to overwrite a newer checkpoint."""


class CorruptedCheckpointError(LedgerPersistenceError):
    """Raised when persisted checkpoint data is malformed or fails integrity verification."""


class UnsupportedCheckpointVersionError(LedgerPersistenceError):
    """Raised when a persisted checkpoint uses an unsupported schema version."""


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
                parsed_meta = json.loads(raw_meta) if raw_meta else {}
                meta = (
                    {}
                    if parsed_meta is None
                    else (parsed_meta if isinstance(parsed_meta, dict) else {"_malformed_metadata": True})
                )
            except Exception:
                meta = {"_malformed_metadata": True}
        elif isinstance(raw_meta, dict):
            meta = dict(raw_meta)
        else:
            meta = {"_malformed_metadata": True} if raw_meta is not None else {}
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
                parsed_meta = json.loads(raw_meta) if raw_meta else {}
                meta = (
                    {}
                    if parsed_meta is None
                    else (parsed_meta if isinstance(parsed_meta, dict) else {"_malformed_metadata": True})
                )
            except Exception:
                meta = {"_malformed_metadata": True}
        elif isinstance(raw_meta, dict):
            meta = dict(raw_meta)
        else:
            meta = {"_malformed_metadata": True} if raw_meta is not None else {}
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
    next_seq INTEGER NOT NULL DEFAULT 1,
    latest_checkpoint_seq INTEGER NOT NULL DEFAULT 0
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

CREATE TABLE IF NOT EXISTS task_checkpoints (
    checkpoint_id TEXT PRIMARY KEY,
    execution_id TEXT NOT NULL,
    task_id TEXT NOT NULL,
    checkpoint_version INTEGER NOT NULL DEFAULT 1,
    sequence_number INTEGER NOT NULL,
    last_event_id TEXT,
    last_event_seq INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL,
    verification_status TEXT NOT NULL,
    task_prompt TEXT NOT NULL DEFAULT '',
    planning_json TEXT NOT NULL DEFAULT '{}',
    environment_json TEXT NOT NULL DEFAULT '{}',
    external_tools_json TEXT NOT NULL DEFAULT '[]',
    verified_targets_json TEXT NOT NULL DEFAULT '[]',
    recovery_summary_json TEXT NOT NULL DEFAULT '{}',
    pending_actions_json TEXT NOT NULL DEFAULT '[]',
    task_graph_json TEXT NOT NULL DEFAULT '{}',
    checkpoint_digest TEXT NOT NULL,
    redacted INTEGER NOT NULL DEFAULT 0,
    truncated INTEGER NOT NULL DEFAULT 0,
    created_at REAL NOT NULL,
    FOREIGN KEY (execution_id) REFERENCES executions(execution_id) ON DELETE CASCADE,
    UNIQUE (execution_id, sequence_number)
);

CREATE INDEX IF NOT EXISTS idx_exec_created ON executions(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_exec_status ON executions(status, updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_events_exec_seq ON execution_events(execution_id, sequence_number ASC);
CREATE INDEX IF NOT EXISTS idx_events_tool ON execution_events(tool_id);
CREATE INDEX IF NOT EXISTS idx_events_type ON execution_events(event_type);
CREATE INDEX IF NOT EXISTS idx_checkpoints_exec_seq ON task_checkpoints(execution_id, sequence_number DESC);
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
        self._memory_checkpoints: Dict[str, List[Dict[str, Any]]] = {}
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
                        cols = {str(r["name"]) for r in conn.execute("PRAGMA table_info(executions)").fetchall()}
                        if "latest_checkpoint_seq" not in cols:
                            conn.execute(
                                "ALTER TABLE executions ADD COLUMN latest_checkpoint_seq INTEGER NOT NULL DEFAULT 0"
                            )
                        cp_cols = {str(r["name"]) for r in conn.execute("PRAGMA table_info(task_checkpoints)").fetchall()}
                        if "pending_actions_json" not in cp_cols:
                            conn.execute(
                                "ALTER TABLE task_checkpoints ADD COLUMN pending_actions_json TEXT NOT NULL DEFAULT '[]'"
                            )
                        if "task_graph_json" not in cp_cols:
                            conn.execute(
                                "ALTER TABLE task_checkpoints ADD COLUMN task_graph_json TEXT NOT NULL DEFAULT '{}'"
                            )
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
        """Finalize an execution record and emit its terminal lifecycle event idempotently."""
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

        _terminal_statuses = {
            ExecutionStatus.COMPLETED.value,
            ExecutionStatus.COMPLETED_UNVERIFIED.value,
            ExecutionStatus.FAILED.value,
            ExecutionStatus.CANCELLED.value,
            ExecutionStatus.TIMED_OUT.value,
            ExecutionStatus.INTERRUPTED.value,
            ExecutionStatus.ABANDONED.value,
        }
        existing_rec = self.get_execution(execution_id)
        if existing_rec is None:
            return None

        existing_events = self.get_events(execution_id)
        last_term_seq = max(
            (ev.sequence_number for ev in existing_events if ev.event_type == LedgerEventType.TASK_TERMINATED.value),
            default=0,
        )
        last_res_seq = max(
            (ev.sequence_number for ev in existing_events if ev.event_type == LedgerEventType.TASK_RESUMED.value),
            default=0,
        )
        resume_epoch = sum(
            1 for ev in existing_events if ev.event_type == LedgerEventType.TASK_RESUMED.value
        )

        if existing_rec.status in _terminal_statuses and existing_rec.completed_at is not None:
            if last_term_seq > last_res_seq:
                return existing_rec
            eff_ver_existing = ver_str or existing_rec.verification_status
            meta_unchanged = all(
                k in existing_rec.metadata and existing_rec.metadata[k] == v
                for k, v in safe_meta.items()
            )
            if (
                existing_rec.status == status_str
                and existing_rec.verification_status == eff_ver_existing
                and (safe_summary is None or existing_rec.final_summary == safe_summary)
                and (safe_error is None or existing_rec.error == safe_error)
                and meta_unchanged
            ):
                return existing_rec

        if emit_terminal_event and last_term_seq <= last_res_seq:
            term_meta = dict(safe_meta)
            if safe_summary:
                term_meta["summary"] = safe_summary
            if safe_error:
                term_meta["error"] = safe_error
            if ver_str:
                term_meta["verification_status"] = ver_str
            idem_key = (
                f"{execution_id}:terminated:epoch{resume_epoch}"
                if resume_epoch > 0
                else f"{execution_id}:terminated"
            )
            try:
                self.record_event(
                    execution_id,
                    LedgerEventType.TASK_TERMINATED,
                    status=status_str,
                    metadata=term_meta,
                    idempotency_key=idem_key,
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

    def list_child_executions(self, parent_execution_id: str) -> List[ExecutionRecord]:
        """List persisted or in-memory child executions ordered chronologically by creation time."""
        parent_id = str(parent_execution_id or "").strip()
        if not parent_id:
            return []
        with self._lock:
            conn = self._ensure_connection(raise_on_error=False, execution_id=parent_id)
            if conn is not None:
                try:
                    rows = conn.execute(
                        "SELECT * FROM executions WHERE parent_execution_id = ? ORDER BY created_at ASC, execution_id ASC",
                        (parent_id,),
                    ).fetchall()
                    return [ExecutionRecord.from_row(r) for r in rows]
                except Exception as exc:
                    self._record_persistence_failure(
                        "list_child_executions",
                        exc,
                        execution_id=parent_id,
                        strict=False,
                    )
            children = [
                rec
                for rec in self._memory_executions.values()
                if rec.parent_execution_id == parent_id
            ]
            children.sort(key=lambda r: (r.created_at, r.execution_id))
            return children

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

    def query_authorization_decisions(
        self,
        *,
        limit: int = 20,
        outcome: Optional[str] = None,
        tool_id: Optional[str] = None,
        execution_id: Optional[str] = None,
    ) -> List[ExecutionEvent]:
        """Return recent AUTHORIZATION_DECISION events ordered deterministically newest-first."""
        eff_limit = max(1, min(int(limit or 20), 500))
        norm_outcome = str(outcome or "").strip().lower() or None
        outcome_aliases: Dict[str, Set[str]] = {
            "allow": {"allow", "allowed"},
            "allowed": {"allow", "allowed"},
            "require_confirmation": {
                "require_confirmation",
                "requires_confirmation",
                "confirmation_required",
            },
            "requires_confirmation": {
                "require_confirmation",
                "requires_confirmation",
                "confirmation_required",
            },
            "deny": {"deny", "denied", "rejected"},
            "denied": {"deny", "denied", "rejected"},
        }
        accepted_outcomes = outcome_aliases.get(norm_outcome, {norm_outcome}) if norm_outcome else None

        def _matches_outcome(ev: ExecutionEvent) -> bool:
            if accepted_outcomes is None:
                return True
            meta = ev.metadata if isinstance(ev.metadata, dict) else {}
            pol_dec = str(meta.get("policy_decision") or "").strip().lower()
            st = str(ev.status or "").strip().lower()
            return (pol_dec in accepted_outcomes) or (st in accepted_outcomes)

        with self._lock:
            conn = self._ensure_connection(raise_on_error=False, execution_id=execution_id)
            if conn is not None:
                try:
                    clauses = ["event_type = ?"]
                    params: List[Any] = [LedgerEventType.AUTHORIZATION_DECISION.value]
                    if tool_id:
                        clauses.append("tool_id = ?")
                        params.append(str(tool_id))
                    if execution_id:
                        clauses.append("execution_id = ?")
                        params.append(str(execution_id))
                    where_sql = " AND ".join(clauses)
                    rows = conn.execute(
                        f"SELECT * FROM execution_events WHERE {where_sql} "
                        "ORDER BY timestamp DESC, sequence_number DESC, id DESC",
                        params,
                    ).fetchall()
                    out_events: List[ExecutionEvent] = []
                    for r in rows:
                        ev = ExecutionEvent.from_row(r)
                        if _matches_outcome(ev):
                            out_events.append(ev)
                            if len(out_events) >= eff_limit:
                                break
                    return out_events
                except Exception as exc:
                    self._record_persistence_failure(
                        "query_authorization_decisions",
                        exc,
                        execution_id=execution_id,
                        strict=False,
                    )

            mem_events: List[ExecutionEvent] = []
            for exec_events in self._memory_events.values():
                for ev in exec_events:
                    if ev.event_type != LedgerEventType.AUTHORIZATION_DECISION.value:
                        continue
                    if tool_id and ev.tool_id != str(tool_id):
                        continue
                    if execution_id and ev.execution_id != str(execution_id):
                        continue
                    if _matches_outcome(ev):
                        mem_events.append(ev)
            mem_events.sort(
                key=lambda e: (float(e.timestamp or 0.0), int(e.sequence_number or 0), str(e.event_id or "")),
                reverse=True,
            )
            return mem_events[:eff_limit]

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

    def record_observation_event(
        self,
        execution_id: str,
        observation: Any,
        *,
        correlation_id: Optional[str] = None,
        strict: Optional[bool] = None,
    ) -> Optional[ExecutionEvent]:
        """Persist an ObservationFramingEngine StructuredObservation into execution_events.

        Uses content digest inside idempotency_key so duplicate recordings deduplicate while
        updated verification evidence appends a new ordered event.
        """
        if not execution_id:
            return None

        if hasattr(observation, "to_dict"):
            obs_dict = observation.to_dict()
        elif isinstance(observation, dict):
            obs_dict = dict(observation)
        else:
            obs_dict = {"structured_result": {"summary": str(observation)}}

        if hasattr(observation, "content_digest"):
            obs_digest = str(observation.content_digest())
        else:
            import hashlib

            raw_json = json.dumps(obs_dict, sort_keys=True, default=str).encode("utf-8", errors="replace")
            obs_digest = hashlib.sha256(raw_json).hexdigest()[:16]

        eff_corr = correlation_id or obs_dict.get("correlation_id")
        tool_id = obs_dict.get("source_tool_id") or obs_dict.get("tool")
        ext = obs_dict.get("external_source") if isinstance(obs_dict.get("external_source"), dict) else None
        is_ext = bool(ext and ext.get("is_external", True))
        src_id = ext.get("source_id") if ext else None
        src_type = ext.get("source_type") if ext else None

        ver_status_raw = str(obs_dict.get("verification_status") or "").lower()
        mapped_ver: Optional[VerificationStatus] = None
        if ver_status_raw == "verified":
            mapped_ver = VerificationStatus.VERIFIED
        elif ver_status_raw in ("verification_failed", "contradicted"):
            mapped_ver = VerificationStatus.VERIFICATION_FAILED
        elif ver_status_raw == "claimed_unverified":
            mapped_ver = VerificationStatus.CLAIMED_UNVERIFIED
        elif ver_status_raw == "not_verified":
            mapped_ver = VerificationStatus.NOT_VERIFIED

        compact_obs = (
            observation.to_compact_dict(max_chars=1600)
            if hasattr(observation, "to_compact_dict")
            else dict(obs_dict)
        )
        # Keep key structured fields accessible directly in event metadata
        obs_meta: Dict[str, Any] = {
            "observation_id": obs_dict.get("observation_id"),
            "evidence_level": obs_dict.get("evidence_level"),
            "execution_status": obs_dict.get("execution_status"),
            "verification_status": obs_dict.get("verification_status"),
            "remaining_uncertainty": obs_dict.get("remaining_uncertainty"),
            "suggested_next_action": obs_dict.get("suggested_next_action"),
            "state_changes": list(obs_dict.get("state_changes") or [])[:6],
            "independent_verification": obs_dict.get("independent_verification"),
            "error_classification": obs_dict.get("error_classification"),
            "recovery_guidance": obs_dict.get("recovery_guidance"),
            "redacted": bool(obs_dict.get("redacted", False)),
            "truncated": bool(obs_dict.get("truncated", False)),
            "observation": compact_obs,
        }

        ev = self.record_event(
            execution_id,
            LedgerEventType.OBSERVATION_UPDATED,
            tool_id=str(tool_id) if tool_id else None,
            source_id=str(src_id) if src_id else None,
            source_type=str(src_type) if src_type else None,
            is_external=is_ext,
            status=str(obs_dict.get("execution_status") or "recorded"),
            correlation_id=eff_corr,
            metadata=obs_meta,
            idempotency_key=f"{execution_id}:{eff_corr or 'obs'}:obs:{obs_digest}",
            verification_status=mapped_ver,
            strict=strict,
        )
        if ev is not None and hasattr(observation, "ledger_refs") and isinstance(observation.ledger_refs, dict):
            refs = observation.ledger_refs
            refs["execution_id"] = execution_id
            if eff_corr:
                refs["correlation_id"] = eff_corr
            ev_ids = list(refs.get("event_ids") or [])
            if ev.event_id not in ev_ids:
                ev_ids.append(ev.event_id)
            refs["event_ids"] = ev_ids
            seqs = list(refs.get("sequence_numbers") or [])
            if ev.sequence_number not in seqs:
                seqs.append(ev.sequence_number)
            refs["sequence_numbers"] = seqs
        return ev

    def get_observations(
        self,
        execution_id: str,
        *,
        correlation_id: Optional[str] = None,
        tool_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieve deterministically ordered structured observation events for an execution."""
        events = self.get_events(
            execution_id,
            event_type=LedgerEventType.OBSERVATION_UPDATED.value,
            tool_id=tool_id,
            correlation_id=correlation_id,
        )
        out: List[Dict[str, Any]] = []
        for ev in events:
            meta = dict(ev.metadata or {})
            out.append({
                "event_id": ev.event_id,
                "execution_id": ev.execution_id,
                "sequence_number": ev.sequence_number,
                "timestamp": ev.timestamp,
                "tool_id": ev.tool_id,
                "correlation_id": ev.correlation_id,
                "is_external": ev.is_external,
                "source_id": ev.source_id,
                "source_type": ev.source_type,
                "status": ev.status,
                "evidence_level": meta.get("evidence_level"),
                "execution_status": meta.get("execution_status") or ev.status,
                "verification_status": meta.get("verification_status"),
                "remaining_uncertainty": meta.get("remaining_uncertainty"),
                "redacted": bool(ev.redacted or meta.get("redacted")),
                "truncated": bool(ev.truncated or meta.get("truncated")),
                "metadata": meta,
            })
        return out

    @staticmethod
    def _compute_checkpoint_digest(payload: Dict[str, Any]) -> str:
        """Compute deterministic SHA-256 digest for checkpoint integrity verification."""
        canonical = {
            "checkpoint_version": int(payload.get("checkpoint_version", 1)),
            "checkpoint_id": str(payload.get("checkpoint_id") or ""),
            "execution_id": str(payload.get("execution_id") or ""),
            "task_id": str(payload.get("task_id") or ""),
            "sequence_number": int(payload.get("sequence_number", 0)),
            "last_event_id": payload.get("last_event_id"),
            "last_event_seq": int(payload.get("last_event_seq", 0)),
            "status": str(payload.get("status") or ""),
            "verification_status": str(payload.get("verification_status") or ""),
            "task_prompt": str(payload.get("task_prompt") or ""),
            "planning_state": payload.get("planning_state") or {},
            "environment_snapshot": payload.get("environment_snapshot") or {},
            "external_tools_used": payload.get("external_tools_used") or [],
            "verified_targets": payload.get("verified_targets") or [],
            "recovery_summary": payload.get("recovery_summary") or {},
        }
        pending_actions = payload.get("pending_actions") or []
        if pending_actions:
            canonical["pending_actions"] = list(pending_actions)
        task_graph = payload.get("task_graph") or {}
        if task_graph:
            canonical["task_graph"] = dict(task_graph)
        raw = json.dumps(canonical, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str).encode(
            "utf-8", errors="replace"
        )
        return hashlib.sha256(raw).hexdigest()

    @classmethod
    def _row_to_checkpoint_dict(cls, row: Union[sqlite3.Row, Dict[str, Any]], *, verify_integrity: bool = True) -> Dict[str, Any]:
        """Deserialize and validate a persisted task_checkpoints row."""
        d = dict(row)
        raw_ver = d.get("checkpoint_version")
        if not isinstance(raw_ver, int) or isinstance(raw_ver, bool):
            raise CorruptedCheckpointError(f"Malformed checkpoint_version: {raw_ver!r}")
        if raw_ver != 1:
            raise UnsupportedCheckpointVersionError(f"Unsupported checkpoint version: {raw_ver}")

        def _parse_json_field(raw_val: Any, expected_type: type, field_name: str) -> Any:
            if isinstance(raw_val, expected_type):
                return raw_val
            if not isinstance(raw_val, str) or not raw_val.strip():
                raise CorruptedCheckpointError(f"Checkpoint field '{field_name}' is empty or missing.")
            try:
                parsed = json.loads(raw_val)
            except Exception as exc:
                raise CorruptedCheckpointError(f"Checkpoint field '{field_name}' contains invalid JSON: {exc}") from exc
            if not isinstance(parsed, expected_type):
                raise CorruptedCheckpointError(
                    f"Checkpoint field '{field_name}' has wrong type {type(parsed).__name__}."
                )
            return parsed

        raw_pa_field = d.get("pending_actions_json") if "pending_actions_json" in d else "[]"
        raw_tg_field = d.get("task_graph_json") if "task_graph_json" in d else "{}"
        cp_dict: Dict[str, Any] = {
            "checkpoint_id": str(d.get("checkpoint_id") or ""),
            "execution_id": str(d.get("execution_id") or ""),
            "task_id": str(d.get("task_id") or ""),
            "checkpoint_version": raw_ver,
            "sequence_number": int(d.get("sequence_number") or 0),
            "last_event_id": d.get("last_event_id"),
            "last_event_seq": int(d.get("last_event_seq") or 0),
            "status": str(d.get("status") or ""),
            "verification_status": str(d.get("verification_status") or ""),
            "task_prompt": str(d.get("task_prompt") or ""),
            "planning_state": _parse_json_field(d.get("planning_json"), dict, "planning_json"),
            "environment_snapshot": _parse_json_field(d.get("environment_json"), dict, "environment_json"),
            "external_tools_used": _parse_json_field(d.get("external_tools_json"), list, "external_tools_json"),
            "verified_targets": _parse_json_field(d.get("verified_targets_json"), list, "verified_targets_json"),
            "recovery_summary": _parse_json_field(d.get("recovery_summary_json"), dict, "recovery_summary_json"),
            "pending_actions": _parse_json_field(raw_pa_field, list, "pending_actions_json"),
            "task_graph": _parse_json_field(raw_tg_field, dict, "task_graph_json"),
            "checkpoint_digest": str(d.get("checkpoint_digest") or ""),
            "redacted": bool(d.get("redacted", False)),
            "truncated": bool(d.get("truncated", False)),
            "timestamp": float(d.get("created_at") or 0.0),
        }
        if not cp_dict["checkpoint_id"] or not cp_dict["execution_id"] or cp_dict["sequence_number"] < 1:
            raise CorruptedCheckpointError("Checkpoint record missing mandatory identity/sequence metadata.")
        if verify_integrity:
            expected_digest = cls._compute_checkpoint_digest(cp_dict)
            if not cp_dict["checkpoint_digest"] or cp_dict["checkpoint_digest"] != expected_digest:
                raise CorruptedCheckpointError(
                    f"Checkpoint '{cp_dict['checkpoint_id']}' failed digest verification."
                )
        return cp_dict

    def save_checkpoint(
        self,
        execution_id: str,
        *,
        checkpoint_version: int = 1,
        task_id: Optional[str] = None,
        last_event_id: Optional[str] = None,
        last_event_seq: Optional[int] = None,
        status: Optional[Union[ExecutionStatus, str]] = None,
        verification_status: Optional[Union[VerificationStatus, str]] = None,
        task_prompt: Optional[str] = None,
        planning_state: Optional[Dict[str, Any]] = None,
        environment_snapshot: Optional[Dict[str, Any]] = None,
        external_tools_used: Optional[List[Dict[str, Any]]] = None,
        verified_targets: Optional[List[str]] = None,
        recovery_summary: Optional[Dict[str, Any]] = None,
        pending_actions: Optional[List[Dict[str, Any]]] = None,
        task_graph: Optional[Dict[str, Any]] = None,
        expected_checkpoint_seq: Optional[int] = None,
        emit_event: bool = True,
        strict: Optional[bool] = None,
    ) -> Optional[Dict[str, Any]]:
        """Atomically persist a versioned task checkpoint with stale-overwrite protection."""
        if not execution_id:
            raise ValueError("execution_id must be non-empty")
        if checkpoint_version != 1:
            raise UnsupportedCheckpointVersionError(f"Unsupported checkpoint version: {checkpoint_version}")

        now = time.time()
        cp_id = f"cp_{uuid.uuid4().hex[:14]}"
        eff_task_id = str(task_id or execution_id)

        safe_prompt, prompt_red, prompt_trunc = sanitize_ledger_text(
            task_prompt or "",
            max_bytes=MAX_LEDGER_TEXT_BYTES,
            strip_injection=True,
        )
        safe_plan, p_red, p_trunc = sanitize_ledger_payload(planning_state, max_bytes=MAX_LEDGER_METADATA_BYTES)
        safe_env, e_red, e_trunc = sanitize_ledger_payload(environment_snapshot, max_bytes=4096)
        safe_ext_list_wrap, x_red, x_trunc = sanitize_ledger_payload({"items": external_tools_used or []}, max_bytes=4096)
        safe_ext = list(safe_ext_list_wrap.get("items") or []) if isinstance(safe_ext_list_wrap.get("items"), list) else []
        safe_vt_wrap, v_red, v_trunc = sanitize_ledger_payload({"targets": verified_targets or []}, max_bytes=4096)
        safe_vt = [str(x) for x in (safe_vt_wrap.get("targets") or [])] if isinstance(safe_vt_wrap.get("targets"), list) else []
        safe_rec, r_red, r_trunc = sanitize_ledger_payload(recovery_summary, max_bytes=4096)
        safe_pa_wrap, pa_red, pa_trunc = sanitize_ledger_payload({"items": pending_actions or []}, max_bytes=16384)
        safe_pa = list(safe_pa_wrap.get("items") or []) if isinstance(safe_pa_wrap.get("items"), list) else []
        safe_tg: Dict[str, Any] = {}
        tg_red, tg_trunc = False, False
        if task_graph:
            safe_tg_base = dict(task_graph)
            if "objective" in safe_tg_base and safe_tg_base["objective"] is not None:
                obj_s, o_r, o_t = sanitize_ledger_text(safe_tg_base["objective"], max_bytes=1024, strip_injection=True)
                safe_tg_base["objective"] = obj_s or ""
                tg_red = tg_red or o_r
                tg_trunc = tg_trunc or o_t
            if isinstance(safe_tg_base.get("summary"), dict):
                s_sum, s_r, s_t = sanitize_ledger_payload(safe_tg_base["summary"], max_bytes=2048)
                safe_tg_base["summary"] = s_sum
                tg_red = tg_red or s_r
                tg_trunc = tg_trunc or s_t
            if isinstance(safe_tg_base.get("nodes"), list):
                if len(safe_tg_base["nodes"]) > 64:
                    tg_trunc = True
                clean_nodes: List[Any] = []
                for nd in safe_tg_base["nodes"][:64]:
                    if isinstance(nd, dict):
                        sn, n_r, n_t = sanitize_ledger_payload(nd, max_bytes=4096)
                        clean_nodes.append(sn)
                        tg_red = tg_red or n_r
                        tg_trunc = tg_trunc or n_t
                    else:
                        clean_nodes.append(nd)
                safe_tg_base["nodes"] = clean_nodes
            if isinstance(safe_tg_base.get("edges"), list):
                if len(safe_tg_base["edges"]) > 128:
                    tg_trunc = True
                clean_edges: List[Any] = []
                for ed in safe_tg_base["edges"][:128]:
                    if isinstance(ed, dict):
                        se, e_r, e_t = sanitize_ledger_payload(ed, max_bytes=1024)
                        clean_edges.append(se)
                        tg_red = tg_red or e_r
                        tg_trunc = tg_trunc or e_t
                    else:
                        clean_edges.append(ed)
                safe_tg_base["edges"] = clean_edges
            safe_tg = safe_tg_base

        any_red = bool(prompt_red or p_red or e_red or x_red or v_red or r_red or pa_red or tg_red)
        any_trunc = bool(prompt_trunc or p_trunc or e_trunc or x_trunc or v_trunc or r_trunc or pa_trunc or tg_trunc)

        with self._lock:
            eff_strict = self.raise_on_persistence_error if strict is None else strict
            conn = self._ensure_connection(raise_on_error=eff_strict, execution_id=execution_id)
            if conn is not None:
                try:
                    from zelyr.db.connection import managed_transaction

                    with managed_transaction(conn):
                        exec_row = conn.execute(
                            "SELECT * FROM executions WHERE execution_id = ?",
                            (execution_id,),
                        ).fetchone()
                        if exec_row is None:
                            raise ValueError(f"Unknown execution_id '{execution_id}' for save_checkpoint")

                        latest_row = conn.execute(
                            "SELECT COALESCE(MAX(sequence_number), 0) AS max_cp_seq, COALESCE(MAX(last_event_seq), 0) AS max_ev_seq "
                            "FROM task_checkpoints WHERE execution_id = ?",
                            (execution_id,),
                        ).fetchone()
                        max_cp_seq = int(latest_row["max_cp_seq"]) if latest_row else 0
                        max_ev_seq = int(latest_row["max_ev_seq"]) if latest_row else 0

                        if expected_checkpoint_seq is not None and max_cp_seq != expected_checkpoint_seq:
                            raise StaleCheckpointError(
                                f"Concurrent checkpoint conflict for '{execution_id}': expected seq {expected_checkpoint_seq}, latest is {max_cp_seq}."
                            )

                        eff_last_ev_seq = int(last_event_seq) if last_event_seq is not None else max(0, int(exec_row["next_seq"]) - 1)
                        if eff_last_ev_seq < max_ev_seq:
                            raise StaleCheckpointError(
                                f"Stale checkpoint rejected for '{execution_id}': event cursor {eff_last_ev_seq} precedes persisted {max_ev_seq}."
                            )

                        eff_last_ev_id = last_event_id
                        if eff_last_ev_id is None and eff_last_ev_seq > 0:
                            ev_r = conn.execute(
                                "SELECT event_id FROM execution_events WHERE execution_id = ? AND sequence_number = ?",
                                (execution_id, eff_last_ev_seq),
                            ).fetchone()
                            if ev_r is not None:
                                eff_last_ev_id = str(ev_r["event_id"])

                        cp_seq = max_cp_seq + 1
                        eff_status = (
                            (status.value if isinstance(status, ExecutionStatus) else str(status))
                            if status is not None
                            else str(exec_row["status"])
                        )
                        eff_ver = (
                            (verification_status.value if isinstance(verification_status, VerificationStatus) else str(verification_status))
                            if verification_status is not None
                            else str(exec_row["verification_status"])
                        )
                        eff_prompt = safe_prompt if (task_prompt is not None and task_prompt != "") else str(exec_row["task_prompt"] or "")

                        cp_payload: Dict[str, Any] = {
                            "checkpoint_id": cp_id,
                            "execution_id": execution_id,
                            "task_id": eff_task_id,
                            "checkpoint_version": checkpoint_version,
                            "sequence_number": cp_seq,
                            "last_event_id": eff_last_ev_id,
                            "last_event_seq": eff_last_ev_seq,
                            "status": eff_status,
                            "verification_status": eff_ver,
                            "task_prompt": eff_prompt,
                            "planning_state": safe_plan,
                            "environment_snapshot": safe_env,
                            "external_tools_used": safe_ext,
                            "verified_targets": safe_vt,
                            "recovery_summary": safe_rec,
                            "pending_actions": safe_pa,
                            "task_graph": safe_tg,
                            "redacted": any_red,
                            "truncated": any_trunc,
                            "timestamp": now,
                        }
                        cp_digest = self._compute_checkpoint_digest(cp_payload)
                        cp_payload["checkpoint_digest"] = cp_digest

                        conn.execute(
                            """
                            INSERT INTO task_checkpoints (
                                checkpoint_id, execution_id, task_id, checkpoint_version,
                                sequence_number, last_event_id, last_event_seq, status,
                                verification_status, task_prompt, planning_json, environment_json,
                                external_tools_json, verified_targets_json, recovery_summary_json,
                                pending_actions_json, task_graph_json, checkpoint_digest, redacted, truncated, created_at
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """,
                            (
                                cp_id,
                                execution_id,
                                eff_task_id,
                                checkpoint_version,
                                cp_seq,
                                eff_last_ev_id,
                                eff_last_ev_seq,
                                eff_status,
                                eff_ver,
                                eff_prompt,
                                json.dumps(safe_plan, ensure_ascii=False, separators=(",", ":"), default=str),
                                json.dumps(safe_env, ensure_ascii=False, separators=(",", ":"), default=str),
                                json.dumps(safe_ext, ensure_ascii=False, separators=(",", ":"), default=str),
                                json.dumps(safe_vt, ensure_ascii=False, separators=(",", ":"), default=str),
                                json.dumps(safe_rec, ensure_ascii=False, separators=(",", ":"), default=str),
                                json.dumps(safe_pa, ensure_ascii=False, separators=(",", ":"), default=str),
                                json.dumps(safe_tg, ensure_ascii=False, separators=(",", ":"), default=str),
                                cp_digest,
                                1 if any_red else 0,
                                1 if any_trunc else 0,
                                now,
                            ),
                        )

                        next_seq = int(exec_row["next_seq"])
                        if emit_event:
                            ev_id = f"evt_{uuid.uuid4().hex[:12]}"
                            ev_meta = {
                                "checkpoint_id": cp_id,
                                "checkpoint_seq": cp_seq,
                                "checkpoint_version": checkpoint_version,
                                "last_event_seq": eff_last_ev_seq,
                                "checkpoint_digest": cp_digest[:16],
                                "pending_action_count": len(safe_pa),
                                "has_task_graph": bool(safe_tg),
                                **({"graph_id": safe_tg.get("graph_id")} if safe_tg.get("graph_id") else {}),
                            }
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
                                    execution_id,
                                    next_seq,
                                    now,
                                    LedgerEventType.TASK_CHECKPOINTED.value,
                                    eff_status,
                                    exec_row["caller_ring"],
                                    exec_row["task_local_scoped"],
                                    1 if any_red else 0,
                                    1 if any_trunc else 0,
                                    f"{execution_id}:cp:{cp_seq}",
                                    json.dumps(ev_meta, ensure_ascii=False, separators=(",", ":"), default=str),
                                ),
                            )
                            next_seq += 1

                        # Never accidentally promote interrupted tasks to COMPLETED on checkpoint
                        conn.execute(
                            """
                            UPDATE executions
                            SET next_seq = ?, latest_checkpoint_seq = ?, updated_at = ?,
                                status = ?, verification_status = ?,
                                redacted = MAX(redacted, ?), truncated = MAX(truncated, ?)
                            WHERE execution_id = ?
                            """,
                            (
                                next_seq,
                                cp_seq,
                                now,
                                eff_status,
                                eff_ver,
                                1 if any_red else 0,
                                1 if any_trunc else 0,
                                execution_id,
                            ),
                        )
                        return cp_payload
                except (ValueError, StaleCheckpointError, UnsupportedCheckpointVersionError):
                    raise
                except Exception as exc:
                    self._record_persistence_failure(
                        "save_checkpoint",
                        exc,
                        execution_id=execution_id,
                        strict=strict,
                    )
                    return None

            # Only reach here if _ensure_connection returned None without raising (strict=False DB failure)
            return None

    def get_latest_checkpoint(
        self,
        execution_id: str,
        *,
        verify_integrity: bool = True,
    ) -> Optional[Dict[str, Any]]:
        """Fetch and validate the latest persisted checkpoint for an execution."""
        if not execution_id:
            return None
        with self._lock:
            conn = self._ensure_connection(raise_on_error=False, execution_id=execution_id)
            if conn is not None:
                try:
                    row = conn.execute(
                        "SELECT * FROM task_checkpoints WHERE execution_id = ? ORDER BY sequence_number DESC LIMIT 1",
                        (execution_id,),
                    ).fetchone()
                    if row is None:
                        return None
                    return self._row_to_checkpoint_dict(row, verify_integrity=verify_integrity)
                except (CorruptedCheckpointError, UnsupportedCheckpointVersionError):
                    raise
                except Exception as exc:
                    self._record_persistence_failure(
                        "get_latest_checkpoint",
                        exc,
                        execution_id=execution_id,
                        strict=False,
                    )
            return None

    def list_checkpoints(
        self,
        execution_id: str,
        *,
        verify_integrity: bool = True,
    ) -> List[Dict[str, Any]]:
        """List all persisted checkpoints for an execution in ascending order."""
        if not execution_id:
            return []
        with self._lock:
            conn = self._ensure_connection(raise_on_error=False, execution_id=execution_id)
            if conn is not None:
                try:
                    rows = conn.execute(
                        "SELECT * FROM task_checkpoints WHERE execution_id = ? ORDER BY sequence_number ASC",
                        (execution_id,),
                    ).fetchall()
                    return [self._row_to_checkpoint_dict(r, verify_integrity=verify_integrity) for r in rows]
                except (CorruptedCheckpointError, UnsupportedCheckpointVersionError):
                    raise
                except Exception as exc:
                    self._record_persistence_failure("list_checkpoints", exc, execution_id=execution_id, strict=False)
            return []

    def list_resumable_executions(
        self,
        *,
        limit: int = 50,
        include_cancelled: bool = True,
        include_failed: bool = True,
    ) -> List[ExecutionRecord]:
        """List interrupted or incomplete executions ordered most-recently-updated first."""
        allowed = [
            ExecutionStatus.INITIALIZED.value,
            ExecutionStatus.RUNNING.value,
            ExecutionStatus.INTERRUPTED.value,
            ExecutionStatus.TIMED_OUT.value,
        ]
        if include_failed:
            allowed.append(ExecutionStatus.FAILED.value)
        if include_cancelled:
            allowed.append(ExecutionStatus.CANCELLED.value)

        eff_limit = max(1, min(int(limit or 50), 500))
        with self._lock:
            conn = self._ensure_connection(raise_on_error=False)
            if conn is not None:
                try:
                    placeholders = ",".join("?" for _ in allowed)
                    rows = conn.execute(
                        f"SELECT * FROM executions WHERE status IN ({placeholders}) ORDER BY updated_at DESC, rowid DESC LIMIT ?",
                        (*allowed, eff_limit),
                    ).fetchall()
                    return [ExecutionRecord.from_row(r) for r in rows]
                except Exception as exc:
                    self._record_persistence_failure("list_resumable_executions", exc, strict=False)
            recs = [r for r in self._memory_executions.values() if r.status in allowed]
            recs.sort(key=lambda r: r.updated_at, reverse=True)
            return recs[:eff_limit]


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
