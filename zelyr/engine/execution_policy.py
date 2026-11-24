"""Centralized, deterministic execution-policy and risk-aware confirmation engine for Zelyr.

Evaluates every proposed tool dispatch *after* canonical normalization and security/capability
authorization using trusted runtime configuration only. Model output, TaskGraph payloads,
plan text, and checkpoints can never lower risk classifications, override policy precedence,
or forge confirmation grants.
"""

from __future__ import annotations

from contextlib import contextmanager
import contextvars
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import hmac
import json
from pathlib import Path
import re
import secrets
import sqlite3
import threading
import time
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple, Union

from zelyr.engine.execution_ledger import (
    LedgerEventType,
    sanitize_ledger_payload,
    sanitize_ledger_text,
)
from zelyr.engine.normalizer import RESERVED_AUTHORIZATION_KEYS, SafeArgumentNormalizer


POLICY_VERSION: int = 1
SUPPORTED_POLICY_VERSIONS: Set[int] = {1}

SUPPORTED_POLICY_MODES: Tuple[str, ...] = (
    "conservative",
    "strict",
    "balanced",
    "basic",
    "autopilot",
)

# Model/checkpoint keys that must never appear in tool arguments or override policy.
UNTRUSTED_POLICY_OVERRIDE_KEYS: Set[str] = set(RESERVED_AUTHORIZATION_KEYS) | {
    "policy_override",
    "risk_override",
    "skip_confirmation",
    "bypass_policy",
    "confirmation_token",
    "confirm_token",
    "confirmation_id",
    "policy_mode",
    "policy_decision",
}


class PolicyError(RuntimeError):
    """Base error for execution-policy evaluation, configuration, or confirmation failures."""


class UnsupportedPolicyVersionError(PolicyError):
    """Raised when an unsupported policy schema/version is encountered."""


class PolicyConfirmationError(PolicyError):
    """Raised when confirmation issuance, verification, or consumption fails."""


class PolicyDecisionOutcome(str, Enum):
    """Final deterministic gate outcome for a normalized tool call."""

    ALLOW = "allow"
    REQUIRE_CONFIRMATION = "require_confirmation"
    DENY = "deny"


class RiskLevel(str, Enum):
    """Unified tool/action risk taxonomy across Zelyr dispatch boundaries."""

    READ_ONLY = "read_only"
    REVERSIBLE_WRITE = "reversible_write"
    MUTATING = "mutating"
    EXTERNAL_ACTION = "external_action"
    PRIVILEGED = "privileged"
    IRREVERSIBLE = "irreversible"
    UNKNOWN = "unknown"


_RISK_SEVERITY: Dict[RiskLevel, int] = {
    RiskLevel.READ_ONLY: 0,
    RiskLevel.REVERSIBLE_WRITE: 1,
    RiskLevel.MUTATING: 2,
    RiskLevel.EXTERNAL_ACTION: 3,
    RiskLevel.PRIVILEGED: 4,
    RiskLevel.IRREVERSIBLE: 5,
    RiskLevel.UNKNOWN: 6,
}

# Known built-in/plugin catalog classification baselines (tool_id -> base RiskLevel).
_READ_ONLY_TOOL_IDS: Set[str] = {
    "echo",
    "file_read",
    "read_file",
    "safe_file_search",
    "list_files",
    "sys_info",
    "clipboard_read",
    "read_document_content",
    "gui_inspect",
    "capture_screen",
    "capture_som_screen",
    "blackboard_read",
    "semantic_memory_search",
}

_REVERSIBLE_TOOL_IDS: Set[str] = {
    "manage_workspace_file",
}

_MUTATING_TOOL_IDS: Set[str] = {
    "file_write",
    "write_file",
    "file_edit",
    "edit_file",
    "file_opener",
    "open_app",
    "clipboard_write",
    "blackboard_write",
    "file_teleport",
    "task_scheduler",
    "delegate_subtask",
    "subagent_delegation",
    "file",
    "fetch_web_page",
    "http_fetch",
    "browser",
}

_PRIVILEGED_TOOL_IDS: Set[str] = {
    "shell",
    "execute_command",
    "shell_exec",
    "shell_command",
    "bash",
    "sh",
    "python_exec",
    "interact_with_ui",
    "gui_actuate",
    "desktop_control",
    "desktop_automation",
    "click_tag",
    "type_text",
    "press_key",
    "interact_with_browser",
    "playwright_web",
    "manage_system_process",
    "modify_system_settings",
}

_IRREVERSIBLE_TOOL_IDS: Set[str] = {
    "delete_file",
    "remove_file",
    "unlink_file",
    "rmdir",
    "wipe_file",
    "format_disk",
    "shred_file",
}

_WORKSPACE_SAFE_READ_ACTIONS: Set[str] = {
    "read",
    "list_tree",
    "list",
    "stat",
    "exists",
    "search",
    "inspect",
}

_WORKSPACE_REVERSIBLE_WRITE_ACTIONS: Set[str] = {
    "write",
    "patch",
    "edit",
    "update",
    "create",
    "append",
    "rollback",
}

_DESTRUCTIVE_ACTIONS: Set[str] = {
    "delete",
    "remove",
    "unlink",
    "rmdir",
    "purge",
    "wipe",
    "truncate",
    "format",
    "destroy",
    "kill",
}

# Sensitive paths always blocked by hard security rules (Rule 1).
_HARD_RESTRICTED_PATH_MARKERS: Tuple[str, ...] = (
    "/etc/shadow",
    "/etc/gshadow",
    "/etc/sudoers",
    "/root/.ssh",
    "/.ssh/id_",
    "/.ssh/authorized_keys",
    "/.aws/credentials",
    "/.gnupg/",
    "/.kube/config",
)

_SENSITIVE_FILENAMES: Set[str] = {
    ".env",
    ".env.local",
    ".env.production",
    ".env.secret",
    "id_rsa",
    "id_ed25519",
    "shadow",
    "gshadow",
    "sudoers",
}

_MUTATING_SYSTEM_FILES: Tuple[str, ...] = (
    "/etc/passwd",
    "/etc/group",
    "/etc/hosts",
    "/etc/fstab",
    "/etc/ssh/",
    "/boot/",
)

_IRREVERSIBLE_SHELL_PATTERNS: Tuple[re.Pattern[str], ...] = (
    re.compile(
        r"(?i)(?:^|[;&|\s])(?:sudo\s+)?rm\s+[^\n]*?(?:-[a-zA-Z]*[rR][a-zA-Z]*f|-[a-zA-Z]*f[a-zA-Z]*[rR]|--no-preserve-root|--recursive)"
    ),
    re.compile(r"(?i)\b(?:mkfs(?:\.[a-z0-9_]+)?|wipefs|fdisk|parted|shred|srm)\b"),
    re.compile(r"(?i)\bdd\s+[^\n]*\bof=/dev/(?:sd|nvme|vd|hd|loop|mmcblk|mapper/)"),
    re.compile(r"(?i)>\s*/dev/(?:sd|nvme|vd|hd|loop|mmcblk)"),
    re.compile(r"(?i)\b(?:chmod|chown)\s+(?:-R|--recursive)\s+[^\n]*\s+/(?:\s|$)"),
    re.compile(r"(?i)\bgit\s+(?:reset\s+--hard|clean\s+-[a-zA-Z]*f[a-zA-Z]*d|push\s+(?:-f|--force|--force-with-lease))\b"),
    re.compile(r"(?i)\b(?:DROP\s+(?:DATABASE|SCHEMA|TABLE)|TRUNCATE\s+TABLE)\b"),
    re.compile(r"(?i)\b(?:shutdown|reboot|poweroff|halt|init\s+[06])\b"),
    re.compile(r"(?i)\b(?:curl|wget)\b[^\n|]*\|\s*(?:sudo\s+)?(?:sh|bash|zsh)\b"),
    re.compile(r":\(\)\s*\{\s*:\|\s*:\s*&\s*\}\s*;?:?"),
)

_DANGEROUS_UI_INSTRUCTION_PATTERNS: Tuple[re.Pattern[str], ...] = (
    re.compile(r"(?i)\brm\s+-[a-zA-Z]*r[a-zA-Z]*f\b"),
    re.compile(r"(?i)\bformat\s+(?:disk|drive|partition)\b"),
    re.compile(r"(?i)\bwipe\s+(?:disk|drive|filesystem|system)\b"),
)

_PATH_ARG_KEYS: Tuple[str, ...] = (
    "path",
    "file_path",
    "target",
    "target_path",
    "dest",
    "destination",
    "source",
    "src",
    "directory",
    "dir",
    "cwd",
    "filename",
    "file",
    "local_path",
)


def compute_canonical_argument_digest(
    tool_id: str,
    normalized_arguments: Mapping[str, Any],
    resource_scope: Sequence[str] = (),
) -> str:
    """Compute a deterministic SHA-256 digest over canonical tool_id, arguments, and scope."""
    canonical_payload = {
        "tool_id": str(tool_id or ""),
        "arguments": dict(normalized_arguments) if isinstance(normalized_arguments, Mapping) else {},
        "resource_scope": sorted(str(r) for r in resource_scope),
    }
    encoded = json.dumps(
        canonical_payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _canonicalize_path_str(raw: str, workspace_root: Optional[Path]) -> str:
    text = str(raw or "").strip()
    if not text:
        return ""
    try:
        expanded = Path(text).expanduser()
        if not expanded.is_absolute() and workspace_root is not None:
            resolved = (workspace_root.resolve(strict=False) / expanded).resolve(strict=False)
        else:
            resolved = expanded.resolve(strict=False)
        return str(resolved)
    except Exception:
        return text


def _is_path_within(path_str: str, root: Path) -> bool:
    try:
        target = Path(path_str).resolve(strict=False)
        base = root.resolve(strict=False)
        target.relative_to(base)
        return True
    except Exception:
        return False


def _looks_like_filesystem_path(val: str) -> bool:
    if not val or "://" in val:
        return False
    return "/" in val or "\\" in val or val.startswith(("~", ".")) or "." in Path(val).name


def extract_resource_scope(
    tool_id: str,
    normalized_arguments: Mapping[str, Any],
    *,
    workspace_root: Optional[Union[str, Path]] = None,
    node_target_resources: Sequence[str] = (),
) -> List[str]:
    """Extract canonicalized target resource identifiers from normalized arguments and node metadata."""
    ws_path = Path(workspace_root).expanduser().resolve(strict=False) if workspace_root else None
    scopes: Set[str] = set()

    for item in node_target_resources or ():
        s = str(item or "").strip()
        if s:
            scopes.add(s)

    if isinstance(normalized_arguments, Mapping):
        for key in _PATH_ARG_KEYS:
            val = normalized_arguments.get(key)
            if isinstance(val, str) and val.strip():
                canon = _canonicalize_path_str(val, ws_path)
                if canon:
                    scopes.add(f"path:{canon}")
        url_val = normalized_arguments.get("url") or normalized_arguments.get("endpoint")
        if isinstance(url_val, str) and url_val.strip():
            scopes.add(f"url:{url_val.strip()}")

    if tool_id in {"interact_with_ui", "gui_actuate", "desktop_control", "desktop_automation", "click_tag", "type_text", "press_key"}:
        scopes.add("exclusive:ui_desktop")
    elif tool_id in {"interact_with_browser", "playwright_web", "browser"}:
        scopes.add("exclusive:browser")
    elif tool_id in {"shell", "execute_command", "shell_exec", "shell_command", "bash", "sh"}:
        scopes.add("exclusive:shell")

    return sorted(scopes)


def _extract_raw_paths(normalized_arguments: Mapping[str, Any], workspace_root: Optional[Path]) -> List[Tuple[str, str]]:
    """Return list of (raw_value, canonical_path) for filesystem-like argument values."""
    out: List[Tuple[str, str]] = []
    if not isinstance(normalized_arguments, Mapping):
        return out
    for key in _PATH_ARG_KEYS:
        val = normalized_arguments.get(key)
        if isinstance(val, str) and val.strip() and _looks_like_filesystem_path(val.strip()):
            raw = val.strip()
            out.append((raw, _canonicalize_path_str(raw, workspace_root)))
    return out


def _normalize_risk_level(val: Union[RiskLevel, str]) -> RiskLevel:
    if isinstance(val, RiskLevel):
        return val
    if isinstance(val, str):
        try:
            return RiskLevel(val.strip().lower())
        except ValueError:
            pass
    raise PolicyError(f"Invalid RiskLevel: {val!r}")


def _normalize_outcome(val: Union[PolicyDecisionOutcome, str]) -> PolicyDecisionOutcome:
    if isinstance(val, PolicyDecisionOutcome):
        return val
    if isinstance(val, str):
        try:
            return PolicyDecisionOutcome(val.strip().lower())
        except ValueError:
            pass
    raise PolicyError(f"Invalid PolicyDecisionOutcome: {val!r}")


def _max_risk(a: RiskLevel, b: RiskLevel) -> RiskLevel:
    return a if _RISK_SEVERITY[a] >= _RISK_SEVERITY[b] else b


def _safe_policy_text(text: Optional[str]) -> str:
    """Return secret-scrubbed string from sanitize_ledger_text without returning tuple flags."""
    if text is None:
        return ""
    cleaned, _, _ = sanitize_ledger_text(str(text))
    return str(cleaned or "")


def sanitize_policy_metadata(value: Any) -> Any:
    """Scrub secrets and never emit raw confirmation tokens into ledger/logs."""
    if isinstance(value, Mapping):
        scrubbed: Dict[str, Any] = {}
        for k, v in value.items():
            kl = str(k).lower()
            if any(secret_key in kl for secret_key in ("confirmation_token", "confirm_token", "raw_token", "token_secret")):
                scrubbed[str(k)] = "[REDACTED]"
            elif isinstance(v, Mapping):
                scrubbed[str(k)] = sanitize_policy_metadata(v)
            else:
                scrubbed[str(k)] = v
        cleaned_map, _, _ = sanitize_ledger_payload(scrubbed)
        return cleaned_map
    if isinstance(value, list):
        return [sanitize_policy_metadata(v) if isinstance(v, (Mapping, list, str)) else v for v in value]
    if isinstance(value, str):
        return _safe_policy_text(value)
    return value


@dataclass
class ExecutionPolicyConfig:
    """Trusted configuration driving deterministic policy decisions.

    Defaults are strictly conservative (`policy_mode="conservative"`):
    - READ_ONLY -> ALLOW
    - REVERSIBLE_WRITE / MUTATING / EXTERNAL_ACTION / PRIVILEGED -> REQUIRE_CONFIRMATION
    - IRREVERSIBLE / UNKNOWN -> DENY
    """

    policy_version: int = POLICY_VERSION
    policy_mode: str = "conservative"
    risk_defaults: Dict[str, str] = field(default_factory=dict)
    tool_overrides: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    resource_restrictions: Dict[str, Any] = field(default_factory=dict)
    confirmation_ttl_seconds: float = 300.0
    max_confirmation_lifetime_seconds: float = 900.0
    governor_strict_mode: bool = False

    def __post_init__(self) -> None:
        if self.policy_version not in SUPPORTED_POLICY_VERSIONS:
            raise UnsupportedPolicyVersionError(
                f"Unsupported execution policy version: {self.policy_version!r}"
            )
        mode_norm = str(self.policy_mode or "conservative").strip().lower()
        if mode_norm not in SUPPORTED_POLICY_MODES:
            # Fail conservative on unknown mode strings.
            mode_norm = "conservative"
        self.policy_mode = mode_norm

        try:
            ttl = float(self.confirmation_ttl_seconds)
        except (TypeError, ValueError):
            ttl = 300.0
        ttl = max(1.0, min(ttl, 86400.0))
        self.confirmation_ttl_seconds = ttl

        try:
            max_lt = float(self.max_confirmation_lifetime_seconds)
        except (TypeError, ValueError):
            max_lt = 900.0
        max_lt = max(ttl, min(max_lt, 86400.0))
        self.max_confirmation_lifetime_seconds = max_lt

        clean_risks: Dict[str, str] = {}
        for rk, out in (self.risk_defaults or {}).items():
            clean_risks[_normalize_risk_level(rk).value] = _normalize_outcome(out).value
        self.risk_defaults = clean_risks

        clean_overrides: Dict[str, Dict[str, Any]] = {}
        for tid, spec in (self.tool_overrides or {}).items():
            if isinstance(tid, str) and tid.strip() and isinstance(spec, Mapping):
                clean_overrides[tid.strip()] = dict(spec)
        self.tool_overrides = clean_overrides
        self.resource_restrictions = dict(self.resource_restrictions or {})

    @classmethod
    def from_zelyr_config(
        cls,
        config: Optional[Any] = None,
        *,
        governor_strict_mode: Optional[bool] = None,
    ) -> "ExecutionPolicyConfig":
        """Build an ExecutionPolicyConfig from a trusted ZelyrConfig instance."""
        if config is None:
            from zelyr.config import get_config

            config = get_config()

        auth_val = getattr(getattr(config, "auth_mode", None), "value", getattr(config, "auth_mode", "basic"))
        auth_str = str(auth_val or "basic").strip().lower()

        raw_mode = getattr(config, "execution_policy_mode", None)
        if raw_mode is not None and str(raw_mode).strip():
            eff_mode = str(raw_mode).strip().lower()
            if eff_mode not in SUPPORTED_POLICY_MODES:
                eff_mode = "conservative"
        elif auth_str == "strict":
            eff_mode = "strict"
        elif auth_str == "autopilot":
            eff_mode = "autopilot"
        else:
            eff_mode = "basic"

        strict_gov = bool(governor_strict_mode) if governor_strict_mode is not None else (auth_str == "strict")
        if governor_strict_mode is None:
            try:
                from zelyr.services.governor import GovernorService

                if GovernorService.instance().is_strict_mode():
                    strict_gov = True
            except Exception:
                pass
        if strict_gov and eff_mode == "basic":
            eff_mode = "strict"

        return cls(
            policy_version=POLICY_VERSION,
            policy_mode=eff_mode,
            risk_defaults=dict(getattr(config, "execution_policy_risk_defaults", {}) or {}),
            tool_overrides=dict(getattr(config, "execution_policy_tool_overrides", {}) or {}),
            resource_restrictions=dict(getattr(config, "execution_policy_resource_restrictions", {}) or {}),
            confirmation_ttl_seconds=getattr(config, "execution_policy_confirmation_ttl_seconds", 300.0),
            max_confirmation_lifetime_seconds=getattr(
                config, "execution_policy_max_confirmation_lifetime_seconds", 900.0
            ),
            governor_strict_mode=strict_gov,
        )

    def resolve_default_outcome_for_risk(self, risk_level: RiskLevel, *, workspace_scoped: bool = True) -> PolicyDecisionOutcome:
        """Resolve baseline PolicyDecisionOutcome for a RiskLevel under current mode & trusted overrides."""
        # Check explicit trusted risk_defaults first (UNKNOWN cannot be blindly auto-allowed).
        explicit = self.risk_defaults.get(risk_level.value)
        if explicit is not None:
            parsed = PolicyDecisionOutcome(explicit)
            if risk_level == RiskLevel.UNKNOWN and parsed == PolicyDecisionOutcome.ALLOW:
                return PolicyDecisionOutcome.DENY
            return parsed

        if risk_level == RiskLevel.UNKNOWN:
            return PolicyDecisionOutcome.DENY
        if risk_level == RiskLevel.IRREVERSIBLE:
            return PolicyDecisionOutcome.DENY
        if risk_level == RiskLevel.READ_ONLY:
            return PolicyDecisionOutcome.ALLOW

        if self.policy_mode in ("conservative", "strict"):
            return PolicyDecisionOutcome.REQUIRE_CONFIRMATION

        if self.policy_mode == "balanced":
            if risk_level == RiskLevel.REVERSIBLE_WRITE and workspace_scoped:
                return PolicyDecisionOutcome.ALLOW
            return PolicyDecisionOutcome.REQUIRE_CONFIRMATION

        if self.policy_mode == "basic":
            # Compatible with legacy AuthMode.BASIC once authorization, scope,
            # resource restrictions, and irreversible/unknown fail-closed checks pass.
            if self.governor_strict_mode and risk_level in (
                RiskLevel.PRIVILEGED,
                RiskLevel.EXTERNAL_ACTION,
                RiskLevel.MUTATING,
            ):
                return PolicyDecisionOutcome.REQUIRE_CONFIRMATION
            if not workspace_scoped and risk_level in (RiskLevel.REVERSIBLE_WRITE, RiskLevel.MUTATING):
                return PolicyDecisionOutcome.REQUIRE_CONFIRMATION
            return PolicyDecisionOutcome.ALLOW

        if self.policy_mode == "autopilot":
            return PolicyDecisionOutcome.ALLOW

        return PolicyDecisionOutcome.DENY


@dataclass
class ConfirmationRecord:
    """Server-side bound single-use confirmation record (never exposes raw token)."""

    confirmation_id: str
    token_hash: str
    tool_id: str
    arg_digest: str
    resource_scope: Tuple[str, ...]
    policy_version: int = POLICY_VERSION
    execution_id: Optional[str] = None
    node_id: Optional[str] = None
    issued_at: float = 0.0
    expires_at: float = 0.0
    max_expires_at: float = 0.0
    consumed: bool = False
    consumed_at: Optional[float] = None
    approved_by: str = "user"
    revoked: bool = False

    def to_safe_dict(self) -> Dict[str, Any]:
        """Return ledger/checkpoint-safe view without token secrets."""
        return {
            "confirmation_id": self.confirmation_id,
            "tool_id": self.tool_id,
            "arg_digest": self.arg_digest,
            "resource_scope": list(self.resource_scope),
            "policy_version": self.policy_version,
            "execution_id": self.execution_id,
            "node_id": self.node_id,
            "issued_at": self.issued_at,
            "expires_at": self.expires_at,
            "max_expires_at": self.max_expires_at,
            "consumed": self.consumed,
            "consumed_at": self.consumed_at,
            "approved_by": self.approved_by,
            "revoked": self.revoked,
        }


@dataclass
class ConfirmationVerificationResult:
    """Outcome of checking or consuming a bound confirmation against an evaluated action."""

    valid: bool
    status: str  # missing, verified, consumed, expired, replayed, mismatch, revoked, version_mismatch, invalid
    reason: str
    confirmation_id: Optional[str] = None


class ConfirmationStore:
    """Thread-safe atomic one-time confirmation store supporting memory and optional SQLite persistence."""

    def __init__(self, db_path: Optional[Union[str, Path]] = None) -> None:
        self._lock = threading.RLock()
        self._by_id: Dict[str, ConfirmationRecord] = {}
        self._id_by_hash: Dict[str, str] = {}
        self._db_path: Optional[Path] = Path(db_path) if db_path else None
        if self._db_path is not None:
            self._init_db()

    def _connect(self) -> sqlite3.Connection:
        assert self._db_path is not None
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self._db_path), timeout=30.0, isolation_level="EXCLUSIVE")
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS policy_confirmations (
                        confirmation_id TEXT PRIMARY KEY,
                        token_hash TEXT UNIQUE NOT NULL,
                        tool_id TEXT NOT NULL,
                        arg_digest TEXT NOT NULL,
                        resource_scope_json TEXT NOT NULL,
                        policy_version INTEGER NOT NULL,
                        execution_id TEXT,
                        node_id TEXT,
                        issued_at REAL NOT NULL,
                        expires_at REAL NOT NULL,
                        max_expires_at REAL NOT NULL,
                        consumed INTEGER NOT NULL DEFAULT 0,
                        consumed_at REAL,
                        approved_by TEXT NOT NULL DEFAULT 'user',
                        revoked INTEGER NOT NULL DEFAULT 0
                    )
                    """
                )
                conn.commit()
            finally:
                conn.close()

    @staticmethod
    def _hash_token(raw_token: str) -> str:
        return hashlib.sha256(raw_token.encode("utf-8")).hexdigest()

    def issue_confirmation(
        self,
        *,
        tool_id: str,
        arg_digest: str,
        resource_scope: Sequence[str] = (),
        execution_id: Optional[str] = None,
        node_id: Optional[str] = None,
        ttl_seconds: float = 300.0,
        max_lifetime_seconds: float = 900.0,
        policy_version: int = POLICY_VERSION,
        approved_by: str = "user",
        now: Optional[float] = None,
    ) -> Tuple[str, ConfirmationRecord]:
        """Issue a bound single-use confirmation returning `(raw_token, safe_record)`.

        Callers must keep `raw_token` out of model-visible prompts, TaskGraphs, and ledger text.
        """
        if not tool_id or not arg_digest:
            raise PolicyConfirmationError("Confirmation requires non-empty tool_id and arg_digest.")
        ts = float(now if now is not None else time.time())
        eff_ttl = max(0.001, float(ttl_seconds))
        eff_max = max(eff_ttl, float(max_lifetime_seconds))
        raw_token = secrets.token_urlsafe(32)
        token_hash = self._hash_token(raw_token)
        conf_id = f"conf_{secrets.token_hex(12)}"
        rec = ConfirmationRecord(
            confirmation_id=conf_id,
            token_hash=token_hash,
            tool_id=str(tool_id),
            arg_digest=str(arg_digest),
            resource_scope=tuple(sorted(str(x) for x in resource_scope)),
            policy_version=int(policy_version),
            execution_id=str(execution_id) if execution_id is not None else None,
            node_id=str(node_id) if node_id is not None else None,
            issued_at=ts,
            expires_at=ts + eff_ttl,
            max_expires_at=ts + eff_max,
            consumed=False,
            consumed_at=None,
            approved_by=str(approved_by or "user"),
            revoked=False,
        )
        with self._lock:
            self._by_id[conf_id] = rec
            self._id_by_hash[token_hash] = conf_id
            if self._db_path is not None:
                conn = self._connect()
                try:
                    conn.execute(
                        """
                        INSERT INTO policy_confirmations (
                            confirmation_id, token_hash, tool_id, arg_digest, resource_scope_json,
                            policy_version, execution_id, node_id, issued_at, expires_at,
                            max_expires_at, consumed, consumed_at, approved_by, revoked
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, NULL, ?, 0)
                        """,
                        (
                            rec.confirmation_id,
                            rec.token_hash,
                            rec.tool_id,
                            rec.arg_digest,
                            json.dumps(list(rec.resource_scope), separators=(",", ":")),
                            rec.policy_version,
                            rec.execution_id,
                            rec.node_id,
                            rec.issued_at,
                            rec.expires_at,
                            rec.max_expires_at,
                            rec.approved_by,
                        ),
                    )
                    conn.commit()
                finally:
                    conn.close()
        return raw_token, rec

    def _load_record_locked(
        self,
        *,
        confirmation_token: Optional[str] = None,
        confirmation_id: Optional[str] = None,
    ) -> Optional[ConfirmationRecord]:
        cid = confirmation_id
        tok_hash: Optional[str] = None
        if confirmation_token:
            tok_hash = self._hash_token(str(confirmation_token))
            cid = cid or self._id_by_hash.get(tok_hash)

        if self._db_path is not None:
            conn = self._connect()
            try:
                if tok_hash:
                    row = conn.execute(
                        "SELECT * FROM policy_confirmations WHERE token_hash = ?", (tok_hash,)
                    ).fetchone()
                elif cid:
                    row = conn.execute(
                        "SELECT * FROM policy_confirmations WHERE confirmation_id = ?", (cid,)
                    ).fetchone()
                else:
                    row = None
                if row is not None:
                    rec = ConfirmationRecord(
                        confirmation_id=str(row["confirmation_id"]),
                        token_hash=str(row["token_hash"]),
                        tool_id=str(row["tool_id"]),
                        arg_digest=str(row["arg_digest"]),
                        resource_scope=tuple(json.loads(row["resource_scope_json"] or "[]")),
                        policy_version=int(row["policy_version"]),
                        execution_id=row["execution_id"],
                        node_id=row["node_id"],
                        issued_at=float(row["issued_at"]),
                        expires_at=float(row["expires_at"]),
                        max_expires_at=float(row["max_expires_at"]),
                        consumed=bool(row["consumed"]),
                        consumed_at=float(row["consumed_at"]) if row["consumed_at"] is not None else None,
                        approved_by=str(row["approved_by"]),
                        revoked=bool(row["revoked"]),
                    )
                    self._by_id[rec.confirmation_id] = rec
                    self._id_by_hash[rec.token_hash] = rec.confirmation_id
                    if confirmation_id and rec.confirmation_id != confirmation_id:
                        return None
                    return rec
                return None
            finally:
                conn.close()

        if cid and cid in self._by_id:
            rec = self._by_id[cid]
            if tok_hash and not hmac.compare_digest(rec.token_hash, tok_hash):
                return None
            return rec
        return None

    def revoke_node_confirmations(self, execution_id: str, node_id: str) -> int:
        """Invalidate all unconsumed confirmations bound to `(execution_id, node_id)` on retry/replan/change."""
        count = 0
        with self._lock:
            for rec in self._by_id.values():
                if rec.execution_id == execution_id and rec.node_id == node_id and not rec.revoked:
                    rec.revoked = True
                    count += 1
            if self._db_path is not None:
                conn = self._connect()
                try:
                    cur = conn.execute(
                        """
                        UPDATE policy_confirmations SET revoked = 1
                        WHERE execution_id = ? AND node_id = ? AND revoked = 0
                        """,
                        (str(execution_id), str(node_id)),
                    )
                    count = max(count, cur.rowcount)
                    conn.commit()
                finally:
                    conn.close()
        return count

    def verify_or_consume(
        self,
        *,
        confirmation_token: Optional[str] = None,
        confirmation_id: Optional[str] = None,
        tool_id: str,
        arg_digest: str,
        resource_scope: Sequence[str] = (),
        execution_id: Optional[str] = None,
        node_id: Optional[str] = None,
        policy_version: int = POLICY_VERSION,
        consume: bool = False,
        now: Optional[float] = None,
    ) -> ConfirmationVerificationResult:
        """Validate secret confirmation token binding and atomically consume when `consume=True`."""
        if not confirmation_token or not isinstance(confirmation_token, str):
            return ConfirmationVerificationResult(
                valid=False,
                status="missing" if not confirmation_id else "invalid",
                reason="Explicit secret confirmation token required.",
                confirmation_id=confirmation_id,
            )

        ts = float(now if now is not None else time.time())
        norm_scope = tuple(sorted(str(x) for x in resource_scope))

        with self._lock:
            rec = self._load_record_locked(
                confirmation_token=confirmation_token,
                confirmation_id=confirmation_id,
            )
            if rec is None:
                return ConfirmationVerificationResult(
                    valid=False,
                    status="invalid",
                    reason="Confirmation token/id not recognized or mismatched.",
                    confirmation_id=confirmation_id,
                )
            if rec.revoked:
                return ConfirmationVerificationResult(
                    valid=False,
                    status="revoked",
                    reason=f"Confirmation '{rec.confirmation_id}' has been revoked.",
                    confirmation_id=rec.confirmation_id,
                )
            if rec.consumed:
                return ConfirmationVerificationResult(
                    valid=False,
                    status="replayed",
                    reason=f"Confirmation '{rec.confirmation_id}' was already consumed (replay rejected).",
                    confirmation_id=rec.confirmation_id,
                )
            if ts > rec.expires_at or ts > rec.max_expires_at:
                return ConfirmationVerificationResult(
                    valid=False,
                    status="expired",
                    reason=f"Confirmation '{rec.confirmation_id}' has expired.",
                    confirmation_id=rec.confirmation_id,
                )
            if rec.policy_version != int(policy_version):
                return ConfirmationVerificationResult(
                    valid=False,
                    status="version_mismatch",
                    reason=(
                        f"Confirmation policy version mismatch: expected {policy_version}, "
                        f"got {rec.policy_version}."
                    ),
                    confirmation_id=rec.confirmation_id,
                )
            if rec.tool_id != str(tool_id):
                return ConfirmationVerificationResult(
                    valid=False,
                    status="mismatch",
                    reason=f"Confirmation bound to tool '{rec.tool_id}', cannot apply to '{tool_id}'.",
                    confirmation_id=rec.confirmation_id,
                )
            if not hmac.compare_digest(rec.arg_digest, str(arg_digest)):
                return ConfirmationVerificationResult(
                    valid=False,
                    status="mismatch",
                    reason="Confirmation argument digest mismatch (action arguments changed).",
                    confirmation_id=rec.confirmation_id,
                )
            if rec.resource_scope != norm_scope:
                return ConfirmationVerificationResult(
                    valid=False,
                    status="mismatch",
                    reason="Confirmation resource scope mismatch.",
                    confirmation_id=rec.confirmation_id,
                )
            if rec.execution_id is not None and execution_id is not None and rec.execution_id != execution_id:
                return ConfirmationVerificationResult(
                    valid=False,
                    status="mismatch",
                    reason="Confirmation bound to a different execution_id.",
                    confirmation_id=rec.confirmation_id,
                )
            if rec.execution_id is not None and execution_id is None:
                return ConfirmationVerificationResult(
                    valid=False,
                    status="mismatch",
                    reason="Confirmation requires matching execution_id context.",
                    confirmation_id=rec.confirmation_id,
                )
            if rec.node_id is not None and node_id is not None and rec.node_id != node_id:
                return ConfirmationVerificationResult(
                    valid=False,
                    status="mismatch",
                    reason="Confirmation bound to a different node_id.",
                    confirmation_id=rec.confirmation_id,
                )
            if rec.node_id is not None and node_id is None:
                return ConfirmationVerificationResult(
                    valid=False,
                    status="mismatch",
                    reason="Confirmation requires matching node_id context.",
                    confirmation_id=rec.confirmation_id,
                )

            if consume:
                if self._db_path is not None:
                    conn = self._connect()
                    try:
                        cur = conn.execute(
                            """
                            UPDATE policy_confirmations
                            SET consumed = 1, consumed_at = ?
                            WHERE confirmation_id = ? AND consumed = 0 AND revoked = 0
                            """,
                            (ts, rec.confirmation_id),
                        )
                        conn.commit()
                        if cur.rowcount != 1:
                            rec.consumed = True
                            return ConfirmationVerificationResult(
                                valid=False,
                                status="replayed",
                                reason=f"Confirmation '{rec.confirmation_id}' already consumed concurrently.",
                                confirmation_id=rec.confirmation_id,
                            )
                    finally:
                        conn.close()
                rec.consumed = True
                rec.consumed_at = ts
                return ConfirmationVerificationResult(
                    valid=True,
                    status="consumed",
                    reason="Bound single-use confirmation verified and consumed.",
                    confirmation_id=rec.confirmation_id,
                )

            return ConfirmationVerificationResult(
                valid=True,
                status="verified",
                reason="Bound confirmation verified.",
                confirmation_id=rec.confirmation_id,
            )


_DEFAULT_CONFIRMATION_STORE = ConfirmationStore()


def get_default_confirmation_store() -> ConfirmationStore:
    """Return process-wide default thread-safe ConfirmationStore."""
    return _DEFAULT_CONFIRMATION_STORE


@dataclass
class ExecutionPolicyDecision:
    """Deterministic structured policy evaluation result for a tool call."""

    decision_id: str
    outcome: PolicyDecisionOutcome
    risk_level: RiskLevel
    tool_id: str
    normalized_arguments: Dict[str, Any]
    arg_digest: str
    resource_scope: List[str]
    policy_version: int = POLICY_VERSION
    rule_ids: List[str] = field(default_factory=list)
    reason: str = ""
    confirmation_required: bool = False
    confirmation_status: str = "not_required"
    confirmation_id: Optional[str] = None
    execution_id: Optional[str] = None
    node_id: Optional[str] = None
    source_surface: str = "unknown"
    is_external: bool = False
    source_id: Optional[str] = None
    source_type: Optional[str] = None
    remediation_hint: str = ""

    @property
    def allowed(self) -> bool:
        return self.outcome == PolicyDecisionOutcome.ALLOW

    def to_safe_dict(self) -> Dict[str, Any]:
        """Serialize sanitized policy decision safe for ledgers, error responses, and model feedback."""
        return {
            "decision_id": self.decision_id,
            "outcome": self.outcome.value,
            "risk_level": self.risk_level.value,
            "tool_id": self.tool_id,
            "arg_digest": self.arg_digest,
            "resource_scope": list(self.resource_scope),
            "policy_version": self.policy_version,
            "rule_ids": list(self.rule_ids),
            "reason": _safe_policy_text(self.reason),
            "confirmation_required": self.confirmation_required,
            "confirmation_status": self.confirmation_status,
            "confirmation_id": self.confirmation_id,
            "execution_id": self.execution_id,
            "node_id": self.node_id,
            "source_surface": self.source_surface,
            "is_external": self.is_external,
            "remediation_hint": _safe_policy_text(self.remediation_hint),
        }

    def to_tool_error_payload(self) -> Dict[str, Any]:
        """Build deterministic error dict suitable for tool runners and orchestrator messages."""
        status = "confirmation_required" if self.outcome == PolicyDecisionOutcome.REQUIRE_CONFIRMATION else "denied"
        err_type = (
            "POLICY_CONFIRMATION_REQUIRED"
            if self.outcome == PolicyDecisionOutcome.REQUIRE_CONFIRMATION
            else "POLICY_DENIED"
        )
        return {
            "success": False,
            "status": status,
            "error_type": err_type,
            "reason": "confirmation_required" if self.outcome == PolicyDecisionOutcome.REQUIRE_CONFIRMATION else "policy_denied",
            "error": _safe_policy_text(self.reason),
            "tool": self.tool_id,
            "risk_level": self.risk_level.value,
            "decision_id": self.decision_id,
            "arg_digest": self.arg_digest,
            "resource_scope": list(self.resource_scope),
            "rule_ids": list(self.rule_ids),
            "confirmation_status": self.confirmation_status,
            "remediation_hint": _safe_policy_text(self.remediation_hint),
        }


class RiskClassifier:
    """Deterministic tool/action risk classifier using trusted metadata and structural inspection.

    Never lowers risk based on untrusted description prose or model-provided hints.
    """

    @staticmethod
    def classify(
        tool_id: str,
        normalized_arguments: Mapping[str, Any],
        *,
        tool_obj: Optional[Any] = None,
        schema: Optional[Mapping[str, Any]] = None,
        required_ring: Optional[int] = None,
        is_registered: bool = True,
        is_external: bool = False,
        source_id: Optional[str] = None,
        source_type: Optional[str] = None,
        external_metadata: Optional[Mapping[str, Any]] = None,
        workspace_root: Optional[Union[str, Path]] = None,
    ) -> Tuple[RiskLevel, List[str]]:
        """Classify `(tool_id, normalized_arguments)` into `(RiskLevel, rule_ids)`."""
        tid = str(tool_id or "").strip()
        rules: List[str] = []
        ws_path = Path(workspace_root).expanduser().resolve(strict=False) if workspace_root else None
        raw_paths = _extract_raw_paths(normalized_arguments, ws_path)
        action = str(normalized_arguments.get("action") or "").strip().lower()
        cmd = str(normalized_arguments.get("command") or normalized_arguments.get("cmd") or "").strip()

        # 1. Fail closed on unknown / unrecognized tools.
        if not tid or not is_registered:
            rules.append("risk:unregistered_unknown_tool")
            return RiskLevel.UNKNOWN, rules

        # 2. Structural irreversible detection (always highest precedence among known tools).
        if tid in _IRREVERSIBLE_TOOL_IDS:
            rules.append(f"risk:irreversible_tool:{tid}")
            return RiskLevel.IRREVERSIBLE, rules
        if action in _DESTRUCTIVE_ACTIONS:
            rules.append(f"risk:destructive_action:{action}")
            return RiskLevel.IRREVERSIBLE, rules
        if bool(normalized_arguments.get("delete")) or bool(normalized_arguments.get("wipe")):
            rules.append("risk:destructive_flag")
            return RiskLevel.IRREVERSIBLE, rules
        if cmd:
            for pat in _IRREVERSIBLE_SHELL_PATTERNS:
                if pat.search(cmd):
                    rules.append("risk:irreversible_shell_command")
                    return RiskLevel.IRREVERSIBLE, rules
        if isinstance(normalized_arguments.get("instruction"), str):
            instr = str(normalized_arguments.get("instruction"))
            for pat in _DANGEROUS_UI_INSTRUCTION_PATTERNS:
                if pat.search(instr):
                    rules.append("risk:irreversible_ui_instruction")
                    return RiskLevel.IRREVERSIBLE, rules
        if isinstance(external_metadata, Mapping) and (
            external_metadata.get("irreversible") is True
            or external_metadata.get("destructiveHint") is True
            or external_metadata.get("destructive_hint") is True
        ):
            rules.append("risk:external_destructive_hint")
            return RiskLevel.IRREVERSIBLE, rules

        # 3. External tools default to EXTERNAL_ACTION (never trusted solely by description text).
        if is_external or tid.startswith("ext__"):
            rules.append("risk:external_tool")
            return RiskLevel.EXTERNAL_ACTION, rules

        # 4. Privileged / Ring-0 / OS automation / shell execution.
        # EchoTool is explicitly safe read-only even when Ring 0 in test registries.
        if tid in _PRIVILEGED_TOOL_IDS or (
            required_ring == 0 and tid not in _READ_ONLY_TOOL_IDS and tid != "manage_workspace_file"
        ):
            # Read-only inspection action inside gui_inspect stays READ_ONLY; mutating GUI/shell stay PRIVILEGED.
            if tid == "desktop_automation" and action == "open_url":
                url = str(normalized_arguments.get("url") or "").strip().lower()
                if url.startswith(("https://", "http://")):
                    rules.append("risk:desktop_fast_path_url")
                    return RiskLevel.MUTATING, rules
            rules.append("risk:privileged_or_ring0")
            return RiskLevel.PRIVILEGED, rules

        # 5. Builtin read-only / reversible / mutating semantics.
        if tid == "manage_workspace_file":
            if action in _WORKSPACE_SAFE_READ_ACTIONS:
                rules.append("risk:workspace_read_only")
                return RiskLevel.READ_ONLY, rules
            if action in _WORKSPACE_REVERSIBLE_WRITE_ACTIONS:
                # Reversible only if target stays in workspace (when workspace_root known).
                if ws_path is not None and any(not _is_path_within(canon, ws_path) for _, canon in raw_paths):
                    rules.append("risk:outside_workspace_write")
                    return RiskLevel.MUTATING, rules
                rules.append("risk:workspace_reversible_write")
                return RiskLevel.REVERSIBLE_WRITE, rules
            rules.append("risk:workspace_unknown_action")
            return RiskLevel.MUTATING, rules

        if tid in _READ_ONLY_TOOL_IDS:
            rules.append(f"risk:known_read_only:{tid}")
            return RiskLevel.READ_ONLY, rules

        # Explicit schema/tool metadata marks safe read_only tool ONLY when non-external & non-privileged.
        fn_meta = (
            dict(schema.get("function", schema))
            if isinstance(schema, Mapping)
            else {}
        )
        tool_ro_attr = bool(getattr(tool_obj, "read_only", False)) if tool_obj is not None else False
        schema_ro = bool(fn_meta.get("read_only", False)) or bool(
            isinstance(fn_meta.get("annotations"), Mapping) and fn_meta["annotations"].get("readOnlyHint") is True
        )
        if (tool_ro_attr or schema_ro) and action not in _WORKSPACE_REVERSIBLE_WRITE_ACTIONS:
            rules.append("risk:metadata_read_only")
            return RiskLevel.READ_ONLY, rules

        # Explicit reversible metadata or safe workspace file write.
        tool_rev_attr = bool(getattr(tool_obj, "reversible", False)) if tool_obj is not None else False
        schema_rev = bool(fn_meta.get("reversible", False))
        if (tool_rev_attr or schema_rev) and not (
            ws_path is not None and any(not _is_path_within(canon, ws_path) for _, canon in raw_paths)
        ):
            rules.append("risk:metadata_reversible_write")
            return RiskLevel.REVERSIBLE_WRITE, rules

        if tid in _MUTATING_TOOL_IDS:
            rules.append(f"risk:known_mutating:{tid}")
            return RiskLevel.MUTATING, rules

        # Custom registered/TaskLocalToolScope tools without explicit read_only/privileged classification:
        # classify as REVERSIBLE_WRITE if all targets stay inside workspace_root or tool has a safe write/read signature,
        # otherwise MUTATING (never READ_ONLY without proof!).
        if ws_path is not None and raw_paths and all(_is_path_within(canon, ws_path) for _, canon in raw_paths):
            rules.append("risk:scoped_mutating")
            return RiskLevel.MUTATING, rules
        rules.append("risk:default_mutating")
        return RiskLevel.MUTATING, rules


class ExecutionPolicyEngine:
    """Centralized, deterministic execution-policy evaluator.

    Precedence order (strictly enforced, cannot be weakened by model input):
      1. Hard security boundaries (untrusted keys, sensitive paths, destructive commands, invalid shapes)
      2. Runtime authorization & isolation (`authorize_and_validate_tool_call`, Ring, allowlist, scope, external auth)
      3. Trusted tool-specific policy overrides & restrictions
      4. Trusted resource-scope & workspace containment restrictions
      5. Configured user risk policy defaults & policy mode
      6. Bound single-use confirmation validation / atomic consumption
    """

    def __init__(
        self,
        config: Optional[ExecutionPolicyConfig] = None,
        confirmation_store: Optional[ConfirmationStore] = None,
    ) -> None:
        self.config = config or ExecutionPolicyConfig()
        self.confirmation_store = confirmation_store or get_default_confirmation_store()

    def evaluate(
        self,
        tool_id: str,
        raw_arguments: Optional[Mapping[str, Any]] = None,
        *,
        normalized_arguments: Optional[Mapping[str, Any]] = None,
        schema: Optional[Mapping[str, Any]] = None,
        tool_obj: Optional[Any] = None,
        required_ring: Optional[int] = None,
        auth_decision: Optional[Any] = None,
        is_registered: bool = True,
        is_external: bool = False,
        source_id: Optional[str] = None,
        source_type: Optional[str] = None,
        external_metadata: Optional[Mapping[str, Any]] = None,
        workspace_root: Optional[Union[str, Path]] = None,
        node_target_resources: Sequence[str] = (),
        execution_id: Optional[str] = None,
        node_id: Optional[str] = None,
        correlation_id: Optional[str] = None,
        confirmation_token: Optional[str] = None,
        confirmation_id: Optional[str] = None,
        consume_confirmation: bool = False,
        confirm_callback: Optional[Callable[[ExecutionPolicyDecision], Any]] = None,
        source_surface: str = "runtime",
        central_ledger: Optional[Any] = None,
        record_to_ledger: bool = True,
        now: Optional[float] = None,
    ) -> ExecutionPolicyDecision:
        """Evaluate execution policy deterministically across Rules 1..6 and optionally log to ledger."""
        tid = str(getattr(auth_decision, "tool_name", None) or tool_id or "").strip()
        decision_id = f"pol_{secrets.token_hex(8)}"
        ws_path = Path(workspace_root).expanduser().resolve(strict=False) if workspace_root else None
        rule_ids: List[str] = []

        # Obtain normalized arguments without trusting override/auth keys.
        raw_map: Dict[str, Any] = {}
        raw_input_ok = True
        if normalized_arguments is not None:
            if isinstance(normalized_arguments, Mapping):
                raw_map = dict(normalized_arguments)
            else:
                raw_input_ok = False
        elif raw_arguments is not None:
            if isinstance(raw_arguments, Mapping):
                raw_map = dict(raw_arguments)
            else:
                raw_input_ok = False

        # Scan raw_arguments (if supplied) AND raw_map for smuggled untrusted policy/auth keys.
        smuggled_keys: List[str] = []
        for source_map in (raw_arguments, raw_map):
            if isinstance(source_map, Mapping):
                for k in source_map.keys():
                    if str(k).strip() in UNTRUSTED_POLICY_OVERRIDE_KEYS:
                        smuggled_keys.append(str(k).strip())

        if (
            normalized_arguments is None
            and raw_input_ok
            and isinstance(schema, Mapping)
        ):
            clean_args, _, norm_err = SafeArgumentNormalizer.normalize_and_validate(
                tid,
                dict(schema),
                raw_map,
                argument_source="model",
            )
            if clean_args is not None and not norm_err:
                eff_args = dict(clean_args)
            else:
                eff_args = dict(raw_map)
        else:
            eff_args = dict(raw_map)

        resource_scope = extract_resource_scope(
            tid,
            eff_args,
            workspace_root=ws_path,
            node_target_resources=node_target_resources,
        )
        arg_digest = compute_canonical_argument_digest(tid, eff_args, resource_scope)

        # Classify risk deterministically from trusted inputs only.
        risk_level, risk_rules = RiskClassifier.classify(
            tid,
            eff_args,
            tool_obj=tool_obj,
            schema=schema,
            required_ring=required_ring,
            is_registered=is_registered,
            is_external=is_external,
            external_metadata=external_metadata,
            workspace_root=ws_path,
        )
        rule_ids.extend(risk_rules)

        def _finalize(
            outcome: PolicyDecisionOutcome,
            reason: str,
            *,
            conf_req: bool = False,
            conf_status: str = "not_required",
            conf_id: Optional[str] = None,
            remediation: str = "",
        ) -> ExecutionPolicyDecision:
            dec = ExecutionPolicyDecision(
                decision_id=decision_id,
                outcome=outcome,
                risk_level=risk_level,
                tool_id=tid,
                normalized_arguments=eff_args,
                arg_digest=arg_digest,
                resource_scope=resource_scope,
                policy_version=self.config.policy_version,
                rule_ids=rule_ids,
                reason=reason,
                confirmation_required=conf_req,
                confirmation_status=conf_status,
                confirmation_id=conf_id,
                execution_id=execution_id,
                node_id=node_id,
                source_surface=source_surface,
                is_external=is_external,
                source_id=source_id,
                source_type=source_type,
                remediation_hint=remediation,
            )
            if record_to_ledger and central_ledger is not None and execution_id:
                self._record_decision_to_ledger(
                    central_ledger,
                    dec,
                    correlation_id=correlation_id,
                )
            return dec

        # ── Rule 1: Hard Security Boundaries ───────────────────────────────────
        if not raw_input_ok:
            rule_ids.append("rule1:invalid_argument_container")
            return _finalize(
                PolicyDecisionOutcome.DENY,
                f"Policy denied '{tid}': arguments must be a dictionary object.",
                remediation="Provide structured dictionary arguments.",
            )

        if smuggled_keys:
            rule_ids.append("rule1:untrusted_policy_or_auth_keys")
            return _finalize(
                PolicyDecisionOutcome.DENY,
                f"Policy denied '{tid}': untrusted authorization/policy keys {sorted(set(smuggled_keys))} rejected.",
                remediation="Remove authorization and policy override fields from tool arguments.",
            )

        raw_paths = _extract_raw_paths(eff_args, ws_path)
        cmd_str = str(eff_args.get("command") or eff_args.get("cmd") or "").strip().lower()
        action = str(eff_args.get("action") or "").strip().lower()
        is_mutating_action = risk_level not in (RiskLevel.READ_ONLY,)

        if cmd_str and any(marker in cmd_str for marker in _HARD_RESTRICTED_PATH_MARKERS):
            rule_ids.append("rule1:restricted_credential_or_security_command")
            return _finalize(
                PolicyDecisionOutcome.DENY,
                f"Policy denied '{tid}': command references protected system or credential path.",
                remediation="Do not access or modify protected system or credential files.",
            )

        for raw_p, canon_p in raw_paths:
            low_canon = canon_p.lower()
            low_raw = raw_p.lower()
            base_name = Path(canon_p).name.lower()
            if any(marker in low_canon or marker in low_raw for marker in _HARD_RESTRICTED_PATH_MARKERS):
                rule_ids.append("rule1:restricted_credential_or_security_path")
                return _finalize(
                    PolicyDecisionOutcome.DENY,
                    f"Policy denied '{tid}': target path references protected system/credential resource.",
                    remediation="Target workspace files rather than system security or credential stores.",
                )
            if base_name in _SENSITIVE_FILENAMES and (
                is_mutating_action or base_name in {"shadow", "gshadow", "sudoers", "id_rsa", "id_ed25519"}
            ):
                rule_ids.append("rule1:sensitive_file_target")
                return _finalize(
                    PolicyDecisionOutcome.DENY,
                    f"Policy denied '{tid}': sensitive target '{base_name}' is protected by hard security policy.",
                    remediation="Do not access or modify secret/credential files.",
                )
            if is_mutating_action and any(low_canon.startswith(prefix) or low_canon == prefix.rstrip("/") for prefix in _MUTATING_SYSTEM_FILES):
                rule_ids.append("rule1:protected_system_file_mutation")
                return _finalize(
                    PolicyDecisionOutcome.DENY,
                    f"Policy denied '{tid}': mutating system path '{canon_p}' is forbidden.",
                    remediation="Confine mutations to authorized workspace paths.",
                )

        # ── Rule 2: Runtime Authorization & Isolation Boundaries ──────────────
        if auth_decision is not None:
            auth_ok = bool(getattr(auth_decision, "allowed", True))
            if not auth_ok:
                err_msg = str(
                    getattr(auth_decision, "detail", None)
                    or getattr(auth_decision, "error_message", None)
                    or "Authorization denied."
                )
                err_type = str(getattr(auth_decision, "error_type", None) or "AUTHORIZATION_DENIED")
                rule_ids.append(f"rule2:auth_denied:{err_type}")
                return _finalize(
                    PolicyDecisionOutcome.DENY,
                    err_msg,
                    remediation="Ensure tool is registered, authorized for caller ring/allowlist/scope, and valid.",
                )
            rule_ids.append("rule2:auth_passed")

        # ── Rule 3: Trusted Tool-Specific Overrides ───────────────────────────
        override = self.config.tool_overrides.get(tid) or {}
        if not override and tool_id and tool_id != tid:
            override = self.config.tool_overrides.get(tool_id) or {}

        # ── Rule 4: Resource-Scope & Workspace Containment Restrictions ───────
        restrictions = self.config.resource_restrictions or {}
        denied_prefixes = [str(p).lower() for p in (restrictions.get("denied_path_prefixes") or []) if p]
        denied_hosts = {str(h).lower() for h in (restrictions.get("denied_hosts") or []) if h}
        enforce_ws = bool(restrictions.get("enforce_workspace_containment", False))
        restricted_ws = restrictions.get("workspace_root") or ws_path
        restricted_ws_path = Path(restricted_ws).expanduser().resolve(strict=False) if restricted_ws else None

        outside_workspace = False
        for _, canon_p in raw_paths:
            low_c = canon_p.lower()
            for pref in denied_prefixes:
                if low_c.startswith(pref):
                    rule_ids.append("rule4:denied_path_prefix")
                    return _finalize(
                        PolicyDecisionOutcome.DENY,
                        f"Policy denied '{tid}': path '{canon_p}' falls under restricted prefix.",
                        remediation="Select an allowed path outside restricted directories.",
                    )
            if restricted_ws_path is not None and not _is_path_within(canon_p, restricted_ws_path):
                outside_workspace = True
                if enforce_ws and is_mutating_action:
                    rule_ids.append("rule4:outside_workspace_mutation_denied")
                    return _finalize(
                        PolicyDecisionOutcome.DENY,
                        f"Policy denied '{tid}': mutating path '{canon_p}' escapes configured workspace '{restricted_ws_path}'.",
                        remediation="Keep file modifications inside the configured workspace root.",
                    )

        url_val = str(eff_args.get("url") or eff_args.get("endpoint") or "").strip().lower()
        if url_val and denied_hosts:
            for dh in denied_hosts:
                if dh in url_val:
                    rule_ids.append("rule4:denied_host")
                    return _finalize(
                        PolicyDecisionOutcome.DENY,
                        f"Policy denied '{tid}': target host is restricted by resource policy.",
                        remediation="Target an approved host/endpoint.",
                    )

        # Evaluate override outcomes AFTER hard security & resource restrictions (Rules 1-4 stay supreme!).
        base_outcome: Optional[PolicyDecisionOutcome] = None
        if override:
            if "outcome" in override:
                base_outcome = _normalize_outcome(override["outcome"])
                rule_ids.append(f"rule3:tool_override_outcome:{base_outcome.value}")
                if base_outcome == PolicyDecisionOutcome.DENY:
                    return _finalize(
                        PolicyDecisionOutcome.DENY,
                        f"Policy denied '{tid}' by explicit trusted tool policy override.",
                        remediation="Tool is explicitly disallowed by policy configuration.",
                    )
            elif override.get("deny") is True:
                rule_ids.append("rule3:tool_override_deny")
                return _finalize(
                    PolicyDecisionOutcome.DENY,
                    f"Policy denied '{tid}' by explicit trusted tool denial rule.",
                    remediation="Tool is explicitly disallowed by policy configuration.",
                )
            if "risk_level" in override:
                ov_risk = _normalize_risk_level(override["risk_level"])
                # Never allow an override to downgrade UNKNOWN or IRREVERSIBLE unless explicitly confirming/allowing via outcome.
                risk_level = _max_risk(risk_level, ov_risk)
                rule_ids.append(f"rule3:tool_override_risk:{risk_level.value}")

        # Escalate outside-workspace writes if not already handled.
        if outside_workspace and is_mutating_action and risk_level == RiskLevel.REVERSIBLE_WRITE:
            risk_level = RiskLevel.MUTATING
            rule_ids.append("rule4:outside_workspace_escalated_to_mutating")

        # ── Rule 5: Configured User Risk Policy ───────────────────────────────
        if base_outcome is None:
            base_outcome = self.config.resolve_default_outcome_for_risk(
                risk_level,
                workspace_scoped=not outside_workspace,
            )
            rule_ids.append(f"rule5:risk_policy:{self.config.policy_mode}:{risk_level.value}->{base_outcome.value}")

        # Unknown and Irreversible default to DENY under Rule 5 and NEVER accept casual overrides unless trusted policy explicitly permitted confirmation.
        if base_outcome == PolicyDecisionOutcome.DENY:
            return _finalize(
                PolicyDecisionOutcome.DENY,
                (
                    f"Policy denied '{tid}' (risk={risk_level.value}) under policy mode '{self.config.policy_mode}'."
                ),
                remediation=(
                    "Tool is unknown or classified irreversible and blocked by default safety policy."
                    if risk_level in (RiskLevel.UNKNOWN, RiskLevel.IRREVERSIBLE)
                    else "Operation blocked by execution policy."
                ),
            )

        if base_outcome == PolicyDecisionOutcome.ALLOW:
            return _finalize(
                PolicyDecisionOutcome.ALLOW,
                f"Policy allowed '{tid}' (risk={risk_level.value}) automatically under mode '{self.config.policy_mode}'.",
                conf_req=False,
                conf_status="not_required",
            )

        # ── Rule 6: Explicit Bound Confirmation Validation ────────────────────
        if confirmation_token or confirmation_id:
            ver = self.confirmation_store.verify_or_consume(
                confirmation_token=confirmation_token,
                confirmation_id=confirmation_id,
                tool_id=tid,
                arg_digest=arg_digest,
                resource_scope=resource_scope,
                execution_id=execution_id,
                node_id=node_id,
                policy_version=self.config.policy_version,
                consume=consume_confirmation,
                now=now,
            )
            if ver.valid:
                rule_ids.append(f"rule6:confirmation_{ver.status}")
                return _finalize(
                    PolicyDecisionOutcome.ALLOW,
                    f"Policy allowed '{tid}' with bound single-use confirmation ({ver.confirmation_id}).",
                    conf_req=True,
                    conf_status=ver.status,
                    conf_id=ver.confirmation_id,
                )
            rule_ids.append(f"rule6:confirmation_{ver.status}")
            # Expired, replayed, revoked, or mismatched confirmations fail closed cleanly.
            fail_outcome = (
                PolicyDecisionOutcome.DENY
                if ver.status in ("replayed", "expired", "revoked", "version_mismatch", "invalid", "mismatch")
                else PolicyDecisionOutcome.REQUIRE_CONFIRMATION
            )
            return _finalize(
                fail_outcome,
                f"Policy rejected '{tid}': {ver.reason}",
                conf_req=True,
                conf_status=ver.status,
                conf_id=ver.confirmation_id,
                remediation="Obtain a fresh confirmation bound specifically to these exact tool arguments and target.",
            )

        # Optional trusted interactive/programmatic confirmation callback (sync callbacks resolved here).
        if confirm_callback is not None:
            pre_decision = ExecutionPolicyDecision(
                decision_id=decision_id,
                outcome=PolicyDecisionOutcome.REQUIRE_CONFIRMATION,
                risk_level=risk_level,
                tool_id=tid,
                normalized_arguments=eff_args,
                arg_digest=arg_digest,
                resource_scope=resource_scope,
                policy_version=self.config.policy_version,
                rule_ids=list(rule_ids),
                reason=(
                    f"Explicit confirmation required for '{tid}' (risk={risk_level.value}, scope={resource_scope})."
                ),
                confirmation_required=True,
                confirmation_status="pending",
                execution_id=execution_id,
                node_id=node_id,
                source_surface=source_surface,
                is_external=is_external,
                remediation_hint="Confirm exact normalized action to proceed.",
            )
            approved = bool(confirm_callback(pre_decision))
            if approved:
                raw_tok, rec = self.confirmation_store.issue_confirmation(
                    tool_id=tid,
                    arg_digest=arg_digest,
                    resource_scope=resource_scope,
                    execution_id=execution_id,
                    node_id=node_id,
                    ttl_seconds=self.config.confirmation_ttl_seconds,
                    max_lifetime_seconds=self.config.max_confirmation_lifetime_seconds,
                    policy_version=self.config.policy_version,
                    approved_by="confirm_callback",
                    now=now,
                )
                ver = self.confirmation_store.verify_or_consume(
                    confirmation_token=raw_tok,
                    confirmation_id=rec.confirmation_id,
                    tool_id=tid,
                    arg_digest=arg_digest,
                    resource_scope=resource_scope,
                    execution_id=execution_id,
                    node_id=node_id,
                    policy_version=self.config.policy_version,
                    consume=consume_confirmation,
                    now=now,
                )
                if ver.valid:
                    rule_ids.append("rule6:callback_confirmed")
                    return _finalize(
                        PolicyDecisionOutcome.ALLOW,
                        f"Policy allowed '{tid}' after interactive confirmation ({rec.confirmation_id}).",
                        conf_req=True,
                        conf_status=ver.status,
                        conf_id=rec.confirmation_id,
                    )
            rule_ids.append("rule6:callback_denied")
            return _finalize(
                PolicyDecisionOutcome.DENY,
                f"Policy denied '{tid}': interactive confirmation declined.",
                conf_req=True,
                conf_status="declined",
                remediation="User declined confirmation for this action.",
            )

        rule_ids.append("rule6:confirmation_missing")
        return _finalize(
            PolicyDecisionOutcome.REQUIRE_CONFIRMATION,
            (
                f"Explicit confirmation required before executing '{tid}' "
                f"(risk={risk_level.value}, mode='{self.config.policy_mode}')."
            ),
            conf_req=True,
            conf_status="missing",
            remediation="Issue a bound confirmation for this exact action digest or confirm interactively.",
        )

    @staticmethod
    def _record_decision_to_ledger(
        central_ledger: Any,
        decision: ExecutionPolicyDecision,
        *,
        correlation_id: Optional[str] = None,
    ) -> None:
        if not decision.execution_id:
            return
        try:
            status_map = {
                PolicyDecisionOutcome.ALLOW: "allowed",
                PolicyDecisionOutcome.REQUIRE_CONFIRMATION: "requires_confirmation",
                PolicyDecisionOutcome.DENY: "denied",
            }
            central_ledger.record_event(
                decision.execution_id,
                LedgerEventType.AUTHORIZATION_DECISION,
                tool_id=decision.tool_id,
                source_id=decision.source_id,
                source_type=decision.source_type,
                is_external=decision.is_external,
                correlation_id=correlation_id or decision.decision_id,
                status=status_map.get(decision.outcome, "denied"),
                metadata={
                    "policy_decision": decision.outcome.value,
                    "decision_id": decision.decision_id,
                    "node_id": decision.node_id,
                    "risk_level": decision.risk_level.value,
                    "arg_digest": decision.arg_digest,
                    "resource_scope": decision.resource_scope,
                    "rule_ids": decision.rule_ids,
                    "confirmation_required": decision.confirmation_required,
                    "confirmation_status": decision.confirmation_status,
                    "confirmation_id": decision.confirmation_id,
                    "policy_version": decision.policy_version,
                    "source_surface": decision.source_surface,
                    "is_external": decision.is_external,
                },
                idempotency_key=f"{decision.execution_id}:{correlation_id or decision.decision_id}:policy:{decision.decision_id}",
            )
        except Exception:
            # Never break fail-closed policy evaluation if ledger logging fails.
            pass

    async def evaluate_async(
        self,
        tool_id: str,
        raw_arguments: Optional[Mapping[str, Any]] = None,
        *,
        confirm_callback: Optional[Callable[[ExecutionPolicyDecision], Any]] = None,
        **kwargs: Any,
    ) -> ExecutionPolicyDecision:
        """Async wrapper over `evaluate` supporting awaitable confirmation callbacks."""
        import inspect

        consume_flag = bool(kwargs.pop("consume_confirmation", False))
        if confirm_callback is None:
            return self.evaluate(
                tool_id,
                raw_arguments,
                consume_confirmation=consume_flag,
                **kwargs,
            )

        # Evaluate without consuming confirmation first; if REQUIRE_CONFIRMATION without token,
        # invoke callback (supporting async awaitables) and re-evaluate cleanly with issued token.
        pre = self.evaluate(
            tool_id,
            raw_arguments,
            consume_confirmation=False,
            record_to_ledger=False,
            confirm_callback=None,
            **kwargs,
        )
        if pre.outcome != PolicyDecisionOutcome.REQUIRE_CONFIRMATION or kwargs.get("confirmation_token"):
            if consume_flag or kwargs.get("record_to_ledger", True):
                return self.evaluate(
                    tool_id,
                    raw_arguments,
                    consume_confirmation=consume_flag,
                    confirm_callback=None,
                    **kwargs,
                )
            return pre

        cb_res = confirm_callback(pre)
        approved = bool(await cb_res if inspect.isawaitable(cb_res) else cb_res)
        if not approved:
            kwargs_no_rec = dict(kwargs)
            central = kwargs_no_rec.get("central_ledger")
            corr = kwargs_no_rec.get("correlation_id")
            denied = ExecutionPolicyDecision(
                decision_id=pre.decision_id,
                outcome=PolicyDecisionOutcome.DENY,
                risk_level=pre.risk_level,
                tool_id=pre.tool_id,
                normalized_arguments=pre.normalized_arguments,
                arg_digest=pre.arg_digest,
                resource_scope=pre.resource_scope,
                policy_version=pre.policy_version,
                rule_ids=list(pre.rule_ids) + ["rule6:callback_denied"],
                reason=f"Policy denied '{pre.tool_id}': interactive confirmation declined.",
                confirmation_required=True,
                confirmation_status="declined",
                execution_id=pre.execution_id,
                node_id=pre.node_id,
                source_surface=pre.source_surface,
                is_external=pre.is_external,
                remediation_hint="User declined confirmation for this action.",
            )
            if kwargs_no_rec.get("record_to_ledger", True) and central is not None and denied.execution_id:
                self._record_decision_to_ledger(central, denied, correlation_id=corr)
            return denied

        raw_tok, rec = self.confirmation_store.issue_confirmation(
            tool_id=pre.tool_id,
            arg_digest=pre.arg_digest,
            resource_scope=pre.resource_scope,
            execution_id=pre.execution_id,
            node_id=pre.node_id,
            ttl_seconds=self.config.confirmation_ttl_seconds,
            max_lifetime_seconds=self.config.max_confirmation_lifetime_seconds,
            policy_version=self.config.policy_version,
            approved_by="confirm_callback",
            now=kwargs.get("now"),
        )
        kwargs["confirmation_token"] = raw_tok
        kwargs["confirmation_id"] = rec.confirmation_id
        return self.evaluate(
            tool_id,
            raw_arguments,
            consume_confirmation=consume_flag,
            confirm_callback=None,
            **kwargs,
        )


_ACTIVE_POLICY_DECISION: contextvars.ContextVar[Optional[ExecutionPolicyDecision]] = contextvars.ContextVar(
    "zelyr_active_policy_decision",
    default=None,
)


def get_active_policy_decision() -> Optional[ExecutionPolicyDecision]:
    """Return the currently active pre-authorized policy decision in this async/thread context, if any."""
    return _ACTIVE_POLICY_DECISION.get()


def matches_active_policy_decision(
    tool_id: str,
    normalized_arguments: Mapping[str, Any],
    *,
    workspace_root: Optional[Union[str, Path]] = None,
    node_target_resources: Sequence[str] = (),
    execution_id: Optional[str] = None,
    node_id: Optional[str] = None,
) -> Optional[ExecutionPolicyDecision]:
    """Return the active ALLOW decision iff canonical tool_id, arg_digest, and execution identity match."""
    active = _ACTIVE_POLICY_DECISION.get()
    if active is None or not active.allowed:
        return None
    if active.tool_id != str(tool_id or "").strip():
        return None
    if active.execution_id is not None and execution_id is not None and active.execution_id != execution_id:
        return None
    if active.node_id is not None and node_id is not None and active.node_id != node_id:
        return None
    ws_path = Path(workspace_root).expanduser().resolve(strict=False) if workspace_root else None
    scope = extract_resource_scope(
        active.tool_id,
        normalized_arguments,
        workspace_root=ws_path,
        node_target_resources=node_target_resources,
    )
    digest = compute_canonical_argument_digest(active.tool_id, normalized_arguments, scope)
    if not hmac.compare_digest(active.arg_digest, digest):
        return None
    return active


@contextmanager
def authorized_policy_context(decision: ExecutionPolicyDecision):
    """Bind an ALLOW ExecutionPolicyDecision across nested dispatch layers without double-consuming tokens."""
    if decision is None or not decision.allowed:
        raise PolicyError("authorized_policy_context requires an ALLOW ExecutionPolicyDecision.")
    token = _ACTIVE_POLICY_DECISION.set(decision)
    try:
        yield decision
    finally:
        _ACTIVE_POLICY_DECISION.reset(token)


def build_execution_policy_engine(
    *,
    policy_engine: Optional[ExecutionPolicyEngine] = None,
    policy_config: Optional[ExecutionPolicyConfig] = None,
    zelyr_config: Optional[Any] = None,
    confirmation_store: Optional[ConfirmationStore] = None,
) -> ExecutionPolicyEngine:
    """Resolve an ExecutionPolicyEngine cleanly from explicit engine/config or active ZelyrConfig."""
    if policy_engine is not None:
        return policy_engine
    eff_cfg = policy_config or ExecutionPolicyConfig.from_zelyr_config(zelyr_config)
    return ExecutionPolicyEngine(config=eff_cfg, confirmation_store=confirmation_store)


def _prepare_direct_policy_context(
    tool_id: str,
    arguments: Optional[Mapping[str, Any]],
    *,
    workspace_root: Optional[Union[str, Path]] = None,
    execution_id: Optional[str] = None,
    node_id: Optional[str] = None,
) -> Tuple[Optional[ExecutionPolicyDecision], Optional[str], Optional[str], Any, Optional[str]]:
    from zelyr.engine.execution_ledger import get_active_execution_context

    act_ctx = get_active_execution_context()
    eff_exec_id = execution_id or (getattr(act_ctx, "execution_id", None) if act_ctx else None)
    eff_node_id = node_id or (getattr(act_ctx, "node_id", None) if act_ctx else None)
    eff_central = getattr(act_ctx, "central_ledger", None) if act_ctx else None
    eff_corr = getattr(act_ctx, "correlation_id", None) if act_ctx else None

    args_map = dict(arguments) if isinstance(arguments, Mapping) else {}
    matched = matches_active_policy_decision(
        tool_id,
        args_map,
        workspace_root=workspace_root,
        execution_id=eff_exec_id,
        node_id=eff_node_id,
    )
    return matched, eff_exec_id, eff_node_id, eff_central, eff_corr


def evaluate_direct_tool_policy(
    tool_id: str,
    arguments: Optional[Mapping[str, Any]] = None,
    *,
    tool_obj: Optional[Any] = None,
    schema: Optional[Mapping[str, Any]] = None,
    required_ring: Optional[int] = None,
    auth_decision: Optional[Any] = None,
    is_registered: bool = True,
    is_external: bool = False,
    external_metadata: Optional[Mapping[str, Any]] = None,
    workspace_root: Optional[Union[str, Path]] = None,
    execution_id: Optional[str] = None,
    node_id: Optional[str] = None,
    correlation_id: Optional[str] = None,
    confirmation_token: Optional[str] = None,
    confirmation_id: Optional[str] = None,
    confirm_callback: Optional[Callable[[ExecutionPolicyDecision], Any]] = None,
    policy_engine: Optional[ExecutionPolicyEngine] = None,
    policy_config: Optional[ExecutionPolicyConfig] = None,
    zelyr_config: Optional[Any] = None,
    source_surface: str = "direct_tool",
) -> ExecutionPolicyDecision:
    """Shared sync policy evaluation for leaf/direct tools respecting active authorized_policy_context."""
    matched, eff_exec_id, eff_node_id, eff_central, eff_corr = _prepare_direct_policy_context(
        tool_id,
        arguments,
        workspace_root=workspace_root,
        execution_id=execution_id,
        node_id=node_id,
    )
    if matched is not None:
        return matched
    engine = build_execution_policy_engine(
        policy_engine=policy_engine,
        policy_config=policy_config,
        zelyr_config=zelyr_config,
    )
    return engine.evaluate(
        tool_id,
        arguments,
        normalized_arguments=arguments if isinstance(arguments, Mapping) else None,
        schema=schema,
        tool_obj=tool_obj,
        required_ring=required_ring,
        auth_decision=auth_decision,
        is_registered=is_registered,
        is_external=is_external,
        external_metadata=external_metadata,
        workspace_root=workspace_root,
        execution_id=eff_exec_id,
        node_id=eff_node_id,
        correlation_id=correlation_id or eff_corr,
        confirmation_token=confirmation_token,
        confirmation_id=confirmation_id,
        consume_confirmation=True,
        confirm_callback=confirm_callback,
        source_surface=source_surface,
        central_ledger=eff_central,
    )


async def evaluate_direct_tool_policy_async(
    tool_id: str,
    arguments: Optional[Mapping[str, Any]] = None,
    *,
    tool_obj: Optional[Any] = None,
    schema: Optional[Mapping[str, Any]] = None,
    required_ring: Optional[int] = None,
    auth_decision: Optional[Any] = None,
    is_registered: bool = True,
    is_external: bool = False,
    external_metadata: Optional[Mapping[str, Any]] = None,
    workspace_root: Optional[Union[str, Path]] = None,
    execution_id: Optional[str] = None,
    node_id: Optional[str] = None,
    correlation_id: Optional[str] = None,
    confirmation_token: Optional[str] = None,
    confirmation_id: Optional[str] = None,
    confirm_callback: Optional[Callable[[ExecutionPolicyDecision], Any]] = None,
    policy_engine: Optional[ExecutionPolicyEngine] = None,
    policy_config: Optional[ExecutionPolicyConfig] = None,
    zelyr_config: Optional[Any] = None,
    source_surface: str = "direct_tool",
) -> ExecutionPolicyDecision:
    """Shared async policy evaluation for leaf/direct tools respecting active authorized_policy_context."""
    matched, eff_exec_id, eff_node_id, eff_central, eff_corr = _prepare_direct_policy_context(
        tool_id,
        arguments,
        workspace_root=workspace_root,
        execution_id=execution_id,
        node_id=node_id,
    )
    if matched is not None:
        return matched
    engine = build_execution_policy_engine(
        policy_engine=policy_engine,
        policy_config=policy_config,
        zelyr_config=zelyr_config,
    )
    return await engine.evaluate_async(
        tool_id,
        arguments,
        normalized_arguments=arguments if isinstance(arguments, Mapping) else None,
        schema=schema,
        tool_obj=tool_obj,
        required_ring=required_ring,
        auth_decision=auth_decision,
        is_registered=is_registered,
        is_external=is_external,
        external_metadata=external_metadata,
        workspace_root=workspace_root,
        execution_id=eff_exec_id,
        node_id=eff_node_id,
        correlation_id=correlation_id or eff_corr,
        confirmation_token=confirmation_token,
        confirmation_id=confirmation_id,
        consume_confirmation=True,
        confirm_callback=confirm_callback,
        source_surface=source_surface,
        central_ledger=eff_central,
    )


__all__ = [
    "POLICY_VERSION",
    "SUPPORTED_POLICY_VERSIONS",
    "SUPPORTED_POLICY_MODES",
    "UNTRUSTED_POLICY_OVERRIDE_KEYS",
    "PolicyError",
    "UnsupportedPolicyVersionError",
    "PolicyConfirmationError",
    "PolicyDecisionOutcome",
    "RiskLevel",
    "ExecutionPolicyConfig",
    "ConfirmationRecord",
    "ConfirmationVerificationResult",
    "ConfirmationStore",
    "ExecutionPolicyDecision",
    "RiskClassifier",
    "ExecutionPolicyEngine",
    "authorized_policy_context",
    "build_execution_policy_engine",
    "compute_canonical_argument_digest",
    "evaluate_direct_tool_policy",
    "evaluate_direct_tool_policy_async",
    "extract_resource_scope",
    "get_active_policy_decision",
    "get_default_confirmation_store",
    "matches_active_policy_decision",
    "sanitize_policy_metadata",
]
