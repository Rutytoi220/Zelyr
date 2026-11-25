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


def _sanitize_scope_item(item: Any) -> str:
    """Scrub URL credentials/tokens and sensitive text from a resource scope string."""
    raw = str(item or "").strip()
    if not raw:
        return ""
    scrubbed = re.sub(r"(?i)(https?://)([^/@\s]+@)", r"\1[REDACTED]@", raw)
    scrubbed = re.sub(
        r"(?i)([?&](?:token|key|api_key|secret|password|auth|sig|access_token)=)([^&\s]+)",
        r"\1[REDACTED]",
        scrubbed,
    )
    return _safe_policy_text(scrubbed)


def _redact_policy_config_value(value: Any, *, key_name: str = "") -> Any:
    """Recursively redact sensitive leaf values inside trusted policy configuration views."""
    if isinstance(value, Mapping):
        return {str(k): _redact_policy_config_value(v, key_name=str(k)) for k, v in value.items()}
    low_k = str(key_name or "").lower()
    if any(
        tok in low_k
        for tok in (
            "secret",
            "token",
            "password",
            "passwd",
            "api_key",
            "apikey",
            "credential",
            "private_key",
            "bearer",
            "authorization",
        )
    ):
        return "[REDACTED]"
    if isinstance(value, (list, tuple, set)):
        return [_redact_policy_config_value(v) for v in value]
    if isinstance(value, str):
        return _sanitize_scope_item(value)
    return value


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
        s = _sanitize_scope_item(item)
        if s:
            scopes.add(s)

    if isinstance(normalized_arguments, Mapping):
        for key in _PATH_ARG_KEYS:
            val = normalized_arguments.get(key)
            if isinstance(val, str) and val.strip():
                canon = _canonicalize_path_str(val, ws_path)
                if canon:
                    scopes.add(_sanitize_scope_item(f"path:{canon}"))
        url_val = normalized_arguments.get("url") or normalized_arguments.get("endpoint")
        if isinstance(url_val, str) and url_val.strip():
            scrubbed_url = _sanitize_scope_item(f"url:{url_val.strip()}")
            if scrubbed_url:
                scopes.add(scrubbed_url)

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
    config_sources: Dict[str, str] = field(default_factory=dict)
    _config_file_path: Optional[str] = field(default=None, repr=False, compare=False)
    _auth_mode_str: Optional[str] = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        raw_sources = dict(self.config_sources or {})
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

        inferred_sources: Dict[str, str] = {
            "policy_mode": "explicit" if self.policy_mode != "conservative" else "default",
            "risk_defaults": "explicit" if self.risk_defaults else "default",
            "tool_overrides": "explicit" if self.tool_overrides else "default",
            "resource_restrictions": "explicit" if self.resource_restrictions else "default",
            "confirmation_ttl_seconds": "explicit" if self.confirmation_ttl_seconds != 300.0 else "default",
            "max_confirmation_lifetime_seconds": "explicit" if self.max_confirmation_lifetime_seconds != 900.0 else "default",
            "governor_strict_mode": "explicit" if self.governor_strict_mode else "default",
        }
        inferred_sources.update({str(k): str(v) for k, v in raw_sources.items()})
        self.config_sources = inferred_sources

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
        explicit_fields = set(getattr(config, "_explicit_fields", set()) or ())
        cfg_file_path = getattr(config, "_config_source_path", None)

        def _src(field_name: str, is_non_default: bool) -> str:
            if field_name in explicit_fields:
                return f"config_file:{cfg_file_path}" if cfg_file_path else "explicit"
            if is_non_default:
                return "explicit"
            return "default"

        raw_mode = getattr(config, "execution_policy_mode", None)
        mode_explicit = raw_mode is not None and bool(str(raw_mode).strip())
        if mode_explicit:
            eff_mode = str(raw_mode).strip().lower()
            if eff_mode not in SUPPORTED_POLICY_MODES:
                eff_mode = "conservative"
            mode_source = f"config_file:{cfg_file_path}" if ("execution_policy_mode" in explicit_fields and cfg_file_path) else "explicit"
        elif auth_str == "strict":
            eff_mode = "strict"
            mode_source = f"derived:auth_mode:{auth_str}"
        elif auth_str == "autopilot":
            eff_mode = "autopilot"
            mode_source = f"derived:auth_mode:{auth_str}"
        else:
            eff_mode = "basic"
            mode_source = f"derived:auth_mode:{auth_str}"

        strict_gov = bool(governor_strict_mode) if governor_strict_mode is not None else (auth_str == "strict")
        gov_source = "explicit" if governor_strict_mode is not None else (f"derived:auth_mode:{auth_str}" if auth_str == "strict" else "default")
        if governor_strict_mode is None:
            try:
                from zelyr.services.governor import GovernorService

                if GovernorService.instance().is_strict_mode():
                    strict_gov = True
                    if auth_str != "strict":
                        gov_source = "derived:governor_strict_mode"
            except Exception:
                pass
        if strict_gov and eff_mode == "basic":
            eff_mode = "strict"
            if not mode_explicit:
                mode_source = "derived:governor_strict_mode"

        risk_defaults = dict(getattr(config, "execution_policy_risk_defaults", {}) or {})
        tool_overrides = dict(getattr(config, "execution_policy_tool_overrides", {}) or {})
        resource_restrictions = dict(getattr(config, "execution_policy_resource_restrictions", {}) or {})
        ttl = getattr(config, "execution_policy_confirmation_ttl_seconds", 300.0)
        max_lt = getattr(config, "execution_policy_max_confirmation_lifetime_seconds", 900.0)

        return cls(
            policy_version=POLICY_VERSION,
            policy_mode=eff_mode,
            risk_defaults=risk_defaults,
            tool_overrides=tool_overrides,
            resource_restrictions=resource_restrictions,
            confirmation_ttl_seconds=ttl,
            max_confirmation_lifetime_seconds=max_lt,
            governor_strict_mode=strict_gov,
            config_sources={
                "policy_mode": mode_source,
                "risk_defaults": _src("execution_policy_risk_defaults", bool(risk_defaults)),
                "tool_overrides": _src("execution_policy_tool_overrides", bool(tool_overrides)),
                "resource_restrictions": _src("execution_policy_resource_restrictions", bool(resource_restrictions)),
                "confirmation_ttl_seconds": _src("execution_policy_confirmation_ttl_seconds", float(ttl) != 300.0),
                "max_confirmation_lifetime_seconds": _src("execution_policy_max_confirmation_lifetime_seconds", float(max_lt) != 900.0),
                "governor_strict_mode": gov_source,
                "auth_mode": _src("auth_mode", auth_str != "basic"),
            },
            _config_file_path=cfg_file_path,
            _auth_mode_str=auth_str,
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


_PRECEDENCE_STAGE_LABELS: Dict[str, str] = {
    "rule1:hard_security": "Rule 1 — Hard Security Boundaries",
    "rule2:runtime_authorization": "Rule 2 — Runtime Authorization & Isolation",
    "rule3:tool_override": "Rule 3 — Trusted Tool-Specific Override",
    "rule4:resource_restriction": "Rule 4 — Resource-Scope & Workspace Containment",
    "rule5:risk_policy": "Rule 5 — Configured Risk Policy Defaults",
    "rule6:confirmation_validation": "Rule 6 — Bound Single-Use Confirmation Validation",
}


@dataclass(frozen=True)
class PolicyExplanation:
    """Structured, human- and machine-readable explanation derived from a real policy evaluation."""

    decision_id: str
    outcome: str
    risk_level: str
    tool_id: str
    reason_code: str
    precedence_stage: str
    precedence_stage_label: str
    determining_rule_id: str
    rule_ids: Tuple[str, ...]
    policy_mode: str
    policy_version: int
    config_source: Dict[str, str]
    resource_scope: Tuple[str, ...]
    confirmation_required: bool
    confirmation_status: str
    is_preview: bool
    summary: str
    remediation_hint: str
    auth_error_type: Optional[str] = None
    execution_id: Optional[str] = None
    node_id: Optional[str] = None
    source_surface: str = "unknown"
    is_external: bool = False

    def to_safe_dict(self) -> Dict[str, Any]:
        """Serialize sanitized explanation without exposing raw arguments or confirmation tokens."""
        return {
            "decision_id": self.decision_id,
            "outcome": self.outcome,
            "risk_level": self.risk_level,
            "tool_id": self.tool_id,
            "reason_code": self.reason_code,
            "precedence_stage": self.precedence_stage,
            "precedence_stage_label": self.precedence_stage_label,
            "determining_rule_id": self.determining_rule_id,
            "rule_ids": list(self.rule_ids),
            "policy_mode": self.policy_mode,
            "policy_version": self.policy_version,
            "config_source": dict(self.config_source),
            "resource_scope": [_sanitize_scope_item(s) for s in self.resource_scope if _sanitize_scope_item(s)],
            "confirmation_required": self.confirmation_required,
            "confirmation_status": self.confirmation_status,
            "is_preview": self.is_preview,
            "summary": _safe_policy_text(self.summary),
            "remediation_hint": _safe_policy_text(self.remediation_hint),
            "auth_error_type": self.auth_error_type,
            "execution_id": self.execution_id,
            "node_id": self.node_id,
            "source_surface": self.source_surface,
            "is_external": self.is_external,
        }


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
    reason_code: str = ""
    precedence_stage: str = ""
    determining_rule_id: str = ""
    policy_mode: str = ""
    config_source: Dict[str, str] = field(default_factory=dict)
    is_preview: bool = False
    auth_error_type: Optional[str] = None
    confirmation_wait_ms: Optional[float] = None

    @property
    def allowed(self) -> bool:
        return self.outcome == PolicyDecisionOutcome.ALLOW

    def explain(self) -> PolicyExplanation:
        """Build a structured PolicyExplanation directly from recorded evaluation metadata."""
        eff_stage = self.precedence_stage or "unrecorded_stage"
        eff_stage_label = _PRECEDENCE_STAGE_LABELS.get(eff_stage, eff_stage)
        eff_rule = self.determining_rule_id or (self.rule_ids[-1] if self.rule_ids else "unspecified")
        eff_reason_code = self.reason_code or eff_rule
        eff_mode = self.policy_mode or "unspecified"
        sanitized_scope = tuple(s for s in (_sanitize_scope_item(x) for x in self.resource_scope) if s)
        preview_tag = "[PREVIEW] " if self.is_preview else ""
        base_reason = _safe_policy_text(self.reason) or f"Outcome {self.outcome.value} for '{self.tool_id}'."
        trigger_rules = [
            r for r in self.rule_ids if r.startswith(("rule3:tool_override", "rule5:risk_policy", "rule4:", "rule1:", "rule2:"))
        ]
        chain_note = f" Evaluated chain: {', '.join(trigger_rules)}." if trigger_rules else ""
        summary = (
            f"{preview_tag}{self.outcome.value.upper()} '{self.tool_id}' "
            f"(risk={self.risk_level.value}, mode={eff_mode}, stage={eff_stage}, "
            f"rule={eff_rule}, reason_code={eff_reason_code}): {base_reason}{chain_note}"
        )
        return PolicyExplanation(
            decision_id=self.decision_id,
            outcome=self.outcome.value,
            risk_level=self.risk_level.value,
            tool_id=self.tool_id,
            reason_code=eff_reason_code,
            precedence_stage=eff_stage,
            precedence_stage_label=eff_stage_label,
            determining_rule_id=eff_rule,
            rule_ids=tuple(self.rule_ids),
            policy_mode=eff_mode,
            policy_version=self.policy_version,
            config_source=dict(self.config_source),
            resource_scope=sanitized_scope,
            confirmation_required=self.confirmation_required,
            confirmation_status=self.confirmation_status,
            is_preview=bool(self.is_preview),
            summary=_safe_policy_text(summary),
            remediation_hint=_safe_policy_text(self.remediation_hint),
            auth_error_type=self.auth_error_type,
            execution_id=self.execution_id,
            node_id=self.node_id,
            source_surface=self.source_surface,
            is_external=self.is_external,
        )

    def to_explanation_dict(self) -> Dict[str, Any]:
        """Return a sanitized dictionary representation of `explain()`."""
        return self.explain().to_safe_dict()

    def to_safe_dict(self) -> Dict[str, Any]:
        """Serialize sanitized policy decision safe for ledgers, error responses, and model feedback."""
        sanitized_scope = [_sanitize_scope_item(s) for s in self.resource_scope if _sanitize_scope_item(s)]
        payload: Dict[str, Any] = {
            "decision_id": self.decision_id,
            "outcome": self.outcome.value,
            "risk_level": self.risk_level.value,
            "tool_id": self.tool_id,
            "arg_digest": self.arg_digest,
            "resource_scope": sanitized_scope,
            "policy_version": self.policy_version,
            "rule_ids": list(self.rule_ids),
            "reason": _safe_policy_text(self.reason),
            "reason_code": self.reason_code,
            "precedence_stage": self.precedence_stage,
            "determining_rule_id": self.determining_rule_id,
            "policy_mode": self.policy_mode,
            "config_source": dict(self.config_source),
            "is_preview": bool(self.is_preview),
            "confirmation_required": self.confirmation_required,
            "confirmation_status": self.confirmation_status,
            "confirmation_id": self.confirmation_id,
            "execution_id": self.execution_id,
            "node_id": self.node_id,
            "source_surface": self.source_surface,
            "is_external": self.is_external,
            "remediation_hint": _safe_policy_text(self.remediation_hint),
        }
        if self.auth_error_type is not None:
            payload["auth_error_type"] = self.auth_error_type
        if self.confirmation_wait_ms is not None:
            payload["confirmation_wait_ms"] = self.confirmation_wait_ms
        return payload

    def to_tool_error_payload(self) -> Dict[str, Any]:
        """Build deterministic error dict suitable for tool runners and orchestrator messages."""
        status = "confirmation_required" if self.outcome == PolicyDecisionOutcome.REQUIRE_CONFIRMATION else "denied"
        err_type = (
            "POLICY_CONFIRMATION_REQUIRED"
            if self.outcome == PolicyDecisionOutcome.REQUIRE_CONFIRMATION
            else "POLICY_DENIED"
        )
        sanitized_scope = [_sanitize_scope_item(s) for s in self.resource_scope if _sanitize_scope_item(s)]
        return {
            "success": False,
            "status": status,
            "error_type": err_type,
            "reason": "confirmation_required" if self.outcome == PolicyDecisionOutcome.REQUIRE_CONFIRMATION else "policy_denied",
            "reason_code": self.reason_code,
            "precedence_stage": self.precedence_stage,
            "determining_rule_id": self.determining_rule_id,
            "policy_mode": self.policy_mode,
            "is_preview": bool(self.is_preview),
            "error": _safe_policy_text(self.reason),
            "tool": self.tool_id,
            "risk_level": self.risk_level.value,
            "decision_id": self.decision_id,
            "arg_digest": self.arg_digest,
            "resource_scope": sanitized_scope,
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
        preview: bool = False,
        _confirmation_wait_ms: Optional[float] = None,
    ) -> ExecutionPolicyDecision:
        """Evaluate execution policy deterministically across Rules 1..6 and optionally log to ledger."""
        if preview:
            consume_confirmation = False
            confirm_callback = None
            record_to_ledger = False
            central_ledger = None

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
            reason_code: str,
            precedence_stage: str,
            determining_rule_id: str,
            conf_req: bool = False,
            conf_status: str = "not_required",
            conf_id: Optional[str] = None,
            remediation: str = "",
            auth_error_type: Optional[str] = None,
            confirmation_wait_ms: Optional[float] = None,
        ) -> ExecutionPolicyDecision:
            sanitized_scope = [_sanitize_scope_item(s) for s in resource_scope if _sanitize_scope_item(s)]
            eff_wait_ms = confirmation_wait_ms if confirmation_wait_ms is not None else _confirmation_wait_ms
            dec = ExecutionPolicyDecision(
                decision_id=decision_id,
                outcome=outcome,
                risk_level=risk_level,
                tool_id=tid,
                normalized_arguments=eff_args,
                arg_digest=arg_digest,
                resource_scope=sanitized_scope,
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
                reason_code=reason_code,
                precedence_stage=precedence_stage,
                determining_rule_id=determining_rule_id,
                policy_mode=self.config.policy_mode,
                config_source=dict(self.config.config_sources),
                is_preview=bool(preview),
                auth_error_type=auth_error_type,
                confirmation_wait_ms=eff_wait_ms,
            )
            if (not preview) and record_to_ledger and central_ledger is not None and execution_id:
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
                reason_code="invalid_argument_container",
                precedence_stage="rule1:hard_security",
                determining_rule_id="rule1:invalid_argument_container",
                remediation="Provide structured dictionary arguments.",
            )

        if smuggled_keys:
            rule_ids.append("rule1:untrusted_policy_or_auth_keys")
            return _finalize(
                PolicyDecisionOutcome.DENY,
                f"Policy denied '{tid}': untrusted authorization/policy keys {sorted(set(smuggled_keys))} rejected.",
                reason_code="untrusted_policy_or_auth_keys",
                precedence_stage="rule1:hard_security",
                determining_rule_id="rule1:untrusted_policy_or_auth_keys",
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
                reason_code="restricted_credential_or_security_command",
                precedence_stage="rule1:hard_security",
                determining_rule_id="rule1:restricted_credential_or_security_command",
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
                    reason_code="restricted_credential_or_security_path",
                    precedence_stage="rule1:hard_security",
                    determining_rule_id="rule1:restricted_credential_or_security_path",
                    remediation="Target workspace files rather than system security or credential stores.",
                )
            if base_name in _SENSITIVE_FILENAMES and (
                is_mutating_action or base_name in {"shadow", "gshadow", "sudoers", "id_rsa", "id_ed25519"}
            ):
                rule_ids.append("rule1:sensitive_file_target")
                return _finalize(
                    PolicyDecisionOutcome.DENY,
                    f"Policy denied '{tid}': sensitive target '{base_name}' is protected by hard security policy.",
                    reason_code="sensitive_file_target",
                    precedence_stage="rule1:hard_security",
                    determining_rule_id="rule1:sensitive_file_target",
                    remediation="Do not access or modify secret/credential files.",
                )
            if is_mutating_action and any(low_canon.startswith(prefix) or low_canon == prefix.rstrip("/") for prefix in _MUTATING_SYSTEM_FILES):
                rule_ids.append("rule1:protected_system_file_mutation")
                return _finalize(
                    PolicyDecisionOutcome.DENY,
                    f"Policy denied '{tid}': mutating system path '{canon_p}' is forbidden.",
                    reason_code="protected_system_file_mutation",
                    precedence_stage="rule1:hard_security",
                    determining_rule_id="rule1:protected_system_file_mutation",
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
                auth_rule_id = f"rule2:auth_denied:{err_type}"
                rule_ids.append(auth_rule_id)
                return _finalize(
                    PolicyDecisionOutcome.DENY,
                    err_msg,
                    reason_code=f"auth_denied:{err_type}",
                    precedence_stage="rule2:runtime_authorization",
                    determining_rule_id=auth_rule_id,
                    auth_error_type=err_type,
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
                        reason_code="denied_path_prefix",
                        precedence_stage="rule4:resource_restriction",
                        determining_rule_id="rule4:denied_path_prefix",
                        remediation="Select an allowed path outside restricted directories.",
                    )
            if restricted_ws_path is not None and not _is_path_within(canon_p, restricted_ws_path):
                outside_workspace = True
                if enforce_ws and is_mutating_action:
                    rule_ids.append("rule4:outside_workspace_mutation_denied")
                    return _finalize(
                        PolicyDecisionOutcome.DENY,
                        f"Policy denied '{tid}': mutating path '{canon_p}' escapes configured workspace '{restricted_ws_path}'.",
                        reason_code="outside_workspace_mutation_denied",
                        precedence_stage="rule4:resource_restriction",
                        determining_rule_id="rule4:outside_workspace_mutation_denied",
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
                        reason_code="denied_host",
                        precedence_stage="rule4:resource_restriction",
                        determining_rule_id="rule4:denied_host",
                        remediation="Target an approved host/endpoint.",
                    )

        # Evaluate override outcomes AFTER hard security & resource restrictions (Rules 1-4 stay supreme!).
        base_outcome: Optional[PolicyDecisionOutcome] = None
        baseline_stage = "rule5:risk_policy"
        baseline_rule_id = ""
        baseline_reason_code = ""
        if override:
            if "outcome" in override:
                base_outcome = _normalize_outcome(override["outcome"])
                ov_rule_id = f"rule3:tool_override_outcome:{base_outcome.value}"
                rule_ids.append(ov_rule_id)
                baseline_stage = "rule3:tool_override"
                baseline_rule_id = ov_rule_id
                if base_outcome == PolicyDecisionOutcome.DENY:
                    return _finalize(
                        PolicyDecisionOutcome.DENY,
                        f"Policy denied '{tid}' by explicit trusted tool policy override.",
                        reason_code="tool_override_deny",
                        precedence_stage="rule3:tool_override",
                        determining_rule_id=ov_rule_id,
                        remediation="Tool is explicitly disallowed by policy configuration.",
                    )
                baseline_reason_code = "tool_override_allow" if base_outcome == PolicyDecisionOutcome.ALLOW else "tool_override_require_confirmation"
            elif override.get("deny") is True:
                rule_ids.append("rule3:tool_override_deny")
                return _finalize(
                    PolicyDecisionOutcome.DENY,
                    f"Policy denied '{tid}' by explicit trusted tool denial rule.",
                    reason_code="tool_override_deny",
                    precedence_stage="rule3:tool_override",
                    determining_rule_id="rule3:tool_override_deny",
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
            r5_rule_id = f"rule5:risk_policy:{self.config.policy_mode}:{risk_level.value}->{base_outcome.value}"
            rule_ids.append(r5_rule_id)
            baseline_stage = "rule5:risk_policy"
            baseline_rule_id = r5_rule_id
            if risk_level == RiskLevel.UNKNOWN:
                baseline_reason_code = "unknown_tool_blocked"
            elif risk_level == RiskLevel.IRREVERSIBLE:
                baseline_reason_code = "irreversible_action_blocked"
            elif risk_level.value in self.config.risk_defaults:
                baseline_reason_code = f"risk_default_{base_outcome.value}:{risk_level.value}"
            else:
                baseline_reason_code = f"risk_policy_{base_outcome.value}:{risk_level.value}"

        # Unknown and Irreversible default to DENY under Rule 5 and NEVER accept casual overrides unless trusted policy explicitly permitted confirmation.
        if base_outcome == PolicyDecisionOutcome.DENY:
            return _finalize(
                PolicyDecisionOutcome.DENY,
                (
                    f"Policy denied '{tid}' (risk={risk_level.value}) under policy mode '{self.config.policy_mode}'."
                ),
                reason_code=baseline_reason_code,
                precedence_stage=baseline_stage,
                determining_rule_id=baseline_rule_id,
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
                reason_code=baseline_reason_code,
                precedence_stage=baseline_stage,
                determining_rule_id=baseline_rule_id,
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
            conf_rule_id = f"rule6:confirmation_{ver.status}"
            if ver.valid:
                rule_ids.append(conf_rule_id)
                return _finalize(
                    PolicyDecisionOutcome.ALLOW,
                    f"Policy allowed '{tid}' with bound single-use confirmation ({ver.confirmation_id}).",
                    reason_code=f"confirmation_{ver.status}",
                    precedence_stage="rule6:confirmation_validation",
                    determining_rule_id=conf_rule_id,
                    conf_req=True,
                    conf_status=ver.status,
                    conf_id=ver.confirmation_id,
                )
            rule_ids.append(conf_rule_id)
            # Expired, replayed, revoked, or mismatched confirmations fail closed cleanly.
            fail_outcome = (
                PolicyDecisionOutcome.DENY
                if ver.status in ("replayed", "expired", "revoked", "version_mismatch", "invalid", "mismatch")
                else PolicyDecisionOutcome.REQUIRE_CONFIRMATION
            )
            return _finalize(
                fail_outcome,
                f"Policy rejected '{tid}': {ver.reason}",
                reason_code=f"confirmation_{ver.status}",
                precedence_stage="rule6:confirmation_validation",
                determining_rule_id=conf_rule_id,
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
                resource_scope=sanitized_scope if (sanitized_scope := [_sanitize_scope_item(s) for s in resource_scope if _sanitize_scope_item(s)]) else [],
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
                reason_code=baseline_reason_code or "confirmation_required",
                precedence_stage=baseline_stage,
                determining_rule_id=baseline_rule_id or "rule6:confirmation_pending",
                policy_mode=self.config.policy_mode,
                config_source=dict(self.config.config_sources),
                is_preview=bool(preview),
            )
            cb_t0 = time.monotonic()
            try:
                approved = bool(confirm_callback(pre_decision))
            finally:
                cb_wait_ms = round((time.monotonic() - cb_t0) * 1000.0, 3)
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
                        reason_code="callback_confirmed",
                        precedence_stage="rule6:confirmation_validation",
                        determining_rule_id="rule6:callback_confirmed",
                        conf_req=True,
                        conf_status=ver.status,
                        conf_id=rec.confirmation_id,
                        confirmation_wait_ms=cb_wait_ms,
                    )
            rule_ids.append("rule6:callback_denied")
            return _finalize(
                PolicyDecisionOutcome.DENY,
                f"Policy denied '{tid}': interactive confirmation declined.",
                reason_code="callback_denied",
                precedence_stage="rule6:confirmation_validation",
                determining_rule_id="rule6:callback_denied",
                conf_req=True,
                conf_status="declined",
                remediation="User declined confirmation for this action.",
                confirmation_wait_ms=cb_wait_ms,
            )

        rule_ids.append("rule6:confirmation_missing")
        return _finalize(
            PolicyDecisionOutcome.REQUIRE_CONFIRMATION,
            (
                f"Explicit confirmation required before executing '{tid}' "
                f"(risk={risk_level.value}, mode='{self.config.policy_mode}')."
            ),
            reason_code="confirmation_missing",
            precedence_stage="rule6:confirmation_validation",
            determining_rule_id="rule6:confirmation_missing",
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
        if decision.is_preview or not decision.execution_id:
            return
        try:
            status_map = {
                PolicyDecisionOutcome.ALLOW: "allowed",
                PolicyDecisionOutcome.REQUIRE_CONFIRMATION: "requires_confirmation",
                PolicyDecisionOutcome.DENY: "denied",
            }
            sanitized_scope = [_sanitize_scope_item(s) for s in decision.resource_scope if _sanitize_scope_item(s)]
            meta_payload = sanitize_policy_metadata(
                {
                    "policy_decision": decision.outcome.value,
                    "decision_id": decision.decision_id,
                    "node_id": decision.node_id,
                    "risk_level": decision.risk_level.value,
                    "arg_digest": decision.arg_digest,
                    "resource_scope": sanitized_scope,
                    "rule_ids": list(decision.rule_ids),
                    "reason": _safe_policy_text(decision.reason),
                    "reason_code": decision.reason_code,
                    "precedence_stage": decision.precedence_stage,
                    "determining_rule_id": decision.determining_rule_id,
                    "policy_mode": decision.policy_mode,
                    "config_source": dict(decision.config_source),
                    "remediation_hint": _safe_policy_text(decision.remediation_hint),
                    "auth_error_type": decision.auth_error_type,
                    "confirmation_required": decision.confirmation_required,
                    "confirmation_status": decision.confirmation_status,
                    "confirmation_id": decision.confirmation_id,
                    "policy_version": decision.policy_version,
                    "source_surface": decision.source_surface,
                    "is_external": decision.is_external,
                    **(
                        {"confirmation_wait_ms": decision.confirmation_wait_ms}
                        if decision.confirmation_wait_ms is not None
                        else {}
                    ),
                }
            )
            central_ledger.record_event(
                decision.execution_id,
                LedgerEventType.AUTHORIZATION_DECISION,
                tool_id=decision.tool_id,
                source_id=decision.source_id,
                source_type=decision.source_type,
                is_external=decision.is_external,
                correlation_id=correlation_id or decision.decision_id,
                status=status_map.get(decision.outcome, "denied"),
                metadata=meta_payload,
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

        if kwargs.get("preview"):
            kwargs["consume_confirmation"] = False
            kwargs["record_to_ledger"] = False
            kwargs["central_ledger"] = None
            confirm_callback = None

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

        cb_t0 = time.monotonic()
        try:
            cb_res = confirm_callback(pre)
            approved = bool(await cb_res if inspect.isawaitable(cb_res) else cb_res)
        finally:
            cb_wait_ms = round((time.monotonic() - cb_t0) * 1000.0, 3)
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
                reason_code="callback_denied",
                precedence_stage="rule6:confirmation_validation",
                determining_rule_id="rule6:callback_denied",
                policy_mode=self.config.policy_mode,
                config_source=dict(self.config.config_sources),
                is_preview=False,
                confirmation_wait_ms=cb_wait_ms,
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
            _confirmation_wait_ms=cb_wait_ms,
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
    eff_central = (
        (getattr(act_ctx, "central_ledger", None) or getattr(act_ctx, "ledger", None))
        if act_ctx
        else None
    )
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


@dataclass(frozen=True)
class PolicyPreviewResult:
    """Result of a strictly non-executing, non-mutating execution-policy preview."""

    decision: ExecutionPolicyDecision
    explanation: PolicyExplanation
    fact_status: str  # "verified" | "unknown" | "unverified"
    is_registered: bool
    is_external: bool
    external_available: Optional[bool]
    auth_allowed: bool
    auth_error_type: Optional[str] = None

    def to_safe_dict(self) -> Dict[str, Any]:
        """Serialize safe preview result without exposing raw arguments or token secrets."""
        return {
            "is_preview": True,
            "fact_status": self.fact_status,
            "is_registered": self.is_registered,
            "is_external": self.is_external,
            "external_available": self.external_available,
            "auth_allowed": self.auth_allowed,
            "auth_error_type": self.auth_error_type,
            "decision": self.decision.to_safe_dict(),
            "explanation": self.explanation.to_safe_dict(),
        }


def _extract_schema_for_preview(
    canonical_name: str,
    *,
    tool_obj: Optional[Any],
    registry: Optional[Any],
    active_scope: Optional[Any],
    active_schemas: Optional[List[Dict[str, Any]]],
) -> Optional[Mapping[str, Any]]:
    if active_scope is not None and hasattr(active_scope, "find_schema"):
        try:
            s = active_scope.find_schema(canonical_name, active_schemas=active_schemas)
            if isinstance(s, Mapping):
                return s
        except Exception:
            pass
    if active_schemas:
        for s in active_schemas:
            if not isinstance(s, Mapping):
                continue
            fn = s.get("function") if isinstance(s.get("function"), Mapping) else {}
            s_name = fn.get("name") or s.get("name")
            if s_name == canonical_name:
                return s
    if tool_obj is not None and hasattr(tool_obj, "schema") and isinstance(getattr(tool_obj, "schema"), Mapping):
        return {
            "type": "function",
            "function": {
                "name": canonical_name,
                "parameters": dict(getattr(tool_obj, "schema")),
            },
        }
    if registry is not None:
        try:
            from zelyr.tool_registry import _find_schema_for_tool

            s = _find_schema_for_tool(
                canonical_name,
                tool=tool_obj,
                tool_registry=registry,
                active_schemas=active_schemas,
            )
            if isinstance(s, Mapping):
                return s
        except Exception:
            pass
    return None


def preview_tool_policy(
    tool_id: str,
    arguments: Optional[Union[Mapping[str, Any], str]] = None,
    *,
    tool_registry: Optional[Any] = None,
    task_local_scope: Optional[Any] = None,
    caller_ring: Optional[int] = None,
    allowed_tools: Optional[Iterable[str]] = None,
    active_schemas: Optional[List[Dict[str, Any]]] = None,
    workspace_root: Optional[Union[str, Path]] = None,
    node_target_resources: Sequence[str] = (),
    policy_engine: Optional[ExecutionPolicyEngine] = None,
    policy_config: Optional[ExecutionPolicyConfig] = None,
    zelyr_config: Optional[Any] = None,
    source_surface: str = "policy_preview",
) -> PolicyPreviewResult:
    """Preview policy evaluation safely without executing tools, minting tokens, or writing ledger state."""
    from zelyr.engine.normalizer import _strict_json_loads_object
    from zelyr.tool_registry import (
        BUILTIN_TOOLS,
        SWARM_TOOLS,
        ToolRegistry,
        authorize_and_validate_tool_call,
        canonicalize_tool_invocation,
        get_active_task_local_scope,
        resolve_tool_ring,
    )

    eff_input_args: Any = {} if arguments is None else arguments
    raw_for_eval: Any = eff_input_args
    if isinstance(eff_input_args, str):
        parsed_obj, _ = _strict_json_loads_object(eff_input_args)
        if parsed_obj is not None:
            raw_for_eval = parsed_obj

    auth_decision = authorize_and_validate_tool_call(
        tool_id,
        eff_input_args,
        caller_ring=caller_ring,
        allowed_tools=allowed_tools,
        active_schemas=active_schemas,
        tool_registry=tool_registry,
        task_local_scope=task_local_scope,
        require_registered=True,
        normalize_args=True,
        validate_schema=True,
        argument_source="model",
    )

    canonical_name = str(getattr(auth_decision, "tool_name", None) or "").strip()
    if not canonical_name:
        canonical_name, _ = canonicalize_tool_invocation(str(tool_id or ""), {})
    canonical_name = canonical_name or str(tool_id or "").strip()

    active_scope = task_local_scope if task_local_scope is not None else get_active_task_local_scope()
    eff_reg = tool_registry
    if eff_reg is None and active_scope is None:
        try:
            eff_reg = ToolRegistry()
        except Exception:
            eff_reg = None

    tool_obj: Optional[Any] = None
    ext_entry: Optional[Any] = None
    is_mock_reg = eff_reg is not None and getattr(type(eff_reg), "__module__", "").startswith("unittest.mock")
    if eff_reg is not None:
        if hasattr(eff_reg, "get_external_tool_entry") and not is_mock_reg:
            try:
                ext_entry = eff_reg.get_external_tool_entry(canonical_name)
            except Exception:
                ext_entry = None
        if hasattr(eff_reg, "get_tool"):
            try:
                tool_obj = eff_reg.get_tool(canonical_name)
            except Exception:
                tool_obj = None

    eff_schema = _extract_schema_for_preview(
        canonical_name,
        tool_obj=tool_obj,
        registry=eff_reg,
        active_scope=active_scope,
        active_schemas=active_schemas,
    )

    resolved_ring: Optional[int] = getattr(auth_decision, "required_ring", None)
    if resolved_ring is None:
        if active_scope is not None and hasattr(active_scope, "resolve_ring"):
            try:
                resolved_ring = active_scope.resolve_ring(canonical_name, active_schemas=active_schemas)
            except Exception:
                resolved_ring = None
        elif eff_reg is not None or tool_obj is not None:
            try:
                resolved_ring = resolve_tool_ring(
                    canonical_name,
                    tool=tool_obj,
                    tool_registry=eff_reg,
                    active_schemas=active_schemas,
                )
            except Exception:
                resolved_ring = None

    auth_err_type = getattr(auth_decision, "error_type", None)
    if active_scope is not None:
        is_registered = bool(canonical_name and canonical_name in active_scope.get_known_tools())
    elif auth_err_type == "TOOL_NOT_FOUND":
        is_registered = False
    else:
        is_registered = bool(
            tool_obj is not None
            or ext_entry is not None
            or canonical_name in BUILTIN_TOOLS
            or canonical_name in SWARM_TOOLS
            or eff_schema is not None
            or auth_decision.allowed
        )

    is_external = bool(
        (tool_obj is not None and getattr(tool_obj, "is_external", False))
        or ext_entry is not None
        or canonical_name.startswith("ext__")
    )
    source_id = (
        getattr(ext_entry, "source_id", None)
        if ext_entry is not None
        else getattr(tool_obj, "source_id", None)
    )
    source_type = (
        getattr(ext_entry, "source_type", None)
        if ext_entry is not None
        else getattr(tool_obj, "source_type", None)
    )

    ext_meta: Optional[Dict[str, Any]] = None
    if is_external:
        ext_meta = {}
        if ext_entry is not None:
            ext_meta["authorized"] = bool(getattr(ext_entry, "authorized", False))
            ext_meta["lifecycle"] = str(
                getattr(getattr(ext_entry, "lifecycle", None), "value", getattr(ext_entry, "lifecycle", ""))
            )
            for attr in ("destructive_hint", "destructiveHint", "irreversible", "annotations"):
                if hasattr(ext_entry, attr):
                    ext_meta[attr] = getattr(ext_entry, attr)
        if tool_obj is not None:
            for attr in ("destructiveHint", "destructive_hint", "irreversible", "annotations"):
                if hasattr(tool_obj, attr):
                    ext_meta[attr] = getattr(tool_obj, attr)

    external_available: Optional[bool] = None
    if is_external:
        if auth_err_type == "TOOL_UNAVAILABLE":
            external_available = False
        elif ext_entry is not None:
            ext_lc = getattr(getattr(ext_entry, "lifecycle", None), "value", str(getattr(ext_entry, "lifecycle", "")))
            obj_ok = bool(tool_obj.is_available()) if (tool_obj is not None and hasattr(tool_obj, "is_available")) else True
            external_available = (ext_lc == "AVAILABLE") and obj_ok
        elif tool_obj is not None and hasattr(tool_obj, "is_available"):
            try:
                external_available = bool(tool_obj.is_available())
            except Exception:
                external_available = False

    if (not is_registered) or auth_err_type == "TOOL_NOT_FOUND":
        fact_status = "unknown"
    elif auth_err_type in ("TOOL_UNAVAILABLE", "SCHEMA_VALIDATION_ERROR") or (
        is_external and external_available is False
    ):
        fact_status = "unverified"
    elif not isinstance(raw_for_eval, Mapping):
        fact_status = "unverified"
    else:
        fact_status = "verified"

    engine = build_execution_policy_engine(
        policy_engine=policy_engine,
        policy_config=policy_config,
        zelyr_config=zelyr_config,
    )
    decision = engine.evaluate(
        canonical_name,
        raw_for_eval,
        normalized_arguments=(
            auth_decision.normalized_arguments
            if (auth_decision.allowed and isinstance(auth_decision.normalized_arguments, Mapping))
            else None
        ),
        schema=eff_schema,
        tool_obj=tool_obj,
        required_ring=resolved_ring,
        auth_decision=auth_decision,
        is_registered=is_registered,
        is_external=is_external,
        source_id=source_id,
        source_type=source_type,
        external_metadata=ext_meta,
        workspace_root=workspace_root,
        node_target_resources=node_target_resources,
        execution_id=None,
        node_id=None,
        correlation_id=None,
        confirmation_token=None,
        confirmation_id=None,
        consume_confirmation=False,
        confirm_callback=None,
        source_surface=source_surface,
        central_ledger=None,
        record_to_ledger=False,
        preview=True,
    )
    explanation = decision.explain()
    return PolicyPreviewResult(
        decision=decision,
        explanation=explanation,
        fact_status=fact_status,
        is_registered=is_registered,
        is_external=is_external,
        external_available=external_available,
        auth_allowed=bool(auth_decision.allowed),
        auth_error_type=auth_err_type,
    )


def inspect_policy_configuration(
    *,
    zelyr_config: Optional[Any] = None,
    policy_config: Optional[ExecutionPolicyConfig] = None,
    governor_strict_mode: Optional[bool] = None,
) -> Dict[str, Any]:
    """Return a sanitized inspection view of effective policy configuration and field provenance."""
    cfg = policy_config or ExecutionPolicyConfig.from_zelyr_config(
        zelyr_config,
        governor_strict_mode=governor_strict_mode,
    )
    risk_matrix: Dict[str, Dict[str, str]] = {}
    for lvl in RiskLevel:
        risk_matrix[lvl.value] = {
            "workspace_scoped": cfg.resolve_default_outcome_for_risk(lvl, workspace_scoped=True).value,
            "outside_workspace": cfg.resolve_default_outcome_for_risk(lvl, workspace_scoped=False).value,
        }

    return {
        "policy_version": cfg.policy_version,
        "policy_mode": cfg.policy_mode,
        "auth_mode": cfg._auth_mode_str,
        "governor_strict_mode": cfg.governor_strict_mode,
        "confirmation_ttl_seconds": cfg.confirmation_ttl_seconds,
        "max_confirmation_lifetime_seconds": cfg.max_confirmation_lifetime_seconds,
        "risk_defaults": dict(cfg.risk_defaults),
        "effective_risk_matrix": risk_matrix,
        "tool_overrides": _redact_policy_config_value(cfg.tool_overrides),
        "resource_restrictions": _redact_policy_config_value(cfg.resource_restrictions),
        "config_sources": dict(cfg.config_sources),
        "config_file_path": cfg._config_file_path,
    }


def format_policy_configuration(view: Mapping[str, Any]) -> str:
    """Render an effective policy configuration view cleanly for terminal/REPL display."""
    sources = view.get("config_sources") or {}
    lines: List[str] = [
        f"Execution Policy Configuration (v{view.get('policy_version', POLICY_VERSION)}):",
        f"  policy_mode:           {view.get('policy_mode')}  (source: {sources.get('policy_mode', 'unknown')})",
        f"  auth_mode:             {view.get('auth_mode') or 'n/a'}  (source: {sources.get('auth_mode', 'n/a')})",
        f"  governor_strict_mode:  {view.get('governor_strict_mode')}  (source: {sources.get('governor_strict_mode', 'unknown')})",
        f"  confirmation_ttl_s:    {view.get('confirmation_ttl_seconds')}  (source: {sources.get('confirmation_ttl_seconds', 'unknown')})",
        f"  max_confirm_life_s:    {view.get('max_confirmation_lifetime_seconds')}  (source: {sources.get('max_confirmation_lifetime_seconds', 'unknown')})",
    ]
    if view.get("config_file_path"):
        lines.append(f"  config_file:           {view.get('config_file_path')}")
    lines.append("  effective_risk_matrix:")
    for rk, outcomes in (view.get("effective_risk_matrix") or {}).items():
        lines.append(
            f"    {rk:<18} in_workspace={outcomes.get('workspace_scoped')}  outside_workspace={outcomes.get('outside_workspace')}"
        )
    overrides = view.get("tool_overrides") or {}
    lines.append(
        f"  tool_overrides:        {len(overrides)} rule(s) (source: {sources.get('tool_overrides', 'unknown')})"
    )
    for tid, spec in sorted(overrides.items()):
        lines.append(f"    {tid}: {json.dumps(spec, sort_keys=True)}")
    restrictions = view.get("resource_restrictions") or {}
    lines.append(
        f"  resource_restrictions: (source: {sources.get('resource_restrictions', 'unknown')}) "
        f"{json.dumps(restrictions, sort_keys=True) if restrictions else '{}'}"
    )
    return "\n".join(lines)


def format_policy_explanation(target: Union[PolicyPreviewResult, PolicyExplanation, ExecutionPolicyDecision]) -> str:
    """Format a PolicyPreviewResult, ExecutionPolicyDecision, or PolicyExplanation for CLI/REPL."""
    preview_res: Optional[PolicyPreviewResult] = target if isinstance(target, PolicyPreviewResult) else None
    if preview_res is not None:
        exp = preview_res.explanation
    elif isinstance(target, ExecutionPolicyDecision):
        exp = target.explain()
    else:
        exp = target

    header = "Execution Policy Preview (non-executing)" if exp.is_preview else "Execution Policy Explanation"
    lines: List[str] = [
        f"{header}: {exp.outcome.upper()} for '{exp.tool_id}'",
        f"  decision_id:         {exp.decision_id}",
        f"  risk_level:          {exp.risk_level}",
        f"  precedence_stage:    {exp.precedence_stage_label} [{exp.precedence_stage}]",
        f"  determining_rule_id: {exp.determining_rule_id}",
        f"  reason_code:         {exp.reason_code}",
        f"  policy_mode:         {exp.policy_mode} (v{exp.policy_version})",
        f"  confirmation:        required={exp.confirmation_required} status={exp.confirmation_status}",
    ]
    if preview_res is not None:
        lines.append(
            f"  fact_status:         {preview_res.fact_status} "
            f"(registered={preview_res.is_registered}, external={preview_res.is_external}, "
            f"external_available={preview_res.external_available}, auth_allowed={preview_res.auth_allowed})"
        )
    if exp.auth_error_type:
        lines.append(f"  auth_error_type:     {exp.auth_error_type}")
    lines.append(f"  resource_scope:      {list(exp.resource_scope)}")
    lines.append(f"  rule_ids:            {list(exp.rule_ids)}")
    lines.append(f"  summary:             {_safe_policy_text(exp.summary)}")
    if exp.remediation_hint:
        lines.append(f"  remediation_hint:    {_safe_policy_text(exp.remediation_hint)}")
    return "\n".join(lines)


def format_authorization_history_entry(event: Any) -> Dict[str, Any]:
    """Convert an ExecutionEvent (AUTHORIZATION_DECISION) into a safe redacted history entry."""
    raw_meta = getattr(event, "metadata", None)
    meta = dict(raw_meta) if isinstance(raw_meta, Mapping) else {}
    malformed = bool(meta.get("_malformed_metadata", False)) or not isinstance(raw_meta, Mapping)

    # Strip any legacy/unexpected argument or token fields before sanitizing.
    for forbidden_key in (
        "normalized_arguments",
        "raw_arguments",
        "arguments",
        "args",
        "confirmation_token",
        "confirm_token",
        "raw_token",
        "token_secret",
    ):
        meta.pop(forbidden_key, None)

    safe_meta = sanitize_policy_metadata(meta) if isinstance(meta, Mapping) else {}
    if not isinstance(safe_meta, dict):
        safe_meta = {}
        malformed = True

    status_str = str(getattr(event, "status", "recorded") or "recorded")
    raw_outcome = safe_meta.get("policy_decision")
    if isinstance(raw_outcome, str) and raw_outcome.strip():
        outcome = raw_outcome.strip().lower()
    else:
        status_to_outcome = {
            "allowed": "allow",
            "allow": "allow",
            "requires_confirmation": "require_confirmation",
            "require_confirmation": "require_confirmation",
            "confirmation_required": "require_confirmation",
            "denied": "deny",
            "deny": "deny",
            "rejected": "deny",
        }
        outcome = status_to_outcome.get(status_str.strip().lower(), status_str.strip().lower())

    raw_scope = safe_meta.get("resource_scope")
    if isinstance(raw_scope, (list, tuple)):
        clean_scope = [_sanitize_scope_item(x) for x in raw_scope if _sanitize_scope_item(x)]
    elif isinstance(raw_scope, str) and raw_scope.strip():
        sc = _sanitize_scope_item(raw_scope)
        clean_scope = [sc] if sc else []
    else:
        clean_scope = []

    raw_rules = safe_meta.get("rule_ids")
    clean_rules = [str(r) for r in raw_rules] if isinstance(raw_rules, (list, tuple)) else []

    reason_code = safe_meta.get("reason_code")
    precedence_stage = safe_meta.get("precedence_stage")
    legacy_metadata = (not malformed) and (not reason_code or not precedence_stage)

    return {
        "event_id": str(getattr(event, "event_id", "")),
        "execution_id": str(getattr(event, "execution_id", "")),
        "sequence_number": int(getattr(event, "sequence_number", 0) or 0),
        "timestamp": float(getattr(event, "timestamp", 0.0) or 0.0),
        "tool_id": getattr(event, "tool_id", None) or "unknown",
        "outcome": outcome,
        "status": status_str,
        "decision_id": safe_meta.get("decision_id"),
        "risk_level": safe_meta.get("risk_level") or "unknown",
        "reason_code": reason_code or (clean_rules[-1] if clean_rules else ("malformed_metadata" if malformed else "legacy_decision")),
        "precedence_stage": precedence_stage or ("malformed_metadata" if malformed else "legacy_unrecorded"),
        "determining_rule_id": safe_meta.get("determining_rule_id") or (clean_rules[-1] if clean_rules else None),
        "policy_mode": safe_meta.get("policy_mode"),
        "reason": _safe_policy_text(safe_meta.get("reason") or ""),
        "remediation_hint": _safe_policy_text(safe_meta.get("remediation_hint") or ""),
        "resource_scope": clean_scope,
        "rule_ids": clean_rules,
        "confirmation_required": bool(safe_meta.get("confirmation_required", False)),
        "confirmation_status": safe_meta.get("confirmation_status") or "unknown",
        "confirmation_id": safe_meta.get("confirmation_id"),
        "source_surface": safe_meta.get("source_surface") or "unknown",
        "is_external": bool(getattr(event, "is_external", safe_meta.get("is_external", False))),
        "node_id": safe_meta.get("node_id"),
        "policy_version": safe_meta.get("policy_version"),
        "legacy_metadata": legacy_metadata,
        "malformed_metadata": malformed,
    }


def query_policy_history(
    central_ledger: Any,
    *,
    limit: int = 20,
    outcome: Optional[str] = None,
    tool_id: Optional[str] = None,
    execution_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Query persistent AUTHORIZATION_DECISION ledger events safely and redact all output fields."""
    if central_ledger is None:
        return {
            "count": 0,
            "db_path": None,
            "persistence_errors": [{"operation": "query_policy_history", "error": "No central_ledger available."}],
            "entries": [],
        }
    events = central_ledger.query_authorization_decisions(
        limit=limit,
        outcome=outcome,
        tool_id=tool_id,
        execution_id=execution_id,
    )
    persistence_errors = (
        list(central_ledger.get_persistence_errors())
        if hasattr(central_ledger, "get_persistence_errors")
        else []
    )
    entries = [format_authorization_history_entry(ev) for ev in events]
    return {
        "count": len(entries),
        "db_path": str(getattr(central_ledger, "db_path", "")),
        "persistence_errors": persistence_errors,
        "entries": entries,
    }


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
    "PolicyExplanation",
    "ExecutionPolicyDecision",
    "PolicyPreviewResult",
    "RiskClassifier",
    "ExecutionPolicyEngine",
    "authorized_policy_context",
    "build_execution_policy_engine",
    "compute_canonical_argument_digest",
    "evaluate_direct_tool_policy",
    "evaluate_direct_tool_policy_async",
    "extract_resource_scope",
    "format_authorization_history_entry",
    "format_policy_configuration",
    "format_policy_explanation",
    "get_active_policy_decision",
    "get_default_confirmation_store",
    "inspect_policy_configuration",
    "matches_active_policy_decision",
    "preview_tool_policy",
    "query_policy_history",
    "sanitize_policy_metadata",
]
