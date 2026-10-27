"""Compatibility helpers for the AXIOM -> Zelyr identity migration.

New installations use ``ZELYR_*`` environment variables and paths under
``zelyr`` / ``.zelyr``.  Existing installations that already have AXIOM
configuration or data continue to use their original locations.  Nothing in
this module copies, moves, or deletes user data.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional


RPC_PREFIX = "zelyr."
LEGACY_RPC_PREFIX = "axiom."


def rpc_method_prefix(method: str) -> str:
    """Return the JSON-RPC dialect used by a client method name.

    Unknown methods default to the canonical Zelyr dialect so callers can
    safely use the result when emitting notifications.
    """
    return LEGACY_RPC_PREFIX if method.startswith(LEGACY_RPC_PREFIX) else RPC_PREFIX


def canonical_rpc_method(method: str) -> str:
    """Normalize a legacy ``axiom.*`` JSON-RPC method to ``zelyr.*``."""
    if method.startswith(LEGACY_RPC_PREFIX):
        return RPC_PREFIX + method[len(LEGACY_RPC_PREFIX):]
    return method


def get_env(name: str, default: Optional[str] = None) -> Optional[str]:
    """Return a Zelyr environment variable with an AXIOM fallback.

    Both ``ZELYR_MEMORY_DB`` and ``AXIOM_MEMORY_DB`` are accepted.  The new
    name wins when both are set.
    """
    if not name.startswith("ZELYR_"):
        name = f"ZELYR_{name}"
    legacy = f"AXIOM_{name[len('ZELYR_'):]}"
    value = os.environ.get(name)
    if value is None:
        value = os.environ.get(legacy)
    return default if value is None else value


def get_env_flag(name: str, default: bool = False) -> bool:
    """Read a boolean Zelyr/AXIOM environment flag."""
    value = get_env(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def sync_legacy_env() -> None:
    """Mirror process-level ``AXIOM_*`` variables to ``ZELYR_*`` names.

    This keeps legacy launch scripts and service units working without
    overwriting an explicitly supplied Zelyr value.  Callers that need
    post-import changes should use :func:`get_env` instead.
    """
    for key, value in list(os.environ.items()):
        if not key.startswith("AXIOM_"):
            continue
        new_key = f"ZELYR_{key[len('AXIOM_'):]}"
        os.environ.setdefault(new_key, value)


def _xdg_config_home() -> Path:
    value = os.environ.get("XDG_CONFIG_HOME")
    return Path(value).expanduser() if value else Path.home() / ".config"


def _xdg_data_home() -> Path:
    value = os.environ.get("XDG_DATA_HOME")
    return Path(value).expanduser() if value else Path.home() / ".local" / "share"


def resolve_file(new_path: Path, legacy_path: Path) -> Path:
    """Prefer the new path, but read an existing legacy path when needed.

    A new file is only selected when it already exists or no legacy file
    exists.  The returned path is therefore safe to open for reading or
    writing without silently creating a second copy of user data.
    """
    if new_path.exists():
        return new_path
    if legacy_path.exists():
        return legacy_path
    return new_path


def config_dir() -> Path:
    """Resolve the configuration directory.

    Precedence: ``ZELYR_CONFIG_DIR`` -> ``AXIOM_CONFIG_DIR`` -> an existing
    legacy ``~/.config/axiom`` -> the new ``~/.config/zelyr``.
    """
    override = get_env("CONFIG_DIR")
    if override:
        return Path(override).expanduser()
    base = _xdg_config_home()
    return resolve_file(base / "zelyr", base / "axiom")


def data_dir() -> Path:
    """Resolve the XDG data directory with legacy AXIOM discovery."""
    base = _xdg_data_home()
    return resolve_file(base / "zelyr", base / "axiom")


def state_dir() -> Path:
    """Resolve the hidden runtime state directory (sockets, tokens, logs)."""
    return resolve_file(Path.home() / ".zelyr", Path.home() / ".axiom")


def _resolve_in_roots(
    new_root: Path,
    legacy_root: Path,
    new_name: str,
    legacy_name: Optional[str] = None,
) -> Path:
    """Resolve a file while keeping a pre-existing legacy root in use.

    File-level reuse takes precedence when either file exists.  If neither
    exists, an existing legacy directory wins so that a first run does not
    silently split user state across two directories.
    """
    new_file = new_root / new_name
    legacy_file = legacy_root / (legacy_name or new_name)
    if new_file.exists():
        return new_file
    if legacy_file.exists():
        return legacy_file
    if new_root.exists():
        return new_file
    if legacy_root.exists():
        return legacy_file
    return new_file


def config_file(name: str = "config.json", legacy_name: Optional[str] = None) -> Path:
    """Resolve a file below the config directory, honouring a legacy filename."""
    base = _xdg_config_home()
    override = get_env("CONFIG_DIR")
    if override:
        root = Path(override).expanduser()
        return resolve_file(root / name, root / (legacy_name or name))
    return _resolve_in_roots(base / "zelyr", base / "axiom", name, legacy_name)


def data_file(name: str, legacy_name: Optional[str] = None) -> Path:
    """Resolve a file below the XDG data directory, honouring a legacy filename."""
    base = _xdg_data_home()
    return _resolve_in_roots(base / "zelyr", base / "axiom", name, legacy_name)


def state_file(name: str, legacy_name: Optional[str] = None) -> Path:
    """Resolve a file below the hidden state directory."""
    return _resolve_in_roots(
        Path.home() / ".zelyr",
        Path.home() / ".axiom",
        name,
        legacy_name,
    )


def repository_memory_db(repo_root: Optional[Path] = None) -> Path:
    """Resolve the repository-root memory database without creating a copy."""
    env_path = get_env("MEMORY_DB")
    if env_path:
        return Path(env_path).expanduser().resolve()
    root = Path(repo_root) if repo_root is not None else Path.cwd()
    return resolve_file(root / "zelyr_memory.db", root / "axiom_memory.db")
