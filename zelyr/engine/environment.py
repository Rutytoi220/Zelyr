"""Runtime Environment Introspection and Truthful Capability Manifest for Zelyr.

Provides deterministic host facts, inference locality classification, desktop
session detection, tool registration/availability/authorization inspection, and
compact Markdown system prompt manifest compilation.
"""

from __future__ import annotations

import logging
import os
import platform
import shutil
import socket
import urllib.parse
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_SHELL_TOOL_IDS = frozenset({"execute_command", "shell", "bash", "sh"})
_FILESYSTEM_TOOL_IDS = frozenset(
    {
        "manage_workspace_file",
        "file_read",
        "file_write",
        "file_search",
        "file_opener",
        "read_file",
        "write_file",
    }
)
_DESKTOP_TOOL_IDS = frozenset({"manage_desktop_window", "hyprland_ipc"})
_BROWSER_TOOL_IDS = frozenset({"interact_with_browser", "browser_cdp"})
_VISION_UI_TOOL_IDS = frozenset({"interact_with_ui"})
_MEMORY_TOOL_IDS = frozenset({"manage_memory", "memory_store", "search_memories"})


class RuntimeEnvironmentManifest:
    """Truthful runtime environment introspection and system prompt compiler."""

    @classmethod
    def _extract_endpoint_url(cls, config: Any = None) -> str:
        """Extract configured endpoint URL without falling back to guessed defaults."""
        url = ""
        if isinstance(config, str):
            return config.strip()
        if isinstance(config, dict):
            raw = (
                config.get("ollama_base_url")
                or config.get("base_url")
                or config.get("endpoint")
                or config.get("api_base")
                or ""
            )
            url = str(raw).strip() if raw else ""
        elif config is not None:
            raw = (
                getattr(config, "ollama_base_url", None)
                or getattr(config, "base_url", None)
                or getattr(config, "endpoint", None)
                or getattr(config, "api_base", None)
                or ""
            )
            url = str(raw).strip() if raw else ""

        if not url and not isinstance(config, str):
            try:
                from zelyr.config import get_config

                cfg = get_config()
                raw_cfg = getattr(cfg, "ollama_base_url", None)
                url = str(raw_cfg).strip() if raw_cfg else ""
            except Exception:
                url = ""

        return url

    @classmethod
    def detect_inference_locality(cls, config: Any = None) -> str:
        """Inspect provider endpoint URL and classify locality.

        Returns:
            - 'local_daemon' for localhost, loopback, or unix/ipc sockets.
            - 'remote_provider' for non-local host endpoints.
            - 'unknown' when no valid endpoint can be established or parsed.
        """
        url = cls._extract_endpoint_url(config)
        if not url:
            return "unknown"

        # Check local sockets and schemes
        if url.startswith("/") or url.startswith("unix://") or "ipc://" in url:
            return "local_daemon"

        if "://" not in url:
            url = f"http://{url}"

        try:
            parsed = urllib.parse.urlparse(url)
            host = (parsed.hostname or "").lower()
        except Exception:
            return "unknown"

        if not host:
            return "unknown"
        if host in ("localhost", "127.0.0.1", "::1", "0.0.0.0", "local"):
            return "local_daemon"
        return "remote_provider"

    @classmethod
    def locality_category(cls, locality: str) -> str:
        """Map detailed inference locality string to 'local', 'remote', or 'unknown'."""
        if locality == "local_daemon":
            return "local"
        if locality == "remote_provider":
            return "remote"
        return "unknown"

    @classmethod
    def detect_desktop_state(cls) -> str:
        """Determine desktop session type and Wayland/Hyprland IPC availability.

        Returns:
            - 'hyprland_wayland' if direct Hyprland IPC socket exists.
            - 'generic_wayland' if WAYLAND_DISPLAY is set but no Hyprland socket.
            - 'headless_or_unsupported' otherwise.
        """
        try:
            from zelyr.engine.hyprland_ipc import HyprlandIPCClient

            client = HyprlandIPCClient()
            if client.is_available():
                return "hyprland_wayland"
        except Exception:
            pass

        if os.environ.get("WAYLAND_DISPLAY"):
            return "generic_wayland"

        return "headless_or_unsupported"

    @classmethod
    def _is_browser_active(cls) -> bool:
        """Check if browser extension bridge or Chrome DevTools Protocol port 9222 responds."""
        try:
            from zelyr.tools.browser_extension import BrowserExtensionBridge

            if (
                BrowserExtensionBridge._instance is not None
                and BrowserExtensionBridge._instance.is_connected()
            ):
                return True
        except Exception:
            pass

        try:
            with socket.create_connection(("127.0.0.1", 9222), timeout=0.05):
                return True
        except (OSError, socket.timeout):
            pass

        return False

    @classmethod
    def _is_shell_runtime_available(cls) -> bool:
        """Substantiate local shell runtime availability without guessing."""
        if os.name == "nt":
            return bool(shutil.which("cmd") or os.environ.get("COMSPEC"))
        return bool(shutil.which("sh") or Path("/bin/sh").exists())

    @classmethod
    def _is_filesystem_runtime_available(cls) -> bool:
        """Substantiate current workspace filesystem accessibility."""
        try:
            cwd = Path.cwd()
            return cwd.exists() and cwd.is_dir()
        except Exception:
            return False

    @classmethod
    def _is_memory_runtime_available(cls) -> bool:
        """Substantiate whether local memory store module is importable."""
        try:
            import importlib

            importlib.import_module("zelyr.db.memory")
            return True
        except Exception:
            return False

    @classmethod
    def get_tool_capability_summary(
        cls,
        tool_registry: Optional[Any] = None,
        caller_ring: Optional[int] = None,
        allowed_tools: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Distinguish registered, available, and authorized tools from a registry.

        - A tool being registered does not imply it is enabled/available.
        - A tool being available does not imply it is authorized for the caller context.
        - If no registry is provided, registered/available/authorized sets are reported as
          unverified ('unknown' / None) rather than fabricated.
        """
        if tool_registry is None:
            return {
                "registry_known": False,
                "authorization_scoped": False,
                "registered": None,
                "available": None,
                "authorized": None,
            }

        raw_registered: Dict[str, Any] = {}
        core_reg = getattr(tool_registry, "_core_registry", None)
        if core_reg is not None and callable(getattr(core_reg, "list_tools", None)):
            candidate = core_reg.list_tools()
            if isinstance(candidate, dict):
                raw_registered = dict(candidate)

        enabled_tools: Dict[str, Any] = {}
        if callable(getattr(tool_registry, "list_tools", None)):
            candidate = tool_registry.list_tools()
            if isinstance(candidate, dict):
                enabled_tools = dict(candidate)

        if not raw_registered and enabled_tools:
            raw_registered = dict(enabled_tools)

        registered_names = sorted(raw_registered.keys())

        desktop_state = cls.detect_desktop_state()
        hyprland_ok = desktop_state == "hyprland_wayland"
        browser_ok = cls._is_browser_active()
        shell_ok = cls._is_shell_runtime_available()
        fs_ok = cls._is_filesystem_runtime_available()
        mem_ok = cls._is_memory_runtime_available()

        available_map: Dict[str, Any] = {}
        for tid, tool_obj in sorted(enabled_tools.items()):
            if tid in _DESKTOP_TOOL_IDS and not hyprland_ok:
                continue
            if tid in _BROWSER_TOOL_IDS and not browser_ok:
                continue
            if tid in _VISION_UI_TOOL_IDS and desktop_state == "headless_or_unsupported":
                continue
            if tid in _SHELL_TOOL_IDS and not shell_ok:
                continue
            if tid in _FILESYSTEM_TOOL_IDS and not fs_ok:
                continue
            if tid in _MEMORY_TOOL_IDS and not mem_ok:
                continue
            available_map[tid] = tool_obj

        available_names = sorted(available_map.keys())

        allowed_set: Optional[set[str]] = None
        if allowed_tools is not None:
            allowed_set = set(allowed_tools)

        authorized_names: List[str] = []
        get_ring = getattr(tool_registry, "_get_tool_ring", None)
        for tid in available_names:
            if allowed_set is not None and tid not in allowed_set:
                continue
            if caller_ring is not None and callable(get_ring):
                try:
                    req_ring = get_ring(tid, available_map[tid])
                    if isinstance(req_ring, int) and caller_ring > req_ring:
                        continue
                except Exception:
                    continue
            authorized_names.append(tid)

        return {
            "registry_known": True,
            "authorization_scoped": (caller_ring is not None or allowed_set is not None),
            "registered": registered_names,
            "available": available_names,
            "authorized": authorized_names,
        }

    @classmethod
    def get_active_actuators(
        cls,
        tool_registry: Optional[Any] = None,
        caller_ring: Optional[int] = None,
        allowed_tools: Optional[Any] = None,
    ) -> Dict[str, bool]:
        """Report substantiated runtime availability of core actuator groups."""
        desktop_active = cls.detect_desktop_state() == "hyprland_wayland"
        browser_active = cls._is_browser_active()
        shell_active = cls._is_shell_runtime_available()
        fs_active = cls._is_filesystem_runtime_available()
        mem_active = cls._is_memory_runtime_available()

        if tool_registry is not None:
            summary = cls.get_tool_capability_summary(
                tool_registry=tool_registry,
                caller_ring=caller_ring,
                allowed_tools=allowed_tools,
            )
            if summary.get("registry_known") and summary.get("registered"):
                auth = set(summary.get("authorized") or [])
                return {
                    "shell": shell_active and bool(auth & _SHELL_TOOL_IDS),
                    "filesystem": fs_active and bool(auth & _FILESYSTEM_TOOL_IDS),
                    "desktop_window": desktop_active and bool(auth & _DESKTOP_TOOL_IDS),
                    "browser": browser_active and bool(auth & _BROWSER_TOOL_IDS),
                    "memory": mem_active and bool(auth & _MEMORY_TOOL_IDS),
                }

        return {
            "shell": shell_active,
            "filesystem": fs_active,
            "desktop_window": desktop_active,
            "browser": browser_active,
            "memory": mem_active,
        }

    @classmethod
    def collect_manifest(
        cls,
        config: Optional[Any] = None,
        tool_registry: Optional[Any] = None,
        caller_ring: Optional[int] = None,
        allowed_tools: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Collect structured, secret-free runtime environment facts."""
        locality_detail = cls.detect_inference_locality(config)
        locality = cls.locality_category(locality_detail)

        model_name = ""
        if isinstance(config, dict):
            model_name = str(config.get("model") or config.get("ollama_model") or "").strip()
        elif config is not None and not isinstance(config, str):
            raw_m = getattr(config, "model", None) or getattr(config, "ollama_model", None)
            model_name = str(raw_m).strip() if raw_m else ""
        else:
            try:
                from zelyr.config import get_config

                cfg = get_config()
                raw_m = getattr(cfg, "ollama_model", None)
                model_name = str(raw_m).strip() if raw_m else ""
            except Exception:
                model_name = ""

        if not model_name:
            model_name = "unknown"

        sys_name = platform.system() or "unknown"
        arch = platform.machine() or "unknown"
        try:
            cwd = str(Path.cwd())
        except Exception:
            cwd = "unknown"

        desktop_state = cls.detect_desktop_state()
        tools_summary = cls.get_tool_capability_summary(
            tool_registry=tool_registry,
            caller_ring=caller_ring,
            allowed_tools=allowed_tools,
        )
        actuators = cls.get_active_actuators(
            tool_registry=tool_registry,
            caller_ring=caller_ring,
            allowed_tools=allowed_tools,
        )

        return {
            "inference_locality": locality_detail,
            "locality": locality,
            "tool_execution_locality": "local",
            "model": model_name,
            "platform": {
                "system": sys_name,
                "machine": arch,
                "cwd": cwd,
            },
            "desktop_state": desktop_state,
            "actuators": actuators,
            "tools": tools_summary,
        }

    @classmethod
    def build_manifest_block(
        cls,
        config: Optional[Any] = None,
        tool_registry: Optional[Any] = None,
        caller_ring: Optional[int] = None,
        allowed_tools: Optional[Any] = None,
    ) -> str:
        """Format a compact, truthful Markdown block bounded strictly under 120 tokens."""
        manifest = cls.collect_manifest(
            config=config,
            tool_registry=tool_registry,
            caller_ring=caller_ring,
            allowed_tools=allowed_tools,
        )
        locality_detail = manifest["inference_locality"]
        model_name = manifest["model"]
        sys_name = manifest["platform"]["system"]
        arch = manifest["platform"]["machine"]
        cwd = manifest["platform"]["cwd"]
        desktop_state = manifest["desktop_state"]
        actuators = manifest["actuators"]

        active = [k for k, v in actuators.items() if v]
        unavailable = [k for k, v in actuators.items() if not v]

        active_str = ", ".join(active) if active else "none"
        unavailable_str = ", ".join(unavailable) if unavailable else "none"

        return (
            "[RUNTIME ENVIRONMENT & CAPABILITY MANIFEST]\n"
            f"- Inference: {locality_detail} ({model_name}) | Tool Exec: {manifest['tool_execution_locality']}\n"
            f"- Target Host: {sys_name} ({arch}) | Working Directory: {cwd}\n"
            f"- Desktop Session: {desktop_state}\n"
            f"- Active Actuators: {active_str}\n"
            f"- Degraded/Unavailable: {unavailable_str}\n"
            "- Constraints: Non-interactive execution, DEVNULL stdin, no interactive sudo."
        )
