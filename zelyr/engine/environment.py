"""Runtime Environment Introspection and Truthful Capability Manifest for Zelyr.

Provides deterministic host facts, inference locality classification, desktop
session detection, tool registration/availability/authorization inspection,
integration health probing with bounded timeouts, secret redaction, caching,
and compact Markdown system prompt manifest compilation.
"""

from __future__ import annotations

from enum import Enum
import importlib
import importlib.util
import json
import logging
import os
from pathlib import Path
import platform
import re
import shutil
import socket
import threading
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple
import urllib.error
import urllib.parse
import urllib.request

logger = logging.getLogger(__name__)


class CapabilityStatus(str, Enum):
    """Canonical availability status for runtime capabilities, tools, and integrations."""

    AVAILABLE = "AVAILABLE"
    UNAVAILABLE = "UNAVAILABLE"
    DEGRADED = "DEGRADED"
    UNKNOWN = "UNKNOWN"


_SHELL_TOOL_IDS = frozenset({"execute_command", "shell", "bash", "sh", "shell_command", "run_command"})
_FILESYSTEM_TOOL_IDS = frozenset(
    {
        "manage_workspace_file",
        "file_read",
        "file_write",
        "file_search",
        "file_opener",
        "read_file",
        "write_file",
        "file",
    }
)
_DESKTOP_TOOL_IDS = frozenset({"manage_desktop_window", "hyprland_ipc"})
_BROWSER_TOOL_IDS = frozenset({"interact_with_browser", "browser_cdp"})
_VISION_UI_TOOL_IDS = frozenset(
    {"interact_with_ui", "vision", "capture_som_screen", "click_tag", "som_click", "som_type", "som_key"}
)
_MEMORY_TOOL_IDS = frozenset(
    {"manage_memory", "memory_store", "search_memories", "remember_fact", "search_memory"}
)
_MEDIA_TOOL_IDS = frozenset({"manage_media_playback"})
_NOTIFICATION_TOOL_IDS = frozenset({"send_desktop_notification"})
_CLIPBOARD_TOOL_IDS = frozenset(
    {"clipboard", "manage_clipboard", "clipboard_read", "clipboard_write"}
)

_SENSITIVE_ENV_KEY_TOKENS = (
    "KEY",
    "TOKEN",
    "SECRET",
    "PASSWORD",
    "PASSWD",
    "CREDENTIAL",
    "AUTH",
    "PRIVATE",
)
_INLINE_SECRET_RE = re.compile(
    r"(?i)(?:"
    r"\bsk-[A-Za-z0-9_-]{8,}\b"
    r"|\bgh[pousr]_[A-Za-z0-9]{8,}\b"
    r"|\bgithub_pat_[A-Za-z0-9_]{8,}\b"
    r"|\bBearer\s+[A-Za-z0-9._~+/=-]{6,}"
    r"|(?:api[_-]?key|secret|token|password|passwd)\s*[:=]\s*[^\s\"'`,;)]+"
    r")"
)


def _is_module_importable(module_name: str) -> bool:
    """Return True if a Python module spec exists without importing heavy weights."""
    if module_name in os.sys.modules:
        return os.sys.modules[module_name] is not None
    try:
        from zelyr.utils.optional_deps import is_extra_available

        return bool(is_extra_available(module_name))
    except Exception:
        try:
            return importlib.util.find_spec(module_name) is not None
        except Exception:
            return False


def _collect_sensitive_env_values() -> List[str]:
    """Collect active secret values from os.environ so they can be scrubbed if leaked."""
    values: List[str] = []
    for k, v in os.environ.items():
        if not v or not isinstance(v, str):
            continue
        cleaned = v.strip()
        if len(cleaned) < 6:
            continue
        upper_k = k.upper()
        if any(tok in upper_k for tok in _SENSITIVE_ENV_KEY_TOKENS):
            values.append(cleaned)
    values.sort(key=len, reverse=True)
    return values


def _scrub_secret_text(text: Any, sensitive_values: Optional[List[str]] = None) -> str:
    """Scrub API keys, tokens, credentials, and sensitive env values from a string."""
    raw = str(text or "")
    if not raw:
        return ""
    try:
        from zelyr.tools import _redact_sensitive_text

        raw = _redact_sensitive_text(raw)
    except Exception:
        pass

    raw = _INLINE_SECRET_RE.sub("[REDACTED]", raw)
    secrets_list = (
        sensitive_values
        if sensitive_values is not None
        else _collect_sensitive_env_values()
    )
    for secret_val in secrets_list:
        if secret_val and secret_val in raw:
            raw = raw.replace(secret_val, "[REDACTED]")
    return raw


def _scrub_structure(obj: Any, sensitive_values: Optional[List[str]] = None) -> Any:
    """Recursively scrub secrets from a manifest dictionary or collection."""
    secrets_list = (
        sensitive_values
        if sensitive_values is not None
        else _collect_sensitive_env_values()
    )
    if isinstance(obj, str):
        return _scrub_secret_text(obj, secrets_list)
    if isinstance(obj, dict):
        return {
            _scrub_secret_text(k, secrets_list) if isinstance(k, str) else k: _scrub_structure(
                v, secrets_list
            )
            for k, v in obj.items()
        }
    if isinstance(obj, list):
        return [_scrub_structure(item, secrets_list) for item in obj]
    if isinstance(obj, tuple):
        return tuple(_scrub_structure(item, secrets_list) for item in obj)
    return obj


class RuntimeEnvironmentManifest:
    """Truthful runtime environment introspection and system prompt compiler."""

    _cache_lock = threading.Lock()
    _manifest_cache: Dict[Tuple[Any, ...], Tuple[float, Dict[str, Any]]] = {}
    _probe_cache: Dict[Tuple[str, str], Tuple[float, Dict[str, Any]]] = {}
    DEFAULT_CACHE_TTL: float = 5.0
    DEFAULT_PROBE_TIMEOUT: float = 0.5

    @classmethod
    def clear_cache(cls) -> None:
        """Clear all cached manifest snapshots and endpoint probe results."""
        with cls._cache_lock:
            cls._manifest_cache.clear()
            cls._probe_cache.clear()

    @classmethod
    def refresh(
        cls,
        config: Optional[Any] = None,
        tool_registry: Optional[Any] = None,
        caller_ring: Optional[int] = None,
        allowed_tools: Optional[Any] = None,
        *,
        probe_services: bool = False,
        probe_timeout: float = DEFAULT_PROBE_TIMEOUT,
    ) -> Dict[str, Any]:
        """Force-refresh and return a freshly collected runtime environment manifest."""
        cls.clear_cache()
        return cls.collect_manifest(
            config=config,
            tool_registry=tool_registry,
            caller_ring=caller_ring,
            allowed_tools=allowed_tools,
            probe_services=probe_services,
            probe_timeout=probe_timeout,
            refresh=True,
        )

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
    def _sanitize_endpoint_url(cls, url: str) -> str:
        """Strip userinfo credentials, query tokens, and fragments from an endpoint URL."""
        raw = str(url or "").strip()
        if not raw:
            return ""
        if raw.startswith("/") or raw.startswith("unix://") or "ipc://" in raw:
            return _scrub_secret_text(raw.split("?", 1)[0].split("#", 1)[0])

        has_scheme = "://" in raw
        parse_target = raw if has_scheme else f"http://{raw}"
        try:
            parsed = urllib.parse.urlparse(parse_target)
            host = parsed.hostname or ""
            if not host:
                return ""
            if ":" in host and not host.startswith("["):
                host = f"[{host}]"
            port_part = f":{parsed.port}" if parsed.port else ""
            netloc = f"{host}{port_part}"
            path = parsed.path.rstrip("/")
            if has_scheme:
                rebuilt = f"{parsed.scheme}://{netloc}{path}"
            else:
                rebuilt = f"{netloc}{path}"
            return _scrub_secret_text(rebuilt)
        except Exception:
            return ""

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
            from zelyr.tools.browser_native_host import NativeBrowserClient

            if NativeBrowserClient().is_socket_present():
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
            importlib.import_module("zelyr.db.memory")
            return True
        except Exception:
            return False

    @classmethod
    def _inspect_vision_runtime(cls, desktop_state: Optional[str] = None) -> Dict[str, Any]:
        """Inspect Vision / Spotless-Arms VLA / VLM grounding capabilities without loading weights."""
        eff_desktop = desktop_state or cls.detect_desktop_state()
        has_wayland = eff_desktop in ("hyprland_wayland", "generic_wayland") or bool(
            os.environ.get("WAYLAND_DISPLAY")
        )
        has_x11 = bool(os.environ.get("DISPLAY"))
        has_display = has_wayland or has_x11

        spotless_enabled = True
        spotless_available = False
        spotless_checkpoint_present = False
        spotless_error: Optional[str] = None

        try:
            from zelyr.core.vision_engine import singleton as vision_singleton

            spotless_enabled = vision_singleton._is_spotless_arms_enabled_by_env()
            checkpoint_dir = getattr(
                vision_singleton, "_SPOTLESS_CHECKPOINT_DIR", None
            )
            if isinstance(checkpoint_dir, Path):
                try:
                    spotless_checkpoint_present = checkpoint_dir.is_dir()
                except Exception:
                    spotless_checkpoint_present = False
            spotless_available = bool(vision_singleton.is_spotless_arms_available())
        except Exception as exc:
            spotless_available = False
            spotless_error = _scrub_secret_text(f"{type(exc).__name__}: {exc}")

        torch_available = _is_module_importable("torch")
        transformers_available = _is_module_importable("transformers")
        vlm_available = bool(torch_available and transformers_available)

        has_grim = shutil.which("grim") is not None
        has_scrot = shutil.which("scrot") is not None
        capture_tool = (
            "grim"
            if (has_wayland and has_grim)
            else ("scrot" if (has_x11 and has_scrot) else None)
        )

        limitations: List[str] = []
        if not has_display:
            limitations.append("No active Wayland or X11 display session detected.")
        if not spotless_enabled:
            limitations.append("Spotless-Arms VLA is disabled via ZELYR_USE_SPOTLESS_ARMS.")
        elif spotless_error:
            limitations.append(f"Spotless-Arms import failed ({spotless_error}).")
        elif not spotless_available:
            limitations.append("Spotless-Arms VLA package is not installed.")
        elif not spotless_checkpoint_present:
            limitations.append("Spotless-Arms checkpoint directory is missing.")

        if not vlm_available:
            missing_ml = [
                pkg
                for pkg, ok in (("torch", torch_available), ("transformers", transformers_available))
                if not ok
            ]
            limitations.append(
                f"Local VLM grounding dependencies missing ({', '.join(missing_ml)})."
            )
        if has_wayland and not has_grim:
            limitations.append("Wayland screenshot utility 'grim' is not installed in PATH.")

        if not has_display:
            status = CapabilityStatus.UNAVAILABLE.value
            backend = "none"
            details = "Headless session; visual grounding and UI interaction are unavailable."
        elif spotless_available and spotless_checkpoint_present:
            status = (
                CapabilityStatus.AVAILABLE.value
                if (capture_tool or not has_wayland or has_grim)
                else CapabilityStatus.DEGRADED.value
            )
            backend = "spotless_arms_vla"
            details = "Spotless-Arms Tier 3 VLA motor cortex available."
        elif spotless_available and not spotless_checkpoint_present:
            status = CapabilityStatus.DEGRADED.value
            backend = "spotless_arms_vla" if not vlm_available else "zelyr_grounding_vlm"
            details = "Spotless-Arms package present but checkpoint directory is absent."
        elif vlm_available:
            status = (
                CapabilityStatus.DEGRADED.value
                if (not spotless_available or (has_wayland and not has_grim))
                else CapabilityStatus.AVAILABLE.value
            )
            backend = "zelyr_grounding_vlm"
            details = "Fallback VLM grounding engine available (Spotless-Arms VLA absent)."
        else:
            status = CapabilityStatus.UNAVAILABLE.value
            backend = "none"
            details = "Neither Spotless-Arms VLA nor local VLM (torch/transformers) is available."

        return {
            "status": status,
            "verified": True,
            "source": "detected",
            "backend": backend,
            "spotless_arms_enabled": spotless_enabled,
            "spotless_arms_available": spotless_available,
            "spotless_checkpoint_present": spotless_checkpoint_present,
            "vlm_fallback_available": vlm_available,
            "display_available": has_display,
            "capture_utility": capture_tool,
            "details": details,
            "limitations": limitations,
        }

    @classmethod
    def collect_system_info(cls, desktop_state: Optional[str] = None) -> Dict[str, Any]:
        """Collect structured host OS, architecture, Python, CPU, memory, and display environment facts."""
        sys_name = platform.system() or "unknown"
        sys_release = platform.release() or "unknown"
        sys_version = platform.version() or "unknown"
        if sys_name == "Linux":
            try:
                os_rel = Path("/etc/os-release")
                if os_rel.is_file():
                    for line in os_rel.read_text(encoding="utf-8", errors="ignore").splitlines():
                        if line.startswith("PRETTY_NAME="):
                            sys_version = line.split("=", 1)[1].strip().strip('"')
                            break
            except Exception:
                pass

        arch = platform.machine() or "unknown"
        py_ver = platform.python_version()
        try:
            cwd = str(Path.cwd())
        except Exception:
            cwd = "unknown"

        # CPU introspection
        logical_cores = os.cpu_count()
        physical_cores: Optional[int] = None
        load_1m: Optional[float] = None
        if hasattr(os, "getloadavg"):
            try:
                load_1m = round(float(os.getloadavg()[0]), 2)
            except Exception:
                load_1m = None

        # Memory introspection (psutil with /proc/meminfo fallback)
        total_mb: Optional[int] = None
        available_mb: Optional[int] = None
        percent_used: Optional[float] = None
        mem_status = CapabilityStatus.UNKNOWN.value
        mem_source = "unknown"

        try:
            import psutil

            try:
                physical_cores = psutil.cpu_count(logical=False)
            except Exception:
                physical_cores = None
            vm = psutil.virtual_memory()
            total_mb = int(vm.total // (1024 * 1024))
            available_mb = int(vm.available // (1024 * 1024))
            percent_used = round(float(vm.percent), 1)
            mem_status = CapabilityStatus.AVAILABLE.value
            mem_source = "detected"
        except Exception:
            try:
                meminfo = Path("/proc/meminfo")
                if meminfo.is_file():
                    kv: Dict[str, int] = {}
                    for line in meminfo.read_text(encoding="utf-8", errors="ignore").splitlines():
                        parts = line.split(":")
                        if len(parts) == 2:
                            key = parts[0].strip()
                            val_parts = parts[1].strip().split()
                            if val_parts and val_parts[0].isdigit():
                                kv[key] = int(val_parts[0])
                    if "MemTotal" in kv:
                        total_mb = kv["MemTotal"] // 1024
                        avail_kb = kv.get("MemAvailable", kv.get("MemFree", 0))
                        available_mb = avail_kb // 1024
                        if total_mb > 0:
                            percent_used = round(
                                ((total_mb - available_mb) / total_mb) * 100.0, 1
                            )
                        mem_status = CapabilityStatus.AVAILABLE.value
                        mem_source = "detected"
            except Exception:
                pass

        eff_desktop = desktop_state or cls.detect_desktop_state()
        wayland_env = bool(os.environ.get("WAYLAND_DISPLAY"))
        x11_env = bool(os.environ.get("DISPLAY"))
        if eff_desktop == "hyprland_wayland":
            session_type = "wayland"
            compositor = "hyprland"
            headless = False
            disp_status = CapabilityStatus.AVAILABLE.value
        elif eff_desktop == "generic_wayland":
            session_type = "wayland"
            compositor = "generic_wayland"
            headless = False
            disp_status = CapabilityStatus.DEGRADED.value
        elif x11_env:
            session_type = "x11"
            compositor = "x11"
            headless = False
            disp_status = CapabilityStatus.DEGRADED.value
        else:
            session_type = "headless"
            compositor = "none"
            headless = True
            disp_status = CapabilityStatus.UNAVAILABLE.value

        cpu_status = (
            CapabilityStatus.AVAILABLE.value
            if isinstance(logical_cores, int) and logical_cores > 0
            else CapabilityStatus.UNKNOWN.value
        )

        return {
            "os_name": sys_name,
            "os_release": sys_release,
            "os_version": sys_version,
            "architecture": arch,
            "python_version": py_ver,
            "cwd": cwd,
            "cpu": {
                "logical_cores": logical_cores,
                "physical_cores": physical_cores,
                "load_avg_1m": load_1m,
                "status": cpu_status,
                "verified": cpu_status == CapabilityStatus.AVAILABLE.value,
                "source": "detected" if cpu_status == CapabilityStatus.AVAILABLE.value else "unknown",
            },
            "memory": {
                "total_mb": total_mb,
                "available_mb": available_mb,
                "percent_used": percent_used,
                "status": mem_status,
                "verified": mem_status == CapabilityStatus.AVAILABLE.value,
                "source": mem_source,
            },
            "display": {
                "session_type": session_type,
                "compositor": compositor,
                "desktop_state": eff_desktop,
                "wayland_display": wayland_env or eff_desktop in ("hyprland_wayland", "generic_wayland"),
                "x11_display": x11_env,
                "headless": headless,
                "status": disp_status,
                "verified": True,
                "source": "detected",
            },
        }

    @classmethod
    def _probe_model_endpoint(
        cls,
        endpoint_url: str,
        model_name: str,
        *,
        timeout: float = DEFAULT_PROBE_TIMEOUT,
        refresh: bool = False,
    ) -> Dict[str, Any]:
        """Probe model endpoint availability with a bounded timeout without loading model weights."""
        sanitized_url = cls._sanitize_endpoint_url(endpoint_url)
        if not sanitized_url:
            return {
                "status": CapabilityStatus.UNKNOWN.value,
                "verified": False,
                "source": "unknown",
                "model_installed": None,
                "installed_models": [],
                "details": "No model endpoint URL configured.",
            }

        cache_key = (sanitized_url, model_name)
        now = time.monotonic()
        if not refresh:
            with cls._cache_lock:
                cached = cls._probe_cache.get(cache_key)
                if cached and (now - cached[0]) <= cls.DEFAULT_CACHE_TTL:
                    return dict(cached[1])

        bounded_timeout = max(0.05, min(float(timeout or cls.DEFAULT_PROBE_TIMEOUT), 5.0))
        base = sanitized_url if "://" in sanitized_url else f"http://{sanitized_url}"
        base = base.rstrip("/")
        probe_url = (
            f"{base}/models"
            if base.endswith("/v1")
            else f"{base}/api/tags"
        )

        try:
            req = urllib.request.Request(probe_url, method="GET")
            with urllib.request.urlopen(req, timeout=bounded_timeout) as resp:
                status_code = getattr(resp, "status", 200)
                raw_body = resp.read().decode("utf-8", errors="replace")
                if status_code != 200:
                    result = {
                        "status": CapabilityStatus.DEGRADED.value,
                        "verified": True,
                        "source": "detected",
                        "model_installed": None,
                        "installed_models": [],
                        "details": f"Endpoint returned HTTP {status_code}.",
                    }
                else:
                    data = json.loads(raw_body) if raw_body.strip() else {}
                    models_list: List[str] = []
                    if isinstance(data, dict) and isinstance(data.get("models"), list):
                        for item in data["models"]:
                            if isinstance(item, dict):
                                m_id = item.get("name") or item.get("model") or item.get("id")
                                if isinstance(m_id, str) and m_id.strip():
                                    models_list.append(m_id.strip())
                    elif isinstance(data, dict) and isinstance(data.get("data"), list):
                        for item in data["data"]:
                            if isinstance(item, dict) and isinstance(item.get("id"), str):
                                models_list.append(item["id"].strip())

                    if model_name and model_name != "unknown" and isinstance(data, dict) and (
                        "models" in data or "data" in data
                    ):
                        target_base = model_name.split(":")[0].lower()
                        matched = any(
                            m.lower() == model_name.lower()
                            or m.split(":")[0].lower() == target_base
                            for m in models_list
                        )
                        if matched:
                            result = {
                                "status": CapabilityStatus.AVAILABLE.value,
                                "verified": True,
                                "source": "detected",
                                "model_installed": True,
                                "installed_models": models_list,
                                "details": f"Endpoint online; model '{model_name}' available.",
                            }
                        else:
                            result = {
                                "status": CapabilityStatus.DEGRADED.value,
                                "verified": True,
                                "source": "detected",
                                "model_installed": False,
                                "installed_models": models_list,
                                "details": (
                                    f"Endpoint online, but configured model '{model_name}' "
                                    "was not found in installed tags."
                                ),
                            }
                    else:
                        result = {
                            "status": CapabilityStatus.AVAILABLE.value,
                            "verified": True,
                            "source": "detected",
                            "model_installed": None,
                            "installed_models": models_list,
                            "details": "Endpoint online.",
                        }
        except (TimeoutError, socket.timeout):
            result = {
                "status": CapabilityStatus.UNAVAILABLE.value,
                "verified": True,
                "source": "detected",
                "model_installed": None,
                "installed_models": [],
                "details": f"Endpoint health check timed out after {bounded_timeout:.2f}s.",
            }
        except urllib.error.HTTPError as exc:
            result = {
                "status": CapabilityStatus.DEGRADED.value
                if exc.code in (401, 403, 404)
                else CapabilityStatus.UNAVAILABLE.value,
                "verified": True,
                "source": "detected",
                "model_installed": None,
                "installed_models": [],
                "details": f"Endpoint returned HTTP {exc.code}.",
            }
        except Exception as exc:
            safe_err = _scrub_secret_text(exc)
            if isinstance(exc, urllib.error.URLError) and isinstance(
                getattr(exc, "reason", None), (TimeoutError, socket.timeout)
            ):
                details_msg = f"Endpoint health check timed out after {bounded_timeout:.2f}s."
            else:
                details_msg = f"Endpoint unreachable ({safe_err})."
            result = {
                "status": CapabilityStatus.UNAVAILABLE.value,
                "verified": True,
                "source": "detected",
                "model_installed": None,
                "installed_models": [],
                "details": details_msg,
            }

        with cls._cache_lock:
            cls._probe_cache[cache_key] = (now, dict(result))
        return result

    @classmethod
    def collect_ai_infrastructure(
        cls,
        config: Optional[Any] = None,
        *,
        probe_services: bool = False,
        probe_timeout: float = DEFAULT_PROBE_TIMEOUT,
        refresh: bool = False,
    ) -> Dict[str, Any]:
        """Collect configured AI model provider, model ID, endpoint status, capabilities, and context limits."""
        raw_url = cls._extract_endpoint_url(config)
        sanitized_url = cls._sanitize_endpoint_url(raw_url)
        locality_detail = cls.detect_inference_locality(config)
        locality = cls.locality_category(locality_detail)

        model_name = ""
        explicit_ctx_limit: Optional[int] = None
        if isinstance(config, dict):
            model_name = str(
                config.get("model") or config.get("ollama_model") or ""
            ).strip()
            raw_ctx = config.get("max_context_tokens") or config.get("context_length")
            if isinstance(raw_ctx, int) and not isinstance(raw_ctx, bool) and raw_ctx > 0:
                explicit_ctx_limit = raw_ctx
        elif config is not None and not isinstance(config, str):
            raw_m = getattr(config, "model", None) or getattr(
                config, "ollama_model", None
            )
            model_name = str(raw_m).strip() if raw_m else ""
            raw_ctx = getattr(config, "max_context_tokens", None) or getattr(
                config, "context_length", None
            )
            if isinstance(raw_ctx, int) and not isinstance(raw_ctx, bool) and raw_ctx > 0:
                explicit_ctx_limit = raw_ctx
        elif not isinstance(config, str):
            try:
                from zelyr.config import get_config

                cfg = get_config()
                raw_m = getattr(cfg, "ollama_model", None)
                model_name = str(raw_m).strip() if raw_m else ""
            except Exception:
                model_name = ""

        if not model_name:
            model_name = "unknown"
        model_name = _scrub_secret_text(model_name)

        if locality_detail == "unknown" and not sanitized_url:
            provider = "unknown"
        elif (
            locality_detail == "local_daemon"
            or "11434" in sanitized_url
            or "ollama" in sanitized_url.lower()
        ):
            provider = "ollama"
        else:
            provider = "openai_compatible"

        if probe_services:
            probe = cls._probe_model_endpoint(
                raw_url,
                model_name,
                timeout=probe_timeout,
                refresh=refresh,
            )
        else:
            probe = {
                "status": CapabilityStatus.UNKNOWN.value,
                "verified": False,
                "source": "configured" if sanitized_url else "unknown",
                "model_installed": None,
                "installed_models": [],
                "details": (
                    "Configured endpoint (live health probe not executed)."
                    if sanitized_url
                    else "No endpoint configured."
                ),
            }

        # Context limit resolution
        if explicit_ctx_limit is not None:
            ctx_tokens = explicit_ctx_limit
            ctx_source = "configured"
        else:
            try:
                from zelyr.engine.budget_manager import (
                    MODEL_CONTEXT_CATALOG,
                    get_model_context_limit,
                )

                ctx_tokens = get_model_context_limit(
                    None if model_name == "unknown" else model_name
                )
                clean_m = model_name.lower().split("/")[-1].split(":")[0]
                in_catalog = model_name != "unknown" and (
                    clean_m in MODEL_CONTEXT_CATALOG
                    or any(
                        k in model_name.lower() or model_name.lower() in k
                        for k in MODEL_CONTEXT_CATALOG
                    )
                )
                ctx_source = "catalog_heuristic" if in_catalog else "default_fallback"
            except Exception:
                ctx_tokens = 8192
                ctx_source = "default_fallback"

        lower_model = model_name.lower()
        vision_hint = any(
            tok in lower_model
            for tok in ("vl", "vision", "llava", "minicpm-v", "pixtral", "gpt-4o", "claude-3", "gemini")
        )
        known_model = model_name != "unknown"

        return {
            "provider": provider,
            "model": model_name,
            "endpoint": sanitized_url or "unknown",
            "locality": locality,
            "inference_locality": locality_detail,
            "endpoint_status": probe["status"],
            "verified": probe["verified"],
            "source": probe["source"],
            "model_installed": probe["model_installed"],
            "details": probe["details"],
            "context_limit": {
                "max_tokens": ctx_tokens,
                "source": ctx_source,
                "verified": False,
            },
            "capabilities": {
                "tool_calling": {
                    "supported": True if known_model else None,
                    "source": "assumed" if known_model else "unknown",
                    "verified": False,
                },
                "streaming": {
                    "supported": True if known_model else None,
                    "source": "assumed" if known_model else "unknown",
                    "verified": False,
                },
                "vision": {
                    "supported": vision_hint if known_model else None,
                    "source": "assumed" if known_model else "unknown",
                    "verified": False,
                },
            },
        }

    @classmethod
    def _inspect_single_tool(
        cls,
        tool_id: str,
        tool_obj: Any,
        *,
        enabled: bool,
        desktop_state: str,
        hyprland_ok: bool,
        browser_ok: bool,
        shell_ok: bool,
        fs_ok: bool,
        mem_ok: bool,
        vision_info: Dict[str, Any],
        tool_registry: Optional[Any],
        caller_ring: Optional[int],
        allowed_set: Optional[set[str]],
    ) -> Dict[str, Any]:
        """Inspect availability, dependencies, security ring, execution mode, and limitations for one tool."""
        required_deps: List[str] = []
        missing_deps: List[str] = []
        limitations: List[str] = []
        degraded = False

        # Resolve required security ring
        req_ring: Optional[int] = None
        get_ring = getattr(tool_registry, "_get_tool_ring", None) if tool_registry is not None else None
        if callable(get_ring):
            try:
                candidate_ring = get_ring(tool_id, tool_obj)
                if isinstance(candidate_ring, int) and not isinstance(candidate_ring, bool):
                    req_ring = candidate_ring
            except Exception:
                req_ring = None
        if req_ring is None and tool_obj is not None:
            raw_r = getattr(tool_obj, "required_ring", getattr(tool_obj, "REQUIRED_RING", None))
            if isinstance(raw_r, int) and not isinstance(raw_r, bool):
                req_ring = raw_r
        if req_ring is None:
            req_ring = 0 if tool_id in _SHELL_TOOL_IDS else 1

        # Resolve tier & execution mode
        cap = getattr(tool_obj, "capability", None)
        tier = int(getattr(cap, "tier", 0 if tool_id in _SHELL_TOOL_IDS else 1))
        if tool_id in _SHELL_TOOL_IDS:
            execution_mode = "external_subprocess"
            required_deps.append("cmd" if os.name == "nt" else "sh")
            if not shell_ok:
                missing_deps.append("cmd" if os.name == "nt" else "sh")
            limitations.append(
                "Non-interactive execution (stdin=DEVNULL, no interactive sudo or TTY prompts)."
            )
        elif tool_id in _DESKTOP_TOOL_IDS:
            execution_mode = "ipc_bridge"
            tier = 1
            required_deps.append("hyprland_ipc_socket")
            if not hyprland_ok:
                missing_deps.append("hyprland_ipc_socket")
                limitations.append("Requires an active Hyprland Wayland compositor session.")
        elif tool_id in _BROWSER_TOOL_IDS:
            execution_mode = "ipc_bridge"
            tier = 2
            required_deps.append("browser_extension_bridge_or_cdp_9222")
            if not browser_ok:
                missing_deps.append("browser_extension_bridge_or_cdp_9222")
                limitations.append(
                    "Requires an active BrowserExtensionBridge client or Chrome DevTools Protocol on 127.0.0.1:9222."
                )
        elif tool_id in _VISION_UI_TOOL_IDS:
            execution_mode = "builtin_in_process"
            tier = 3
            required_deps.extend(["graphical_display_session", "vision_backend_vla_or_vlm"])
            if desktop_state == "headless_or_unsupported":
                missing_deps.append("graphical_display_session")
                limitations.append("Unavailable in headless sessions without a display server.")
            else:
                v_status = vision_info.get("status")
                if v_status == CapabilityStatus.UNAVAILABLE.value:
                    missing_deps.append("vision_backend_vla_or_vlm")
                    limitations.extend(vision_info.get("limitations", []))
                elif v_status == CapabilityStatus.DEGRADED.value:
                    degraded = True
                    limitations.extend(vision_info.get("limitations", []))
        elif tool_id in _FILESYSTEM_TOOL_IDS:
            execution_mode = "builtin_in_process"
            required_deps.append("accessible_workspace_cwd")
            if not fs_ok:
                missing_deps.append("accessible_workspace_cwd")
                limitations.append("Current working directory is inaccessible.")
        elif tool_id in _MEMORY_TOOL_IDS:
            execution_mode = "builtin_in_process"
            required_deps.append("zelyr.db.memory")
            if not mem_ok:
                missing_deps.append("zelyr.db.memory")
                limitations.append("Local SQLite memory store module is unavailable.")
        elif tool_id in _MEDIA_TOOL_IDS:
            execution_mode = "external_subprocess"
            required_deps.append("playerctl")
            if shutil.which("playerctl") is None:
                missing_deps.append("playerctl")
                limitations.append("Requires 'playerctl' binary in PATH for MPRIS media control.")
        elif tool_id in _NOTIFICATION_TOOL_IDS:
            execution_mode = "external_subprocess"
            required_deps.extend(["notify-send", "graphical_display_session"])
            if shutil.which("notify-send") is None:
                missing_deps.append("notify-send")
            if desktop_state == "headless_or_unsupported" and not os.environ.get("DISPLAY"):
                missing_deps.append("graphical_display_session")
        elif tool_id in _CLIPBOARD_TOOL_IDS:
            execution_mode = "external_subprocess"
            if desktop_state in ("hyprland_wayland", "generic_wayland"):
                required_deps.extend(["wl-copy", "wl-paste"])
                for b_name in ("wl-copy", "wl-paste"):
                    if shutil.which(b_name) is None:
                        missing_deps.append(b_name)
            elif os.environ.get("DISPLAY"):
                required_deps.append("xclip_or_xsel")
                if shutil.which("xclip") is None and shutil.which("xsel") is None:
                    missing_deps.append("xclip_or_xsel")
            else:
                required_deps.append("graphical_display_session")
                missing_deps.append("graphical_display_session")
        else:
            is_ext = bool(getattr(tool_obj, "is_external", False))
            if is_ext:
                src_type = str(getattr(tool_obj, "source_type", "external") or "external")
                execution_mode = f"external_{src_type}" if not src_type.startswith("external") else src_type
            else:
                mod_name = type(tool_obj).__module__ if tool_obj is not None else ""
                execution_mode = (
                    "sandboxed_plugin"
                    if "dynamic_tools" in mod_name or "plugin" in mod_name
                    else "builtin_in_process"
                )

        # Honor explicit tool object dependency declarations & is_available() hook
        if tool_obj is not None:
            req_bins = getattr(tool_obj, "required_binaries", None)
            if isinstance(req_bins, (list, tuple, set, frozenset)):
                for b_name in req_bins:
                    if isinstance(b_name, str) and b_name.strip():
                        clean_b = b_name.strip()
                        if clean_b not in required_deps:
                            required_deps.append(clean_b)
                        if shutil.which(clean_b) is None and clean_b not in missing_deps:
                            missing_deps.append(clean_b)
                            limitations.append(f"Missing required binary '{clean_b}' in PATH.")

            req_mods = getattr(tool_obj, "required_modules", None)
            if isinstance(req_mods, (list, tuple, set, frozenset)):
                for m_name in req_mods:
                    if isinstance(m_name, str) and m_name.strip():
                        clean_m = m_name.strip()
                        if clean_m not in required_deps:
                            required_deps.append(clean_m)
                        if not _is_module_importable(clean_m) and clean_m not in missing_deps:
                            missing_deps.append(clean_m)
                            limitations.append(f"Missing optional Python dependency '{clean_m}'.")

            is_avail_fn = getattr(tool_obj, "is_available", None)
            if callable(is_avail_fn):
                try:
                    if not bool(is_avail_fn()):
                        if "tool_runtime_check" not in missing_deps:
                            missing_deps.append("tool_runtime_check")
                        limitations.append("Tool runtime self-check reported unavailable.")
                except Exception as exc:
                    if "tool_runtime_check" not in missing_deps:
                        missing_deps.append("tool_runtime_check")
                    limitations.append(f"Tool runtime self-check raised {_scrub_secret_text(exc)}.")

        if not enabled:
            limitations.append("Tool is disabled in user configuration.")

        is_available = bool(enabled and not missing_deps)

        # Evaluate authorization under caller_ring and allowed_tools
        is_authorized = is_available
        if is_available:
            if allowed_set is not None and tool_id not in allowed_set:
                is_authorized = False
                limitations.append("Excluded by active allowed_tools policy.")
            if caller_ring is not None:
                if not isinstance(req_ring, int) or isinstance(req_ring, bool):
                    is_authorized = False
                    limitations.append("Unknown or invalid required security ring.")
                elif caller_ring > req_ring:
                    is_authorized = False
                    limitations.append(
                        f"Requires security ring {req_ring} (caller ring is {caller_ring})."
                    )

        if not is_available:
            status = CapabilityStatus.UNAVAILABLE.value
        elif degraded:
            status = CapabilityStatus.DEGRADED.value
        else:
            status = CapabilityStatus.AVAILABLE.value

        is_ext_tool = bool(getattr(tool_obj, "is_external", False)) if tool_obj is not None else False
        detail_entry: Dict[str, Any] = {
            "tool_id": tool_id,
            "origin": "external" if is_ext_tool else "native",
            "status": status,
            "enabled": enabled,
            "available": is_available,
            "authorized": is_authorized,
            "required_ring": req_ring,
            "caller_ring": caller_ring,
            "tier": tier,
            "execution_mode": execution_mode,
            "required_dependencies": required_deps,
            "missing_dependencies": missing_deps,
            "limitations": limitations,
            "verified": True,
            "verification_source": "detected",
        }
        if is_ext_tool:
            detail_entry["source_id"] = str(getattr(tool_obj, "source_id", "external") or "external")
            detail_entry["source_type"] = str(getattr(tool_obj, "source_type", "external") or "external")
            detail_entry["external_tool_name"] = str(
                getattr(tool_obj, "external_tool_name", tool_id) or tool_id
            )
        return detail_entry

    @classmethod
    def get_tool_capability_summary(
        cls,
        tool_registry: Optional[Any] = None,
        caller_ring: Optional[int] = None,
        allowed_tools: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Distinguish registered, available, degraded, and authorized tools from a registry.

        - A tool being registered does not imply it is enabled/available.
        - A tool being available does not imply it is authorized for the caller context.
        - Never claims a tool is available merely because its Python module imports.
        - If no registry is provided, registered/available/authorized sets are reported as
          unverified (None) rather than fabricated.
        """
        if tool_registry is None:
            return {
                "registry_known": False,
                "authorization_scoped": False,
                "registered": None,
                "available": None,
                "authorized": None,
                "degraded": None,
                "unavailable": None,
                "unauthorized": None,
                "details": None,
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

        # If a fresh ToolRegistry has not yet lazily instantiated its built-in tools,
        # populate built-in descriptors when get_builtin_tools() is present and raw_registered is empty.
        if (
            not raw_registered
            and callable(getattr(tool_registry, "get_builtin_tools", None))
            and callable(getattr(tool_registry, "get_tool", None))
        ):
            try:
                for b_id in tool_registry.get_builtin_tools():
                    t_inst = tool_registry.get_tool(b_id)
                    if t_inst is not None:
                        raw_registered[b_id] = t_inst
                if callable(getattr(tool_registry, "list_tools", None)):
                    cand = tool_registry.list_tools()
                    if isinstance(cand, dict):
                        enabled_tools = dict(cand)
            except Exception:
                pass

        registered_names = sorted(raw_registered.keys())

        desktop_state = cls.detect_desktop_state()
        hyprland_ok = desktop_state == "hyprland_wayland"
        browser_ok = cls._is_browser_active()
        shell_ok = cls._is_shell_runtime_available()
        fs_ok = cls._is_filesystem_runtime_available()
        mem_ok = cls._is_memory_runtime_available()
        vision_info = cls._inspect_vision_runtime(desktop_state=desktop_state)

        allowed_set: Optional[set[str]] = None
        if allowed_tools is not None:
            if isinstance(allowed_tools, (str, bytes, dict)) or not isinstance(
                allowed_tools, Iterable
            ):
                allowed_set = set()
            else:
                allowed_set = {
                    str(item).strip()
                    for item in allowed_tools
                    if isinstance(item, str) and str(item).strip()
                }

        invalid_caller_ring = caller_ring is not None and (
            not isinstance(caller_ring, int)
            or isinstance(caller_ring, bool)
            or caller_ring < 0
            or caller_ring > 3
        )

        available_names: List[str] = []
        authorized_names: List[str] = []
        degraded_names: List[str] = []
        unavailable_names: List[str] = []
        unauthorized_names: List[str] = []
        details_map: Dict[str, Dict[str, Any]] = {}

        for tid in registered_names:
            tool_obj = raw_registered.get(tid)
            is_enabled = tid in enabled_tools
            info = cls._inspect_single_tool(
                tid,
                tool_obj,
                enabled=is_enabled,
                desktop_state=desktop_state,
                hyprland_ok=hyprland_ok,
                browser_ok=browser_ok,
                shell_ok=shell_ok,
                fs_ok=fs_ok,
                mem_ok=mem_ok,
                vision_info=vision_info,
                tool_registry=tool_registry,
                caller_ring=caller_ring,
                allowed_set=allowed_set,
            )
            if invalid_caller_ring:
                info["authorized"] = False
                info["limitations"].append(f"Invalid caller_ring {caller_ring!r}.")

            details_map[tid] = info
            if info["available"]:
                available_names.append(tid)
                if info["status"] == CapabilityStatus.DEGRADED.value:
                    degraded_names.append(tid)
                if info["authorized"]:
                    authorized_names.append(tid)
                else:
                    unauthorized_names.append(tid)
            else:
                unavailable_names.append(tid)

        return {
            "registry_known": True,
            "authorization_scoped": (caller_ring is not None or allowed_tools is not None),
            "registered": registered_names,
            "available": available_names,
            "authorized": authorized_names,
            "degraded": degraded_names,
            "unavailable": unavailable_names,
            "unauthorized": unauthorized_names,
            "details": details_map,
        }

    @classmethod
    def get_external_tools_summary(
        cls,
        tool_registry: Optional[Any] = None,
        caller_ring: Optional[int] = None,
        allowed_tools: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Collect sanitized external/fallback tool source health, counts, and lifecycle details."""
        empty_counts = {
            "sources": 0,
            "healthy_sources": 0,
            "discovered": 0,
            "validated": 0,
            "authorized": 0,
            "available": 0,
            "unavailable": 0,
            "degraded": 0,
            "schema_failures": 0,
        }
        if tool_registry is None:
            return {
                "registry_known": False,
                "sources": {},
                "counts": dict(empty_counts),
                "available": [],
                "unavailable": [],
                "degraded": [],
                "authorized": [],
                "validation_failures": [],
                "details": {},
            }

        ext_fn = getattr(tool_registry, "get_external_manifest_summary", None)
        if callable(ext_fn):
            try:
                summary = ext_fn(caller_ring=caller_ring, allowed_tools=allowed_tools)
                if isinstance(summary, dict):
                    out = dict(summary)
                    out.setdefault("registry_known", True)
                    return _scrub_structure(out)
            except Exception as exc:
                logger.debug("External manifest summary failed: %s", exc)

        # Fallback introspection for lightweight/legacy registries holding is_external=True tools
        raw_tools = {}
        if callable(getattr(tool_registry, "list_tools", None)):
            cand = tool_registry.list_tools()
            if isinstance(cand, dict):
                raw_tools = dict(cand)

        sources: Dict[str, Dict[str, Any]] = {}
        avail: List[str] = []
        unavail: List[str] = []
        auth_ids: List[str] = []
        details: Dict[str, Dict[str, Any]] = {}
        for tid, tobj in sorted(raw_tools.items()):
            if not getattr(tobj, "is_external", False):
                continue
            sid = str(getattr(tobj, "source_id", "external") or "external")
            stype = str(getattr(tobj, "source_type", "external") or "external")
            ep = cls._sanitize_endpoint_url(str(getattr(tobj, "source_endpoint", "") or "")) or sid
            is_avail = True
            if callable(getattr(tobj, "is_available", None)):
                try:
                    is_avail = bool(tobj.is_available())
                except Exception:
                    is_avail = False
            if is_avail:
                avail.append(tid)
                auth_ids.append(tid)
            else:
                unavail.append(tid)
            src_rec = sources.setdefault(
                sid,
                {
                    "source_id": sid,
                    "source_type": stype,
                    "endpoint": ep,
                    "health": "HEALTHY" if is_avail else "UNAVAILABLE",
                    "discovered_count": 0,
                    "authorized_count": 0,
                    "available_count": 0,
                    "unavailable_count": 0,
                    "validation_failure_count": 0,
                    "last_error": None,
                },
            )
            src_rec["discovered_count"] += 1
            if is_avail:
                src_rec["authorized_count"] += 1
                src_rec["available_count"] += 1
            else:
                src_rec["unavailable_count"] += 1
                src_rec["health"] = "UNAVAILABLE"
            details[tid] = {
                "tool_id": tid,
                "source_id": sid,
                "source_type": stype,
                "external_tool_name": str(getattr(tobj, "external_tool_name", tid) or tid),
                "lifecycle": "AVAILABLE" if is_avail else "UNAVAILABLE",
                "authorized": True,
                "schema_valid": True,
                "is_external": True,
            }

        return _scrub_structure(
            {
                "registry_known": True,
                "sources": sources,
                "counts": {
                    "sources": len(sources),
                    "healthy_sources": sum(1 for s in sources.values() if s["health"] == "HEALTHY"),
                    "discovered": len(details),
                    "validated": len(details),
                    "authorized": len(auth_ids),
                    "available": len(avail),
                    "unavailable": len(unavail),
                    "degraded": 0,
                    "schema_failures": 0,
                },
                "available": avail,
                "unavailable": unavail,
                "degraded": [],
                "authorized": auth_ids,
                "validation_failures": [],
                "details": details,
            }
        )

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
    def collect_integrations(
        cls,
        *,
        ai_info: Optional[Dict[str, Any]] = None,
        desktop_state: Optional[str] = None,
        browser_active: Optional[bool] = None,
        mem_active: Optional[bool] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """Collect structured availability status for all external and host integrations."""
        eff_desktop = desktop_state or cls.detect_desktop_state()
        eff_browser = (
            browser_active if browser_active is not None else cls._is_browser_active()
        )
        eff_mem = (
            mem_active
            if mem_active is not None
            else cls._is_memory_runtime_available()
        )
        eff_ai = ai_info or cls.collect_ai_infrastructure()

        # 1. Ollama integration
        ollama_status = eff_ai.get("endpoint_status", CapabilityStatus.UNKNOWN.value)
        ollama_verified = bool(eff_ai.get("verified", False))
        ollama_source = eff_ai.get("source", "unknown")
        ollama_limitations: List[str] = []
        if ollama_status == CapabilityStatus.DEGRADED.value:
            ollama_limitations.append(str(eff_ai.get("details", "Endpoint degraded.")))
        elif ollama_status == CapabilityStatus.UNAVAILABLE.value:
            ollama_limitations.append(str(eff_ai.get("details", "Endpoint unreachable.")))

        ollama_integration = {
            "status": ollama_status,
            "verified": ollama_verified,
            "source": ollama_source,
            "endpoint": eff_ai.get("endpoint", "unknown"),
            "model": eff_ai.get("model", "unknown"),
            "model_installed": eff_ai.get("model_installed"),
            "details": eff_ai.get("details", ""),
            "limitations": ollama_limitations,
        }

        # 2. Browser integration (Extension Bridge + CDP 9222 + Playwright package)
        bridge_listening = False
        bridge_connected = False
        try:
            from zelyr.tools.browser_extension import BrowserExtensionBridge

            inst = getattr(BrowserExtensionBridge, "_instance", None)
            if inst is not None:
                bridge_listening = getattr(inst, "server", None) is not None
                bridge_connected = bool(inst.is_connected())
        except Exception:
            pass

        native_host_connected = False
        try:
            from zelyr.tools.browser_native_host import NativeBrowserClient

            native_host_connected = bool(NativeBrowserClient().is_socket_present())
        except Exception:
            pass

        playwright_installed = _is_module_importable("playwright")
        browser_limitations: List[str] = []
        if eff_browser:
            browser_status = CapabilityStatus.AVAILABLE.value
            browser_details = "Active browser automation channel (WebExtension bridge, Native Messaging host, or CDP 9222)."
        elif bridge_listening or playwright_installed:
            browser_status = CapabilityStatus.DEGRADED.value
            browser_details = (
                "Browser bridge or Playwright installed, but no active browser session is connected."
            )
            browser_limitations.append(
                "No connected BrowserExtensionBridge/Native Messaging client and CDP port 9222 is closed."
            )
        else:
            browser_status = CapabilityStatus.UNAVAILABLE.value
            browser_details = "No browser extension bridge, native messaging host, CDP endpoint, or Playwright runtime detected."
            browser_limitations.append(
                "Connect the Zelyr browser extension (or install Chromium native host via 'zelyr browser install-host') or launch Chromium with --remote-debugging-port=9222."
            )

        browser_integration = {
            "status": browser_status,
            "verified": True,
            "source": "detected",
            "bridge_listening": bridge_listening,
            "bridge_connected": bridge_connected,
            "native_host_connected": native_host_connected,
            "playwright_installed": playwright_installed,
            "details": browser_details,
            "limitations": browser_limitations,
        }

        # 3. Vision / Spotless-Arms VLA integration
        vision_integration = cls._inspect_vision_runtime(desktop_state=eff_desktop)

        # 4. GitHub / Git integration
        git_installed = shutil.which("git") is not None
        gh_installed = shutil.which("gh") is not None
        token_configured = bool(
            os.environ.get("GH_TOKEN", "").strip()
            or os.environ.get("GITHUB_TOKEN", "").strip()
        )
        in_git_repo = False
        try:
            cur = Path.cwd().resolve()
            in_git_repo = any((p / ".git").exists() for p in (cur, *cur.parents))
        except Exception:
            in_git_repo = False

        github_limitations: List[str] = []
        if not git_installed:
            github_status = CapabilityStatus.UNAVAILABLE.value
            github_details = "Git binary is not installed in PATH."
            github_limitations.append("Install 'git' to enable repository operations.")
        elif gh_installed:
            github_status = CapabilityStatus.AVAILABLE.value
            github_details = "git and GitHub CLI (gh) are installed."
            if not in_git_repo:
                github_limitations.append("Current working directory is not inside a Git worktree.")
        else:
            github_status = CapabilityStatus.DEGRADED.value
            github_details = "git is installed, but GitHub CLI (gh) is not found in PATH."
            github_limitations.append("GitHub CLI ('gh') is unavailable for PR/Actions operations.")

        github_integration = {
            "status": github_status,
            "verified": True,
            "source": "detected",
            "git_installed": git_installed,
            "gh_cli_installed": gh_installed,
            "in_git_repository": in_git_repo,
            "token_configured": token_configured,
            "details": github_details,
            "limitations": github_limitations,
        }

        # 5. Desktop Compositor integration
        if eff_desktop == "hyprland_wayland":
            comp_status = CapabilityStatus.AVAILABLE.value
            comp_details = "Hyprland Wayland IPC socket active."
            comp_limitations: List[str] = []
        elif eff_desktop == "generic_wayland" or os.environ.get("DISPLAY"):
            comp_status = CapabilityStatus.DEGRADED.value
            comp_details = "Graphical session present without Hyprland IPC window management."
            comp_limitations = ["Hyprland IPC tools (manage_desktop_window) are unavailable."]
        else:
            comp_status = CapabilityStatus.UNAVAILABLE.value
            comp_details = "No Wayland or X11 display server detected."
            comp_limitations = ["Graphical desktop automation is unavailable."]

        desktop_integration = {
            "status": comp_status,
            "verified": True,
            "source": "detected",
            "desktop_state": eff_desktop,
            "details": comp_details,
            "limitations": comp_limitations,
        }

        # 6. Clipboard integration
        has_wl_clip = bool(shutil.which("wl-copy") and shutil.which("wl-paste"))
        has_x11_clip = bool(shutil.which("xclip") or shutil.which("xsel"))
        has_display = eff_desktop in ("hyprland_wayland", "generic_wayland") or bool(
            os.environ.get("DISPLAY")
        )
        if has_display and (has_wl_clip or has_x11_clip):
            clip_status = CapabilityStatus.AVAILABLE.value
            clip_details = "Clipboard utilities and active display session detected."
            clip_limitations: List[str] = []
        elif has_wl_clip or has_x11_clip:
            clip_status = CapabilityStatus.DEGRADED.value
            clip_details = "Clipboard binaries installed, but no active display session detected."
            clip_limitations = ["Clipboard operations require an active Wayland or X11 display."]
        else:
            clip_status = CapabilityStatus.UNAVAILABLE.value
            clip_details = "No clipboard utilities (wl-copy/wl-paste or xclip/xsel) found in PATH."
            clip_limitations = ["Install wl-clipboard (Wayland) or xclip (X11)."]

        clipboard_integration = {
            "status": clip_status,
            "verified": True,
            "source": "detected",
            "wayland_clipboard": has_wl_clip,
            "x11_clipboard": has_x11_clip,
            "details": clip_details,
            "limitations": clip_limitations,
        }

        # 7. Media / MPRIS integration
        has_playerctl = shutil.which("playerctl") is not None
        media_integration = {
            "status": (
                CapabilityStatus.AVAILABLE.value
                if has_playerctl
                else CapabilityStatus.UNAVAILABLE.value
            ),
            "verified": True,
            "source": "detected",
            "playerctl_installed": has_playerctl,
            "details": (
                "playerctl installed for MPRIS media control."
                if has_playerctl
                else "playerctl not installed in PATH."
            ),
            "limitations": [] if has_playerctl else ["Install 'playerctl' for media playback control."],
        }

        # 8. Memory Store integration
        memory_integration = {
            "status": (
                CapabilityStatus.AVAILABLE.value
                if eff_mem
                else CapabilityStatus.UNAVAILABLE.value
            ),
            "verified": True,
            "source": "detected",
            "details": (
                "Local SQLite semantic memory store available."
                if eff_mem
                else "SQLite memory module unavailable."
            ),
            "limitations": [] if eff_mem else ["Persistent memory tools are unavailable."],
        }

        return {
            "ollama": ollama_integration,
            "browser": browser_integration,
            "vision_vla": vision_integration,
            "github": github_integration,
            "desktop_compositor": desktop_integration,
            "clipboard": clipboard_integration,
            "media_mpris": media_integration,
            "memory_store": memory_integration,
        }

    @classmethod
    def _build_cache_key(
        cls,
        *,
        endpoint_url: str,
        model_name: str,
        desktop_state: str,
        browser_active: bool,
        shell_ok: bool,
        fs_ok: bool,
        mem_ok: bool,
        tool_registry: Optional[Any],
        caller_ring: Optional[int],
        allowed_tools: Optional[Any],
        probe_services: bool,
    ) -> Tuple[Any, ...]:
        """Build a deterministic cache key that automatically invalidates on tool or state changes."""
        reg_sig: Optional[Tuple[Any, ...]] = None
        if tool_registry is not None:
            core_reg = getattr(tool_registry, "_core_registry", None)
            raw_keys: Tuple[str, ...] = ()
            if core_reg is not None and callable(getattr(core_reg, "list_tools", None)):
                try:
                    c = core_reg.list_tools()
                    if isinstance(c, dict):
                        raw_keys = tuple(sorted(c.keys()))
                except Exception:
                    raw_keys = ()
            enabled_keys: Tuple[str, ...] = ()
            if callable(getattr(tool_registry, "list_tools", None)):
                try:
                    c = tool_registry.list_tools()
                    if isinstance(c, dict):
                        enabled_keys = tuple(sorted(c.keys()))
                except Exception:
                    enabled_keys = ()
            ext_mgr = getattr(tool_registry, "_external_manager", None)
            ext_sig = (
                ext_mgr.state_signature()
                if ext_mgr is not None and callable(getattr(ext_mgr, "state_signature", None))
                else ()
            )
            reg_sig = (id(tool_registry), raw_keys, enabled_keys, ext_sig)

        if allowed_tools is None:
            allowed_sig: Any = None
        elif isinstance(allowed_tools, (str, bytes, dict)) or not isinstance(
            allowed_tools, Iterable
        ):
            allowed_sig = ("__invalid__", type(allowed_tools).__name__)
        else:
            allowed_sig = tuple(sorted(str(x) for x in allowed_tools))

        return (
            endpoint_url,
            model_name,
            desktop_state,
            browser_active,
            shell_ok,
            fs_ok,
            mem_ok,
            reg_sig,
            caller_ring,
            allowed_sig,
            probe_services,
            os.environ.get("ZELYR_USE_SPOTLESS_ARMS", ""),
            os.environ.get("WAYLAND_DISPLAY", ""),
            os.environ.get("DISPLAY", ""),
        )

    @classmethod
    def collect_manifest(
        cls,
        config: Optional[Any] = None,
        tool_registry: Optional[Any] = None,
        caller_ring: Optional[int] = None,
        allowed_tools: Optional[Any] = None,
        *,
        probe_services: bool = False,
        probe_timeout: float = DEFAULT_PROBE_TIMEOUT,
        refresh: bool = False,
        use_cache: bool = True,
    ) -> Dict[str, Any]:
        """Collect structured, secret-free runtime environment facts."""
        raw_url = cls._extract_endpoint_url(config)
        sanitized_url = cls._sanitize_endpoint_url(raw_url)
        desktop_state = cls.detect_desktop_state()
        browser_active = cls._is_browser_active()
        shell_ok = cls._is_shell_runtime_available()
        fs_ok = cls._is_filesystem_runtime_available()
        mem_ok = cls._is_memory_runtime_available()

        ai_info = cls.collect_ai_infrastructure(
            config=config,
            probe_services=probe_services,
            probe_timeout=probe_timeout,
            refresh=refresh,
        )
        model_name = ai_info["model"]

        cache_key = cls._build_cache_key(
            endpoint_url=sanitized_url,
            model_name=model_name,
            desktop_state=desktop_state,
            browser_active=browser_active,
            shell_ok=shell_ok,
            fs_ok=fs_ok,
            mem_ok=mem_ok,
            tool_registry=tool_registry,
            caller_ring=caller_ring,
            allowed_tools=allowed_tools,
            probe_services=probe_services,
        )

        now = time.monotonic()
        if use_cache and not refresh:
            with cls._cache_lock:
                cached = cls._manifest_cache.get(cache_key)
                if cached and (now - cached[0]) <= cls.DEFAULT_CACHE_TTL:
                    return _scrub_structure(cached[1])

        system_info = cls.collect_system_info(desktop_state=desktop_state)
        tools_summary = cls.get_tool_capability_summary(
            tool_registry=tool_registry,
            caller_ring=caller_ring,
            allowed_tools=allowed_tools,
        )
        external_summary = cls.get_external_tools_summary(
            tool_registry=tool_registry,
            caller_ring=caller_ring,
            allowed_tools=allowed_tools,
        )
        tools_summary["external"] = dict(external_summary.get("counts") or {})
        actuators = cls.get_active_actuators(
            tool_registry=tool_registry,
            caller_ring=caller_ring,
            allowed_tools=allowed_tools,
        )
        integrations = cls.collect_integrations(
            ai_info=ai_info,
            desktop_state=desktop_state,
            browser_active=browser_active,
            mem_active=mem_ok,
        )

        manifest: Dict[str, Any] = {
            "schema_version": "1.4",
            "inference_locality": ai_info["inference_locality"],
            "locality": ai_info["locality"],
            "tool_execution_locality": "local",
            "model": model_name,
            "platform": {
                "system": system_info["os_name"],
                "machine": system_info["architecture"],
                "cwd": system_info["cwd"],
            },
            "desktop_state": desktop_state,
            "actuators": actuators,
            "system": system_info,
            "ai_infrastructure": ai_info,
            "tools": tools_summary,
            "external_tools": external_summary,
            "integrations": integrations,
        }

        scrubbed = _scrub_structure(manifest)
        if use_cache:
            with cls._cache_lock:
                cls._manifest_cache[cache_key] = (now, scrubbed)
        return scrubbed

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
            probe_services=False,
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

        block = (
            "[RUNTIME ENVIRONMENT & CAPABILITY MANIFEST]\n"
            f"- Inference: {locality_detail} ({model_name}) | Tool Exec: {manifest['tool_execution_locality']}\n"
            f"- Target Host: {sys_name} ({arch}) | Working Directory: {cwd}\n"
            f"- Desktop Session: {desktop_state}\n"
            f"- Active Actuators: {active_str}\n"
            f"- Degraded/Unavailable: {unavailable_str}\n"
            "- Constraints: Non-interactive execution, DEVNULL stdin, no interactive sudo."
        )
        ext_tools = manifest.get("external_tools") or {}
        ext_counts = ext_tools.get("counts") or {}
        if ext_counts.get("sources", 0) > 0 or ext_counts.get("discovered", 0) > 0:
            ext_line = (
                f"\n- External: {ext_counts.get('available', 0)}/{ext_counts.get('discovered', 0)} "
                f"avail ({ext_counts.get('healthy_sources', 0)}/{ext_counts.get('sources', 0)} src)"
            )
            if len(block) + len(ext_line) <= 480:
                block += ext_line
        return _scrub_secret_text(block)

    @classmethod
    def format_diagnostic_report(
        cls,
        manifest: Optional[Dict[str, Any]] = None,
        *,
        config: Optional[Any] = None,
        tool_registry: Optional[Any] = None,
        caller_ring: Optional[int] = None,
        allowed_tools: Optional[Any] = None,
        probe_services: bool = True,
        probe_timeout: float = DEFAULT_PROBE_TIMEOUT,
    ) -> str:
        """Render a detailed, human-readable diagnostic report for CLI inspection."""
        data = manifest or cls.collect_manifest(
            config=config,
            tool_registry=tool_registry,
            caller_ring=caller_ring,
            allowed_tools=allowed_tools,
            probe_services=probe_services,
            probe_timeout=probe_timeout,
            refresh=probe_services,
        )
        sys_info = data.get("system", {})
        cpu_info = sys_info.get("cpu", {})
        mem_info = sys_info.get("memory", {})
        disp_info = sys_info.get("display", {})
        ai_info = data.get("ai_infrastructure", {})
        ctx_info = ai_info.get("context_limit", {})
        tools_info = data.get("tools", {})
        integrations = data.get("integrations", {})

        lines: List[str] = [
            "Zelyr Runtime Environment Manifest (v1.4)",
            "=" * 44,
            "",
            "[System]",
            f"  OS:            {sys_info.get('os_name', 'unknown')} ({sys_info.get('os_version', 'unknown')})",
            f"  Kernel/Rel:    {sys_info.get('os_release', 'unknown')}",
            f"  Architecture:  {sys_info.get('architecture', 'unknown')}",
            f"  Python:        {sys_info.get('python_version', 'unknown')}",
            f"  CPU Cores:     {cpu_info.get('logical_cores', 'unknown')} logical"
            + (
                f" ({cpu_info.get('physical_cores')} physical)"
                if cpu_info.get("physical_cores")
                else ""
            )
            + (
                f" | 1m load: {cpu_info.get('load_avg_1m')}"
                if cpu_info.get("load_avg_1m") is not None
                else ""
            ),
            f"  Memory:        {mem_info.get('available_mb', 'unknown')} MB available / "
            f"{mem_info.get('total_mb', 'unknown')} MB total "
            f"({mem_info.get('percent_used', 'unknown')}% used) [{mem_info.get('status', 'UNKNOWN')}]",
            f"  Display:       {disp_info.get('session_type', 'unknown')} "
            f"(compositor={disp_info.get('compositor', 'none')}, state={disp_info.get('desktop_state')}) "
            f"[{disp_info.get('status', 'UNKNOWN')}]",
            f"  Working Dir:   {sys_info.get('cwd', 'unknown')}",
            "",
            "[AI Infrastructure]",
            f"  Provider:      {ai_info.get('provider', 'unknown')} ({ai_info.get('inference_locality', 'unknown')})",
            f"  Model:         {ai_info.get('model', 'unknown')}",
            f"  Endpoint:      {ai_info.get('endpoint', 'unknown')} "
            f"[{ai_info.get('endpoint_status', 'UNKNOWN')} · source={ai_info.get('source', 'unknown')}]",
            f"  Context Limit: {ctx_info.get('max_tokens', 'unknown')} tokens "
            f"(source={ctx_info.get('source', 'unknown')})",
            f"  Details:       {ai_info.get('details', '')}",
            "",
            "[Tools]",
        ]

        if not tools_info.get("registry_known"):
            lines.append("  Registry:      Not attached (unverified)")
        else:
            reg_list = tools_info.get("registered") or []
            avail_list = tools_info.get("available") or []
            auth_list = tools_info.get("authorized") or []
            deg_list = tools_info.get("degraded") or []
            unavail_list = tools_info.get("unavailable") or []
            lines.append(
                f"  Counts:        {len(reg_list)} registered · {len(avail_list)} available · "
                f"{len(auth_list)} authorized · {len(deg_list)} degraded · {len(unavail_list)} unavailable"
            )
            details = tools_info.get("details") or {}
            for tid in reg_list:
                t_meta = details.get(tid, {})
                st = t_meta.get("status", "UNKNOWN")
                ring = t_meta.get("required_ring", "?")
                mode = t_meta.get("execution_mode", "unknown")
                auth_flag = "authorized" if t_meta.get("authorized") else "unauthorized"
                lim_list = t_meta.get("limitations") or []
                lim_suffix = f" — {'; '.join(lim_list)}" if lim_list else ""
                lines.append(
                    f"  • {tid:<26} [{st:<11}] ring={ring} mode={mode} ({auth_flag}){lim_suffix}"
                )

        lines.extend(["", "[Integrations]"])
        for int_name, int_meta in sorted(integrations.items()):
            st = int_meta.get("status", "UNKNOWN")
            src = int_meta.get("source", "unknown")
            det = int_meta.get("details", "")
            lines.append(f"  • {int_name:<20} [{st:<11}] (source={src}) {det}")

        ext_info = data.get("external_tools") or {}
        ext_counts = ext_info.get("counts") or {}
        if ext_info.get("registry_known") or ext_counts.get("sources", 0) > 0 or ext_counts.get("discovered", 0) > 0:
            lines.extend(["", "[External Tools]"])
            lines.append(
                f"  Counts:        {ext_counts.get('sources', 0)} sources "
                f"({ext_counts.get('healthy_sources', 0)} healthy) · "
                f"{ext_counts.get('discovered', 0)} discovered · "
                f"{ext_counts.get('authorized', 0)} authorized · "
                f"{ext_counts.get('available', 0)} available · "
                f"{ext_counts.get('unavailable', 0)} unavailable · "
                f"{ext_counts.get('schema_failures', 0)} schema failures"
            )
            for sid, smeta in sorted((ext_info.get("sources") or {}).items()):
                lines.append(
                    f"  • source={sid:<18} [{smeta.get('health', 'UNKNOWN'):<12}] "
                    f"type={smeta.get('source_type', 'external')} endpoint={smeta.get('endpoint', 'unknown')}"
                )
            for tid, tmeta in sorted((ext_info.get("details") or {}).items()):
                lines.append(
                    f"    - {tid:<28} [{tmeta.get('lifecycle', 'UNKNOWN'):<11}] "
                    f"source={tmeta.get('source_id', '?')} authorized={tmeta.get('authorized', False)}"
                )
            for verr in ext_info.get("validation_failures") or []:
                lines.append(
                    f"    ! schema_error tool={verr.get('tool_id', '?')} ({verr.get('error_type', 'ERROR')}): "
                    f"{verr.get('error', '')}"
                )

        return _scrub_secret_text("\n".join(lines))
