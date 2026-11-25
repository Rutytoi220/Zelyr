"""Chromium/Chrome Native Messaging host and UNIX IPC client for Zelyr.

Provides a least-privilege, user-scoped Native Messaging host (``ai.zelyr.browser_host``)
over Chromium's 4-byte length-prefixed stdin/stdout stdio channel paired with a
user-private ``AF_UNIX`` socket (``0600``, UID-checked) for local Zelyr tool dispatch.

Security boundaries:
- No unrestricted HTTP control server or TCP listener is opened.
- Validates caller extension origin (``chrome-extension://<32-char-id>/``) against
  explicitly configured ``allowed_origins``.
- Enforces bounded message framing, action allowlists (disabling ``evaluate_script``
  by default unless explicitly opted in), URL scheme checks (``http``, ``https``,
  ``about:blank`` only), cloud-metadata blocking, and host allowlist/denylist rules.
"""

from __future__ import annotations

import asyncio
import hashlib
import ipaddress
import json
import logging
import os
import re
import socket
import stat
import struct
import sys
import threading
import urllib.parse
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Dict, Iterable, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

NATIVE_HOST_NAME = "ai.zelyr.browser_host"
MAX_NATIVE_MESSAGE_BYTES = 1024 * 1024  # 1 MiB strict frame limit
MAX_SELECTOR_LEN = 1024
MAX_TEXT_LEN = 64 * 1024
MAX_URL_LEN = 4096

EXTENSION_ORIGIN_RE = re.compile(r"^chrome-extension://([a-p]{32})/?$")
VALID_SCHEMES = frozenset({"http", "https"})

ALLOWED_NATIVE_ACTIONS = frozenset({
    "ping",
    "get_active_tab",
    "list_tabs",
    "tabs",
    "switch_tab",
    "switch",
    "open_tab",
    "new_tab",
    "navigate_url",
    "navigate",
    "close_tab",
    "extract_page_content",
    "get_dom",
    "get_content",
    "content",
    "get_page_snapshot",
    "snapshot",
    "click",
    "click_element",
    "type",
    "fill_element",
    "scroll_page",
    "scroll",
    "capture_tab_screenshot",
    "screenshot",
    "reload_tab",
    "duplicate_tab",
    "pin_tab",
    "manage_browser_tabs",
})

METADATA_HOSTS = frozenset({
    "169.254.169.254",
    "metadata.google.internal",
    "metadata.internal",
    "metadata",
    "instance-data",
})


@dataclass
class NativeBrowserResult:
    """Structured result returned by ``NativeBrowserClient.send_command``."""

    success: bool
    output: Dict[str, Any]
    error: Optional[str] = None
    remedy_hint: Optional[str] = None
    allowed_actions: Optional[List[str]] = None

    def get(self, key: str, default: Any = None) -> Any:
        return self.output.get(key, default)

    def __contains__(self, key: str) -> bool:
        return key in self.output

    def __getitem__(self, key: str) -> Any:
        return self.output[key]


def default_extension_dir() -> Path:
    """Return path to Zelyr's bundled MV3 WebExtension directory."""
    return Path(__file__).resolve().parent / "extension"


def compute_unpacked_extension_id(extension_dir: Optional[Path] = None) -> str:
    """Compute Chromium's deterministic 32-char unpacked extension ID from path."""
    target = Path(extension_dir) if extension_dir is not None else default_extension_dir()
    real_path = os.path.realpath(str(target))
    digest = hashlib.sha256(real_path.encode("utf-8")).digest()[:16]
    return "".join(chr(ord("a") + (b >> 4)) + chr(ord("a") + (b & 0x0F)) for b in digest)


def default_native_socket_path() -> Path:
    """Return the user-private AF_UNIX socket path for the native messaging host."""
    override = os.environ.get("ZELYR_BROWSER_NATIVE_SOCKET", "").strip()
    if override:
        return Path(os.path.expanduser(override))
    runtime_dir = os.environ.get("XDG_RUNTIME_DIR", "").strip()
    if runtime_dir:
        return Path(runtime_dir) / "zelyr" / "browser_native_host.sock"
    data_home = os.environ.get("XDG_DATA_HOME", "").strip()
    base = Path(os.path.expanduser(data_home)) if data_home else Path.home() / ".local" / "share"
    return base / "zelyr" / "run" / "browser_native_host.sock"


def default_chromium_hosts_dir(user_data_dir: Optional[Path] = None) -> Path:
    """Return Chromium's NativeMessagingHosts directory for current user or profile."""
    if user_data_dir is not None:
        return Path(user_data_dir).expanduser() / "NativeMessagingHosts"
    xdg_cfg = os.environ.get("XDG_CONFIG_HOME", "").strip()
    cfg_base = Path(os.path.expanduser(xdg_cfg)) if xdg_cfg else Path.home() / ".config"
    return cfg_base / "chromium" / "NativeMessagingHosts"


def default_host_wrapper_path() -> Path:
    """Return path to executable launcher script referenced by Chromium's host manifest."""
    data_home = os.environ.get("XDG_DATA_HOME", "").strip()
    base = Path(os.path.expanduser(data_home)) if data_home else Path.home() / ".local" / "share"
    return base / "zelyr" / "bin" / "zelyr-browser-host"


def _normalize_origin(origin: str) -> str:
    m = EXTENSION_ORIGIN_RE.match(origin.strip())
    if not m:
        return ""
    return f"chrome-extension://{m.group(1)}/"


def _host_matches_pattern(hostname: str, pattern: str) -> bool:
    host = hostname.strip().lower().rstrip(".")
    pat = pattern.strip().lower().rstrip(".")
    if not pat:
        return False
    if pat.startswith("*."):
        suffix = pat[1:]  # '.example.com'
        return host == pat[2:] or host.endswith(suffix)
    if pat.startswith("."):
        return host == pat[1:] or host.endswith(pat)
    return host == pat


def _parse_env_host_list(var_name: str) -> List[str]:
    raw = os.environ.get(var_name, "").strip()
    if not raw:
        return []
    return [part.strip().lower() for part in raw.split(",") if part.strip()]


def load_manifest_allowed_origins(manifest_path: Optional[Path] = None) -> List[str]:
    """Load allowed chrome-extension origins from Chromium host manifest and env."""
    origins: List[str] = []
    env_origins = os.environ.get("ZELYR_BROWSER_ALLOWED_ORIGINS", "").strip()
    if env_origins:
        for item in env_origins.split(","):
            norm = _normalize_origin(item)
            if norm and norm not in origins:
                origins.append(norm)
    env_ids = os.environ.get("ZELYR_BROWSER_ALLOWED_EXTENSION_IDS", "").strip()
    if env_ids:
        for ext_id in env_ids.split(","):
            norm = _normalize_origin(f"chrome-extension://{ext_id.strip()}/")
            if norm and norm not in origins:
                origins.append(norm)

    candidate_paths: List[Path] = []
    if manifest_path is not None:
        candidate_paths.append(Path(manifest_path))
    else:
        env_manifest = os.environ.get("ZELYR_BROWSER_NATIVE_MANIFEST", "").strip()
        if env_manifest:
            candidate_paths.append(Path(os.path.expanduser(env_manifest)))
        candidate_paths.append(default_chromium_hosts_dir() / f"{NATIVE_HOST_NAME}.json")
        candidate_paths.append(Path("/etc/chromium/native-messaging-hosts") / f"{NATIVE_HOST_NAME}.json")

    for cand in candidate_paths:
        if cand.is_file():
            try:
                data = json.loads(cand.read_text(encoding="utf-8"))
                for item in data.get("allowed_origins") or []:
                    if isinstance(item, str):
                        norm = _normalize_origin(item)
                        if norm and norm not in origins:
                            origins.append(norm)
            except Exception as exc:
                logger.warning("Failed reading native messaging manifest %s: %s", cand, exc)
    return origins


def validate_extension_origin(
    origin: Optional[str],
    allowed_origins: Optional[Iterable[str]] = None,
) -> Tuple[bool, str]:
    """Validate Chrome extension caller origin against format rules and allowlist."""
    if not origin or not isinstance(origin, str):
        return False, "Missing Chrome extension origin argument."
    normalized = _normalize_origin(origin)
    if not normalized:
        return False, f"Malformed extension origin '{origin}'. Expected chrome-extension://<32-char-id>/."

    if allowed_origins is None:
        allowed_list = load_manifest_allowed_origins()
    else:
        allowed_list = [norm for norm in (_normalize_origin(str(o)) for o in allowed_origins) if norm]

    if not allowed_list:
        return False, "No allowed extension origins configured (fail closed)."
    if normalized not in allowed_list:
        return False, f"Extension origin '{normalized}' is not in allowed_origins."
    return True, normalized


def validate_browser_url(
    url: str,
    allowed_hosts: Optional[Iterable[str]] = None,
    denied_hosts: Optional[Iterable[str]] = None,
) -> Tuple[bool, str]:
    """Validate URL scheme and host against metadata, denylist, and allowlist policies."""
    raw = str(url or "").strip()
    if not raw:
        return False, "Empty URL is not allowed."
    if len(raw) > MAX_URL_LEN:
        return False, f"URL exceeds maximum length ({MAX_URL_LEN} chars)."
    if any(ord(ch) < 32 for ch in raw):
        return False, "URL contains illegal control characters."
    if raw.lower() == "about:blank":
        return True, raw

    candidate = raw if "://" in raw else f"https://{raw}"
    # Catch explicit non-http schemes like javascript: or data: without '://'
    scheme_prefix = raw.split(":", 1)[0].lower() if ":" in raw else ""
    if scheme_prefix and "://" not in raw and not raw.split(":", 1)[1].startswith("//"):
        if scheme_prefix not in VALID_SCHEMES:
            return False, f"Blocked URL scheme '{scheme_prefix}:'. Only http://, https://, and about:blank are allowed."

    parsed = urllib.parse.urlparse(candidate)
    if parsed.scheme.lower() not in VALID_SCHEMES:
        return False, f"Blocked URL scheme '{parsed.scheme}'. Only http and https are allowed."

    hostname = (parsed.hostname or "").strip().lower().rstrip(".")
    if not hostname:
        return False, f"URL '{raw}' is missing a valid hostname."

    if hostname in METADATA_HOSTS or hostname.endswith(".google.internal"):
        return False, f"Blocked cloud metadata host '{hostname}'."
    try:
        ip_obj = ipaddress.ip_address(hostname.strip("[]"))
        if ip_obj.is_link_local:
            return False, f"Blocked link-local metadata IP '{ip_obj}'."
    except ValueError:
        pass

    eff_denied = (
        [str(h).strip().lower() for h in denied_hosts if str(h).strip()]
        if denied_hosts is not None
        else _parse_env_host_list("ZELYR_BROWSER_DENIED_HOSTS")
    )
    for pat in eff_denied:
        if _host_matches_pattern(hostname, pat):
            return False, f"HOST_DENIED: Host '{hostname}' matches denied_hosts policy '{pat}'."

    eff_allowed = (
        [str(h).strip().lower() for h in allowed_hosts if str(h).strip()]
        if allowed_hosts is not None
        else _parse_env_host_list("ZELYR_BROWSER_ALLOWED_HOSTS")
    )
    if eff_allowed and not any(_host_matches_pattern(hostname, pat) for pat in eff_allowed):
        return False, f"HOST_NOT_ALLOWED: Host '{hostname}' is not in allowed_hosts."

    return True, candidate


def validate_browser_command(
    msg: Any,
    allowed_hosts: Optional[Iterable[str]] = None,
    denied_hosts: Optional[Iterable[str]] = None,
    allow_evaluate_script: Optional[bool] = None,
) -> Tuple[bool, str]:
    """Validate structured browser command payload before forwarding to extension."""
    if not isinstance(msg, dict):
        return False, "Browser command payload must be a JSON object."

    try:
        serialized = json.dumps(msg).encode("utf-8")
    except (TypeError, ValueError) as exc:
        return False, f"Browser command is not JSON-serializable: {exc}"
    if len(serialized) > MAX_NATIVE_MESSAGE_BYTES:
        return False, f"Browser command exceeds {MAX_NATIVE_MESSAGE_BYTES} byte limit."

    cmd_id = msg.get("id")
    if not isinstance(cmd_id, str) or not cmd_id.strip() or len(cmd_id) > 128:
        return False, "Browser command missing valid string 'id'."

    raw_action = msg.get("action")
    if not isinstance(raw_action, str) or not raw_action.strip():
        return False, "Browser command missing valid string 'action'."
    action = raw_action.strip().lower()

    if allow_evaluate_script is None:
        allow_eval = os.environ.get("ZELYR_BROWSER_ALLOW_EVALUATE_SCRIPT", "").strip() == "1"
    else:
        allow_eval = bool(allow_evaluate_script)

    permitted = set(ALLOWED_NATIVE_ACTIONS)
    if allow_eval:
        permitted.add("evaluate_script")
    if action not in permitted:
        return False, f"Action '{action}' is not permitted by native host policy."

    if action in ("open_tab", "new_tab", "navigate_url", "navigate"):
        target_url = msg.get("url") or msg.get("query") or msg.get("target")
        if action in ("open_tab", "new_tab") and not target_url:
            target_url = "about:blank"
        if not isinstance(target_url, str) or not target_url.strip():
            return False, f"Action '{action}' requires a valid 'url' parameter."
        ok_url, url_err = validate_browser_url(target_url, allowed_hosts=allowed_hosts, denied_hosts=denied_hosts)
        if not ok_url:
            return False, url_err

    if action == "manage_browser_tabs" and msg.get("url"):
        ok_url, url_err = validate_browser_url(str(msg["url"]), allowed_hosts=allowed_hosts, denied_hosts=denied_hosts)
        if not ok_url:
            return False, url_err

    if action in ("click", "type"):
        sel = msg.get("selector")
        if not isinstance(sel, str) or not sel.strip():
            return False, f"Action '{action}' requires a non-empty string 'selector'."
        if len(sel) > MAX_SELECTOR_LEN:
            return False, f"Selector exceeds {MAX_SELECTOR_LEN} char limit."

    if action in ("click_element", "fill_element"):
        elem_id = msg.get("element_id")
        try:
            parsed_id = int(elem_id)
            if parsed_id < 0:
                return False, "element_id must be a non-negative integer."
        except (TypeError, ValueError):
            return False, f"Action '{action}' requires an integer 'element_id'."

    if "text" in msg and msg["text"] is not None:
        if not isinstance(msg["text"], str) or len(msg["text"]) > MAX_TEXT_LEN:
            return False, f"'text' parameter must be a string <= {MAX_TEXT_LEN} chars."

    return True, "ok"


def read_native_message(stream: BinaryIO, max_bytes: int = MAX_NATIVE_MESSAGE_BYTES) -> Optional[Dict[str, Any]]:
    """Read one 4-byte little-endian length-prefixed JSON message from Native Messaging stream."""
    header = stream.read(4)
    if not header:
        return None
    if len(header) < 4:
        raise ValueError("Truncated Native Messaging header.")
    (length,) = struct.unpack("<I", header)
    if length == 0 or length > max_bytes:
        raise ValueError(f"Invalid Native Messaging frame size: {length} bytes (limit={max_bytes}).")
    body = stream.read(length)
    if len(body) < length:
        raise ValueError("Truncated Native Messaging body.")
    parsed = json.loads(body.decode("utf-8"))
    if not isinstance(parsed, dict):
        raise ValueError("Native Messaging payload must decode to a JSON object.")
    return parsed


def write_native_message(stream: BinaryIO, payload: Dict[str, Any], max_bytes: int = MAX_NATIVE_MESSAGE_BYTES) -> None:
    """Write one 4-byte little-endian length-prefixed JSON message to Native Messaging stream."""
    encoded = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    if len(encoded) > max_bytes:
        raise ValueError(f"Outgoing Native Messaging payload exceeds {max_bytes} byte limit.")
    stream.write(struct.pack("<I", len(encoded)))
    stream.write(encoded)
    stream.flush()


def install_native_messaging_host(
    *,
    extension_dir: Optional[Path] = None,
    extension_id: Optional[str] = None,
    allowed_origins: Optional[Sequence[str]] = None,
    hosts_dir: Optional[Path] = None,
    user_data_dir: Optional[Path] = None,
    wrapper_path: Optional[Path] = None,
    allowed_hosts: Optional[Sequence[str]] = None,
    python_executable: Optional[str] = None,
) -> Dict[str, Any]:
    """Install Chromium NativeMessagingHosts manifest and user-scoped launcher script."""
    ext_dir = Path(extension_dir) if extension_dir is not None else default_extension_dir()
    ext_id = (extension_id or compute_unpacked_extension_id(ext_dir)).strip().lower()
    if allowed_origins:
        origins = [norm for norm in (_normalize_origin(o) for o in allowed_origins) if norm]
    else:
        origins = [f"chrome-extension://{ext_id}/"]

    if not origins:
        raise ValueError("At least one valid chrome-extension://<id>/ origin is required.")

    target_hosts_dir = Path(hosts_dir) if hosts_dir is not None else default_chromium_hosts_dir(user_data_dir)
    target_hosts_dir.mkdir(parents=True, exist_ok=True)
    manifest_file = target_hosts_dir / f"{NATIVE_HOST_NAME}.json"

    launcher = Path(wrapper_path) if wrapper_path is not None else default_host_wrapper_path()
    launcher.parent.mkdir(parents=True, exist_ok=True)
    py_exe = python_executable or sys.executable
    repo_root = str(Path(__file__).resolve().parents[2])

    env_exports = [
        f'export PYTHONPATH="{repo_root}${{PYTHONPATH:+:$PYTHONPATH}}"',
        f'export ZELYR_BROWSER_NATIVE_MANIFEST="{manifest_file}"',
    ]
    if allowed_hosts:
        joined = ",".join(str(h).strip().lower() for h in allowed_hosts if str(h).strip())
        if joined:
            env_exports.append(f'export ZELYR_BROWSER_ALLOWED_HOSTS="{joined}"')

    script_lines = [
        "#!/bin/sh",
        *env_exports,
        f'exec "{py_exe}" -m zelyr.tools.browser_native_host "$@"',
        "",
    ]
    launcher.write_text("\n".join(script_lines), encoding="utf-8")
    os.chmod(launcher, 0o755)

    manifest_data = {
        "name": NATIVE_HOST_NAME,
        "description": "Zelyr Local Browser Native Messaging Host",
        "path": str(launcher),
        "type": "stdio",
        "allowed_origins": origins,
    }
    manifest_file.write_text(json.dumps(manifest_data, indent=2) + "\n", encoding="utf-8")
    os.chmod(manifest_file, 0o644)

    return {
        "host_name": NATIVE_HOST_NAME,
        "extension_dir": str(ext_dir),
        "extension_id": ext_id,
        "allowed_origins": origins,
        "manifest_path": str(manifest_file),
        "launcher_path": str(launcher),
    }


def get_native_host_status(
    *,
    user_data_dir: Optional[Path] = None,
    socket_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Inspect bundled extension, installed manifest, and active native socket status."""
    ext_dir = default_extension_dir()
    ext_id = compute_unpacked_extension_id(ext_dir)
    manifest_path = default_chromium_hosts_dir(user_data_dir) / f"{NATIVE_HOST_NAME}.json"
    sock = Path(socket_path) if socket_path is not None else default_native_socket_path()
    socket_present = sock.exists() and stat.S_ISSOCK(sock.stat().st_mode)
    return {
        "host_name": NATIVE_HOST_NAME,
        "extension_dir": str(ext_dir),
        "extension_exists": ext_dir.is_dir(),
        "extension_id": ext_id,
        "manifest_path": str(manifest_path),
        "manifest_installed": manifest_path.is_file(),
        "allowed_origins": load_manifest_allowed_origins(manifest_path) if manifest_path.is_file() else [],
        "socket_path": str(sock),
        "socket_present": socket_present,
    }


class NativeBrowserClient:
    """Async AF_UNIX client connecting Zelyr tools to a live BrowserNativeHost process."""

    def __init__(self, socket_path: Optional[Path] = None):
        self.socket_path = Path(socket_path) if socket_path is not None else default_native_socket_path()

    def is_socket_present(self) -> bool:
        """Fast filesystem check for active AF_UNIX socket."""
        try:
            st = self.socket_path.stat()
            return stat.S_ISSOCK(st.st_mode)
        except OSError:
            return False

    async def is_available(self, timeout: float = 0.5) -> bool:
        """Verify live responsiveness of the BrowserNativeHost over AF_UNIX."""
        if not self.is_socket_present():
            return False
        try:
            res = await self.send_command("ping", timeout=timeout)
            return bool(res.success)
        except Exception:
            return False

    async def send_command(
        self,
        action: str,
        timeout: float = 10.0,
        **params: Any,
    ) -> NativeBrowserResult:
        """Send one validated browser action over AF_UNIX to BrowserNativeHost."""
        if not self.is_socket_present():
            return NativeBrowserResult(
                success=False,
                output={"status": "error", "error": f"Native browser socket not found: {self.socket_path}"},
                error=f"Native browser socket not found: {self.socket_path}",
            )

        req_id = f"native_{uuid.uuid4().hex[:12]}"
        payload: Dict[str, Any] = {"id": req_id, "action": action, **params}

        try:
            reader, writer = await asyncio.wait_for(
                asyncio.open_unix_connection(str(self.socket_path)),
                timeout=min(timeout, 2.0),
            )
        except Exception as exc:
            return NativeBrowserResult(
                success=False,
                output={"status": "error", "error": f"Failed to connect native browser socket: {exc}"},
                error=f"Failed to connect native browser socket: {exc}",
            )

        try:
            writer.write((json.dumps(payload) + "\n").encode("utf-8"))
            await writer.drain()
            raw_line = await asyncio.wait_for(reader.readline(), timeout=timeout)
            if not raw_line:
                return NativeBrowserResult(
                    success=False,
                    output={"status": "error", "error": "Native browser host closed connection without reply."},
                    error="Native browser host closed connection without reply.",
                )
            resp = json.loads(raw_line.decode("utf-8"))
            ok = bool(resp.get("success", resp.get("status") != "error"))
            return NativeBrowserResult(
                success=ok,
                output=resp,
                error=resp.get("error"),
                remedy_hint=resp.get("remedy_hint"),
                allowed_actions=resp.get("allowed_actions"),
            )
        except asyncio.TimeoutError:
            err = f"Native browser command '{action}' timed out after {timeout}s."
            return NativeBrowserResult(success=False, output={"status": "error", "error": err}, error=err)
        except Exception as exc:
            err = f"Native browser IPC failure for '{action}': {exc}"
            return NativeBrowserResult(success=False, output={"status": "error", "error": err}, error=err)
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:
                pass


class BrowserNativeHost:
    """Chromium Native Messaging host process bridging stdio frames to user-private AF_UNIX."""

    def __init__(
        self,
        *,
        origin: str,
        allowed_origins: Optional[Iterable[str]] = None,
        socket_path: Optional[Path] = None,
        allowed_hosts: Optional[Iterable[str]] = None,
        denied_hosts: Optional[Iterable[str]] = None,
        allow_evaluate_script: Optional[bool] = None,
        stdin_stream: Optional[BinaryIO] = None,
        stdout_stream: Optional[BinaryIO] = None,
    ) -> None:
        ok, detail = validate_extension_origin(origin, allowed_origins=allowed_origins)
        if not ok:
            raise PermissionError(detail)
        self.origin = detail
        self.socket_path = Path(socket_path) if socket_path is not None else default_native_socket_path()
        self.allowed_hosts = list(allowed_hosts) if allowed_hosts is not None else None
        self.denied_hosts = list(denied_hosts) if denied_hosts is not None else None
        self.allow_evaluate_script = allow_evaluate_script
        self._stdin: BinaryIO = stdin_stream or sys.stdin.buffer
        self._stdout: BinaryIO = stdout_stream or sys.stdout.buffer
        self._write_lock = threading.Lock()
        self._pending: Dict[str, asyncio.Future] = {}
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._server: Optional[asyncio.AbstractServer] = None
        self._shutdown_event: Optional[asyncio.Event] = None

    def _send_to_browser(self, payload: Dict[str, Any]) -> None:
        with self._write_lock:
            write_native_message(self._stdout, payload)

    def _stdin_reader_worker(self) -> None:
        try:
            while True:
                msg = read_native_message(self._stdin)
                if msg is None:
                    break
                if self._loop is None:
                    continue
                if msg.get("type") == "ping" or (msg.get("action") == "ping" and msg.get("id") not in self._pending):
                    try:
                        self._send_to_browser({"id": msg.get("id", "ping"), "action": "pong", "success": True})
                    except Exception:
                        pass
                    continue
                msg_id = str(msg.get("id") or "")
                self._loop.call_soon_threadsafe(self._resolve_pending, msg_id, msg)
        except Exception as exc:
            logger.debug("Native stdin reader terminated: %s", exc)
        finally:
            if self._loop is not None and self._shutdown_event is not None:
                self._loop.call_soon_threadsafe(self._shutdown_event.set)

    def _resolve_pending(self, msg_id: str, msg: Dict[str, Any]) -> None:
        fut = self._pending.pop(msg_id, None)
        if fut is not None and not fut.done():
            fut.set_result(msg)

    async def handle_local_request(self, req: Dict[str, Any]) -> Dict[str, Any]:
        """Validate one local AF_UNIX request and dispatch to the connected Chromium extension."""
        ok, reason = validate_browser_command(
            req,
            allowed_hosts=self.allowed_hosts,
            denied_hosts=self.denied_hosts,
            allow_evaluate_script=self.allow_evaluate_script,
        )
        if not ok:
            return {
                "id": str(req.get("id") or ""),
                "success": False,
                "status": "error",
                "error_type": "POLICY_DENIED",
                "error": reason,
            }

        cmd_id = str(req["id"])
        assert self._loop is not None
        fut = self._loop.create_future()
        self._pending[cmd_id] = fut
        try:
            self._send_to_browser(req)
            reply = await asyncio.wait_for(fut, timeout=15.0)
            return reply
        except asyncio.TimeoutError:
            self._pending.pop(cmd_id, None)
            return {
                "id": cmd_id,
                "success": False,
                "status": "error",
                "error": f"Timed out awaiting extension response for '{req.get('action')}'.",
            }
        except Exception as exc:
            self._pending.pop(cmd_id, None)
            return {
                "id": cmd_id,
                "success": False,
                "status": "error",
                "error": f"Failed forwarding command to browser extension: {exc}",
            }

    @staticmethod
    def _verify_peer_uid(writer: asyncio.StreamWriter) -> bool:
        """Verify connecting AF_UNIX peer shares current OS user UID (Linux SO_PEERCRED)."""
        sock = writer.get_extra_info("socket")
        if sock is None or not hasattr(socket, "SO_PEERCRED"):
            return True
        try:
            creds = sock.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, struct.calcsize("3i"))
            _, uid, _ = struct.unpack("3i", creds)
            return uid == os.getuid()
        except Exception:
            return False

    async def _handle_unix_client(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        try:
            if not self._verify_peer_uid(writer):
                err = {"success": False, "status": "error", "error": "Peer UID verification failed."}
                writer.write((json.dumps(err) + "\n").encode("utf-8"))
                await writer.drain()
                return

            raw_line = await asyncio.wait_for(reader.readline(), timeout=5.0)
            if not raw_line:
                return
            if len(raw_line) > MAX_NATIVE_MESSAGE_BYTES:
                err = {"success": False, "status": "error", "error": "Command exceeds 1 MiB frame cap."}
                writer.write((json.dumps(err) + "\n").encode("utf-8"))
                await writer.drain()
                return
            req = json.loads(raw_line.decode("utf-8"))
            resp = await self.handle_local_request(req)
            writer.write((json.dumps(resp) + "\n").encode("utf-8"))
            await writer.drain()
        except Exception as exc:
            try:
                err = {"success": False, "status": "error", "error": f"Malformed local browser IPC request: {exc}"}
                writer.write((json.dumps(err) + "\n").encode("utf-8"))
                await writer.drain()
            except Exception:
                pass
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:
                pass

    async def run(self) -> None:
        """Start AF_UNIX listener and stdio Native Messaging relay until stdin EOF."""
        self._loop = asyncio.get_running_loop()
        self._shutdown_event = asyncio.Event()

        self.socket_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.chmod(self.socket_path.parent, 0o700)
        except OSError:
            pass
        if self.socket_path.exists():
            self.socket_path.unlink()

        self._server = await asyncio.start_unix_server(
            self._handle_unix_client,
            path=str(self.socket_path),
        )
        os.chmod(self.socket_path, 0o600)

        reader_thread = threading.Thread(target=self._stdin_reader_worker, daemon=True)
        reader_thread.start()

        try:
            await self._shutdown_event.wait()
        finally:
            for fut in self._pending.values():
                if not fut.done():
                    fut.set_exception(ConnectionResetError("Native messaging host shutting down."))
            self._pending.clear()
            if self._server is not None:
                self._server.close()
                await self._server.wait_closed()
            if self.socket_path.exists():
                try:
                    self.socket_path.unlink()
                except OSError:
                    pass


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point invoked directly by Chromium/Chrome Native Messaging."""
    args = list(sys.argv[1:] if argv is None else argv)
    origin = args[0] if args else os.environ.get("ZELYR_NATIVE_ORIGIN", "")
    try:
        host = BrowserNativeHost(origin=origin)
    except PermissionError as exc:
        sys.stderr.write(f"Native host origin denied: {exc}\n")
        return 2

    try:
        asyncio.run(host.run())
        return 0
    except KeyboardInterrupt:
        return 0
    except Exception as exc:
        sys.stderr.write(f"Native host error: {exc}\n")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
