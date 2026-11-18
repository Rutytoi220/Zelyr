import hashlib
import logging
import os
import re
import shutil
import sys
import tempfile
import time
import json
import importlib.util
from pathlib import Path
from typing import Any, Dict, List, Callable, Awaitable, Optional, Set

logger = logging.getLogger(__name__)

_ANSI_RE = re.compile(r"\x1b(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")


def _strip_ansi_codes(text: str) -> str:
    """Strip ANSI escape sequences from text while preserving formatting."""
    if not text:
        return ""
    return _ANSI_RE.sub("", text)


strip_ansi_codes = _strip_ansi_codes

HISTORICAL_TEMPLATE_HASHES: Dict[str, Set[str]] = {
    "execute_command": {
        "2a13ac516ff055002b3ca71b13398ee39185d38c155f051c85f77c8c7034bbdf",  # v1 (subprocess.run baseline, escaped)
        "015fc4c94b4085015605b9f04f33ade5ead868b928d1879e2ead3d41f1ebe7ac",  # v1 (subprocess.run baseline, unescaped)
        "ead3aa6b79254f2a587b991c314262683641b3f39a18a6ce3ae25d9476d62627",  # v2 (asyncio.create_subprocess_exec Day 2)
        "f06413f020b18cd892ea1c3e692f48d1527ee1691419963bee230f3aa04a263c",  # v3 (PTY and non-interactive isolation)
    },
}

_EXECUTE_COMMAND_TEMPLATE = (
    "import asyncio\n"
    "import os\n"
    "import re\n"
    "import signal\n"
    "try:\n"
    "    import pty as _pty_mod\n"
    "except ImportError:\n"
    "    _pty_mod = None\n\n"
    "IS_CORE = True\n"
    "TIER = 0\n"
    "TOOL_SCHEMA = {\n"
    '    "type": "function",\n'
    '    "function": {\n'
    '        "name": "execute_command",\n'
    '        "description": "Executes a shell command via bash and returns stdout and stderr.",\n'
    '        "parameters": {\n'
    '            "type": "object",\n'
    '            "properties": {\n'
    '                "command": {"type": "string", "description": "Command to execute"},\n'
    '                "cwd": {"type": "string", "description": "Optional working directory for execution"},\n'
    '                "pty": {\n'
    '                    "type": "boolean",\n'
    '                    "description": "Execute inside a pseudo-terminal (TTY). Enable for commands requiring a controlling terminal or checking isatty().",\n'
    '                    "default": False,\n'
    '                },\n'
    '            },\n'
    '            "required": ["command"],\n'
    '        },\n'
    '    },\n'
    "}\n"
    'TUI_HINT = "⚙️ Executing system command..."\n'
    "REQUIRED_RING = 0\n\n"
    '_ANSI_RE = re.compile(r"\\x1b(?:[@-Z\\\\-_]|\\[[0-?]*[ -/]*[@-~])")\n\n'
    "def _strip_ansi_codes(text: str) -> str:\n"
    '    """Strip ANSI escape sequences from text while preserving formatting."""\n'
    "    if not text:\n"
    '        return ""\n'
    '    return _ANSI_RE.sub("", text)\n\n'
    "async def execute(command: str = '', timeout: float = 60.0, pty: bool = False, cwd: str = None, **kwargs) -> str:\n"
    "    if cwd is None:\n"
    '        cwd = kwargs.get("cwd")\n'
    "    env = os.environ.copy()\n"
    '    env.setdefault("PAGER", "cat")\n'
    '    env.setdefault("TERM", "dumb")\n'
    '    env.setdefault("DEBIAN_FRONTEND", "noninteractive")\n'
    '    env.setdefault("CI", "1")\n'
    '    env.setdefault("PYTHONUNBUFFERED", "1")\n\n'
    "    max_bytes = 2 * 1024 * 1024\n"
    "    use_pty = bool(pty and _pty_mod is not None and hasattr(_pty_mod, 'openpty'))\n\n"
    "    if use_pty and _pty_mod is not None:\n"
    "        master_fd, slave_fd = _pty_mod.openpty()\n"
    "        try:\n"
    "            proc = await asyncio.create_subprocess_exec(\n"
    "                'bash', '-c', command,\n"
    "                stdin=slave_fd,\n"
    "                stdout=slave_fd,\n"
    "                stderr=slave_fd,\n"
    "                start_new_session=True,\n"
    "                env=env,\n"
    "                cwd=cwd,\n"
    "            )\n"
    "        except Exception as e:\n"
    "            try:\n"
    "                os.close(slave_fd)\n"
    "            except OSError:\n"
    "                pass\n"
    "            try:\n"
    "                os.close(master_fd)\n"
    "            except OSError:\n"
    "                pass\n"
    '            return f"Error executing command: {e}"\n'
    "        finally:\n"
    "            try:\n"
    "                os.close(slave_fd)\n"
    "            except OSError:\n"
    "                pass\n\n"
    "        async def _cleanup_pty_proc():\n"
    "            has_pg = hasattr(os, 'killpg') and hasattr(os, 'getpgid')\n"
    "            try:\n"
    "                if has_pg:\n"
    "                    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)\n"
    "                else:\n"
    "                    proc.terminate()\n"
    "            except (ProcessLookupError, OSError):\n"
    "                pass\n\n"
    "            try:\n"
    "                await asyncio.wait_for(proc.wait(), timeout=1.5)\n"
    "            except (asyncio.TimeoutError, TimeoutError):\n"
    "                try:\n"
    "                    if has_pg:\n"
    "                        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)\n"
    "                    else:\n"
    "                        proc.kill()\n"
    "                except (ProcessLookupError, OSError):\n"
    "                    pass\n"
    "                try:\n"
    "                    await proc.wait()\n"
    "                except Exception:\n"
    "                    pass\n\n"
    "        buf = bytearray()\n"
    "        loop = asyncio.get_running_loop()\n"
    "        reader_future = loop.create_future()\n"
    "        os.set_blocking(master_fd, False)\n\n"
    "        def _on_pty_readable():\n"
    "            try:\n"
    "                chunk = os.read(master_fd, 65536)\n"
    "                if chunk:\n"
    "                    if len(buf) < max_bytes:\n"
    "                        buf.extend(chunk[:max_bytes - len(buf)])\n"
    "                else:\n"
    "                    loop.remove_reader(master_fd)\n"
    "                    if not reader_future.done():\n"
    "                        reader_future.set_result(None)\n"
    "            except (BlockingIOError, InterruptedError):\n"
    "                pass\n"
    "            except (OSError, Exception):\n"
    "                loop.remove_reader(master_fd)\n"
    "                if not reader_future.done():\n"
    "                    reader_future.set_result(None)\n\n"
    "        loop.add_reader(master_fd, _on_pty_readable)\n"
    "        try:\n"
    "            try:\n"
    "                await asyncio.wait_for(\n"
    "                    asyncio.gather(reader_future, proc.wait()),\n"
    "                    timeout=timeout,\n"
    "                )\n"
    "            except (asyncio.TimeoutError, TimeoutError):\n"
    "                await _cleanup_pty_proc()\n"
    '                return f"Exit code 124\\nError executing command: Command \'{command}\' timed out after {timeout} seconds"\n'
    "            except (asyncio.CancelledError, BaseException):\n"
    "                await _cleanup_pty_proc()\n"
    "                raise\n"
    "        finally:\n"
    "            try:\n"
    "                loop.remove_reader(master_fd)\n"
    "            except Exception:\n"
    "                pass\n"
    "            try:\n"
    "                os.close(master_fd)\n"
    "            except OSError:\n"
    "                pass\n\n"
    "        raw_out = bytes(buf).decode('utf-8', errors='replace')\n"
    "        body = _strip_ansi_codes(raw_out)\n"
    '        return f"Exit code {proc.returncode}\\n{body}".strip() if body else f"Exit code {proc.returncode}"\n\n'
    "    start_new_session = hasattr(os, 'setsid')\n"
    "    try:\n"
    "        proc = await asyncio.create_subprocess_exec(\n"
    "            'bash', '-c', command,\n"
    "            stdin=asyncio.subprocess.DEVNULL,\n"
    "            stdout=asyncio.subprocess.PIPE,\n"
    "            stderr=asyncio.subprocess.PIPE,\n"
    "            start_new_session=start_new_session,\n"
    "            env=env,\n"
    "            cwd=cwd,\n"
    "        )\n"
    "    except Exception as e:\n"
    '        return f"Error executing command: {e}"\n\n'
    "    async def _read_stream(stream):\n"
    "        buf = bytearray()\n"
    "        try:\n"
    "            while True:\n"
    "                chunk = await stream.read(65536)\n"
    "                if not chunk:\n"
    "                    break\n"
    "                if len(buf) < max_bytes:\n"
    "                    buf.extend(chunk[:max_bytes - len(buf)])\n"
    "        except (asyncio.CancelledError, Exception):\n"
    "            pass\n"
    "        return bytes(buf)\n\n"
    "    async def _cleanup_proc():\n"
    "        has_pg = hasattr(os, 'killpg') and hasattr(os, 'getpgid')\n"
    "        try:\n"
    "            if has_pg:\n"
    "                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)\n"
    "            else:\n"
    "                proc.terminate()\n"
    "        except (ProcessLookupError, OSError):\n"
    "            pass\n\n"
    "        try:\n"
    "            await asyncio.wait_for(proc.wait(), timeout=1.5)\n"
    "        except (asyncio.TimeoutError, TimeoutError):\n"
    "            try:\n"
    "                if has_pg:\n"
    "                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)\n"
    "                else:\n"
    "                    proc.kill()\n"
    "            except (ProcessLookupError, OSError):\n"
    "                pass\n"
    "            try:\n"
    "                await proc.wait()\n"
    "            except Exception:\n"
    "                pass\n\n"
    "    try:\n"
    "        stdout_bytes, stderr_bytes = await asyncio.wait_for(\n"
    "            asyncio.gather(_read_stream(proc.stdout), _read_stream(proc.stderr)),\n"
    "            timeout=timeout,\n"
    "        )\n"
    "        await proc.wait()\n"
    "        out = _strip_ansi_codes(stdout_bytes.decode('utf-8', errors='replace'))\n"
    "        err = _strip_ansi_codes(stderr_bytes.decode('utf-8', errors='replace'))\n"
    "        body = (out or '') + (err or '')\n"
    '        return f"Exit code {proc.returncode}\\n{body}".strip() if body else f"Exit code {proc.returncode}"\n'
    "    except (asyncio.TimeoutError, TimeoutError):\n"
    "        await _cleanup_proc()\n"
    '        return f"Exit code 124\\nError executing command: Command \'{command}\' timed out after {timeout} seconds"\n'
    "    except (asyncio.CancelledError, BaseException):\n"
    "        await _cleanup_proc()\n"
    "        raise\n"
)

def safe_provision_tool(
    target_path: Path,
    canonical_code: str,
    historical_hashes: Optional[Set[str]] = None,
) -> bool:
    """Safely provision or upgrade a tool template in TOOLS_DIR with strict hash provenance.

    Rules:
    - If target_path does not exist: write canonical template atomically.
    - If target_path is a symlink or not a regular file: log warning, DO NOT modify/overwrite, treat as user-managed.
    - If target_path exists: compute complete SHA-256 hash.
      * If hash matches canonical: no action needed.
      * If hash matches an explicitly recognized historical generated template hash:
        a. Create timestamped backup: <filename>.py.bak.<timestamp>
        b. Preserve original POSIX file permissions (os.stat(target_path).st_mode).
        c. Atomically write new template using tempfile in same dir + os.replace.
        d. Wrap in try...except: if atomic replacement fails, ensure backup is retained, attempt rollback, raise clear error.
      * If hash is unknown, missing, or user-modified: NEVER overwrite or replace. Treat as user-managed code.
    """
    if target_path.is_symlink() or (target_path.exists() and not target_path.is_file()):
        logger.warning(
            f"Tool path {target_path} is a symlink or non-regular file; skipping provisioning to protect user configuration."
        )
        return False

    if not target_path.exists():
        target_path.parent.mkdir(parents=True, exist_ok=True)
        temp_file = tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=target_path.parent,
            delete=False,
        )
        temp_path = Path(temp_file.name)
        try:
            temp_file.write(canonical_code)
            temp_file.flush()
            os.fsync(temp_file.fileno())
            temp_file.close()
            os.replace(temp_path, target_path)
            return True
        except Exception as exc:
            if temp_path.exists():
                try:
                    temp_path.unlink()
                except Exception:
                    pass
            logger.error(f"Failed to provision tool {target_path}: {exc}")
            return False

    # Target path exists and is a regular file
    try:
        file_bytes = target_path.read_bytes()
        file_hash = hashlib.sha256(file_bytes).hexdigest()
    except Exception as e:
        logger.warning(f"Could not read tool file {target_path} for hash calculation: {e}")
        return False

    canonical_bytes = canonical_code.encode("utf-8")
    canonical_hash = hashlib.sha256(canonical_bytes).hexdigest()

    if file_hash == canonical_hash:
        return True

    tool_name = target_path.stem
    allowed_hashes: Set[str] = set(historical_hashes) if historical_hashes is not None else set()
    allowed_hashes.update(HISTORICAL_TEMPLATE_HASHES.get(tool_name, set()))
    allowed_hashes.add(canonical_hash)

    if file_hash not in allowed_hashes:
        logger.info(
            f"Tool {target_path.name} (hash: {file_hash[:12]}...) does not match any recognized historical template. Treating as user-managed."
        )
        return False

    # Recognized historical hash! Perform safe upgrade.
    timestamp = int(time.time() * 1000)
    backup_path = target_path.with_name(f"{target_path.name}.bak.{timestamp}")
    orig_mode = os.stat(target_path).st_mode

    try:
        shutil.copy2(target_path, backup_path)
    except Exception as exc:
        logger.error(f"Failed to create backup for {target_path}: {exc}")
        raise RuntimeError(f"Backup creation failed for {target_path} before upgrade: {exc}") from exc

    temp_file = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=target_path.parent,
        delete=False,
    )
    temp_path = Path(temp_file.name)
    try:
        temp_file.write(canonical_code)
        temp_file.flush()
        os.fsync(temp_file.fileno())
        temp_file.close()
        os.chmod(temp_path, orig_mode)
        os.replace(temp_path, target_path)
        logger.info(f"Safely upgraded {target_path.name} to current template (backup saved to {backup_path.name}).")
        return True
    except Exception as exc:
        if temp_path.exists():
            try:
                temp_path.unlink()
            except Exception:
                pass
        # Rollback from backup if target_path was damaged
        if not target_path.exists() and backup_path.exists():
            try:
                shutil.copy2(backup_path, target_path)
            except Exception:
                pass
        logger.error(f"Atomic replacement failed for {target_path}: {exc}; backup retained at {backup_path}")
        raise RuntimeError(
            f"Atomic replacement failed for {target_path}; rollback attempted. Backup retained at {backup_path}"
        ) from exc

def get_tools_dir() -> Path:
    """Return the active tools.d directory respecting ZELYR_CONFIG_DIR."""
    from zelyr.config import get_config_dir
    return get_config_dir() / "tools.d"

TOOLS_DIR = get_tools_dir()

_schemas: List[Dict[str, Any]] = []
_rings: Dict[str, int] = {}
_executors: Dict[str, Callable[..., Awaitable[str]]] = {}
_tui_hints: Dict[str, str] = {}
_capabilities: Dict[str, Any] = {}
_plugin_file_paths: Dict[str, Path] = {}

def get_plugin_file_paths() -> Dict[str, Path]:
    """Return a mapping of registered dynamic tool names to their file paths on disk."""
    return dict(_plugin_file_paths)

def load_plugins() -> None:
    global _schemas, _executors, _tui_hints, _capabilities, _plugin_file_paths, TOOLS_DIR
    TOOLS_DIR = get_tools_dir()
    _schemas.clear()
    _rings.clear()
    _executors.clear()
    _tui_hints.clear()
    _capabilities.clear()
    _plugin_file_paths.clear()

    TOOLS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Ensure core semantic memory and Tier 1 OS tools are provisioned
    provisions = [
        (
            TOOLS_DIR / "remember_fact.py",
            (
                "import json\n"
                "from zelyr.tools.memory import RememberFactTool\n\n"
                "_tool = RememberFactTool()\n\n"
                "TOOL_SCHEMA = {\n"
                '    "type": "function",\n'
                '    "function": {\n'
                '        "name": _tool.name,\n'
                '        "description": _tool.description,\n'
                '        "parameters": _tool.schema,\n'
                "    }\n"
                "}\n"
                'TUI_HINT = "🧠 Remembering fact in semantic memory..."\n'
                "REQUIRED_RING = 0\n\n"
                "async def execute(fact: str = '', **kwargs) -> str:\n"
                '    params = {"fact": fact, **kwargs}\n'
                "    res = await _tool.execute(params)\n"
                "    return json.dumps(res.to_dict(tool=_tool.name, arguments=params))\n"
            ),
        ),
        (
            TOOLS_DIR / "manage_system_process.py",
            (
                "import json\n"
                "from zelyr.tools.os_system import ManageSystemProcessTool\n\n"
                "_tool = ManageSystemProcessTool()\n\n"
                "TOOL_SCHEMA = {\n"
                '    "type": "function",\n'
                '    "function": {\n'
                '        "name": _tool.name,\n'
                '        "description": _tool.description,\n'
                '        "parameters": _tool.schema,\n'
                "    }\n"
                "}\n"
                'TUI_HINT = "⚙️ Managing Linux system processes..."\n'
                "REQUIRED_RING = 0\n\n"
                "async def execute(action: str = 'list', target: str = '', signal: int = 15, **kwargs) -> str:\n"
                '    params = {"action": action, "target": target, "signal": signal, **kwargs}\n'
                "    res = await _tool.execute(params)\n"
                "    return json.dumps(res.to_dict(tool=_tool.name, arguments=params))\n"
            ),
        ),
        (
            TOOLS_DIR / "manage_system_clipboard.py",
            (
                "import json\n"
                "from zelyr.tools.os_system import ManageSystemClipboardTool\n\n"
                "_tool = ManageSystemClipboardTool()\n\n"
                "TOOL_SCHEMA = {\n"
                '    "type": "function",\n'
                '    "function": {\n'
                '        "name": _tool.name,\n'
                '        "description": _tool.description,\n'
                '        "parameters": _tool.schema,\n'
                "    }\n"
                "}\n"
                'TUI_HINT = "📋 Managing system clipboard..."\n'
                "REQUIRED_RING = 0\n\n"
                "async def execute(action: str = 'read', content: str = '', **kwargs) -> str:\n"
                '    params = {"action": action, "content": content, **kwargs}\n'
                "    res = await _tool.execute(params)\n"
                "    return json.dumps(res.to_dict(tool=_tool.name, arguments=params))\n"
            ),
        ),
        (
            TOOLS_DIR / "query_system_journal.py",
            (
                "import json\n"
                "from zelyr.tools.os_system import QuerySystemJournalTool\n\n"
                "_tool = QuerySystemJournalTool()\n\n"
                "TOOL_SCHEMA = {\n"
                '    "type": "function",\n'
                '    "function": {\n'
                '        "name": _tool.name,\n'
                '        "description": _tool.description,\n'
                '        "parameters": _tool.schema,\n'
                "    }\n"
                "}\n"
                'TUI_HINT = "📜 Querying systemd journal logs..."\n'
                "REQUIRED_RING = 0\n\n"
                "async def execute(lines: int = 30, unit: str = '', priority: str = '', **kwargs) -> str:\n"
                '    params = {"lines": lines, "unit": unit, "priority": priority, **kwargs}\n'
                "    res = await _tool.execute(params)\n"
                "    return json.dumps(res.to_dict(tool=_tool.name, arguments=params))\n"
            ),
        ),
        (
            TOOLS_DIR / "manage_media_playback.py",
            (
                "import json\n"
                "from zelyr.tools.os_desktop import ManageMediaPlaybackTool\n\n"
                "_tool = ManageMediaPlaybackTool()\n\n"
                "TOOL_SCHEMA = {\n"
                '    "type": "function",\n'
                '    "function": {\n'
                '        "name": _tool.name,\n'
                '        "description": _tool.description,\n'
                '        "parameters": _tool.schema,\n'
                "    }\n"
                "}\n"
                'TUI_HINT = "🎵 Controlling desktop media & audio..."\n'
                "REQUIRED_RING = 0\n\n"
                "async def execute(action: str = 'status', player: str = '', value: str = '', **kwargs) -> str:\n"
                '    params = {"action": action, "player": player, "value": value, **kwargs}\n'
                "    res = await _tool.execute(params)\n"
                "    return json.dumps(res.to_dict(tool=_tool.name, arguments=params))\n"
            ),
        ),
        (
            TOOLS_DIR / "manage_workspace_file.py",
            (
                "import json\n"
                "from zelyr.tools.workspace_file import ManageWorkspaceFileTool\n\n"
                "_tool = ManageWorkspaceFileTool()\n\n"
                "TOOL_SCHEMA = {\n"
                '    "type": "function",\n'
                '    "function": {\n'
                '        "name": _tool.name,\n'
                '        "description": _tool.description,\n'
                '        "parameters": _tool.schema,\n'
                "    }\n"
                "}\n"
                'TUI_HINT = "📁 Safely managing workspace file..."\n'
                "REQUIRED_RING = 0\n\n"
                "async def execute(action: str = 'read', path: str = '', content: str = '', search_text: str = '', replace_text: str = '', max_lines: int = 200, **kwargs) -> str:\n"
                '    params = {"action": action, "path": path, "content": content, "search_text": search_text, "replace_text": replace_text, "max_lines": max_lines, **kwargs}\n'
                "    res = await _tool.execute(params)\n"
                "    return json.dumps(res.to_dict(tool=_tool.name, arguments=params))\n"
            ),
        ),
        (
            TOOLS_DIR / "send_desktop_notification.py",
            (
                "import json\n"
                "from zelyr.tools.os_desktop import SendDesktopNotificationTool\n\n"
                "_tool = SendDesktopNotificationTool()\n\n"
                "TOOL_SCHEMA = {\n"
                '    "type": "function",\n'
                '    "function": {\n'
                '        "name": _tool.name,\n'
                '        "description": _tool.description,\n'
                '        "parameters": _tool.schema,\n'
                "    }\n"
                "}\n"
                'TUI_HINT = "🔔 Sending native desktop notification..."\n'
                "REQUIRED_RING = 0\n\n"
                "async def execute(title: str = '', message: str = '', urgency: str = 'normal', app_name: str = 'Zelyr', **kwargs) -> str:\n"
                '    params = {"title": title, "message": message, "urgency": urgency, "app_name": app_name, **kwargs}\n'
                "    res = await _tool.execute(params)\n"
                "    return json.dumps(res.to_dict(tool=_tool.name, arguments=params))\n"
            ),
        ),
        (
            TOOLS_DIR / "inspect_network.py",
            (
                "import json\n"
                "from zelyr.tools.os_system import InspectNetworkTool\n\n"
                "_tool = InspectNetworkTool()\n\n"
                "TOOL_SCHEMA = {\n"
                '    "type": "function",\n'
                '    "function": {\n'
                '        "name": _tool.name,\n'
                '        "description": _tool.description,\n'
                '        "parameters": _tool.schema,\n'
                "    }\n"
                "}\n"
                'TUI_HINT = "🌐 Inspecting network interfaces & latency..."\n'
                "REQUIRED_RING = 0\n\n"
                "async def execute(check_target: str = '1.1.1.1', **kwargs) -> str:\n"
                '    params = {"check_target": check_target, **kwargs}\n'
                "    res = await _tool.execute(params)\n"
                "    return json.dumps(res.to_dict(tool=_tool.name, arguments=params))\n"
            ),
        ),
        (
            TOOLS_DIR / "execute_command.py",
            _EXECUTE_COMMAND_TEMPLATE,
        ),
        (
            TOOLS_DIR / "manage_desktop_window.py",
            (
                "import json\n"
                "from zelyr.tools.hyprland_ipc import ManageDesktopWindowTool\n\n"
                "_tool = ManageDesktopWindowTool()\n\n"
                "TOOL_SCHEMA = {\n"
                '    "type": "function",\n'
                '    "function": {\n'
                '        "name": _tool.name,\n'
                '        "description": _tool.description,\n'
                '        "parameters": _tool.schema,\n'
                '    }\n'
                "}\n"
                'TUI_HINT = "🪟 Managing Hyprland window/workspace..."\n'
                "REQUIRED_RING = 0\n\n"
                "async def execute(action: str, target: str = '', **kwargs) -> str:\n"
                '    params = {"action": action, "target": target, **kwargs}\n'
                "    res = await _tool.execute(params)\n"
                "    return json.dumps(res.to_dict(tool=_tool.name, arguments=params))\n"
            ),
        ),
        (
            TOOLS_DIR / "interact_with_browser.py",
            (
                "import json\n"
                "from zelyr.tools.browser_cdp import InteractWithBrowserTool\n\n"
                "_tool = InteractWithBrowserTool()\n\n"
                "TOOL_SCHEMA = {\n"
                '    "type": "function",\n'
                '    "function": {\n'
                '        "name": _tool.name,\n'
                '        "description": _tool.description,\n'
                '        "parameters": _tool.schema,\n'
                '    }\n'
                "}\n"
                'TUI_HINT = "🌐 Interacting with browser (CDP)..."\n'
                "REQUIRED_RING = 0\n\n"
                "async def execute(action: str, selector: str = '', text: str = '', expression: str = '', port: int = 9222, **kwargs) -> str:\n"
                '    params = {"action": action, "selector": selector, "text": text, "expression": expression, "port": port, **kwargs}\n'
                "    res = await _tool.execute(params)\n"
                "    return json.dumps(res.to_dict(tool=_tool.name, arguments=params))\n"
            ),
        ),
        (
            TOOLS_DIR / "web_search.py",
            (
                "import json\n"
                "from zelyr.tools.web_search import WebSearchTool\n\n"
                "_tool = WebSearchTool()\n\n"
                "TOOL_SCHEMA = {\n"
                '    "type": "function",\n'
                '    "function": {\n'
                '        "name": _tool.name,\n'
                '        "description": _tool.description,\n'
                '        "parameters": _tool.schema,\n'
                "    }\n"
                "}\n"
                'TUI_HINT = "🔍 Searching the web..."\n'
                "REQUIRED_RING = 1\n\n"
                "async def execute(query: str = '', num_results: int = 5, **kwargs) -> str:\n"
                '    params = {"query": query, "num_results": num_results, **kwargs}\n'
                "    res = await _tool.execute(params)\n"
                "    return json.dumps(res.to_dict(tool=_tool.name, arguments=params))\n"
            ),
        ),
        (
            TOOLS_DIR / "fetch_web_page.py",
            (
                "import json\n"
                "from zelyr.tools.web_search import FetchWebPageTool\n\n"
                "_tool = FetchWebPageTool()\n\n"
                "TOOL_SCHEMA = {\n"
                '    "type": "function",\n'
                '    "function": {\n'
                '        "name": _tool.name,\n'
                '        "description": _tool.description,\n'
                '        "parameters": _tool.schema,\n'
                "    }\n"
                "}\n"
                'TUI_HINT = "🌐 Fetching and reading webpage content..."\n'
                "REQUIRED_RING = 1\n\n"
                "async def execute(url: str = '', max_chars: int = 5000, **kwargs) -> str:\n"
                '    params = {"url": url, "max_chars": max_chars, **kwargs}\n'
                "    res = await _tool.execute(params)\n"
                "    return json.dumps(res.to_dict(tool=_tool.name, arguments=params))\n"
            ),
        ),
        (
            TOOLS_DIR / "vision.py",
            (
                "import json\n"
                "from zelyr.tools.vision import InteractWithUITool\n\n"
                "_tool = InteractWithUITool()\n\n"
                "TOOL_SCHEMA = {\n"
                '    "type": "function",\n'
                '    "function": {\n'
                '        "name": _tool.name,\n'
                '        "description": _tool.description,\n'
                '        "parameters": _tool.schema,\n'
                "    }\n"
                "}\n"
                'TUI_HINT = "🎯 Interacting with UI..."\n'
                "REQUIRED_RING = 0\n\n"
                "async def execute(instruction: str, **kwargs) -> str:\n"
                '    params = {"instruction": instruction, **kwargs}\n'
                "    res = await _tool.execute(params)\n"
                "    return json.dumps(res.to_dict(tool=_tool.name, arguments=params))\n"
            ),
        ),
        (
            TOOLS_DIR / "delegate_subtask.py",
            (
                "import json\n"
                "from zelyr.tools.subagent_tool import SubagentDelegationTool\n\n"
                "_tool = SubagentDelegationTool()\n\n"
                "TOOL_SCHEMA = {\n"
                '    "type": "function",\n'
                '    "function": {\n'
                '        "name": _tool.name,\n'
                '        "description": _tool.description,\n'
                '        "parameters": _tool.schema,\n'
                "    }\n"
                "}\n"
                'TUI_HINT = "🤖 Delegating subtask to specialized worker agent..."\n'
                "REQUIRED_RING = 1\n\n"
                "async def execute(role: str = '', task: str = '', context: dict = None, timeout_seconds: int = 60, **kwargs) -> str:\n"
                '    params = {"role": role, "task": task, "context": context or {}, "timeout_seconds": timeout_seconds, **kwargs}\n'
                "    res = await _tool.execute(params)\n"
                "    return json.dumps(res.to_dict(tool=_tool.name, arguments=params))\n"
            ),
        ),
    ]
    for target_file, code_content in provisions:
        try:
            safe_provision_tool(target_file, code_content)
        except Exception as exc:
            logger.warning(f"Error provisioning tool {target_file}: {exc}")

    for path in TOOLS_DIR.glob("*.py"):
        if path.name.startswith("__"):
            continue
            
        module_name = f"zelyr.dynamic_tools.{path.stem}"
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec and spec.loader:
            mod = importlib.util.module_from_spec(spec)
            try:
                spec.loader.exec_module(mod)
                
                if hasattr(mod, "TOOL_SCHEMA") and hasattr(mod, "execute"):
                    schema = getattr(mod, "TOOL_SCHEMA")
                    fn = schema.setdefault("function", {})
                    params = fn.get("parameters")
                    if not params or not isinstance(params, dict) or params.get("type") in ("", None) or params.get("properties") is None:
                        fn["parameters"] = {"type": "object", "properties": {}}
                    name = fn["name"]
                    _schemas.append(schema)
                    _executors[name] = getattr(mod, "execute")
                    _rings[name] = getattr(mod, "REQUIRED_RING", 0)
                    
                    if hasattr(mod, "TUI_HINT"):
                        _tui_hints[name] = getattr(mod, "TUI_HINT")

                    _capabilities[name] = _extract_capability(mod, name)
                    _plugin_file_paths[name] = path
            except Exception as e:
                print(f"\033[1;31m[Warning] Failed to load plugin {path.name}: {e}\033[0m")

    # Autonomous Capability Indexing into persistent MemoryStore
    try:
        from zelyr.memory.capability_indexer import index_tool_capabilities
        index_tool_capabilities()
    except Exception:
        pass


TIER_1_BUILTIN = {
    "manage_system_process",
    "manage_system_clipboard",
    "query_system_journal",
    "manage_media_playback",
    "manage_workspace_file",
    "send_desktop_notification",
    "inspect_network",
}


def _extract_capability(mod: Any, name: str):
    from zelyr.tools.core import ToolCapability

    if name in ("create_tool", "manage_tools"):
        return ToolCapability(
            tier=None,
            is_core=False,
            requires_bridge=False,
            requires_window=False,
            token_cost=150,
        )

    if mod is not None:

        if hasattr(mod, "CAPABILITY") and isinstance(mod.CAPABILITY, ToolCapability):
            return mod.CAPABILITY
        if hasattr(mod, "TIER"):
            try:
                tier = int(getattr(mod, "TIER", 1))
            except Exception:
                tier = 1
            is_core = bool(getattr(mod, "IS_CORE", tier == 0))
            requires_bridge = bool(getattr(mod, "REQUIRES_BRIDGE", tier == 2))
            requires_window = bool(getattr(mod, "REQUIRES_WINDOW", False))
            try:
                token_cost = int(getattr(mod, "TOKEN_COST", 150))
            except Exception:
                token_cost = 150
            return ToolCapability(
                tier=tier,
                is_core=is_core,
                requires_bridge=requires_bridge,
                requires_window=requires_window,
                token_cost=token_cost,
            )
        if hasattr(mod, "IS_CORE") and bool(getattr(mod, "IS_CORE")):
            return ToolCapability(
                tier=0,
                is_core=True,
                requires_bridge=bool(getattr(mod, "REQUIRES_BRIDGE", False)),
                requires_window=bool(getattr(mod, "REQUIRES_WINDOW", False)),
                token_cost=int(getattr(mod, "TOKEN_COST", 150)),
            )
        if hasattr(mod, "_tool") and hasattr(mod._tool, "capability"):
            cap = mod._tool.capability
            if name in ("manage_desktop_window", "interact_with_browser", "interact_with_ui") or name in TIER_1_BUILTIN:
                return cap
            if cap.is_core or cap.tier in (2, 3) or cap.requires_bridge:
                return cap

    # Heuristic fallback mapping by tool identifier
    if name in ("execute_command", "manage_desktop_window"):
        return ToolCapability(
            tier=0 if name == "execute_command" else 1,
            is_core=True,
            requires_bridge=False,
            requires_window=(name == "manage_desktop_window"),
            token_cost=150,
        )
    if name in TIER_1_BUILTIN:
        return ToolCapability(
            tier=1,
            is_core=False,
            requires_bridge=False,
            requires_window=False,
            token_cost=150,
        )
    if name in ("interact_with_browser", "web_search", "fetch_web_page"):
        return ToolCapability(
            tier=2,
            is_core=False,
            requires_bridge=(name == "interact_with_browser"),
            requires_window=(name == "interact_with_browser"),
            token_cost=250,
        )
    if name in ("interact_with_ui", "vision", "capture_som_screen", "som_click", "som_type", "som_key"):
        return ToolCapability(
            tier=3,
            is_core=False,
            requires_bridge=False,
            requires_window=True,
            token_cost=600,
        )
    return ToolCapability(
        tier=None,
        is_core=False,
        requires_bridge=False,
        requires_window=False,
        token_cost=150,
    )


def get_tool_capability(name: str):
    if not _capabilities and not _schemas:
        load_plugins()
    if name in _capabilities:
        return _capabilities[name]
    return _extract_capability(None, name)


def get_tool_capabilities() -> Dict[str, Any]:
    if not _capabilities and not _schemas:
        load_plugins()
    return dict(_capabilities)


class DynamicCapabilitySet(set):
    """Compatibility set proxy reflecting capability registry dynamically."""

    def __init__(self, filter_fn: Callable[[str, Any], bool], default_fallback: set):
        super().__init__(default_fallback)
        self._filter_fn = filter_fn
        self._default_fallback = set(default_fallback)

    def _sync(self) -> None:
        if _capabilities:
            res = {name for name, cap in _capabilities.items() if self._filter_fn(name, cap)}
            if res:
                self.clear()
                self.update(res)
                return
        self.clear()
        self.update(self._default_fallback)

    def __contains__(self, item: Any) -> bool:
        self._sync()
        return super().__contains__(item)

    def __iter__(self):
        self._sync()
        return super().__iter__()

    def __len__(self) -> int:
        self._sync()
        return super().__len__()

    def __or__(self, other: Any) -> set:
        self._sync()
        if isinstance(other, DynamicCapabilitySet):
            other._sync()
        return super().__or__(other)

    def __ror__(self, other: Any) -> set:
        self._sync()
        if isinstance(other, DynamicCapabilitySet):
            other._sync()
        return set(other) | set(self)

    def __and__(self, other: Any) -> set:
        self._sync()
        if isinstance(other, DynamicCapabilitySet):
            other._sync()
        return super().__and__(other)

    def __rand__(self, other: Any) -> set:
        self._sync()
        if isinstance(other, DynamicCapabilitySet):
            other._sync()
        return set(other) & set(self)

    def __sub__(self, other: Any) -> set:
        self._sync()
        if isinstance(other, DynamicCapabilitySet):
            other._sync()
        return super().__sub__(other)

    def __rsub__(self, other: Any) -> set:
        self._sync()
        if isinstance(other, DynamicCapabilitySet):
            other._sync()
        return set(other) - set(self)

    def __repr__(self) -> str:
        self._sync()
        return super().__repr__()


CORE_TOOLS = DynamicCapabilitySet(
    lambda n, c: getattr(c, "is_core", False) or getattr(c, "tier", None) == 0,
    {"execute_command", "manage_desktop_window"},
)

TIER_1_TOOLS = DynamicCapabilitySet(
    lambda n, c: getattr(c, "tier", None) == 1 and not getattr(c, "is_core", False),
    {
        "manage_system_process",
        "manage_system_clipboard",
        "query_system_journal",
        "manage_media_playback",
        "manage_workspace_file",
        "send_desktop_notification",
        "inspect_network",
    },
)

TIER_2_TOOLS = DynamicCapabilitySet(
    lambda n, c: getattr(c, "tier", None) == 2,
    {"interact_with_browser", "web_search", "fetch_web_page"},
)

TIER_3_TOOLS = DynamicCapabilitySet(
    lambda n, c: getattr(c, "tier", None) == 3,
    {"interact_with_ui"},
)


def filter_tool_schemas_by_tier(schemas: List[Dict[str, Any]], tier: str, is_browser: bool = False) -> List[Dict[str, Any]]:
    """Filter dynamic tool schemas based on the detected automation tier.

    - Always includes core system tools: execute_command, manage_desktop_window.
    - If Tier 1: includes Core + Tier 1 tools.
    - If Tier 2: includes Core + Tier 2 tools.
    - If Tier 3 / Ambiguous / General: includes the full registry.
    - If is_browser is True, interact_with_ui is strictly purged from exposed schemas.
    """
    tier_lower = (tier or "").lower().strip()
    if tier_lower in ("tier1", "tier1_ipc", "1"):
        target_tier = 1
    elif tier_lower in ("tier2", "tier2_browser", "2"):
        target_tier = 2
    else:
        target_tier = None

    if target_tier is not None:
        filtered = []
        for s in schemas:
            name = s.get("function", {}).get("name") or s.get("name")
            cap = get_tool_capability(name)
            if cap.is_core or cap.tier == 0 or cap.tier == target_tier:
                filtered.append(s)
        result = filtered if filtered else list(schemas)
    else:
        result = list(schemas)

    if is_browser:
        result = [s for s in result if (s.get("function", {}).get("name") or s.get("name")) != "interact_with_ui"]

    return result


def get_tool_schemas(ring: int = 0, tier: Optional[str] = None, is_browser: bool = False) -> List[Dict[str, Any]]:
    if not _schemas:
        load_plugins()
    filtered = []
    for s in _schemas:
        fn = s.setdefault("function", {})
        params = fn.get("parameters")
        if not params or not isinstance(params, dict) or params.get("type") in ("", None) or params.get("properties") is None:
            fn["parameters"] = {"type": "object", "properties": {}}
        name = fn["name"]
        tool_ring = _rings.get(name, 0)
        # Ring 0 has access to everything. Ring 3 only has access to Ring 3+.
        if tool_ring >= ring:
            filtered.append(s)
    if tier or is_browser:
        filtered = filter_tool_schemas_by_tier(filtered, tier or "", is_browser=is_browser)
    return filtered

def get_tui_hints() -> Dict[str, str]:
    if not _tui_hints:
        load_plugins()
    return dict(_tui_hints)

PROVISIONED_TOOLS = {
    "remember_fact",
    "manage_system_process",
    "manage_system_clipboard",
    "query_system_journal",
    "manage_media_playback",
    "manage_workspace_file",
    "send_desktop_notification",
    "inspect_network",
    "interact_with_browser",
    "interact_with_ui",
    "vision",
    "capture_som_screen",
    "click_tag",
    "som_click",
    "som_type",
    "som_key",
    "web_search",
    "fetch_web_page",
    "create_tool",
    "manage_tools",
    "execute_command",
    "manage_desktop_window",
    "delegate_subtask",
}

TIER_TIMEOUTS: Dict[str, float] = {
    "interact_with_ui": 35.0,
    "vision": 35.0,
    "capture_som_screen": 35.0,
    "click_tag": 35.0,
    "interact_with_browser": 10.0,
}

def get_tier_timeout(tool_name: str) -> float:
    """Return execution latency budget for a given tool name:
    Tier 1 (CLI/OS): 8.0s
    Tier 2 (Browser Extension): 10.0s
    Tier 3 (VLM Grounding): 35.0s
    """
    return TIER_TIMEOUTS.get(tool_name, 8.0)

async def execute_tool(
    name: str,
    *,
    caller_ring: Optional[int] = None,
    allowed_tools: Optional[Any] = None,
    **kwargs,
) -> str:
    from zelyr.tool_registry import authorize_and_validate_tool_call, get_active_task_local_scope

    if get_active_task_local_scope() is not None:
        return json.dumps({
            "success": False,
            "error": "AuthorizationError: Isolated task-local test executor cannot access production plugin registry.",
            "error_type": "AUTHORIZATION_DENIED",
            "reason": "authorization_denied",
        })

    if not _executors:
        load_plugins()

    decision = authorize_and_validate_tool_call(
        name,
        kwargs,
        caller_ring=caller_ring,
        allowed_tools=allowed_tools,
        active_schemas=_schemas or None,
        validate_schema=True,
    )
    if not decision.allowed:
        return json.dumps({
            "success": False,
            "error": decision.detail or f"AuthorizationError: Tool '{decision.tool_name or name}' denied.",
            "error_type": decision.error_type or "AUTHORIZATION_DENIED",
            "reason": decision.reason or "authorization_denied",
        })

    name = decision.tool_name
    kwargs = decision.normalized_arguments

    if name not in _executors:
        try:
            from zelyr.tool_registry import ToolRegistry
            registry = ToolRegistry()
            if registry.get_tool(name) is not None:
                res = await registry.execute_async(
                    name,
                    caller_ring=caller_ring,
                    allowed_tools=allowed_tools,
                    **kwargs,
                )
                if hasattr(res, "to_dict"):
                    return json.dumps(res.to_dict(tool=name, arguments=kwargs))
                return str(res)
        except Exception:
            pass
        return f"Tool execution failed: Tool {name} not found in dynamic plugins."
    
    import asyncio
    func = _executors[name]
    
    timeout = get_tier_timeout(name)
    timeout_int = int(timeout)
    is_builtin = (
        name in TIER_1_BUILTIN
        or name in PROVISIONED_TOOLS
        or getattr(get_tool_capability(name), "is_core", False)
    )

    if is_builtin:
        def thread_worker():
            if asyncio.iscoroutinefunction(func):
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    return loop.run_until_complete(func(**kwargs))
                finally:
                    loop.close()
            else:
                return func(**kwargs)

        try:
            res = await asyncio.wait_for(
                asyncio.to_thread(thread_worker),
                timeout=timeout
            )
            return str(res)
        except (TimeoutError, asyncio.TimeoutError):
            return json.dumps({
                "success": False,
                "error": f"Tool execution timed out after {timeout_int}s. Try a faster Tier 1 command or verify active window.",
            })
        except Exception as e:
            return f"Tool execution failed: {e}"
    else:
        from zelyr.core.plugin_sandbox import SandboxedPluginRunner
        runner = SandboxedPluginRunner(func, name=name, timeout=timeout)
        tool_res = await runner.run_async(**kwargs)
        if not tool_res.success:
            return json.dumps({"success": False, "error": tool_res.error})
        if isinstance(tool_res.output, str):
            return tool_res.output
        return json.dumps(tool_res.to_dict(tool=name, arguments=kwargs))


def reload_plugin(path: Any) -> None:
    tools_dir = get_tools_dir()
    if isinstance(path, str):
        path_str = path
        if not path_str.endswith(".py"):
            path = tools_dir / f"{path_str}.py"
        elif not os.path.isabs(path_str):
            path = tools_dir / path_str
        else:
            path = Path(path_str)
    elif not isinstance(path, Path):
        return

    if not path.is_file() or not path.name.endswith(".py"):
        raise FileNotFoundError(f"Plugin file not found: {path}")

    module_name = f"zelyr.dynamic_tools.{path.stem}"

    # Clean up sys.modules cache to force a fresh import
    if module_name in sys.modules:
        del sys.modules[module_name]

    spec = importlib.util.spec_from_file_location(module_name, path)
    if not spec or not spec.loader:
        raise ImportError(f"Could not load spec for plugin at {path}")

    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)

        if hasattr(mod, "TOOL_SCHEMA") and hasattr(mod, "execute"):
            schema = getattr(mod, "TOOL_SCHEMA")
            fn = schema.setdefault("function", {})
            params = fn.get("parameters")
            if not params or not isinstance(params, dict) or params.get("type") in ("", None) or params.get("properties") is None:
                fn["parameters"] = {"type": "object", "properties": {}}
            name = fn["name"]

            # Remove existing schema if it exists
            global _schemas, _executors, _tui_hints, _capabilities, _plugin_file_paths
            _schemas = [s for s in _schemas if (s.get("function", {}).get("name") or s.get("name")) != name]

            _schemas.append(schema)
            _executors[name] = getattr(mod, "execute")
            _rings[name] = getattr(mod, "REQUIRED_RING", 0)
            _capabilities[name] = _extract_capability(mod, name)
            _plugin_file_paths[name] = path

            if hasattr(mod, "TUI_HINT"):
                _tui_hints[name] = getattr(mod, "TUI_HINT")
        else:
            raise ValueError(f"Plugin {path.name} must export TOOL_SCHEMA and execute()")

    except Exception as e:
        print(f"\033[1;31m[Warning] Failed to load plugin {path.name}: {e}\033[0m")
        raise

    try:
        from zelyr.memory.capability_indexer import index_tool_capabilities
        index_tool_capabilities()
    except Exception:
        pass


def unregister_plugin(name: str) -> bool:
    """Unregister a dynamic tool from memory and registries.

    Removes the tool from _schemas, _executors, _rings, _tui_hints, and _capabilities,
    cleans up sys.modules, and resynchronizes DynamicCapabilitySet.
    Returns True if the tool was found and removed, False otherwise.
    """
    global _schemas, _executors, _rings, _tui_hints, _capabilities, _plugin_file_paths

    found = False

    orig_len = len(_schemas)
    _schemas = [s for s in _schemas if (s.get("function", {}).get("name") or s.get("name")) != name]
    if len(_schemas) != orig_len:
        found = True

    if name in _executors:
        del _executors[name]
        found = True

    if name in _rings:
        del _rings[name]
        found = True

    if name in _tui_hints:
        del _tui_hints[name]
        found = True

    if name in _capabilities:
        del _capabilities[name]
        found = True

    if name in _plugin_file_paths:
        del _plugin_file_paths[name]
        found = True

    module_name = f"zelyr.dynamic_tools.{name}"
    if module_name in sys.modules:
        del sys.modules[module_name]
        found = True

    CORE_TOOLS._sync()
    TIER_1_TOOLS._sync()
    TIER_2_TOOLS._sync()
    TIER_3_TOOLS._sync()

    return found

