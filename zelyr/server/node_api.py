from zelyr.compat import state_dir, state_file
import logging
import json
import asyncio
import time
from pathlib import Path
from typing import Any
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from zelyr.config import get_config
from zelyr.core import Engine
from zelyr.memory import SyncMemoryStore
from zelyr.llm.universal_client import UniversalLLMClient
from zelyr.agents.orchestrator_agent import OrchestratorAgent
from zelyr.tool_registry import ToolRegistry
from zelyr.tools.plugin_loader import load_plugins
from zelyr.core.config_service import initialize_model_config
from zelyr.tools import EchoTool, ShellTool, FileReadTool, FileWriteTool, SystemInfoTool
from zelyr.tools.system_admin import SystemAdminTool
from zelyr.legacy_wrapper import create_legacy_tools

logger = logging.getLogger("zelyr.node")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

app = FastAPI(title="Zelyr Node API", description="Headless Distributed Swarm Node")

# ── CORS — allow Tailscale / LAN PySide6 clients to connect cleanly ──── #
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_boot_time = time.time()

class NodeState:
    def __init__(self):
        self.engine = None
        self.memory = None
        self.ollama = None
        self.orchestrator = None
        self.remote_shell_token = None

state = NodeState()


@app.get("/health")
async def health_check():
    """Lightweight liveness probe for Swarm discovery and monitoring."""
    return JSONResponse({
        "status": "online",
        "node": "zelyr-swarm-worker",
        "uptime_seconds": round(time.time() - _boot_time, 1),
        "engine_ready": state.engine is not None,
    })

@app.on_event("startup")
async def startup_event():
    logger.info("Initializing Zelyr Headless Engine...")
    
    config = get_config()
    state.ollama = UniversalLLMClient(default_model=config.ollama_model)
    initialize_model_config(config, state.ollama)
    
    # Setup Memory
    zelyr_dir = state_dir()
    zelyr_dir.mkdir(exist_ok=True, parents=True)
    db_path = str(state_file("zelyr.db", "axiom.db"))
    state.memory = SyncMemoryStore(db_path, embedding_provider=state.ollama)
    
    # Engine & Registry
    state.engine = Engine(memory=state.memory)
    state.engine.initialize()
    
    tool_registry = ToolRegistry(state.engine.registry)
    
    # Load default tools
    tools = [
        EchoTool(), ShellTool(), FileReadTool('.'), FileWriteTool('.'), SystemInfoTool(),
        SystemAdminTool()
    ]
    tools.extend(create_legacy_tools())
    for tool in tools:
        state.engine.registry.register_tool(tool.tool_id, tool)
        
    load_plugins(tool_registry)
    
    # Orchestrator
    state.orchestrator = OrchestratorAgent(
        tool_registry, 
        state.engine.event_bus, 
        state.memory, 
        llm=state.ollama
    )
    state.engine.registry.register_agent(state.orchestrator.name, state.orchestrator)
    state.memory.create_conversation('Zelyr Node Session')
    
    logger.info("Zelyr Headless Engine Online.")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down Zelyr Headless Engine...")
    if state.engine:
        state.engine.shutdown()
    if state.memory:
        state.memory.close()
    if state.ollama:
        state.ollama.close()

@app.websocket("/ws/swarm")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("New WebSocket connection accepted on /ws/swarm")
    
    try:
        while True:
            data = await websocket.receive_text()
            try:
                payload = json.loads(data)
                prompt = payload.get("prompt", "")
            except json.JSONDecodeError:
                prompt = data
                
            if not prompt:
                continue
                
            logger.info(f"Received prompt: {prompt}")
            
            # Send processing status
            await websocket.send_text(json.dumps({"status": "processing"}))
            
            # Phase 1: Blocking execution run in a separate thread to not block the asyncio event loop
            loop = asyncio.get_running_loop()
            response = await loop.run_in_executor(None, state.orchestrator.run, prompt)
            
            output_str = ""
            if response.output:
                if isinstance(response.output, dict):
                    output_str = response.output.get('response', '')
                elif isinstance(response.output, str):
                    output_str = response.output
                else:
                    output_str = json.dumps(response.output, default=str)
                    
            if not output_str and response.error:
                output_str = f"Error: {response.error}"
                
            import re
            output_str = re.sub(r'<think>[\s\S]*?(?:</think>|$)', '', output_str, flags=re.IGNORECASE)
            output_str = output_str.replace('</think>', '').replace('<think>', '').strip()
                
            # Stream the final result back
            await websocket.send_text(json.dumps({
                "status": "complete",
                "response": output_str
            }))
            
            # Record in memory
            state.memory.add_message('user', prompt)
            state.memory.add_message('assistant', output_str)
            
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}", exc_info=True)
        try:
            await websocket.send_text(json.dumps({"status": "error", "message": str(e)}))
        except:
            pass

from fastapi import Request, HTTPException
from pydantic import BaseModel, ConfigDict, Field, field_validator
import hmac
import ipaddress
import os
import re
import signal
import stat
import sys

MAX_SHELL_REQUEST_BYTES = 64 * 1024  # 64 KiB max HTTP request body
MAX_SHELL_COMMAND_BYTES = 16 * 1024  # 16 KiB max command string
MAX_SHELL_OUTPUT_BYTES = 256 * 1024  # 256 KiB combined stdout+stderr cap
MAX_SHELL_TIMEOUT_SECONDS = 300
MIN_REMOTE_SHELL_TOKEN_LENGTH = 16
DEFAULT_MAX_CONCURRENT_SHELL_EXECUTIONS = 4

_TAILSCALE_CGNAT_NETWORK = ipaddress.ip_network("100.64.0.0/10")
_TRUSTED_PRIVATE_NETWORKS = (
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("fc00::/7"),
)
_SAFE_SHELL_ENV_ALLOWLIST = (
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "TZ",
    "TMPDIR",
    "USER",
    "LOGNAME",
)
_SHELL_CONCURRENCY_STATE = {"active": 0}


class ShellRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    command: str = Field(..., description="Shell command to execute on the node.")
    timeout: int = Field(300, description="Execution timeout in seconds (1..300).")
    cwd: str | None = Field(None, description="Optional working directory within allowed workspace roots.")

    @field_validator("command")
    @classmethod
    def _validate_command(cls, value: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("command must be a non-empty string.")
        if "\x00" in value:
            raise ValueError("command must not contain NUL bytes.")
        if len(value.encode("utf-8", errors="replace")) > MAX_SHELL_COMMAND_BYTES:
            raise ValueError(f"command exceeds maximum size ({MAX_SHELL_COMMAND_BYTES} bytes).")
        return value

    @field_validator("timeout", mode="before")
    @classmethod
    def _validate_timeout(cls, value: Any) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("timeout must be an integer between 1 and 300.")
        if value < 1 or value > MAX_SHELL_TIMEOUT_SECONDS:
            raise ValueError(f"timeout must be between 1 and {MAX_SHELL_TIMEOUT_SECONDS} seconds.")
        return value

    @field_validator("cwd")
    @classmethod
    def _validate_cwd(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str) or not value.strip():
            raise ValueError("cwd must be a non-empty string when provided.")
        if "\x00" in value:
            raise ValueError("cwd must not contain NUL bytes.")
        return value.strip()


def _is_trusted_shell_client_ip(raw_host: str) -> bool:
    """Allow only loopback, strict RFC 1918/ULA private LAN, or Tailscale CGNAT (100.64.0.0/10)."""
    if not raw_host or not isinstance(raw_host, str):
        return False
    host = raw_host.split("%", 1)[0].strip()
    try:
        parsed = ipaddress.ip_address(host)
    except ValueError:
        return False
    eff = parsed.ipv4_mapped if isinstance(parsed, ipaddress.IPv6Address) and parsed.ipv4_mapped else parsed
    if eff.is_unspecified or eff.is_multicast or eff.is_link_local:
        return False
    if eff.is_loopback:
        return True
    if eff.is_reserved:
        return False
    if isinstance(eff, ipaddress.IPv4Address) and eff in _TAILSCALE_CGNAT_NETWORK:
        return True
    return any(eff in net for net in _TRUSTED_PRIVATE_NETWORKS)


def _resolve_remote_shell_token() -> str | None:
    """Resolve configured remote shell credential cleanly; returns None (fail-closed) when unconfigured."""
    in_mem = getattr(state, "remote_shell_token", None)
    if in_mem is not None:
        if isinstance(in_mem, str):
            tok = in_mem.strip()
            if len(tok) >= MIN_REMOTE_SHELL_TOKEN_LENGTH:
                return tok
        return None

    env_tok = os.environ.get("ZELYR_REMOTE_SHELL_TOKEN")
    if env_tok is not None:
        tok = env_tok.strip()
        return tok if len(tok) >= MIN_REMOTE_SHELL_TOKEN_LENGTH else None

    token_file_override = os.environ.get("ZELYR_REMOTE_SHELL_TOKEN_FILE")
    token_path = Path(token_file_override).expanduser() if token_file_override else state_file("remote_shell.token")
    try:
        st = os.lstat(token_path)
    except OSError:
        return None
    if stat.S_ISLNK(st.st_mode) or not stat.S_ISREG(st.st_mode):
        logger.warning("Refusing remote_shell.token because it is not a regular file.")
        return None
    if hasattr(os, "getuid") and st.st_uid != os.getuid():
        logger.warning("Refusing remote_shell.token due to unexpected file ownership.")
        return None
    if (st.st_mode & 0o077) != 0:
        logger.warning("Refusing remote_shell.token with group/world-accessible permissions.")
        return None
    try:
        tok = token_path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if len(tok) < MIN_REMOTE_SHELL_TOKEN_LENGTH:
        logger.warning("Refusing remote_shell.token shorter than minimum length.")
        return None
    return tok


def _extract_bearer_token(request: Request) -> str | None:
    """Extract Bearer token strictly from Authorization header (never query params)."""
    auth_header = request.headers.get("authorization")
    if not auth_header:
        return None
    parts = auth_header.strip().split(None, 1)
    if len(parts) != 2 or parts[0].lower() != "bearer":
        return None
    token = parts[1].strip()
    return token or None


def _get_allowed_shell_cwds() -> tuple[Path, list[Path]]:
    """Return (default_cwd, allowed_roots) ensuring workspace exists safely."""
    default_root = state_dir() / "swarm_workspace"
    default_root.mkdir(parents=True, exist_ok=True)
    resolved_default = default_root.resolve()

    allowed_raw = os.environ.get("ZELYR_REMOTE_SHELL_ALLOWED_CWDS", "").strip()
    roots: list[Path] = []
    if allowed_raw:
        for entry in allowed_raw.split(os.pathsep):
            item = entry.strip()
            if item:
                roots.append(Path(item).expanduser().resolve())
    if not roots:
        roots = [resolved_default]
    return resolved_default, roots


def _resolve_safe_shell_cwd(requested_cwd: str | None) -> Path:
    """Resolve and confine the shell working directory strictly inside allowed workspace roots."""
    default_root, allowed_roots = _get_allowed_shell_cwds()
    if requested_cwd is None:
        env_cwd = os.environ.get("ZELYR_REMOTE_SHELL_CWD", "").strip()
        if not env_cwd:
            return default_root
        requested_cwd = env_cwd

    raw_path = Path(requested_cwd).expanduser()
    if ".." in raw_path.parts:
        raise HTTPException(status_code=403, detail="Forbidden: Working directory path traversal is not allowed.")
    candidate = raw_path if raw_path.is_absolute() else (default_root / raw_path)
    try:
        resolved = candidate.resolve(strict=True)
    except OSError:
        raise HTTPException(status_code=400, detail="Requested working directory does not exist.")
    if not resolved.is_dir():
        raise HTTPException(status_code=400, detail="Requested working directory is not a directory.")
    if not any(resolved.is_relative_to(root) for root in allowed_roots):
        raise HTTPException(
            status_code=403,
            detail="Forbidden: Working directory is outside allowed workspace roots.",
        )
    return resolved


def _build_sanitized_shell_env(resolved_cwd: Path) -> dict[str, str]:
    """Construct a minimal, secret-free environment for remote shell subprocesses."""
    env: dict[str, str] = {}
    for key in _SAFE_SHELL_ENV_ALLOWLIST:
        val = os.environ.get(key)
        if val is not None:
            env[key] = val

    trusted_dirs = [str(Path(sys.executable).parent), "/usr/local/sbin", "/usr/local/bin", "/usr/sbin", "/usr/bin", "/sbin", "/bin"]
    seen_dirs: list[str] = []
    for d in trusted_dirs:
        if d and os.path.isabs(d) and d not in seen_dirs and os.path.isdir(d):
            seen_dirs.append(d)
    env["PATH"] = os.pathsep.join(seen_dirs) if seen_dirs else "/usr/bin:/bin"
    env["HOME"] = str(resolved_cwd)
    env["PWD"] = str(resolved_cwd)
    env["PAGER"] = "cat"
    env["GIT_PAGER"] = "cat"
    env["GIT_TERMINAL_PROMPT"] = "0"
    env["GIT_CONFIG_NOSYSTEM"] = "1"
    env["GIT_EXTERNAL_DIFF"] = ""
    env["TERM"] = "dumb"
    env["DEBIAN_FRONTEND"] = "noninteractive"
    env["CI"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONNOUSERSITE"] = "1"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


def _redact_shell_text(text: str, *, extra_secrets: tuple[str, ...] = ()) -> str:
    """Scrub credentials, secret env values, and configured auth tokens from output/logs/errors."""
    from zelyr.engine.normalizer import _redact_sensitive_text

    cleaned = _redact_sensitive_text(str(text or ""))
    for secret in extra_secrets:
        if secret and len(secret) >= 4 and secret in cleaned:
            cleaned = cleaned.replace(secret, "[REDACTED]")
    return cleaned


def _summarize_command_for_log(command: str, *, extra_secrets: tuple[str, ...] = ()) -> str:
    """Produce a compact, single-line, secret-redacted preview + SHA-256 digest for audit logs."""
    import hashlib

    cmd_bytes = command.encode("utf-8", errors="replace")
    digest = hashlib.sha256(cmd_bytes).hexdigest()[:12]
    one_line = re.sub(r"\s+", " ", _redact_shell_text(command, extra_secrets=extra_secrets)).strip()
    if len(one_line) > 120:
        one_line = one_line[:117] + "..."
    return f"sha256={digest} len={len(cmd_bytes)} preview={one_line!r}"


def _get_max_concurrent_shell_executions() -> int:
    raw = os.environ.get("ZELYR_REMOTE_SHELL_MAX_CONCURRENT")
    if not raw:
        return DEFAULT_MAX_CONCURRENT_SHELL_EXECUTIONS
    try:
        val = int(raw)
        return max(1, min(val, 32))
    except ValueError:
        return DEFAULT_MAX_CONCURRENT_SHELL_EXECUTIONS


async def _terminate_subprocess(process: Any) -> None:
    """Safely terminate an async subprocess (and its own process group if created) and reap it."""
    pid = getattr(process, "pid", None)
    if isinstance(pid, int) and pid > 0 and hasattr(os, "killpg") and hasattr(os, "getpgid"):
        try:
            if os.getpgid(pid) == pid:
                os.killpg(pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError, OSError):
            pass
    try:
        kill_fn = getattr(process, "kill", None)
        if callable(kill_fn):
            kill_fn()
    except (ProcessLookupError, OSError):
        pass
    try:
        wait_fn = getattr(process, "wait", None)
        if callable(wait_fn):
            await asyncio.wait_for(wait_fn(), timeout=2.0)
    except Exception:
        pass


async def _read_bounded_streams(
    stdout_stream: Any,
    stderr_stream: Any,
    max_bytes: int,
) -> tuple[bytes, bytes, bool]:
    """Read stdout and stderr concurrently while enforcing a strict combined byte ceiling."""
    stdout_buf = bytearray()
    stderr_buf = bytearray()
    total_read = 0
    truncated = False

    async def _drain(stream: Any, target: bytearray) -> None:
        nonlocal total_read, truncated
        if stream is None:
            return
        while not truncated:
            chunk = await stream.read(4096)
            if not chunk:
                break
            if truncated:
                break
            remaining = max_bytes - total_read
            if len(chunk) > remaining:
                if remaining > 0:
                    target.extend(chunk[:remaining])
                    total_read += remaining
                truncated = True
                break
            target.extend(chunk)
            total_read += len(chunk)

    tasks = {
        asyncio.create_task(_drain(stdout_stream, stdout_buf)),
        asyncio.create_task(_drain(stderr_stream, stderr_buf)),
    }
    try:
        while tasks:
            done, tasks = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for finished in done:
                exc = finished.exception()
                if exc is not None:
                    raise exc
            if truncated:
                for task in tasks:
                    task.cancel()
                if tasks:
                    await asyncio.gather(*tasks, return_exceptions=True)
                tasks = set()
                break
    except BaseException:
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        raise

    return bytes(stdout_buf), bytes(stderr_buf), truncated


async def _read_bounded_request_body(request: Request, max_bytes: int) -> bytes:
    """Read request body cleanly while rejecting oversized payloads before JSON decoding."""
    content_length = request.headers.get("content-length")
    if content_length is not None:
        try:
            declared = int(content_length)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid Content-Length header.")
        if declared < 0:
            raise HTTPException(status_code=400, detail="Invalid Content-Length header.")
        if declared > max_bytes:
            raise HTTPException(status_code=413, detail="Request payload exceeds maximum allowed size.")

    body = bytearray()
    async for chunk in request.stream():
        if not chunk:
            continue
        if len(body) + len(chunk) > max_bytes:
            raise HTTPException(status_code=413, detail="Request payload exceeds maximum allowed size.")
        body.extend(chunk)
    return bytes(body)


@app.post("/execute_shell")
async def execute_shell(request: Request):
    """
    Executes an authenticated shell command on the host inside a controlled environment.
    Requires both trusted network source (loopback, private LAN, or Tailscale 100.64.0.0/10)
    and constant-time Bearer token verification.
    """
    client_ip = request.client.host if request.client else "unknown"
    if not _is_trusted_shell_client_ip(client_ip):
        logger.warning("Rejected /execute_shell from untrusted IP: %s", client_ip)
        raise HTTPException(status_code=403, detail="Forbidden: Tailscale or LAN IP required.")

    provided_token = _extract_bearer_token(request)
    if not provided_token:
        logger.warning("Rejected unauthenticated /execute_shell request from %s", client_ip)
        raise HTTPException(
            status_code=401,
            detail="Unauthorized: Bearer token required.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    expected_token = _resolve_remote_shell_token()
    if not expected_token:
        logger.warning("Rejected /execute_shell from %s: no valid shell auth token configured.", client_ip)
        raise HTTPException(
            status_code=503,
            detail="Remote shell execution is disabled: authentication token not configured.",
        )

    if not hmac.compare_digest(provided_token.encode("utf-8"), expected_token.encode("utf-8")):
        logger.warning("Rejected /execute_shell with invalid token from %s", client_ip)
        raise HTTPException(
            status_code=401,
            detail="Unauthorized: Invalid authentication credentials.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    extra_secrets = tuple(s for s in (expected_token, provided_token) if s)

    raw_body = await _read_bounded_request_body(request, MAX_SHELL_REQUEST_BYTES)
    try:
        body_obj = json.loads(raw_body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise HTTPException(status_code=400, detail="Malformed JSON request body.")
    if not isinstance(body_obj, dict):
        raise HTTPException(status_code=422, detail="Request body must be a JSON object.")

    try:
        payload = ShellRequest.model_validate(body_obj)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=_redact_shell_text(str(exc), extra_secrets=extra_secrets))

    resolved_cwd = _resolve_safe_shell_cwd(payload.cwd)
    sanitized_env = _build_sanitized_shell_env(resolved_cwd)
    command = payload.command
    timeout = payload.timeout
    cmd_summary = _summarize_command_for_log(command, extra_secrets=extra_secrets)

    max_concurrent = _get_max_concurrent_shell_executions()
    if _SHELL_CONCURRENCY_STATE["active"] >= max_concurrent:
        logger.warning("Rejected /execute_shell from %s due to concurrency limit (%d)", client_ip, max_concurrent)
        raise HTTPException(
            status_code=429,
            detail="Too many concurrent shell executions; try again later.",
            headers={"Retry-After": "1"},
        )

    _SHELL_CONCURRENCY_STATE["active"] += 1
    process = None
    try:
        logger.info("Executing remote shell command from %s (%s) cwd=%s", client_ip, cmd_summary, resolved_cwd)
        process = await asyncio.create_subprocess_shell(
            command,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(resolved_cwd),
            env=sanitized_env,
            start_new_session=True,
        )

        async def _collect() -> tuple[bytes, bytes, bool, int | None]:
            out_b, err_b, is_trunc = await _read_bounded_streams(
                process.stdout,
                process.stderr,
                MAX_SHELL_OUTPUT_BYTES,
            )
            if is_trunc:
                await _terminate_subprocess(process)
                return out_b, err_b, True, process.returncode
            rc = await process.wait()
            return out_b, err_b, False, rc

        stdout_bytes, stderr_bytes, truncated, returncode = await asyncio.wait_for(_collect(), timeout=timeout)

        stdout_text = _redact_shell_text(stdout_bytes.decode("utf-8", errors="replace"), extra_secrets=extra_secrets)
        stderr_text = _redact_shell_text(stderr_bytes.decode("utf-8", errors="replace"), extra_secrets=extra_secrets)
        if truncated:
            notice = f"\n[OUTPUT TRUNCATED AT {MAX_SHELL_OUTPUT_BYTES} BYTES]"
            stderr_text = (stderr_text + notice) if stderr_text else notice.lstrip("\n")

        return {
            "stdout": stdout_text,
            "stderr": stderr_text,
            "returncode": returncode if returncode is not None else -1,
            "truncated": truncated,
        }
    except asyncio.TimeoutError:
        logger.error("Remote shell command timed out after %ds (%s)", timeout, cmd_summary)
        if process is not None:
            await _terminate_subprocess(process)
        return {
            "stdout": "",
            "stderr": f"Command timed out after {timeout} seconds.",
            "returncode": -1,
            "timed_out": True,
        }
    except asyncio.CancelledError:
        logger.warning("Remote shell command cancelled (%s)", cmd_summary)
        if process is not None:
            await _terminate_subprocess(process)
        raise
    except HTTPException:
        raise
    except Exception as exc:
        safe_err = _redact_shell_text(str(exc), extra_secrets=extra_secrets)
        logger.error("Shell execution failed (%s): %s", cmd_summary, safe_err)
        if process is not None:
            await _terminate_subprocess(process)
        raise HTTPException(status_code=500, detail=f"Shell execution failed: {safe_err}")
    finally:
        _SHELL_CONCURRENCY_STATE["active"] = max(0, _SHELL_CONCURRENCY_STATE["active"] - 1)

from fastapi import UploadFile, File
from fastapi.responses import FileResponse

@app.post("/teleport/push")
async def teleport_push(request: Request, file: UploadFile = File(...), filename: str = None):
    """
    Securely uploads a file to the remote Swarm Node's workspace.
    """
    client_ip = request.client.host if request.client else "unknown"
    is_allowed = False
    try:
        ip = ipaddress.ip_address(client_ip)
        if client_ip.startswith("100.") or ip.is_loopback or ip.is_private:
            is_allowed = True
    except ValueError:
        pass
        
    if not is_allowed:
        logger.warning(f"Rejected /teleport/push from untrusted IP: {client_ip}")
        raise HTTPException(status_code=403, detail="Forbidden: Tailscale or LAN IP required.")
        
    target_name = filename or file.filename
    if not target_name:
        raise HTTPException(status_code=403, detail="Filename missing.")
        
    if ".." in target_name or target_name.startswith("/") or target_name.startswith("\\"):
        raise HTTPException(status_code=403, detail="Invalid filename format. Path traversal detected.")
        
    workspace_dir = state_dir() / "swarm_workspace"
    workspace_dir.mkdir(parents=True, exist_ok=True)
    
    target_path = (workspace_dir / target_name).resolve()
    
    if not str(target_path).startswith(str(workspace_dir.resolve())):
        raise HTTPException(status_code=403, detail="Path traversal detected.")
        
    try:
        import aiofiles
        async with aiofiles.open(target_path, "wb") as f:
            while content := await file.read(1024 * 1024):
                await f.write(content)
        return {"status": "success", "message": f"File {target_name} uploaded successfully."}
    except Exception as e:
        logger.error(f"Failed to save file: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/teleport/pull")
async def teleport_pull(request: Request, filename: str):
    """
    Securely downloads a file from the remote Swarm Node's workspace.
    """
    client_ip = request.client.host if request.client else "unknown"
    is_allowed = False
    try:
        ip = ipaddress.ip_address(client_ip)
        if client_ip.startswith("100.") or ip.is_loopback or ip.is_private:
            is_allowed = True
    except ValueError:
        pass
        
    if not is_allowed:
        logger.warning(f"Rejected /teleport/pull from untrusted IP: {client_ip}")
        raise HTTPException(status_code=403, detail="Forbidden: Tailscale or LAN IP required.")
        
    if ".." in filename or filename.startswith("/") or filename.startswith("\\"):
        raise HTTPException(status_code=403, detail="Invalid filename format. Path traversal detected.")
        
    workspace_dir = state_dir() / "swarm_workspace"
    target_path = (workspace_dir / filename).resolve()
    
    if not str(target_path).startswith(str(workspace_dir.resolve())):
        raise HTTPException(status_code=403, detail="Path traversal detected.")
        
    if not target_path.exists() or not target_path.is_file():
        raise HTTPException(status_code=404, detail="File not found.")
        
    return FileResponse(path=str(target_path), filename=filename)

from zelyr.network.p2p_sync import get_receiver_protocol, get_receiver_pin

class PairRequest(BaseModel):
    public_key: str

@app.post("/sync/pair")
async def sync_pair(request: Request, payload: PairRequest):
    """Phase 1: Key Exchange"""
    pin = get_receiver_pin()
    if not pin:
        raise HTTPException(status_code=400, detail="Node is not in pairing mode.")
        
    protocol = get_receiver_protocol()
    try:
        # Derive shared key using client's public key and our PIN
        protocol.derive_shared_key(payload.public_key, pin)
    except Exception as e:
        logger.error(f"Pairing failed: {e}")
        raise HTTPException(status_code=400, detail="Invalid public key or handshake failed.")
        
    # Return our public key
    return {"public_key": protocol.get_public_key_pem()}

class CommitRequest(BaseModel):
    nonce: str
    ciphertext: str

@app.post("/sync/commit")
async def sync_commit(request: Request, payload: CommitRequest):
    """Phase 2: Receive Encrypted State"""
    protocol = get_receiver_protocol()
    
    success = protocol.import_state({
        "nonce": payload.nonce,
        "ciphertext": payload.ciphertext
    })
    
    if not success:
        raise HTTPException(status_code=403, detail="Decryption failed. Invalid PIN or corrupted payload.")
        
    # Clear PIN after successful pairing
    import zelyr.network.p2p_sync as p2p_sync
    p2p_sync.set_receiver_pin(None)
    
    return {"status": "success", "message": "Swarm state synchronized successfully."}
