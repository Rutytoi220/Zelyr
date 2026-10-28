import asyncio
from collections import deque
from enum import Enum
import json
import logging
import os
from pathlib import Path
import secrets
import sys
import time
from typing import Any, Callable, Deque, Dict, List, Optional
import urllib.parse
import websockets

from zelyr.compat import data_dir
from zelyr.core.events import Event
from zelyr.server.protocol import (
    JSONRPCRequest,
    MAX_PAYLOAD_SIZE,
    METHOD_NOT_FOUND,
    RATE_LIMIT_EXCEEDED,
    SlidingWindowRateLimiter,
    make_error_response,
    make_success_response,
    parse_and_validate_message,
)

logger = logging.getLogger(__name__)


class WorkerState(str, Enum):
    INITIAL = "INITIAL"
    RUNNING = "RUNNING"
    STOPPED = "STOPPED"
    CRASHED = "CRASHED"
    DEGRADED = "DEGRADED"
    CIRCUIT_BROKEN = "CIRCUIT_BROKEN"


class SupervisedWorker:
    """Descriptor and state tracking for a supervised daemon worker process or async task."""

    def __init__(
        self,
        name: str,
        target: Any,
        is_process: bool = False,
        env: Optional[Dict[str, str]] = None,
        cwd: Optional[str] = None,
    ):
        self.name = name
        self.target = target
        self.is_process = is_process
        self.env = env
        self.cwd = cwd
        self.task: Optional[asyncio.Task] = None
        self.process: Optional[Any] = None
        self.crash_timestamps: Deque[float] = deque()
        self.restart_count: int = 0
        self.state: WorkerState = WorkerState.INITIAL
        self.last_error: Optional[str] = None
        self.next_retry_at: float = 0.0

    def record_crash(self, error: str, now: Optional[float] = None) -> None:
        ts = time.time() if now is None else now
        self.crash_timestamps.append(ts)
        self.restart_count += 1
        self.last_error = error
        self.state = WorkerState.CRASHED
        while self.crash_timestamps and ts - self.crash_timestamps[0] > 60.0:
            self.crash_timestamps.popleft()

    def should_trip_circuit_breaker(self, now: Optional[float] = None) -> bool:
        ts = time.time() if now is None else now
        while self.crash_timestamps and ts - self.crash_timestamps[0] > 60.0:
            self.crash_timestamps.popleft()
        return len(self.crash_timestamps) >= 5

    def get_backoff_delay(self) -> float:
        crashes = len(self.crash_timestamps)
        if crashes <= 1:
            return 1.0
        return min(1.0 * (2 ** (crashes - 1)), 30.0)


class DaemonWorkerSupervisor:
    """Supervises background worker loops and child processes with exponential backoff and circuit breaking."""

    def __init__(self, event_bus: Optional[Any] = None):
        self.event_bus = event_bus
        self.workers: Dict[str, SupervisedWorker] = {}
        self._monitoring_task: Optional[asyncio.Task] = None
        self._is_running: bool = False

    def register_worker(
        self,
        name: str,
        target: Any,
        is_process: bool = False,
        env: Optional[Dict[str, str]] = None,
        cwd: Optional[str] = None,
    ) -> SupervisedWorker:
        worker = SupervisedWorker(name, target, is_process=is_process, env=env, cwd=cwd)
        self.workers[name] = worker
        return worker

    async def start_worker(self, name: str) -> bool:
        worker = self.workers.get(name)
        if not worker:
            raise KeyError(f"Unknown worker: {name}")

        now = time.time()
        if worker.should_trip_circuit_breaker(now):
            worker.state = WorkerState.CIRCUIT_BROKEN
            logger.error("Circuit breaker TRIPPED for worker '%s': 5 crashes within 60s.", name)
            if self.event_bus:
                self.event_bus.publish_sync(
                    "supervisor.circuit_breaker.tripped",
                    {"worker": name, "crashes": len(worker.crash_timestamps), "last_error": worker.last_error},
                )
            return False

        worker.state = WorkerState.RUNNING
        try:
            if worker.is_process:
                cmd = worker.target if isinstance(worker.target, list) else [str(worker.target)]
                worker.process = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    env=worker.env,
                    cwd=worker.cwd,
                )
            else:
                if asyncio.iscoroutinefunction(worker.target):
                    worker.task = asyncio.create_task(worker.target())
                elif callable(worker.target):
                    worker.task = asyncio.create_task(asyncio.to_thread(worker.target))
            return True
        except Exception as e:
            worker.record_crash(str(e), now)
            if worker.should_trip_circuit_breaker(now):
                worker.state = WorkerState.CIRCUIT_BROKEN
                if self.event_bus:
                    self.event_bus.publish_sync(
                        "supervisor.circuit_breaker.tripped",
                        {"worker": name, "crashes": len(worker.crash_timestamps), "last_error": str(e)},
                    )
            return False

    async def monitor_once(self) -> None:
        """Non-blocking polling sweep across registered workers."""
        now = time.time()
        for name, worker in list(self.workers.items()):
            if worker.state in (WorkerState.STOPPED, WorkerState.CIRCUIT_BROKEN):
                continue

            if worker.state == WorkerState.RUNNING:
                if worker.is_process and worker.process:
                    if worker.process.returncode is not None:
                        exit_code = worker.process.returncode
                        stderr_snippet = ""
                        try:
                            if worker.process.stderr:
                                err_bytes = await asyncio.wait_for(worker.process.stderr.read(), timeout=0.1)
                                stderr_snippet = err_bytes.decode(errors="replace").strip()
                        except Exception:
                            pass
                        err_msg = f"Process exited with code {exit_code}. Stderr: {stderr_snippet}"
                        worker.record_crash(err_msg, now)
                        if worker.should_trip_circuit_breaker(now):
                            worker.state = WorkerState.CIRCUIT_BROKEN
                            logger.error("Circuit breaker TRIPPED for worker '%s': 5 crashes within 60s.", name)
                            if self.event_bus:
                                self.event_bus.publish_sync(
                                    "supervisor.circuit_breaker.tripped",
                                    {"worker": name, "crashes": len(worker.crash_timestamps), "last_error": err_msg},
                                )
                        else:
                            delay = worker.get_backoff_delay()
                            worker.next_retry_at = now + delay
                            logger.warning("Worker '%s' crashed (exit %s). Retrying in %ss.", name, exit_code, delay)

                elif not worker.is_process and worker.task:
                    if worker.task.done():
                        exc = None
                        try:
                            exc = worker.task.exception()
                        except (asyncio.CancelledError, asyncio.InvalidStateError):
                            pass
                        err_msg = f"Task failed: {exc}" if exc else "Task finished unexpectedly."
                        worker.record_crash(err_msg, now)
                        if worker.should_trip_circuit_breaker(now):
                            worker.state = WorkerState.CIRCUIT_BROKEN
                            logger.error("Circuit breaker TRIPPED for worker '%s': 5 crashes within 60s.", name)
                            if self.event_bus:
                                self.event_bus.publish_sync(
                                    "supervisor.circuit_breaker.tripped",
                                    {"worker": name, "crashes": len(worker.crash_timestamps), "last_error": err_msg},
                                )
                        else:
                            delay = worker.get_backoff_delay()
                            worker.next_retry_at = now + delay
                            logger.warning("Worker '%s' crashed (%s). Retrying in %ss.", name, err_msg, delay)

            elif worker.state == WorkerState.CRASHED:
                if now >= worker.next_retry_at:
                    await self.start_worker(name)

    async def start(self) -> None:
        self._is_running = True
        for name in list(self.workers.keys()):
            await self.start_worker(name)
        self._monitoring_task = asyncio.create_task(self._supervision_loop())

    async def _supervision_loop(self) -> None:
        while self._is_running:
            try:
                await self.monitor_once()
            except Exception as e:
                logger.error("Error in supervisor monitoring loop: %s", e)
            await asyncio.sleep(0.5)

    async def shutdown(self, timeout: float = 5.0) -> None:
        """Safe shutdown & zombie cleanup: SIGTERM, bounded 5s wait, then SIGKILL."""
        self._is_running = False
        if self._monitoring_task:
            self._monitoring_task.cancel()

        for name, worker in self.workers.items():
            worker.state = WorkerState.STOPPED
            if worker.task and not worker.task.done():
                worker.task.cancel()

            if worker.is_process and worker.process and worker.process.returncode is None:
                try:
                    worker.process.terminate()
                    try:
                        await asyncio.wait_for(worker.process.wait(), timeout=timeout)
                    except asyncio.TimeoutError:
                        logger.warning("Worker process '%s' did not terminate in %ss; sending SIGKILL.", name, timeout)
                        worker.process.kill()
                        await worker.process.wait()
                except ProcessLookupError:
                    pass
                except Exception as e:
                    logger.error("Error shutting down worker process '%s': %s", name, e)


class ZelyrDaemonServer:
    def __init__(self, auth_token: Optional[str] = None):
        from zelyr.core.events import EventBus
        self.event_bus = EventBus()
        self.supervisor = DaemonWorkerSupervisor(event_bus=self.event_bus)
        self.lifecycle = None
        self.cli = None

        self.auth_token = auth_token or os.environ.get("ZELYR_AUTH_TOKEN") or secrets.token_urlsafe(32)
        self.token_file = data_dir() / "daemon.token"
        self._write_token_file()
        
        self.scheduler_service = None
        self.telemetry = None
        self.governor = None
        self.swarm_router = None
        self.sys_watchdog = None
        self.governor_service = None
        self.indexer_service = None
        self._server = None
        self._loop = None
        
        self.clients = set()
        self.event_bus.subscribe("*", self._on_bus_event)

    def _write_token_file(self) -> None:
        """Write active runtime auth token with strict 0600 POSIX permissions."""
        try:
            self.token_file.parent.mkdir(parents=True, exist_ok=True)
            fd = os.open(str(self.token_file), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with open(fd, "w", encoding="utf-8") as f:
                f.write(self.auth_token)
            os.chmod(self.token_file, 0o600)
        except Exception as e:
            logger.warning("[Daemon] Could not write secure daemon.token: %s", e)

    def _extract_token(self, websocket: Any) -> Optional[str]:
        """Extract auth token from Authorization header or ?token= query parameter."""
        req = getattr(websocket, "request", None)
        headers = getattr(req, "headers", None) if req else getattr(websocket, "request_headers", None)
        path = getattr(req, "path", None) if req else getattr(websocket, "path", None)

        if headers:
            auth_header = headers.get("Authorization") or headers.get("authorization")
            if auth_header and auth_header.startswith(("Bearer ", "bearer ")):
                return auth_header.split(" ", 1)[1].strip()

        if path:
            parsed = urllib.parse.urlparse(path)
            q = urllib.parse.parse_qs(parsed.query)
            tokens = q.get("token")
            if tokens and tokens[0]:
                return tokens[0].strip()

        return None

    async def initialize_background(self):
        from zelyr.core.lifecycle import AppLifecycleState, LifecycleManager

        self.lifecycle = LifecycleManager(self.event_bus)
        self.lifecycle.transition(AppLifecycleState.CORE_INITIALIZING)

        def _emit_status(service_name: str, state: AppLifecycleState) -> None:
            self.event_bus.publish_sync("startup.service.update", {"service": service_name, "state": state.value})

        # DirectoryWatchdog — lightweight, safe to instantiate eagerly
        try:
            from zelyr.services.watchdog_service import DirectoryWatchdog
            self.dir_watchdog = DirectoryWatchdog()
            self.dir_watchdog.start(asyncio.get_running_loop())
            _emit_status("dir_watchdog", AppLifecycleState.READY)
        except Exception as e:
            logger.error("Failed to start dir_watchdog: %s", e)
            _emit_status("dir_watchdog", AppLifecycleState.DEGRADED)

        # Heavy CLI Init in a separate thread so we don't block the daemon's WS server
        try:
            from zelyr.api.cli import CLI

            def _build_cli():
                return CLI(bus=self.event_bus)
            
            self.cli = await asyncio.to_thread(_build_cli)
            _emit_status("core", AppLifecycleState.READY)
        except Exception as e:
            logger.error("Failed to initialize CLI: %s", e)
            _emit_status("core", AppLifecycleState.DEGRADED)

        try:
            from zelyr.services.scheduler_service import BackgroundSchedulerService
            self.scheduler_service = BackgroundSchedulerService(event_bus=self.event_bus)
            self.scheduler_service.start()
            _emit_status("scheduler", AppLifecycleState.READY)
        except Exception as e:
            logger.error("Failed to start scheduler: %s", e)
            _emit_status("scheduler", AppLifecycleState.DEGRADED)

        try:
            from zelyr.services.telemetry_service import TelemetryService
            self.telemetry = TelemetryService(event_bus=self.event_bus)
            self.telemetry.start()
            _emit_status("telemetry", AppLifecycleState.READY)
        except Exception as e:
            logger.error("Failed to start telemetry: %s", e)
            _emit_status("telemetry", AppLifecycleState.DEGRADED)

        try:
            from zelyr.core.governor import ThermalGovernor
            from zelyr.services.governor import GovernorService
            self.governor = ThermalGovernor.instance(event_bus=self.event_bus)
            self.governor_service = GovernorService.instance(event_bus=self.event_bus)
            _emit_status("governor", AppLifecycleState.READY)
        except Exception as e:
            logger.error("Failed to start governor: %s", e)
            _emit_status("governor", AppLifecycleState.DEGRADED)

        try:
            from zelyr.core.swarm_router import SwarmRouter
            self.swarm_router = SwarmRouter.instance(event_bus=self.event_bus)
            _emit_status("swarm_router", AppLifecycleState.READY)
        except Exception as e:
            logger.error("Failed to start swarm_router: %s", e)
            _emit_status("swarm_router", AppLifecycleState.DEGRADED)

        try:
            from zelyr.services.sys_watchdog import SystemHealthWatchdog
            self.sys_watchdog = SystemHealthWatchdog(submit_task_callback=self._submit_task)
            _emit_status("sys_watchdog", AppLifecycleState.READY)
        except Exception as e:
            logger.error("Failed to start sys_watchdog: %s", e)
            _emit_status("sys_watchdog", AppLifecycleState.DEGRADED)

        try:
            from zelyr.memory.indexer import IndexerService
            self.indexer_service = IndexerService(event_bus=self.event_bus)
            self.indexer_service.start()
            _emit_status("indexer", AppLifecycleState.READY)
        except Exception as e:
            logger.error("Failed to start indexer: %s", e)
            _emit_status("indexer", AppLifecycleState.DEGRADED)

        try:
            from zelyr.services.cron_service import CronService
            self.cron_service = CronService(event_bus=self.event_bus)
            asyncio.create_task(self.cron_service.start())
            _emit_status("cron", AppLifecycleState.READY)
        except Exception as e:
            logger.error("Failed to start CronService: %s", e)
            _emit_status("cron", AppLifecycleState.DEGRADED)

        try:
            from zelyr.core.wake_word import WakeWordService
            from zelyr.core.audio import AudioManager
            self.wake_word_service = WakeWordService(event_bus=self.event_bus)
            
            from zelyr.config import get_config
            if getattr(get_config(), "wake_word_enabled", False):
                self.wake_word_service.start()
                
            # Init AudioManager for the daemon with bus
            AudioManager.instance(event_bus=self.event_bus)
            _emit_status("wake_word", AppLifecycleState.READY)
        except Exception as e:
            logger.error("Failed to start WakeWordService: %s", e)
            _emit_status("wake_word", AppLifecycleState.DEGRADED)

        # Determine final state: at least one service degraded => overall degraded
        degraded_services = []
        for svc in (
            self.cli, self.scheduler_service, self.telemetry,
            self.governor, self.swarm_router, self.sys_watchdog, self.indexer_service,
        ):
            if svc is None:
                degraded_services.append(type(svc).__name__ if svc else "unknown")

        final_state = AppLifecycleState.DEGRADED if degraded_services else AppLifecycleState.READY
        self.lifecycle.transition(final_state)
        self.event_bus.publish_sync("startup.service.ready", {"state": final_state.value})
        
        # Subscribe to EventBus to broadcast to websockets
        self.event_bus.subscribe("*", self._on_bus_event)

    def _submit_task(self, prompt: str):
        self.event_bus.publish_sync("orchestrator.trigger", data={"prompt": prompt, "source": "daemon_watchdog"})

    def _on_bus_event(self, event):
        """Broadcast internal events to all connected WS clients."""
        if not self.clients:
            return
            
        name = getattr(event, 'name', getattr(event, 'event_type', 'unknown'))
        payload = getattr(event, 'payload', getattr(event, 'data', None))
        
        # Only broadcast relevant events to frontend to save bandwidth
        if not name.startswith(("llm.", "tool.", "swarm.", "telemetry.", "orchestrator.", "lifecycle.", "startup.")):
            return
            
        msg = {
            "type": "event",
            "event_type": name,
            "payload": payload
        }
        msg_str = json.dumps(msg, default=str)
        
        # We may be inside a ThreadPoolExecutor (e.g. from orchestrator LLM calls)
        # We must schedule the async task on the main loop thread-safely
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # We are not in the main thread with a running loop, find the actual loop
            # Fallback for some environments, though get_running_loop might throw here
            pass
            
        # Instead, let's just use the server's main loop if we can capture it
        if hasattr(self, '_loop') and self._loop:
            for ws in list(self.clients):
                asyncio.run_coroutine_threadsafe(self._send_safe(ws, msg_str), self._loop)
        else:
            # Fallback if _loop not set
            for ws in list(self.clients):
                try:
                    asyncio.get_event_loop().call_soon_threadsafe(
                        lambda w=ws, m=msg_str: asyncio.create_task(self._send_safe(w, m))
                    )
                except Exception:
                    pass

    async def _send_safe(self, ws, msg: str):
        try:
            await ws.send(msg)
        except Exception:
            pass

    async def handle_client(self, websocket):
        provided_token = self._extract_token(websocket)
        if not provided_token or not secrets.compare_digest(provided_token, self.auth_token):
            logger.warning("[Daemon] Unauthorized connection attempt rejected.")
            try:
                await websocket.close(code=4401, reason="Unauthorized")
            except Exception:
                pass
            return

        self.clients.add(websocket)
        logger.info("Client connected. Total clients: %d", len(self.clients))
        rate_limiter = SlidingWindowRateLimiter(max_requests=50, window_seconds=1.0)
        try:
            async for message in websocket:
                msg_len = len(message.encode("utf-8") if isinstance(message, str) else message)
                if msg_len > MAX_PAYLOAD_SIZE:
                    logger.warning("[Daemon] Message size %d exceeds 1MB. Closing with code 4409.", msg_len)
                    await websocket.close(code=4409, reason="Message Too Big")
                    break

                if not rate_limiter.allow_request():
                    logger.warning("[Daemon] Rate limit exceeded for client.")
                    err_env = make_error_response(None, RATE_LIMIT_EXCEEDED, "Rate limit exceeded: max 50 requests/sec")
                    await websocket.send(json.dumps(err_env))
                    continue

                parsed_req, val_err = parse_and_validate_message(message)
                if val_err:
                    await websocket.send(json.dumps(val_err))
                    continue

                try:
                    if isinstance(parsed_req, JSONRPCRequest):
                        method = parsed_req.method
                        req_id = parsed_req.id

                        if method in ("sync_state", "get_state"):
                            state_val = self.lifecycle.state.value if self.lifecycle else "BOOTING"
                            await websocket.send(json.dumps(make_success_response(req_id, {"state": state_val})))
                        elif method == "get_tools":
                            tools_list = []
                            if self.cli:
                                tools = self.cli.orchestrator._tool_registry._core_registry.list_tools()
                                from zelyr.config import get_config
                                disabled = getattr(get_config(), "disabled_plugins", [])
                                for tid, t in sorted(tools.items()):
                                    desc = getattr(t, 'description', '') or (t.__doc__.strip().split('\n')[0] if t.__doc__ else "")
                                    tools_list.append({"id": tid, "description": desc, "enabled": tid not in disabled})
                            await websocket.send(json.dumps(make_success_response(req_id, tools_list)))
                        elif method == "ping":
                            await websocket.send(json.dumps(make_success_response(req_id, "pong")))
                        else:
                            await websocket.send(json.dumps(make_error_response(req_id, METHOD_NOT_FOUND, f"Method '{method}' not found")))
                        continue

                    data = parsed_req
                    action = data.get("action")
                    if not self.cli:
                        if action == "get_tools":
                            await websocket.send(json.dumps({"type": "response", "action": "get_tools", "data": []}))
                        elif action == "sync_state":
                            state_val = self.lifecycle.state.value if self.lifecycle else "BOOTING"
                            await websocket.send(json.dumps({
                                "type": "event",
                                "event_type": "lifecycle.state_changed",
                                "payload": {"new_state": state_val}
                            }))
                        else:
                            await websocket.send(json.dumps({"type": "event", "event_type": "system.alert", "payload": {"level": "warning", "message": "Zelyr core is still initializing... please wait."}}))
                        continue
                        

                    action = data.get("action")
                    if action == "sync_state":
                        state_val = self.lifecycle.state.value if self.lifecycle else "BOOTING"
                        await websocket.send(json.dumps({
                            "type": "event",
                            "event_type": "lifecycle.state_changed",
                            "payload": {"new_state": state_val}
                        }))
                    elif action == "submit_task":
                        prompt = data.get("prompt")
                        if prompt:
                            # Run in executor to avoid blocking the websocket loop.
                            # run_in_executor returns a Future and schedules it automatically.
                            loop = asyncio.get_running_loop()
                            loop.run_in_executor(None, self.cli.orchestrator.run, prompt, True)
                    elif action == "reload_plugins":
                        logger.info("Hot-reloading plugins triggered via IPC...")
                        self.cli.orchestrator.reload_plugins()
                        await websocket.send(json.dumps({"type": "response", "action": "reload_plugins", "success": True}))
                    elif action == "config_updated":
                        logger.info("Config update triggered via IPC. Reloading...")
                        from zelyr.config import set_config, ZelyrConfig
                        set_config(ZelyrConfig.load())
                        if self.event_bus:
                            self.event_bus.publish_sync("config.updated", {})
                        await websocket.send(json.dumps({"type": "response", "action": "config_updated", "success": True}))
                    elif action == "get_tools":
                        tools = self.cli.orchestrator._tool_registry._core_registry.list_tools()
                        from zelyr.config import get_config
                        disabled = getattr(get_config(), "disabled_plugins", [])
                        
                        tool_list = []
                        for tid, t in sorted(tools.items()):
                            desc = getattr(t, 'description', '')
                            if not desc and t.__doc__:
                                desc = t.__doc__.strip().split('\n')[0]
                            tool_list.append({
                                "id": tid,
                                "description": desc or "No description provided.",
                                "enabled": tid not in disabled
                            })
                            
                        await websocket.send(json.dumps({
                            "type": "response",
                            "action": "get_tools",
                            "data": tool_list
                        }))
                    elif action == "toggle_tool":
                        tool_id = data.get("tool_id")
                        enabled = data.get("enabled")
                        from zelyr.config import get_config
                        config = get_config()
                        disabled_list = getattr(config, 'disabled_plugins', [])
                        
                        if enabled and tool_id in disabled_list:
                            disabled_list.remove(tool_id)
                        elif not enabled and tool_id not in disabled_list:
                            disabled_list.append(tool_id)
                            
                        config.disabled_plugins = disabled_list
                        config.save()
                        
                        await websocket.send(json.dumps({
                            "type": "response",
                            "action": "toggle_tool",
                            "success": True
                        }))
                    elif action == "add_mcp_server":
                        name = data.get("name")
                        cmd = data.get("command")
                        args = data.get("args", [])
                        if name and cmd and hasattr(self.cli, "mcp_hub"):
                            await self.cli.mcp_hub.add_server_config(name, cmd, args)
                    elif action == "remove_mcp_server":
                        name = data.get("name")
                        if name and hasattr(self.cli, "mcp_hub"):
                            await self.cli.mcp_hub.remove_server_config(name)
                    elif action == "get_mcp_status":
                        if hasattr(self.cli, "mcp_hub"):
                            await self.cli.mcp_hub.broadcast_status()
                except Exception as e:
                    logger.error("[Daemon] Error handling client message: %s", e, exc_info=True)
        except Exception as we:
            logger.error("[Daemon] Client connection error: %s", we)
        finally:
            self.clients.discard(websocket)
            logger.info("Client disconnected.")

    async def start(self, host: str = "127.0.0.1", port: int = 8000, start_background: bool = True):
        loop = asyncio.get_running_loop()
        self._loop = loop

        logger.info(f"Starting Zelyr Daemon WebSocket server on ws://{host}:{port}")
        self._server = await websockets.serve(self.handle_client, host, port)

        if start_background:
            asyncio.create_task(self.initialize_background())
        return self._server

    async def stop(self):
        """Cleanly stop the daemon server, supervisor, and disconnect all clients."""
        if hasattr(self, "token_file") and self.token_file and self.token_file.exists():
            try:
                self.token_file.unlink()
            except Exception:
                pass
        if hasattr(self, "supervisor") and self.supervisor:
            await self.supervisor.shutdown(timeout=5.0)
        for ws in list(self.clients):
            try:
                await ws.close()
            except Exception:
                pass
        self.clients.clear()
        if self._server:
            self._server.close()
            await self._server.wait_closed()
            self._server = None

    async def run(self, host: str = "127.0.0.1", port: int = 8000):
        await self.start(host=host, port=port, start_background=True)
        await asyncio.Future()  # run forever

def main():
    import argparse
    import websockets

    parser = argparse.ArgumentParser(description="Zelyr Background Bridge Daemon")
    parser.add_argument("--host", default="127.0.0.1", help="Host interface to bind (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind (default: 8000)")
    parser.add_argument("--log-level", default="INFO", help="Logging level (default: INFO)")
    args = parser.parse_args()

    level = getattr(logging, args.log_level.upper(), logging.INFO)
    logging.basicConfig(level=level, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
    daemon = ZelyrDaemonServer()
    try:
        asyncio.run(daemon.run(host=args.host, port=args.port))
    except KeyboardInterrupt:
        logger.info("Daemon shutting down.")
        if daemon.cli:
            daemon.cli.close()

if __name__ == "__main__":
    main()

# Legacy AXIOM API compatibility aliases
AxiomDaemonServer = ZelyrDaemonServer
