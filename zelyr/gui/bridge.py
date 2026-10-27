from __future__ import annotations
"""Zelyr GUI — Async Signal Bridge.

Bridges Zelyr's synchronous EventBus and async OrchestratorAgent
to Qt's main thread via thread-safe Signal emissions.
"""


import asyncio
import json
import logging
import threading
import subprocess
from typing import Any

from PySide6.QtCore import QObject, Signal, QThread, Slot
from zelyr.client.ipc_client import ZelyrDaemonClient

logger = logging.getLogger(__name__)


class ZelyrBridge(QObject):
    """Thread-safe bridge between the Zelyr EventBus and the PySide6 UI.

    All Qt signals are emitted from the Qt main thread (via
    ``QMetaObject.invokeMethod`` / ``Signal`` auto-connection), so
    widget access is always safe.

    Signals
    -------
    token_received(str)
        Streamed LLM markdown chunk — connect to the chat renderer.
    tool_status_changed(str, str)
        (tool_id, status_message) emitted when a tool starts/finishes.
    telemetry_updated(dict)
        Routing telemetry snapshot: model, auth_mode, cpu %, ram %.
    response_finished(str)
        Complete aggregated response text when streaming ends.
    error_occurred(str)
        Human-readable error message.
    """

    # --- Qt Signals (all emitted from Qt thread) ---
    token_received: Signal = Signal(str)
    tool_status_changed: Signal = Signal(str, str)
    telemetry_updated: Signal = Signal(dict)
    telemetry_tick_received: Signal = Signal(dict)
    response_finished: Signal = Signal(str)
    error_occurred: Signal = Signal(str)
    system_alert: Signal = Signal(str, str)
    request_gui_auth: Signal = Signal(str, str, dict)
    
    # Swarm Signals
    swarm_agent_started: Signal = Signal(str, str)
    swarm_agent_token: Signal = Signal(str, str)
    swarm_agent_completed: Signal = Signal(str, str)
    swarm_status_changed: Signal = Signal(dict)
    connection_status_changed: Signal = Signal(str)
    tools_received: Signal = Signal(list)
    synapse_event: Signal = Signal(object)
    zelyrfs_status: Signal = Signal(str)
    governor_approval_requested: Signal = Signal(str, dict)
    ui_widget_generated: Signal = Signal(dict)
    startup_service_update: Signal = Signal(dict)
    lifecycle_state_changed: Signal = Signal(dict)
    agent_loop_status: Signal = Signal(dict)
    mcp_servers_updated: Signal = Signal(dict)
    power_mode_changed: Signal = Signal(bool, str)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._loop: asyncio.AbstractEventLoop | None = None
        self._client = ZelyrDaemonClient()
        self._client.on_event = self._on_daemon_event
        self._client.on_connect = self._on_daemon_connect
        self._client.on_disconnect = self._on_daemon_disconnect
        self._lock = threading.Lock()
        
        self.session_id: str | None = None
        from zelyr.memory.sessions import SessionDatabase
        self.session_db = SessionDatabase()

    def _on_daemon_connect(self):
        self.connection_status_changed.emit('connected')
        
    def _on_daemon_disconnect(self):
        self.connection_status_changed.emit('disconnected')

    # ------------------------------------------------------------------
    # Public API (called from the Qt main thread)
    # ------------------------------------------------------------------
    def set_event_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Register the qasync event loop so we can schedule coroutines."""
        self._loop = loop
        asyncio.run_coroutine_threadsafe(self._init_session(), loop)

    async def _init_session(self):
        await self.session_db.initialize()
        self.session_id = await self.session_db.create_session("Zelyr Desktop Session")

    def initialize_client(self) -> None:
        """Attempt to connect to the daemon, starting it if necessary."""
        if self._loop:
            asyncio.run_coroutine_threadsafe(self._connect_to_daemon(), self._loop)

    async def _connect_to_daemon(self):
        success = await self._client.connect()
        if success:
            return

        # Check if auto-start is enabled
        from zelyr.config import get_config
        config = get_config()
        if not getattr(config, 'auto_ollama_start', True):
            logger.info("Daemon offline and auto-start is disabled.")
            return

        logger.info("Daemon offline, attempting to auto-start...")
        self.connection_status_changed.emit('connecting')

        # Strategy 1: Try systemd user service (Linux)
        daemon_started = False
        try:
            result = subprocess.run(
                ['systemctl', '--user', 'start', 'zelyrd.service'],
                check=False, capture_output=True, timeout=5
            )
            if result.returncode == 0:
                daemon_started = True
                logger.info("Started daemon via systemctl --user start zelyrd.service")
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

        # Strategy 2: Spawn daemon subprocess directly as fallback
        if not daemon_started:
            try:
                import sys
                python = sys.executable
                subprocess.Popen(
                    [python, '-m', 'zelyr.api.cli', 'daemon', 'start'],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    start_new_session=True
                )
                daemon_started = True
                logger.info("Started daemon via subprocess fallback.")
                print("[GUI] Waiting 3 seconds for daemon to bind to port 8000...")
                await asyncio.sleep(3)
            except Exception as e:
                logger.error(f"Failed to start daemon subprocess: {e}")

        if not daemon_started:
            logger.error("All daemon auto-start strategies failed.")
            return

        # Retry connection with backoff (daemon needs time to boot, especially if pulling models)
        for attempt in range(15):
            delay = min(0.5 * (1.5 ** attempt), 5.0)
            await asyncio.sleep(delay)
            success = await self._client.connect()
            if success:
                logger.info(f"Connected to daemon on attempt {attempt + 1}.")
                return
            logger.debug(f"Daemon connect attempt {attempt + 1} failed, retrying...")

        logger.error("Daemon started but could not connect after 15 retries.")

    def get_available_models(self) -> list[str]:
        """Fetch available models synchronously for the UI."""
        try:
            result = subprocess.run(["ollama", "list"], capture_output=True, text=True, check=True)
            lines = result.stdout.strip().split('\n')[1:]
            models = set()
            for line in lines:
                if line.strip():
                    name = line.split()[0]
                    if name != "NAME":
                        models.add(name)
            return sorted(list(models))
        except Exception as e:
            logger.error(f"Failed to fetch available models: {e}")
            return []

    def refresh_models(self) -> None:
        """Trigger a background refresh of the available models."""
        pass

    def submit_task(self, user_input: str) -> None:
        """Schedule an orchestrator run on the async event loop.

        This is the *only* correct entry-point for sending a user message.
        It schedules a coroutine on the async loop (running in the Qt
        thread via qasync) so inference never blocks the UI.
        """
        if self._loop is None:
            self.error_occurred.emit("Event loop not initialised — cannot submit task.")
            return
        if not self._client.is_connected:
            self.error_occurred.emit("Daemon offline — cannot submit task.")
            return
            
        if self.session_id:
            msg = {"role": "user", "content": user_input}
            asyncio.run_coroutine_threadsafe(self.session_db.append_message(self.session_id, msg), self._loop)
            
        asyncio.run_coroutine_threadsafe(self._client.submit_task(user_input), self._loop)

    def request_tools(self) -> None:
        if self._loop is None or not self._client.is_connected:
            return
        asyncio.run_coroutine_threadsafe(self._client.request_tools(), self._loop)

    def toggle_tool(self, tool_id: str, enabled: bool) -> None:
        if self._loop is None or not self._client.is_connected:
            return
        asyncio.run_coroutine_threadsafe(self._client.toggle_tool(tool_id, enabled), self._loop)

    # ------------------------------------------------------------------
    def send_approval_response(self, tool_name: str, approved: bool) -> None:
        if self._loop is None or not self._client.is_connected:
            return
        event = {"type": "publish", "event": {"event_type": "governor.approval_response", "source": "gui", "data": {"tool_name": tool_name, "approved": approved}}}
        import json, asyncio
        asyncio.run_coroutine_threadsafe(self._client.websocket.send(json.dumps(event)), self._loop)

    def send_reload_plugins(self):
        if self._loop and self._client:
            asyncio.run_coroutine_threadsafe(
                self._client.ws.send(json.dumps({"action": "reload_plugins"})), 
                self._loop
            )

    def send_config_updated(self):
        if self._loop and self._client and self._client.ws:
            asyncio.run_coroutine_threadsafe(
                self._client.ws.send(json.dumps({"action": "config_updated"})), 
                self._loop
            )

    def send_add_mcp_server(self, name: str, command: str, args: list) -> None:
        if self._loop and self._client and self._client.ws:
            payload = {"action": "add_mcp_server", "name": name, "command": command, "args": args}
            asyncio.run_coroutine_threadsafe(
                self._client.ws.send(json.dumps(payload)), 
                self._loop
            )

    def send_remove_mcp_server(self, name: str) -> None:
        if self._loop and self._client and self._client.ws:
            payload = {"action": "remove_mcp_server", "name": name}
            asyncio.run_coroutine_threadsafe(
                self._client.ws.send(json.dumps(payload)), 
                self._loop
            )

    def send_get_mcp_status(self) -> None:
        if self._loop and self._client and self._client.ws:
            payload = {"action": "get_mcp_status"}
            asyncio.run_coroutine_threadsafe(
                self._client.ws.send(json.dumps(payload)), 
                self._loop
            )

    def set_strict_mode(self, enabled: bool) -> None:
        if self._loop is None or not self._client.is_connected:
            return
        event = {"type": "publish", "event": {"event_type": "governor.set_strict_mode", "source": "gui", "data": {"enabled": enabled}}}
        import json, asyncio
        asyncio.run_coroutine_threadsafe(self._client.websocket.send(json.dumps(event)), self._loop)

    # Internal async task runner
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # EventBus subscription handlers (called from EventBus thread)
    # ------------------------------------------------------------------

    def _on_daemon_event(self, data: dict) -> None:
        """Route incoming daemon JSON events to Qt signals."""
        msg_type = data.get("type", "")
        if msg_type == "response" and data.get("action") == "get_tools":
            self.tools_received.emit(data.get("data", []))
            return
            
        event_type = data.get("event_type", "")
        payload = data.get("payload", {})
        
        if event_type == "llm.token":
            self.token_received.emit(payload.get("token", ""))
        elif event_type.startswith("tool."):
            tool_id = payload.get("tool_id") or payload.get("tool_name", "Unknown Tool")
            if event_type == "tool.started":
                self.tool_status_changed.emit(tool_id, f"Running {tool_id}...")
            elif event_type == "tool.finished":
                self.tool_status_changed.emit(tool_id, f"{tool_id} completed.")
        elif event_type == "telemetry.update":
            self.telemetry_updated.emit(payload)
        elif event_type == "system.telemetry.tick":
            self.telemetry_tick_received.emit(payload)
        elif event_type == "orchestrator.finished":
            resp = payload.get("response", "")
            if self.session_id and resp and self._loop:
                msg = {"role": "assistant", "content": resp}
                asyncio.run_coroutine_threadsafe(self.session_db.append_message(self.session_id, msg), self._loop)
            self.response_finished.emit(resp)
        elif event_type.startswith("synapse."):
            class _Evt:
                def __init__(self, t, d):
                    self.event_type = t
                    self.data = d
            self.synapse_event.emit(_Evt(event_type, payload))
        elif event_type == "system.alert":
            self.system_alert.emit(payload.get("level", "info"), payload.get("message", ""))
        elif event_type == "ui.widget_generated":
            self.ui_widget_generated.emit(payload)
        elif event_type == "governor.approval_requested":
            self.governor_approval_requested.emit(payload.get("tool_name", ""), payload.get("arguments", {}))
        elif event_type == "zelyrfs.status":
            self.zelyrfs_status.emit(payload.get("status", "Unknown"))
        elif event_type == "startup.service.update" or event_type == "startup.service.ready":
            self.startup_service_update.emit(payload)
        elif event_type == "lifecycle.state_changed":
            self.lifecycle_state_changed.emit(payload)
        elif event_type == "agent.loop.status":
            self.agent_loop_status.emit(payload)
        elif event_type == "mcp.servers.updated":
            self.mcp_servers_updated.emit(payload)
        elif event_type == "system.power.throttled":
            msg = "⚠️ **[POWER SAVING MODE ENABLED]:** Battery below 30%. Zelyr has autonomously downshifted to a lightweight cognitive model to preserve system power."
            self.power_mode_changed.emit(True, msg)
        elif event_type == "system.power.restored":
            msg = "🔋 **[POWER RESTORED]:** Charger connected. Zelyr is reverting to the primary cognitive model."
            self.power_mode_changed.emit(False, msg)
        elif event_type.startswith("swarm."):
            self._on_swarm_event(event_type, payload)

    def _on_swarm_event(self, event_type: str, payload: dict) -> None:
        """Relay swarm telemetry to the Qt thread."""
        agent = payload.get("agent_name", "Unknown")
        if event_type == "swarm.agent.started":
            self.swarm_agent_started.emit(agent, str(payload.get("assigned_task")))
        elif event_type == "swarm.agent.token":
            self.swarm_agent_token.emit(agent, str(payload.get("chunk")))
        elif event_type == "swarm.agent.completed":
            self.swarm_agent_completed.emit(agent, str(payload.get("result_summary")))
        elif event_type == "swarm.status.changed":
            self.swarm_status_changed.emit(payload)

# Legacy AXIOM API compatibility aliases
AxiomBridge = ZelyrBridge
