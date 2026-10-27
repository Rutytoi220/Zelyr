import asyncio
import json
import logging
import os
import sys
from pathlib import Path
import websockets

from zelyr.core.events import Event

logger = logging.getLogger(__name__)

class ZelyrDaemonServer:
    def __init__(self):
        from zelyr.core.events import EventBus
        self.event_bus = EventBus()
        self.lifecycle = None
        self.cli = None
        
        self.scheduler_service = None
        self.telemetry = None
        self.governor = None
        self.swarm_router = None
        self.sys_watchdog = None
        self.governor_service = None
        self.indexer_service = None
        
        self.clients = set()
        self.event_bus.subscribe("*", self._on_bus_event)

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
        self.clients.add(websocket)
        logger.info("Client connected. Total clients: %d", len(self.clients))
        try:
            async for message in websocket:
                try:
                    data = json.loads(message)
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

    async def run(self, host: str = "127.0.0.1", port: int = 8000):
        loop = asyncio.get_running_loop()
        self._loop = loop
        
        logger.info(f"Starting Zelyr Daemon WebSocket server on ws://{host}:{port}")
        
        # We start the server in the background
        start_server = websockets.serve(self.handle_client, host, port)
        await start_server
        
        # Now spawn the heavy loading tasks in the background
        asyncio.create_task(self.initialize_background())
        
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
