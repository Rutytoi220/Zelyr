"""Zelyr Monitor App using Textual."""
from zelyr.compat import state_file
import json
import asyncio
import logging
from pathlib import Path
from textual.app import App, ComposeResult
from textual.containers import Vertical, Horizontal, Center, Middle
from textual.widgets import Header, Footer, Static, Label
from textual.reactive import reactive
import socket
import time
from zelyr.client.tui.widgets import TelemetryPanel, SwarmDebatePanel, FlightRecorderLog, RoutinesPanel, TelemetryUpdate, SwarmProposalUpdate, FlightRecordUpdate, RoutineUpdate
logger = logging.getLogger(__name__)

class OfflineOverlay(Static):
    """Displays a pulsing offline message."""

    def compose(self) -> ComposeResult:
        """Auto-generated docstring.


Returns:
    Return value.
"""
        with Center():
            with Middle():
                yield Label('DAEMON OFFLINE - WAITING FOR KERNEL...', classes='offline-pulse')

class ZelyrMonitorApp(App):
    """The Zelyr System Monitor TUI."""
    CSS_PATH = 'monitor.tcss'
    BINDINGS = [('d', 'toggle_dark', 'Toggle dark mode'), ('q', 'quit', 'Quit')]
    daemon_online = reactive(False)

    def __init__(self, **kwargs):
        """Auto-generated docstring.


Returns:
    Return value.
"""
        super().__init__(**kwargs)
        self.hostname = socket.gethostname()
        self.start_time = time.time()

    def compose(self) -> ComposeResult:
        """Auto-generated docstring.


Returns:
    Return value.
"""
        yield Header(show_clock=True)
        with Vertical(id='main-container'):
            with Horizontal(id='top-panels'):
                with Vertical(id='left-sidebar'):
                    yield TelemetryPanel()
                with Vertical(id='center-panel'):
                    yield FlightRecorderLog()
            with Horizontal(id='bottom-panels'):
                yield SwarmDebatePanel()
                yield RoutinesPanel()
        yield OfflineOverlay(id='offline-overlay')
        yield Footer()

    def watch_daemon_online(self, online: bool) -> None:
        """Toggle the offline overlay when state changes."""
        overlay = self.query_one('#offline-overlay')
        if online:
            overlay.display = False
        else:
            overlay.display = True

    async def on_mount(self) -> None:
        """Start the background task to listen to the Unix socket."""
        self.socket_task = asyncio.create_task(self._listen_to_daemon())

    async def _listen_to_daemon(self) -> None:
        """Connect to the Zelyr socket and dispatch events to UI."""
        socket_path = state_file("zelyr.sock", "axiom.sock")
        while True:
            try:
                if not socket_path.exists():
                    self.daemon_online = False
                    await asyncio.sleep(2)
                    continue
                reader, writer = await asyncio.open_unix_connection(str(socket_path))
                self.daemon_online = True
                self.post_message(FlightRecordUpdate(event_type='SYSTEM', source='Monitor', payload='Connected to Zelyr Daemon Unix Socket.'))
                while True:
                    line = await reader.readline()
                    if not line:
                        break
                    try:
                        payload = json.loads(line.decode().strip())
                        self._dispatch_payload(payload)
                    except json.JSONDecodeError:
                        continue
            except Exception as e:
                self.daemon_online = False
                await asyncio.sleep(2)

    def _dispatch_payload(self, payload: dict) -> None:
        """Route incoming JSON-RPC payloads to the appropriate UI widgets."""
        if 'method' not in payload:
            return
        method = payload['method']
        params = payload.get('params', {})
        if method == 'event.bus.published':
            event_type = params.get('event_type', 'unknown')
            source = params.get('source', 'unknown')
            data = params.get('data', {})
            recorder = self.query_one(FlightRecorderLog)
            recorder.post_message(FlightRecordUpdate(event_type=event_type, source=source, payload=str(data)[:200]))
            if event_type == 'telemetry.update':
                telemetry_panel = self.query_one(TelemetryPanel)
                telemetry_panel.post_message(TelemetryUpdate(ram=data.get('ram_percent', 0), vram=data.get('vram_percent', 0), intent=data.get('intent', 'orchestration')))
            elif event_type == 'swarm.proposal':
                swarm_panel = self.query_one(SwarmDebatePanel)
                swarm_panel.post_message(SwarmProposalUpdate(proposal_id=data.get('proposal_id', '?'), tool=data.get('tool', '?'), status='PROPOSED', agent=data.get('agent', '?')))
            elif event_type == 'swarm.vote':
                swarm_panel_vote = self.query_one(SwarmDebatePanel)
                swarm_panel_vote.post_message(SwarmProposalUpdate(proposal_id=data.get('proposal_id', '?'), tool='vote', status=data.get('vote', 'UNKNOWN'), agent=data.get('voter', '?')))
            elif event_type == 'routine.started' or event_type == 'routine.completed':
                routines_panel = self.query_one(RoutinesPanel)
                routines_panel.post_message(RoutineUpdate(routine_name=data.get('routine_name', '?'), next_run=data.get('next_run', 'Scheduled'), status='RUNNING' if 'started' in event_type else 'IDLE'))

# Legacy AXIOM API compatibility aliases
AxiomMonitorApp = ZelyrMonitorApp
