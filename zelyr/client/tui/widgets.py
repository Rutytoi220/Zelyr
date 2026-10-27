"""TUI Widgets for Zelyr Monitor."""
from textual.app import ComposeResult
from textual.containers import Vertical, Horizontal
from textual.widgets import Static, ProgressBar, RichLog, Label
from textual.message import Message

class TelemetryUpdate(Message):
    """Message to update telemetry."""

    def __init__(self, ram: float, vram: float, intent: str):
        """Auto-generated docstring.

Args:
    ram: Argument.
    vram: Argument.
    intent: Argument.

Returns:
    Return value.
"""
        self.ram = ram
        self.vram = vram
        self.intent = intent
        super().__init__()

class SwarmProposalUpdate(Message):
    """Message to update swarm debate."""

    def __init__(self, proposal_id: str, tool: str, status: str, agent: str):
        """Auto-generated docstring.

Args:
    proposal_id: Argument.
    tool: Argument.
    status: Argument.
    agent: Argument.

Returns:
    Return value.
"""
        self.proposal_id = proposal_id
        self.tool = tool
        self.status = status
        self.agent = agent
        super().__init__()

class FlightRecordUpdate(Message):
    """Message for new flight log record."""

    def __init__(self, event_type: str, source: str, payload: str):
        """Auto-generated docstring.

Args:
    event_type: Argument.
    source: Argument.
    payload: Argument.

Returns:
    Return value.
"""
        self.event_type = event_type
        self.source = source
        self.payload = payload
        super().__init__()

class RoutineUpdate(Message):
    """Message to update active routines."""

    def __init__(self, routine_name: str, next_run: str, status: str):
        """Auto-generated docstring.

Args:
    routine_name: Argument.
    next_run: Argument.
    status: Argument.

Returns:
    Return value.
"""
        self.routine_name = routine_name
        self.next_run = next_run
        self.status = status
        super().__init__()

class TelemetryPanel(Static):
    """Displays Hardware RAM/VRAM and active Inference Tier."""

    def compose(self) -> ComposeResult:
        """Auto-generated docstring.


Returns:
    Return value.
"""
        yield Label('Hardware Telemetry', classes='panel-title')
        yield Horizontal(Label('RAM:  ', id='ram-lbl'), ProgressBar(total=100, show_eta=False, id='ram-bar'), classes='telemetry-row')
        yield Horizontal(Label('VRAM: ', id='vram-lbl'), ProgressBar(total=100, show_eta=False, id='vram-bar'), classes='telemetry-row')
        yield Horizontal(Label('Active Intent: '), Label('ORCHESTRATION', id='intent-badge', classes='intent-badge'))

    def on_telemetry_update(self, message: TelemetryUpdate) -> None:
        """Update gauges when telemetry arrives."""
        ram_bar = self.query_one('#ram-bar', ProgressBar)
        vram_bar = self.query_one('#vram-bar', ProgressBar)
        intent_badge = self.query_one('#intent-badge', Label)
        ram_bar.update(progress=message.ram)
        vram_bar.update(progress=message.vram)
        intent_badge.update(message.intent.upper())

class SwarmDebatePanel(Static):
    """Displays active sub-agent proposals and voting status."""

    def __init__(self, **kwargs):
        """Auto-generated docstring.


Returns:
    Return value.
"""
        super().__init__(**kwargs)
        self.proposals = {}
        self.log_widget = None

    def compose(self) -> ComposeResult:
        """Auto-generated docstring.


Returns:
    Return value.
"""
        yield Label('Swarm Consensus Debate', classes='panel-title')
        self.log_widget = RichLog(highlight=True, markup=True)
        yield self.log_widget

    def on_swarm_proposal_update(self, message: SwarmProposalUpdate) -> None:
        """Auto-generated docstring.

Args:
    message: Argument.

Returns:
    Return value.
"""
        if not self.log_widget:
            return
        color = 'white'
        if message.status == 'APPROVED':
            color = 'green'
        elif message.status == 'REJECTED':
            color = 'red'
        elif message.status == 'PROPOSED':
            color = 'yellow'
        text = f'[[{color}]{message.status}[/]] {message.agent} -> {message.tool} (ID: {message.proposal_id[:8]})'
        self.log_widget.write(text)

class FlightRecorderLog(Static):
    """Displays a live streaming log of EventBus bus.published events."""

    def __init__(self, **kwargs):
        """Auto-generated docstring.


Returns:
    Return value.
"""
        super().__init__(**kwargs)
        self.log_widget = None

    def compose(self) -> ComposeResult:
        """Auto-generated docstring.


Returns:
    Return value.
"""
        yield Label('Live EventBus Flight Recorder', classes='panel-title')
        self.log_widget = RichLog(highlight=True, markup=True, wrap=True)
        yield self.log_widget

    def on_flight_record_update(self, message: FlightRecordUpdate) -> None:
        """Auto-generated docstring.

Args:
    message: Argument.

Returns:
    Return value.
"""
        if not self.log_widget:
            return
        color = 'cyan'
        if message.event_type == 'router.intent.classified':
            color = 'magenta'
        elif message.event_type == 'wasm.trap':
            color = 'red'
        elif message.event_type == 'rollback.triggered':
            color = 'yellow'
        text = f'[bold {color}]{message.event_type}[/] [dim]from {message.source}[/]\n  {message.payload}'
        self.log_widget.write(text)

class RoutinesPanel(Static):
    """Displays running background cron tasks."""

    def __init__(self, **kwargs):
        """Auto-generated docstring.


Returns:
    Return value.
"""
        super().__init__(**kwargs)
        self.log_widget = None

    def compose(self) -> ComposeResult:
        """Auto-generated docstring.


Returns:
    Return value.
"""
        yield Label('Active Background Routines', classes='panel-title')
        self.log_widget = RichLog(highlight=True, markup=True)
        yield self.log_widget

    def on_routine_update(self, message: RoutineUpdate) -> None:
        """Auto-generated docstring.

Args:
    message: Argument.

Returns:
    Return value.
"""
        if not self.log_widget:
            return
        color = 'green' if message.status == 'RUNNING' else 'yellow'
        text = f'[{color}]●[/] {message.routine_name} [dim](Next: {message.next_run})[/]'
        self.log_widget.write(text)
