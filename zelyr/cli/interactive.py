"""Interactive Magic DevEx Runner."""
import logging
import uuid
from pathlib import Path
from typing import Any
from rich.console import Console
from rich.status import Status
from zelyr.core.events import EventBus
from zelyr.core.transaction import WorkspaceTransactionManager
from zelyr.indexer.graph_engine import CodeGraphIndex
from zelyr.agents.orchestrator_agent import OrchestratorAgent
logger = logging.getLogger(__name__)

class InteractiveWorkflowRunner:
    """Coordinates complex workflow steps and reports progress interactively."""

    def __init__(self, engine: Any, orchestrator: OrchestratorAgent):
        """Auto-generated docstring.

Args:
    engine: Argument.
    orchestrator: Argument.

Returns:
    Return value.
"""
        self.engine = engine
        self.orchestrator = orchestrator
        self.bus: EventBus = engine.event_bus
        self.console = Console()

    def run(self, prompt: str) -> None:
        """Run the magic DevEx workflow for a single prompt."""
        self.console.print(f"\n[bold blue]zelyr run[/bold blue] '{prompt}'\n")
        with self.console.status('[bold cyan]Indexing AST Workspace...[/bold cyan]') as status:
            index = CodeGraphIndex()
            workspace_dir = Path.cwd()
            index.index_workspace(workspace_dir)
            num_nodes = len(index.graph.nodes)
            self.console.print(f'[green][✓][/green] AST Graph Indexed (Found {num_nodes} nodes)')
        session_id = str(uuid.uuid4())
        txn = WorkspaceTransactionManager(bus=self.bus, transaction_id=session_id)
        if not hasattr(self.engine, 'active_transactions'):
            self.engine.active_transactions = []
        self.engine.active_transactions.append(txn)
        try:
            txn.begin()
            self.console.print('[green][✓][/green] Transaction Staged (Copy-on-write backup active)')
            with self.console.status('[bold magenta][⏳] OrchestratorAgent analyzing and refactoring...[/bold magenta]') as status:
                result = self.orchestrator.run(prompt, use_tools=True, session_id=session_id)
            if result.success:
                self.console.print(f'[green][✓][/green] Orchestrator completed successfully.')
                self.console.print("[yellow][!][/yellow] Ready to commit. Run 'zelyr commit' to apply.")
            else:
                self.console.print(f'[red][✗][/red] Orchestrator failed: {result.error}')
                txn.rollback()
                raise RuntimeError(result.error)
        except Exception as e:
            if txn._active:
                txn.rollback()
            self.console.print(f'\n[bold red]Task Failed[/bold red]: {e}')
            self.console.print('[green][✓][/green] Workspace safely restored to its prior state.')
            raise
