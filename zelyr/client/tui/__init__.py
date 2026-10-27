"""TUI package initialization."""

from zelyr.client.tui.app import ZelyrMonitorApp

# Legacy name retained for existing AXIOM integrations.
AxiomMonitorApp = ZelyrMonitorApp

__all__ = ["ZelyrMonitorApp", "AxiomMonitorApp"]
