from PySide6.QtWidgets import QWidget, QHBoxLayout, QLabel
from PySide6.QtCore import Qt

class SwarmHUD(QWidget):
    """Compact horizontal Swarm Pill status bar."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("swarmHUD")
        self
        
        self.layout = QHBoxLayout(self)
        self.layout.setContentsMargins(16, 4, 16, 4)
        self.layout.setSpacing(10)
        
        self.pills = {}
        
        self.layout.addStretch()

    def add_pill(self, agent_name: str, task: str):
        if agent_name in self.pills:
            return
            
        icon = "🟢"
        if "coder" in agent_name.lower(): icon = "💻"
        elif "research" in agent_name.lower(): icon = "📚"
        elif "vision" in agent_name.lower(): icon = "👁️"
            
        pill = QLabel(f"[ {icon} {agent_name} ]")
        pill
        
        # Insert before the stretch
        self.layout.insertWidget(self.layout.count() - 1, pill)
        self.pills[agent_name] = pill

    def update_pill(self, agent_name: str, status: str):
        if agent_name in self.pills:
            pass # HUD doesn't stream tokens, just shows active state

    def remove_pill(self, agent_name: str):
        if agent_name in self.pills:
            pill = self.pills.pop(agent_name)
            self.layout.removeWidget(pill)
            pill.deleteLater()
