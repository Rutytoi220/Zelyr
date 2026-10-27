"""Zelyr Desktop v3.4 — Swarm UI Component."""

from __future__ import annotations

import html
from typing import Dict

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFrame, QHBoxLayout, QLabel, QVBoxLayout,
    QWidget, QToolButton,
)

class SwarmPill(QFrame):
    """Dynamic multi-pill container for Swarm execution."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("swarmPill")
        self

        self.outer_layout = QVBoxLayout(self)
        self.outer_layout.setContentsMargins(4, 4, 4, 4)
        self.outer_layout.setSpacing(4)

        # Header
        self.header_layout = QHBoxLayout()
        self.header_icon = QLabel("🐝")
        self.header_label = QLabel("<b>Swarm Active — Supervisor Delegating...</b>")
        self.header_label
        self.header_layout.addWidget(self.header_icon)
        self.header_layout.addWidget(self.header_label, 1)
        
        self.outer_layout.addLayout(self.header_layout)
        
        self.agents: Dict[str, SwarmAgentPill] = {}

    def add_agent_task(self, agent_name: str, task: str) -> None:
        """Add a new agent pill to the swarm view."""
        if agent_name in self.agents:
            return
        pill = SwarmAgentPill(agent_name, task)
        self.agents[agent_name] = pill
        self.outer_layout.addWidget(pill)

    def update_agent_status(self, agent_name: str, chunk: str) -> None:
        """Append a token/chunk to an agent's reasoning stream."""
        if agent_name in self.agents:
            self.agents[agent_name].append_text(chunk)

    def complete_agent(self, agent_name: str, result: str) -> None:
        """Mark an agent as complete and show final result."""
        if agent_name in self.agents:
            self.agents[agent_name].set_completed(result)
        
        # Check if all completed
        all_done = all(a.is_completed for a in self.agents.values())
        if all_done:
            self.header_label.setText("<b>Swarm Execution Complete ✅</b>")
            self.header_label


class SwarmAgentPill(QFrame):
    """Individual agent execution pill."""
    
    def __init__(self, agent_name: str, task: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("swarmAgentPill")
        self
        
        self._expanded = False
        self.is_completed = False
        
        icon = "💻" if "Coder" in agent_name else "📚" if "Research" in agent_name else "📸" if "Vision" in agent_name else "🤖"
        
        self.layout_main = QVBoxLayout(self)
        self.layout_main.setContentsMargins(4, 2, 4, 2)
        
        header = QHBoxLayout()
        self._icon = QLabel(icon)
        self._summary = QLabel(f"<b>{html.escape(agent_name)}</b>: {html.escape(task)}")
        self._summary.setWordWrap(True)
        self._status = QLabel("[Running ⏳]")
        self._status
        
        self._toggle = QToolButton()
        self._toggle.setText("▶")
        self._toggle.clicked.connect(self._toggle_detail)
        
        header.addWidget(self._icon)
        header.addWidget(self._summary, 1)
        header.addWidget(self._status)
        header.addWidget(self._toggle)
        
        self.layout_main.addLayout(header)
        
        self._detail = QLabel("")
        self._detail.setWordWrap(True)
        self._detail.setVisible(False)
        self._detail
        self.layout_main.addWidget(self._detail)

    def append_text(self, chunk: str) -> None:
        """Append streamed text to detail."""
        self._detail.setText(self._detail.text() + html.escape(chunk))

    def set_completed(self, result: str) -> None:
        """Mark as completed and append final result."""
        self.is_completed = True
        self._status.setText("[Done ✅]")
        self._status
        if result:
            self._detail.setText(self._detail.text() + f"\n\nResult:\n{html.escape(result)}")

    def _toggle_detail(self) -> None:
        self._expanded = not self._expanded
        self._detail.setVisible(self._expanded)
        self._toggle.setText("▼" if self._expanded else "▶")
