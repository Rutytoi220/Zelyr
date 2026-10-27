"""Singularity UI Dashboard.

PySide6 dialogue displaying local RLHF dataset size, visual REM Sleep
memory pruning stats, and a 1-click [Assimilate Node] input box for
target IP addresses!
"""
from PySide6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QGroupBox,
    QLineEdit,
    QTextEdit,
)
from PySide6.QtCore import Qt
import logging
import asyncio
from pathlib import Path
import os
import json
from zelyr.compat import data_file

logger = logging.getLogger(__name__)

class SingularityControlDialog(QDialog):
    """The ultimate Singularity Dashboard."""

    def __init__(self, parent=None, event_bus=None):
        super().__init__(parent)
        self.setWindowTitle("🌌 Singularity Engine")
        self.setMinimumSize(600, 500)
        self.event_bus = event_bus
        self._init_ui()
        self._load_rlhf_stats()

        if self.event_bus:
            self.event_bus.subscribe("memory.rem_sleep.complete", self._on_rem_sleep)

    def _init_ui(self):
        layout = QVBoxLayout(self)

        # Header
        header = QLabel("<h2>Zelyr Singularity: Self-Optimization & Expansion</h2>")
        header
        layout.addWidget(header)

        # 1. RLHF Self-Improvement
        rlhf_group = QGroupBox("Continuous RLHF Self-Improvement")
        rlhf_group
        rlhf_layout = QVBoxLayout()
        
        self.dataset_label = QLabel("Loading dataset stats...")
        self.dataset_label
        rlhf_layout.addWidget(self.dataset_label)
        
        self.btn_evolve = QPushButton("🧬 Evolve Model (Create Modelfile)")
        self.btn_evolve.clicked.connect(self._trigger_evolution)
        rlhf_layout.addWidget(self.btn_evolve)
        
        rlhf_group.setLayout(rlhf_layout)
        layout.addWidget(rlhf_group)

        # 2. Network Assimilation
        assim_group = QGroupBox("Autonomous Network Assimilation")
        assim_group
        assim_layout = QVBoxLayout()
        
        input_layout = QHBoxLayout()
        self.ssh_input = QLineEdit()
        self.ssh_input.setPlaceholderText("e.g. root@192.168.1.150")
        input_layout.addWidget(self.ssh_input)
        
        self.btn_assimilate = QPushButton("Assimilate Node")
        self.btn_assimilate
        self.btn_assimilate.clicked.connect(self._trigger_assimilation)
        input_layout.addWidget(self.btn_assimilate)
        
        assim_layout.addLayout(input_layout)
        assim_group.setLayout(assim_layout)
        layout.addWidget(assim_group)

        # 3. REM Sleep
        rem_group = QGroupBox("Nightly REM Sleep Log")
        rem_group
        rem_layout = QVBoxLayout()
        
        self.rem_log = QTextEdit()
        self.rem_log.setReadOnly(True)
        self.rem_log
        rem_layout.addWidget(self.rem_log)
        
        self.btn_rem = QPushButton("Trigger Manual REM Sleep")
        self.btn_rem.clicked.connect(self._trigger_rem_sleep)
        rem_layout.addWidget(self.btn_rem)
        
        rem_group.setLayout(rem_layout)
        layout.addWidget(rem_group)

    def _load_rlhf_stats(self):
        preferences_file = data_file("training_data/rlhf_preferences.jsonl")
        count = 0
        if preferences_file.exists():
            with open(preferences_file, "r") as f:
                count = sum(1 for _ in f)
        self.dataset_label.setText(f"RLHF Dataset Size: {count} Successful Trajectories")

    def _trigger_evolution(self):
        from zelyr.engine.self_improvement import RLHFEngine
        engine = RLHFEngine()
        modelfile_path = engine.evolve_model()
        self.rem_log.append(f"[RLHF] Successfully generated Modelfile at {modelfile_path}")

    def _trigger_assimilation(self):
        target = self.ssh_input.text()
        if not target:
            return
            
        from zelyr.server.mesh_deployer import SwarmAssimilatorTool
        tool = SwarmAssimilatorTool()
        
        self.rem_log.append(f"[ASSIMILATION] Initiating zero-touch deployment to {target}...")
        
        # In a real GUI this would be offloaded to a QThread, we just run the mock synchronously for the demo
        loop = asyncio.get_event_loop()
        if loop.is_running():
            asyncio.run_coroutine_threadsafe(self._async_assimilate(tool, target), loop)
        else:
            res = asyncio.run(tool.assimilate_node(target))
            self._handle_assimilate_res(res)

    async def _async_assimilate(self, tool, target):
        res = await tool.assimilate_node(target)
        self._handle_assimilate_res(res)
        
    def _handle_assimilate_res(self, res):
        self.rem_log.append(f"[ASSIMILATION] Success! Target {res['target']} bound to Swarm via {res['package_deployed']}.")
        
    def _trigger_rem_sleep(self):
        from zelyr.memory.rem_sleep import DeepMemoryConsolidation
        from zelyr.core.events import EventBus
        dmc = DeepMemoryConsolidation(EventBus())
        
        mock_graph = [
            "Zelyr uses the EventBus for signaling.",
            "The system is powered by Python and PySide6.",
            "Zelyr uses the EventBus for signaling.",  # Duplicate!
        ]
        
        loop = asyncio.get_event_loop()
        if loop.is_running():
            asyncio.run_coroutine_threadsafe(dmc.trigger_rem_sleep(mock_graph), loop)
        else:
            asyncio.run(dmc.trigger_rem_sleep(mock_graph))

    def _on_rem_sleep(self, event):
        data = event.data
        self.rem_log.append(f"[REM Sleep] Compacted {data['fused_nodes']} redundant nodes. Pruned {data['pruned_bytes']} bytes.")
