from PySide6.QtWidgets import QDialog, QVBoxLayout, QTableWidget, QTableWidgetItem, QHeaderView, QLabel, QHBoxLayout, QPushButton
from zelyr.engine.graph_memory import GraphMemoryEngine

class GraphDialog(QDialog):
    """UI for exploring the Entity Graph (Relational Index)."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("🕸️ Zelyr Entity Graph")
        self.setMinimumSize(800, 500)
        
        self.graph = GraphMemoryEngine()
        self._init_ui()
        
    def _init_ui(self):
        layout = QVBoxLayout(self)
        
        header = QLabel("Entity & Relationship Explorer")
        header
        layout.addWidget(header)
        
        # Table
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["Source Entity", "Type", "Relation", "Target Entity"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table
        layout.addWidget(self.table)
        
        # Toolbar
        toolbar = QHBoxLayout()
        btn_refresh = QPushButton("🔄 Refresh Graph")
        btn_refresh.clicked.connect(self._refresh_data)
        toolbar.addWidget(btn_refresh)
        toolbar.addStretch()
        layout.addLayout(toolbar)
        
        self._refresh_data()
        
    def _refresh_data(self):
        self.table.setRowCount(0)
        edges = self.graph.get_all_relationships()
        nodes = {n["id"]: n for n in self.graph.get_all_entities()}
        
        for i, edge in enumerate(edges):
            self.table.insertRow(i)
            src_node = nodes.get(edge["source_id"], {})
            tgt_node = nodes.get(edge["target_id"], {})
            
            src_name = src_node.get("name", edge["source_id"])
            src_type = src_node.get("type", "Unknown")
            tgt_name = tgt_node.get("name", edge["target_id"])
            
            self.table.setItem(i, 0, QTableWidgetItem(src_name))
            self.table.setItem(i, 1, QTableWidgetItem(src_type))
            self.table.setItem(i, 2, QTableWidgetItem(f"--[{edge['relation_type']}]-->"))
            self.table.setItem(i, 3, QTableWidgetItem(tgt_name))
