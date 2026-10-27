from PySide6.QtWidgets import QDialog, QVBoxLayout, QTableWidget, QTableWidgetItem, QHeaderView, QLabel, QHBoxLayout, QPushButton, QLineEdit, QMessageBox
from zelyr.services.recall_engine import RecallEngine
import datetime

class RecallDialog(QDialog):
    """UI for exploring Local-First Visual Recall OCR history."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("⏱️ Zelyr Recall Timeline")
        self.setMinimumSize(900, 600)
        
        self.recall = RecallEngine()
        self._init_ui()
        
    def _init_ui(self):
        layout = QVBoxLayout(self)
        
        header = QLabel("Continuous Visual Memory (OCR History)")
        header
        layout.addWidget(header)
        
        # Search bar
        search_layout = QHBoxLayout()
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Search past screen contents...")
        self.search_input.returnPressed.connect(self._search_data)
        
        btn_search = QPushButton("Search")
        btn_search.clicked.connect(self._search_data)
        
        search_layout.addWidget(self.search_input)
        search_layout.addWidget(btn_search)
        layout.addLayout(search_layout)
        
        # Table
        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["Timestamp", "Window Title", "OCR Snippet"])
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        self.table
        layout.addWidget(self.table)
        
        # Toolbar
        toolbar = QHBoxLayout()
        btn_delete = QPushButton("🗑️ Clear History")
        btn_delete
        btn_delete.clicked.connect(self._clear_history)
        toolbar.addWidget(btn_delete)
        toolbar.addStretch()
        layout.addLayout(toolbar)
        
        self._search_data()
        
    def _search_data(self):
        query = self.search_input.text().strip()
        self.table.setRowCount(0)
        
        if query:
            results = self.recall.search_history(query)
        else:
            # If no query, just show nothing or recent (FTS requires match, so fallback if empty)
            results = []
            
        for i, row in enumerate(results):
            self.table.insertRow(i)
            ts_str = datetime.datetime.fromtimestamp(row["timestamp"]).strftime('%Y-%m-%d %H:%M:%S')
            
            self.table.setItem(i, 0, QTableWidgetItem(ts_str))
            self.table.setItem(i, 1, QTableWidgetItem(row["window_title"]))
            
            # Truncate snippet for UI
            snippet = str(row["ocr_text"])[:100] + "..."
            self.table.setItem(i, 2, QTableWidgetItem(snippet))

    def _clear_history(self):
        reply = QMessageBox.question(self, "Clear History", "Permanently delete all OCR visual memory?", QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.recall.delete_history()
            self._search_data()
