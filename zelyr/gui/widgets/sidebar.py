import asyncio
from PySide6.QtWidgets import QDockWidget, QWidget, QVBoxLayout, QListWidget, QLabel, QListWidgetItem
from PySide6.QtCore import Qt, Signal
import time

class SessionSidebar(QDockWidget):
    """Left sidebar for session history."""
    
    session_selected = Signal(str)
    
    def __init__(self, bridge, parent=None):
        super().__init__("Sessions", parent)
        self.setObjectName("sessionSidebar")
        self.setAllowedAreas(Qt.DockWidgetArea.LeftDockWidgetArea)
        self.setFeatures(QDockWidget.DockWidgetFeature.DockWidgetClosable)
        self.setMinimumWidth(250)
        
        self.bridge = bridge
        self.session_db = bridge.session_db
        
        self
        
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        self.list_widget = QListWidget()
        self.list_widget.itemClicked.connect(self._on_item_clicked)
        
        layout.addWidget(self.list_widget)
        self.setWidget(container)
        
    def load_sessions(self):
        """Asynchronously loads the latest sessions into the list."""
        if not self.bridge._loop:
            return
            
        async def _fetch():
            sessions = await self.session_db.get_recent_sessions()
            return sessions
            
        future = asyncio.run_coroutine_threadsafe(_fetch(), self.bridge._loop)
        
        # In a real app we'd use a signal or QTimer to check the future. 
        # For simplicity in this local context, we'll wait for it.
        try:
            sessions = future.result(timeout=2.0)
            self.list_widget.clear()
            for s in sessions:
                item = QListWidgetItem(s["title"])
                item.setData(Qt.ItemDataRole.UserRole, s["session_id"])
                self.list_widget.addItem(item)
        except Exception as e:
            pass
            
    def _on_item_clicked(self, item: QListWidgetItem):
        session_id = item.data(Qt.ItemDataRole.UserRole)
        self.session_selected.emit(session_id)
