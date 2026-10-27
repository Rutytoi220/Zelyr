import asyncio
from PySide6.QtCore import Qt, Slot, QTimer
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QFrame, QScrollArea, QWidget, QMessageBox
)
from zelyr.memory.schedules import ScheduleDatabase
import logging

logger = logging.getLogger(__name__)

class TemporalSchedulerDialog(QDialog):
    """UI for managing active Temporal Engine semantic tasks."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Temporal Engine")
        self.setMinimumSize(600, 500)

        self.db = ScheduleDatabase()
        
        layout = QVBoxLayout(self)
        
        header = QLabel("<h2>Semantic Scheduler (Temporal Engine)</h2>")
        header.setObjectName("update_header")
        layout.addWidget(header)
        
        desc = QLabel("Manage background AI tasks scheduled via natural language.")
        desc.setObjectName("hub_desc")
        desc.setWordWrap(True)
        layout.addWidget(desc)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll
        
        self.tasks_container = QWidget()
        self.tasks_layout = QVBoxLayout(self.tasks_container)
        self.tasks_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.scroll.setWidget(self.tasks_container)
        
        layout.addWidget(self.scroll)
        
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        refresh_btn = QPushButton("Refresh")
        refresh_btn.setObjectName("scheduler_refresh")
        refresh_btn.clicked.connect(self.load_tasks)
        btn_layout.addWidget(refresh_btn)
        layout.addLayout(btn_layout)

        loop = asyncio.get_event_loop()
        if loop.is_running():
            asyncio.create_task(self.db.initialize())
        else:
            loop.run_until_complete(self.db.initialize())
            
        QTimer.singleShot(100, self.load_tasks)

    @Slot()
    def load_tasks(self):
        while self.tasks_layout.count():
            child = self.tasks_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
                
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            future = asyncio.run_coroutine_threadsafe(self.db.get_schedules(), loop)
            future.add_done_callback(lambda f: QTimer.singleShot(0, lambda: self._populate_ui(f.result())))
        else:
            tasks = loop.run_until_complete(self.db.get_schedules())
            self._populate_ui(tasks)
            
    def _populate_ui(self, tasks):
        if not tasks:
            empty = QLabel("No semantic tasks scheduled. Tell Zelyr to 'schedule a task...'")
            empty.setObjectName("scheduler_empty")
            self.tasks_layout.addWidget(empty)
            return
            
        for t in tasks:
            row = QFrame()
            row.setObjectName("scheduler_task_row")
            row_layout = QHBoxLayout(row)
            
            text_layout = QVBoxLayout()
            t_label = QLabel(f"<b>Prompt:</b> {t['user_prompt']}")
            t_label.setObjectName("scheduler_task_label")
            t_label.setWordWrap(True)
            
            d_label = QLabel(f"<b>Cron:</b> <code>{t['cron_expression']}</code> | <b>Active:</b> {t['is_active']}")
            d_label.setObjectName("scheduler_cron_label")
            
            text_layout.addWidget(t_label)
            text_layout.addWidget(d_label)
            row_layout.addLayout(text_layout)
            
            row_layout.addStretch()
            
            btn_layout = QVBoxLayout()
            
            toggle_btn = QPushButton("Pause" if t['is_active'] else "Resume")
            toggle_btn.setObjectName("scheduler_toggle")
            toggle_btn.setProperty("status", "warning" if t['is_active'] else "success")
            toggle_btn.style().unpolish(toggle_btn)
            toggle_btn.style().polish(toggle_btn)
            toggle_btn.clicked.connect(lambda _, t_id=t['id'], state=t['is_active']: self.toggle_task(t_id, not state))
            
            del_btn = QPushButton("Delete")
            del_btn.setObjectName("scheduler_delete")
            del_btn.clicked.connect(lambda _, t_id=t['id']: self.delete_task(t_id))
            
            btn_layout.addWidget(toggle_btn)
            btn_layout.addWidget(del_btn)
            
            row_layout.addLayout(btn_layout)
            self.tasks_layout.addWidget(row)

    @Slot(str, bool)
    def toggle_task(self, task_id, new_state):
        loop = asyncio.get_event_loop()
        if loop.is_running():
            future = asyncio.run_coroutine_threadsafe(self.db.toggle_schedule(task_id, new_state), loop)
            future.add_done_callback(lambda f: QTimer.singleShot(0, self.load_tasks))
        else:
            loop.run_until_complete(self.db.toggle_schedule(task_id, new_state))
            self.load_tasks()

    @Slot(str)
    def delete_task(self, task_id):
        confirm = QMessageBox.question(
            self, "Delete Task", 
            "Are you sure you want to delete this scheduled task?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if confirm == QMessageBox.StandardButton.Yes:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                future = asyncio.run_coroutine_threadsafe(self.db.delete_schedule(task_id), loop)
                future.add_done_callback(lambda f: QTimer.singleShot(0, self.load_tasks))
            else:
                loop.run_until_complete(self.db.delete_schedule(task_id))
                self.load_tasks()
