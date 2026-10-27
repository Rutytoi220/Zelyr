from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, 
    QProgressBar, QGroupBox, QGridLayout, QPushButton
)
from PySide6.QtCore import Qt
from zelyr.engine.budget_mgr import TokenBudgetManager
from zelyr.config import get_config

class BudgetDialog(QDialog):
    """Dialog showing Cloud Token Usage Breakdown."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Cloud Token Budget Breakdown")
        self.setMinimumWidth(400)
        
        self.budget_mgr = TokenBudgetManager()
        self.config = get_config()
        
        self._build_ui()
        self._load_data()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(16)
        
        # Today's Usage Group
        today_group = QGroupBox("Today's Cloud Usage")
        today_group
        today_layout = QGridLayout(today_group)
        
        self.lbl_tokens_used = QLabel("Tokens Used:")
        self.val_tokens_used = QLabel("0")
        self.lbl_cost = QLabel("Est. Cost:")
        self.val_cost = QLabel("$0.00")
        
        today_layout.addWidget(self.lbl_tokens_used, 0, 0)
        today_layout.addWidget(self.val_tokens_used, 0, 1)
        today_layout.addWidget(self.lbl_cost, 1, 0)
        today_layout.addWidget(self.val_cost, 1, 1)
        
        # Progress Bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(True)
        self.progress_bar
        today_layout.addWidget(self.progress_bar, 2, 0, 1, 2)
        
        layout.addWidget(today_group)
        
        # Close Button
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        btn_layout.addWidget(close_btn)
        layout.addLayout(btn_layout)

    def _load_data(self):
        usage = self.budget_mgr.get_today_usage()
        total_tokens = usage['total_tokens']
        total_cost = usage['total_cost']
        
        limit = getattr(self.config, 'daily_cloud_token_limit', 50000)
        
        self.val_tokens_used.setText(f"{total_tokens:,} / {limit:,}")
        self.val_cost.setText(f"${total_cost:.4f}")
        
        percent = int((total_tokens / limit) * 100) if limit > 0 else 100
        if percent > 100:
            percent = 100
            
        self.progress_bar.setValue(percent)
        self.progress_bar.setFormat(f"{percent}% of Daily Limit")
        
        if percent >= 90:
            self.progress_bar
        elif percent >= 75:
            self.progress_bar
