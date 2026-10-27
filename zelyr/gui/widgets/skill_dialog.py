from zelyr.compat import data_dir
import os
from pathlib import Path
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QListWidget, 
    QTextEdit, QPushButton, QLabel, QMessageBox
)
from PySide6.QtCore import Qt

class SkillManagerDialog(QDialog):
    """UI for managing auto-compiled python skills."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("🧩 Zelyr Skill Library")
        self.setMinimumSize(700, 500)
        
        self.skills_dir = data_dir() / "skills"
        self._init_ui()
        self._load_skills()
        
    def _init_ui(self):
        layout = QHBoxLayout(self)
        
        # Left: Skill List
        left_layout = QVBoxLayout()
        left_layout.addWidget(QLabel("Compiled Skills:"))
        
        self.skill_list = QListWidget()
        self.skill_list.setFixedWidth(200)
        self.skill_list.currentItemChanged.connect(self._on_skill_selected)
        left_layout.addWidget(self.skill_list)
        
        # Left: Actions
        btn_layout = QHBoxLayout()
        self.delete_btn = QPushButton("🗑️ Delete")
        self.delete_btn
        self.delete_btn.clicked.connect(self._delete_skill)
        self.delete_btn.setEnabled(False)
        btn_layout.addWidget(self.delete_btn)
        
        left_layout.addLayout(btn_layout)
        layout.addLayout(left_layout)
        
        # Right: Code Viewer
        right_layout = QVBoxLayout()
        right_layout.addWidget(QLabel("Skill Source Code:"))
        
        self.code_viewer = QTextEdit()
        self.code_viewer.setReadOnly(True)
        self.code_viewer
        right_layout.addWidget(self.code_viewer)
        
        layout.addLayout(right_layout)
        
    def _load_skills(self):
        self.skill_list.clear()
        if not self.skills_dir.exists():
            return
            
        for file in self.skills_dir.glob("*.py"):
            if file.is_file():
                self.skill_list.addItem(file.name)
                
    def _on_skill_selected(self, current, previous):
        if not current:
            self.code_viewer.clear()
            self.delete_btn.setEnabled(False)
            return
            
        file_path = self.skills_dir / current.text()
        if file_path.exists():
            with open(file_path, "r") as f:
                self.code_viewer.setPlainText(f.read())
            self.delete_btn.setEnabled(True)
            
    def _delete_skill(self):
        current = self.skill_list.currentItem()
        if not current:
            return
            
        file_name = current.text()
        file_path = self.skills_dir / file_name
        
        reply = QMessageBox.question(
            self, 'Delete Skill?',
            f"Are you sure you want to delete {file_name}?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            try:
                if file_path.exists():
                    os.remove(file_path)
                self._load_skills()
                self.code_viewer.clear()
                self.delete_btn.setEnabled(False)
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to delete skill: {e}")
