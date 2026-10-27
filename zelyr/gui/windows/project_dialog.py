import os
from pathlib import Path
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QCursor
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, 
    QTextEdit, QPushButton, QFileDialog, QListWidget, QFrame
)

class ProjectDialog(QDialog):
    """Modal dialog to create a new project with context and files."""
    
    project_created = Signal(str, str, list)  # title, context, file_paths

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("New Project")
        self.setFixedSize(500, 550)
        self

        self.attached_files = []

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(12)

        _label_style = "font-size: 14px; font-weight: 500;"
        _input_style = "font-size: 14px;"
        _btn_style   = "font-size: 14px; border-radius: 8px; padding: 8px 16px;"

        # Title
        lbl_name = QLabel("Project Name")
        lbl_name.setStyleSheet(_label_style)
        layout.addWidget(lbl_name)
        self.title_input = QLineEdit()
        self.title_input.setPlaceholderText("e.g. Zelyr Refactoring")
        self.title_input.setStyleSheet(_input_style)
        layout.addWidget(self.title_input)

        # Context
        lbl_ctx = QLabel("Custom Context Instructions (Optional)")
        lbl_ctx.setStyleSheet(_label_style)
        layout.addWidget(lbl_ctx)
        self.context_input = QTextEdit()
        self.context_input.setPlaceholderText("Provide any context, system prompts, or background information you want the AI to always know in this project...")
        self.context_input.setStyleSheet(_input_style)
        layout.addWidget(self.context_input)

        # Attachments
        attach_layout = QHBoxLayout()
        lbl_files = QLabel("Attached Files")
        lbl_files.setStyleSheet(_label_style)
        attach_layout.addWidget(lbl_files)
        attach_layout.addStretch()
        
        self.btn_attach = QPushButton("Browse Files")
        self.btn_attach.setStyleSheet(_btn_style)
        self.btn_attach.setCursor(Qt.PointingHandCursor)
        self.btn_attach.clicked.connect(self._browse_files)
        attach_layout.addWidget(self.btn_attach)
        layout.addLayout(attach_layout)

        self.file_list = QListWidget()
        self.file_list.setFixedHeight(80)
        layout.addWidget(self.file_list)

        # Buttons
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        
        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.setStyleSheet(_btn_style)
        self.btn_cancel.setCursor(Qt.PointingHandCursor)
        self.btn_cancel.clicked.connect(self.reject)
        
        self.btn_create = QPushButton("Create Project")
        self.btn_create.setObjectName("createBtn")
        self.btn_create.setStyleSheet(_btn_style)
        self.btn_create.setCursor(Qt.PointingHandCursor)
        self.btn_create.clicked.connect(self._on_create)
        
        btn_layout.addWidget(self.btn_cancel)
        btn_layout.addWidget(self.btn_create)
        
        layout.addStretch()
        layout.addLayout(btn_layout)


    def _browse_files(self):
        files, _ = QFileDialog.getOpenFileNames(self, "Select Files to Attach", str(Path.home()))
        for f in files:
            if f not in self.attached_files:
                self.attached_files.append(f)
                self.file_list.addItem(os.path.basename(f))

    def _on_create(self):
        title = self.title_input.text().strip()
        if not title:
            self.title_input
            return
            
        context = self.context_input.toPlainText()
        self.project_created.emit(title, context, self.attached_files)
        self.accept()
