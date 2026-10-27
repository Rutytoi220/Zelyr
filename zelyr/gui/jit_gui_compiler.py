"""Just-In-Time (JIT) Generative UI Compiler.

Takes AI-generated PySide6 Python code strings, safely compiles them
in-memory, and injects the resulting QWidget into the active Wayland HUD.
"""
import logging
import ast
from typing import Optional, Any
from PySide6.QtWidgets import QWidget, QLabel, QVBoxLayout

logger = logging.getLogger(__name__)

class JITSecurityException(Exception):
    """Raised when JIT code violates security constraints."""
    pass

class GenerativeUIEngine:
    """Compiles AI-generated PySide6 code at runtime."""
    
    def __init__(self):
        # We define a strict subset of modules the AI code can access
        self.allowed_modules = {
            "PySide6.QtWidgets": __import__("PySide6.QtWidgets", fromlist=["*"]),
            "PySide6.QtCore": __import__("PySide6.QtCore", fromlist=["*"]),
            "PySide6.QtGui": __import__("PySide6.QtGui", fromlist=["*"])
        }
        
    def _sanitize_code(self, code_str: str) -> bool:
        """Basic AST check to prevent standard malicious ops (e.g. `os.system`)."""
        try:
            tree = ast.parse(code_str)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if not alias.name.startswith("PySide6"):
                            logger.error(f"JIT Compiler: Unauthorized import '{alias.name}' rejected.")
                            return False
                elif isinstance(node, ast.ImportFrom):
                    if not node.module or not node.module.startswith("PySide6"):
                        logger.error(f"JIT Compiler: Unauthorized from-import '{node.module}' rejected.")
                        return False
            return True
        except SyntaxError as e:
            logger.error(f"JIT Compiler: Syntax error in generated code - {e}")
            return False

    def compile_widget(self, class_name: str, code_str: str) -> Optional[QWidget]:
        """Compiles the code string and returns an instance of class_name."""
        logger.info(f"JIT Compiler: Compiling generative widget '{class_name}'...")
        
        if not self._sanitize_code(code_str):
            return self._fallback_error_widget("Security/Syntax check failed.")
            
        try:
            # Create a safe globals dict
            safe_globals = {
                "__builtins__": __builtins__
            }
            # Add allowed modules
            for mod_name, mod in self.allowed_modules.items():
                safe_globals[mod_name.split('.')[-1]] = mod
                
            # Pre-inject common PySide classes to avoid needing imports in the snippet
            import PySide6.QtWidgets as QtWidgets
            import PySide6.QtCore as QtCore
            import PySide6.QtGui as QtGui
            safe_globals.update({
                "QWidget": QtWidgets.QWidget,
                "QLabel": QtWidgets.QLabel,
                "QPushButton": QtWidgets.QPushButton,
                "QVBoxLayout": QtWidgets.QVBoxLayout,
                "QHBoxLayout": QtWidgets.QHBoxLayout,
                "Qt": QtCore.Qt,
            })
            
            # Execute the code in the sandboxed namespace
            exec(code_str, safe_globals)
            
            # Extract the class
            if class_name not in safe_globals:
                logger.error(f"JIT Compiler: Class '{class_name}' not found in compiled namespace.")
                return self._fallback_error_widget(f"Class '{class_name}' not defined.")
                
            widget_class = safe_globals[class_name]
            
            # Instantiate it
            instance = widget_class()
            if not isinstance(instance, QWidget):
                logger.error(f"JIT Compiler: '{class_name}' must inherit from QWidget.")
                return self._fallback_error_widget(f"'{class_name}' is not a QWidget.")
                
            logger.info("JIT Compiler: Successfully instantiated generative UI.")
            return instance
            
        except Exception as e:
            logger.error(f"JIT Compiler: Exception during compilation/instantiation - {e}")
            return self._fallback_error_widget(f"Runtime Error: {str(e)}")

    def _fallback_error_widget(self, error_msg: str) -> QWidget:
        """Returns a generic error widget when compilation fails."""
        w = QWidget()
        lay = QVBoxLayout(w)
        lbl = QLabel("⚠️ Generative UI Compilation Failed")
        lbl
        lay.addWidget(lbl)
        lay.addWidget(QLabel(error_msg))
        return w
