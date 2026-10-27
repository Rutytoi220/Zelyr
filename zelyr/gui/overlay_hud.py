import logging
from PySide6.QtWidgets import QMainWindow, QRubberBand, QWidget, QApplication
from PySide6.QtCore import Qt, QRect, Signal, QPoint
from PySide6.QtGui import QColor, QPainter, QBrush, QPen

logger = logging.getLogger(__name__)

class CropOverlayWindow(QMainWindow):
    """Fullscreen transparent overlay for visual crop selection."""
    
    crop_selected = Signal(int, int, int, int)  # x, y, w, h
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Frameless, transparent, stays on top
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.setAttribute(Qt.WA_TranslucentBackground)
        
        # Cover all screens
        screens = QApplication.screens()
        if screens:
            geom = screens[0].geometry()
            for screen in screens[1:]:
                geom = geom.united(screen.geometry())
            self.setGeometry(geom)
            
        self.setCursor(Qt.CrossCursor)
        
        self.origin = QPoint()
        self.rubberBand = QRubberBand(QRubberBand.Rectangle, self)
        
        # Custom styling for the rubberband is limited, so we can override paintEvent if we want a custom red/cyan box
        # For simplicity, we just use the QRubberBand and a translucent paintEvent
        
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setBrush(QBrush(QColor(0, 0, 0, 100)))  # Semi-transparent dark overlay
        painter.setPen(Qt.NoPen)
        painter.drawRect(self.rect())
        
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.origin = event.pos()
            self.rubberBand.setGeometry(QRect(self.origin, event.pos()).normalized())
            self.rubberBand.show()
            
    def mouseMoveEvent(self, event):
        if not self.origin.isNull():
            self.rubberBand.setGeometry(QRect(self.origin, event.pos()).normalized())
            
    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            rect = self.rubberBand.geometry()
            self.rubberBand.hide()
            self.close()
            
            x, y, w, h = rect.x(), rect.y(), rect.width(), rect.height()
            if w > 10 and h > 10:
                logger.info(f"Crop selected: {x}, {y}, {w}, {h}")
                self.crop_selected.emit(x, y, w, h)
            else:
                logger.debug("Crop selection too small, ignoring.")

