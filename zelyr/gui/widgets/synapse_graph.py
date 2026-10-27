from PySide6.QtWidgets import (QGraphicsView, QGraphicsScene, QGraphicsItem, 
                               QGraphicsEllipseItem, QGraphicsTextItem, QGraphicsLineItem)
from PySide6.QtCore import Qt, QPointF, QTimer, QPropertyAnimation, QObject, Property
from PySide6.QtGui import QColor, QPen, QBrush, QFont, QPainterPath

import logging
logger = logging.getLogger(__name__)

class NodeItem(QGraphicsEllipseItem):
    def __init__(self, text: str, node_type: str, x: float, y: float):
        super().__init__(-30, -30, 60, 60)
        self.setPos(x, y)
        self.setZValue(10)
        
        # Colors based on node type
        if node_type == "thought":
            color = QColor("#3b82f6") # Blue
        elif node_type == "tool":
            color = QColor("#f97316") # Orange
        elif node_type == "memory":
            color = QColor("#22c55e") # Green
        elif node_type == "root":
            color = QColor("#8b5cf6") # Purple
        else:
            color = QColor("#6b7280") # Gray
            
        self.setBrush(QBrush(color))
        self.setPen(QPen(Qt.NoPen))
        
        # Label
        self.label = QGraphicsTextItem(text, self)
        font = QFont("Inter", 9, QFont.Weight.Bold)
        self.label.setFont(font)
        self.label.setDefaultTextColor(Qt.white)
        
        # Center the label
        br = self.label.boundingRect()
        self.label.setPos(-br.width()/2, -br.height()/2)
        
        # Description Text (Below node)
        # Note: We won't show long descriptions, just the type
        self.desc = QGraphicsTextItem(node_type.upper(), self)
        desc_font = QFont("Inter", 8)
        self.desc.setFont(desc_font)
        self.desc.setDefaultTextColor(QColor("#9ca3af"))
        br2 = self.desc.boundingRect()
        self.desc.setPos(-br2.width()/2, 35)


class EdgeItem(QGraphicsLineItem):
    def __init__(self, source_pos: QPointF, target_pos: QPointF):
        super().__init__(source_pos.x(), source_pos.y(), target_pos.x(), target_pos.y())
        pen = QPen(QColor("#4b5563"))
        pen.setWidth(2)
        pen.setStyle(Qt.DashLine)
        self.setPen(pen)
        self.setZValue(1)


class SynapseGraph(QGraphicsView):
    """A real-time animated node graph visualizing AI thoughts and tool executions."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        
        self.setRenderHint(self.renderHints() | self.renderHints().Antialiasing)
        self.setBackgroundBrush(QBrush(QColor("#111827"))) # Dark gray/black
        
        # Hide scrollbars
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        
        self.nodes = []
        self.edges = []
        
        # Layout metrics
        self.start_x = 200
        self.start_y = 50
        self.y_spacing = 100
        self.current_y = self.start_y
        
    def reset_graph(self):
        """Clear the graph for a new session."""
        self.scene.clear()
        self.nodes.clear()
        self.edges.clear()
        self.current_y = self.start_y
        
        # Add a root node
        self.add_node("Prompt", "root")

    def add_node(self, text: str, node_type: str):
        """Add a new node and link it to the previous one."""
        x = self.start_x
        y = self.current_y
        
        node = NodeItem(text, node_type, x, y)
        self.scene.addItem(node)
        
        if self.nodes:
            prev_node = self.nodes[-1]
            edge = EdgeItem(prev_node.pos(), node.pos())
            self.scene.addItem(edge)
            self.edges.append(edge)
            
        self.nodes.append(node)
        self.current_y += self.y_spacing
        
        # Auto-scroll/pan down
        self.centerOn(node)
        
    def handle_telemetry(self, event):
        """Process an incoming synapse event from the EventBus."""
        event_type = getattr(event, 'event_type', '')
        data = getattr(event, 'data', {})
        
        if event_type == 'synapse.agent_thought':
            # Create a short snippet for the thought
            thought = data.get('thought', 'Thinking...')
            snippet = thought[:10] + "..." if len(thought) > 10 else thought
            self.add_node(snippet, "thought")
            
        elif event_type == 'synapse.tool_call_started':
            tool_name = data.get('tool', 'Tool')
            self.add_node(tool_name, "tool")
            
        elif event_type == 'synapse.memory_retrieved':
            self.add_node("Mem", "memory")
