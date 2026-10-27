"""IoT/Physical World Control Dashboard.

PySide6 dialogue exposing manual overrides for physical MQTT actuators.
"""
from PySide6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QGroupBox,
    QSlider,
)
from PySide6.QtCore import Qt
import logging

try:
    from zelyr.tools.iot_mqtt import IoTActuatorTool
except ImportError:
    IoTActuatorTool = None

logger = logging.getLogger(__name__)

class IoTControlDialog(QDialog):
    """Dashboard for physical environment actuation."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("💡 IoT / Physical Environment")
        self.setMinimumSize(400, 300)
        self.iot_tool = IoTActuatorTool() if IoTActuatorTool else None
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)

        # Header
        header = QLabel("<h2>Physical Environment Controls</h2>")
        header
        layout.addWidget(header)

        # Lighting Controls
        light_group = QGroupBox("Office Lighting")
        light_group
        light_layout = QVBoxLayout()
        
        btn_layout = QHBoxLayout()
        self.btn_lights_on = QPushButton("Turn ON")
        self.btn_lights_on.clicked.connect(lambda: self._actuate("home/office/light/set", '{"state": "ON"}'))
        
        self.btn_lights_off = QPushButton("Turn OFF")
        self.btn_lights_off.clicked.connect(lambda: self._actuate("home/office/light/set", '{"state": "OFF"}'))
        
        btn_layout.addWidget(self.btn_lights_on)
        btn_layout.addWidget(self.btn_lights_off)
        light_layout.addLayout(btn_layout)

        # Dimmer
        dimmer_layout = QHBoxLayout()
        dimmer_layout.addWidget(QLabel("Brightness:"))
        self.dimmer_slider = QSlider(Qt.Horizontal)
        self.dimmer_slider.setRange(0, 255)
        self.dimmer_slider.setValue(255)
        self.dimmer_slider.sliderReleased.connect(self._on_dimmer_changed)
        dimmer_layout.addWidget(self.dimmer_slider)
        light_layout.addLayout(dimmer_layout)

        light_group.setLayout(light_layout)
        layout.addWidget(light_group)

        # HVAC / Fan
        fan_group = QGroupBox("HVAC / Fan")
        fan_group
        fan_layout = QHBoxLayout()
        
        self.btn_fan_on = QPushButton("Fan ON")
        self.btn_fan_on.clicked.connect(lambda: self._actuate("home/office/fan/set", '{"state": "ON"}'))
        
        self.btn_fan_off = QPushButton("Fan OFF")
        self.btn_fan_off.clicked.connect(lambda: self._actuate("home/office/fan/set", '{"state": "OFF"}'))
        
        fan_layout.addWidget(self.btn_fan_on)
        fan_layout.addWidget(self.btn_fan_off)
        fan_group.setLayout(fan_layout)
        layout.addWidget(fan_group)

        layout.addStretch()

    def _on_dimmer_changed(self):
        val = self.dimmer_slider.value()
        self._actuate("home/office/light/set", f'{{"brightness": {val}}}')

    def _actuate(self, topic: str, payload: str):
        if self.iot_tool:
            result = self.iot_tool.execute({"topic": topic, "payload": payload})
            if result.success:
                logger.info(f"IoT Dialog: Successfully published to {topic}")
            else:
                logger.error(f"IoT Dialog: Publish failed - {result.error}")
        else:
            logger.warning(f"IoT Dialog: Simulation Mode - {topic} -> {payload}")
