"""IoT MQTT Actuator Tool.

Connects Zelyr to local MQTT brokers (like Mosquitto or Home Assistant)
to translate agent intents into physical world actuation (e.g., lights, fans).
"""
import logging
from typing import Dict, Any

try:
    import paho.mqtt.client as mqtt
    MQTT_AVAILABLE = True
except ImportError:
    MQTT_AVAILABLE = False

from zelyr.tools.core import BaseTool, ToolResult, ToolParameter

logger = logging.getLogger(__name__)

class IoTActuatorTool(BaseTool):
    """Tool for actuating physical IoT devices via MQTT."""
    
    def __init__(self, broker_host: str = "localhost", broker_port: int = 1883):
        self.broker_host = broker_host
        self.broker_port = broker_port

    def get_info(self) -> dict:
        return {
            "name": "iot_actuate",
            "description": "Send MQTT payloads to local IoT devices (e.g., Home Assistant) to actuate physical environment.",
            "parameters": [
                ToolParameter("topic", "str", "The MQTT topic to publish to (e.g., 'home/living_room/light/set')"),
                ToolParameter("payload", "str", "The payload to send (e.g., 'ON', 'OFF', '{\"brightness\": 128}')")
            ],
            "execution_count": getattr(self, "execution_count", 0)
        }
        
    def execute(self, args: Dict[str, Any]) -> ToolResult:
        if not MQTT_AVAILABLE:
            return ToolResult(
                success=False,
                error="paho-mqtt package is not installed. Cannot actuate IoT devices."
            )
            
        topic = args.get("topic")
        payload = args.get("payload")
        
        if not topic or not payload:
            return ToolResult(success=False, error="Missing required parameters: 'topic' and 'payload'")
            
        try:
            logger.info(f"IoTActuator: Publishing to {topic} -> {payload}")
            
            # Simple single-shot publish
            import paho.mqtt.publish as publish
            publish.single(
                topic,
                payload=payload,
                hostname=self.broker_host,
                port=self.broker_port
            )
            
            return ToolResult(
                success=True,
                data={"status": "published", "topic": topic, "payload": payload}
            )
        except Exception as e:
            logger.error(f"IoTActuator: Failed to publish MQTT message: {e}")
            return ToolResult(success=False, error=str(e))
