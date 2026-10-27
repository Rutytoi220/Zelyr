from zelyr.compat import config_dir
import json
import logging
from pathlib import Path
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)

@dataclass
class UIConfig:
    theme: str = "dark"
    accent_color: str = "#2ECC71"
    voice_mode: str = "push_to_talk"

class UIConfigManager:
    """Manages the UI configuration (theme, accent color) separately from core system settings."""
    
    def __init__(self):
        self.config_dir = config_dir()
        self.config_path = self.config_dir / "ui_config.json"
        self._config = UIConfig()
        
    def exists(self) -> bool:
        """Check if the UI configuration file exists on disk."""
        return self.config_path.exists()

    def load(self) -> UIConfig:
        """Load UI configuration from disk."""
        if not self.exists():
            return self._config
            
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            self._config.theme = data.get("theme", "dark")
            self._config.accent_color = data.get("accent_color", "#2ECC71")
            self._config.voice_mode = data.get("voice_mode", "push_to_talk")
            
        except Exception as e:
            logger.error(f"Failed to load UI config: {e}")
            
        return self._config
        
    def save(self, config: UIConfig = None) -> None:
        """Save UI configuration to disk."""
        if config is not None:
            self._config = config
            
        self.config_dir.mkdir(parents=True, exist_ok=True)
        try:
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(asdict(self._config), f, indent=4)
        except Exception as e:
            logger.error(f"Failed to save UI config: {e}")

# Global singleton
_ui_config_manager = UIConfigManager()

def get_ui_config_manager() -> UIConfigManager:
    return _ui_config_manager
