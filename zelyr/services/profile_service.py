"""Profile Management Service.

Handles UI progressive disclosure state (Standard, Advanced, Developer)
using PySide6 signals to dynamically update the HUD.
"""
from PySide6.QtCore import QObject, Signal
from enum import Enum
from zelyr.config import get_config

class ProfileLevel(str, Enum):
    STANDARD = "standard"
    ADVANCED = "advanced"
    DEVELOPER = "developer"

class ProfileService(QObject):
    """Singleton service to manage UI profile disclosure."""
    profile_changed = Signal(ProfileLevel)
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ProfileService, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
        
    def __init__(self):
        if self._initialized:
            return
        super().__init__()
        self._initialized = True
        
        config = get_config()
        self._current = ProfileLevel(getattr(config, "ui_profile_level", "standard"))
        
    @classmethod
    def instance(cls) -> "ProfileService":
        return cls()
        
    def get_profile(self) -> ProfileLevel:
        return self._current
        
    def set_profile(self, level: ProfileLevel) -> None:
        if self._current != level:
            self._current = level
            
            # Persist
            config = get_config()
            config.ui_profile_level = level.value
            config.save()
            
            # Notify subscribers (HUD)
            self.profile_changed.emit(self._current)
