from abc import ABC, abstractmethod

class BaseOSController(ABC):
    """Abstract base class for OS-specific desktop automation."""

    @property
    @abstractmethod
    def can_click(self) -> bool:
        """Indicates if this controller can perform mouse clicks."""
        pass

    @property
    @abstractmethod
    def can_type(self) -> bool:
        """Indicates if this controller can type text."""
        pass

    @property
    @abstractmethod
    def can_capture(self) -> bool:
        """Indicates if this controller can capture the screen."""
        pass
        
    @property
    @abstractmethod
    def can_manage_windows(self) -> bool:
        """Indicates if this controller can manage windows."""
        pass

    @abstractmethod
    def click(self, x: int, y: int, button: str = "left", clicks: int = 1) -> None:
        """Click at a specific coordinate."""
        pass

    @abstractmethod
    def type_text(self, text: str) -> None:
        """Type a string of text into the currently focused window."""
        pass

    @abstractmethod
    def press_key(self, key: str) -> None:
        """Press a specific key (e.g., 'Return', 'Tab')."""
        pass
