from __future__ import annotations

import logging
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from zelyr.core.events import EventBus

logger = logging.getLogger(__name__)


class LifecycleState(str, Enum):
    """Lifecycle states for the Zelyr Daemon and subsystems."""
    BOOTING = "BOOTING"
    GUI_READY = "GUI_READY"
    CORE_INITIALIZING = "CORE_INITIALIZING"
    READY = "READY"
    DEGRADED = "DEGRADED"


class AppLifecycleState(str, Enum):
    """Application-wide lifecycle states emitted via EventBus."""
    BOOTING = "BOOTING"
    GUI_READY = "GUI_READY"
    CORE_INITIALIZING = "CORE_INITIALIZING"
    READY = "READY"
    DEGRADED = "DEGRADED"


class LifecycleManager:
    """Manages application lifecycle state transitions with event emission."""

    def __init__(self, event_bus: "EventBus | None" = None) -> None:
        self._state: AppLifecycleState = AppLifecycleState.BOOTING
        self._event_bus = event_bus

    @property
    def state(self) -> AppLifecycleState:
        return self._state

    def transition(self, new_state: AppLifecycleState) -> None:
        """Transition to a new state and emit a lifecycle.state_changed event."""
        old = self._state
        self._state = new_state
        logger.info("Lifecycle: %s -> %s", old.value, new_state.value)
        if self._event_bus is not None:
            self._event_bus.publish_sync(
                "lifecycle.state_changed",
                {
                    "old_state": old.value,
                    "new_state": new_state.value,
                },
            )
