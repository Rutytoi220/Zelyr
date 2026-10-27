"""Action registry for Zelyr - handles different types of actions."""

import time
from typing import Tuple, Dict, Callable, Optional
from utils.logger import get_logger
from brain.context_memory import get_context_memory

logger = get_logger(__name__)


class ActionRegistry:
    """Registry for handling different action types."""
    
    def __init__(self):
        """Initialize action registry."""
        self.actions: Dict[str, Callable] = {}
        self._register_builtin_actions()
    
    def _register_builtin_actions(self) -> None:
        """Register built-in action handlers."""
        # Browser actions are delegated to the main executor,
        # but we handle some high-level actions here
        self.register('new_tab', self._handle_new_tab)
        self.register('new_window', self._handle_new_window)
        self.register('search', self._handle_search)
        self.register('go_to', self._handle_goto)
    
    def register(self, action_name: str, handler: Callable) -> None:
        """Register an action handler.
        
        Args:
            action_name: Name of the action
            handler: Function that handles the action
        """
        self.actions[action_name] = handler
        logger.debug(f"Registered action: {action_name}")
    
    def execute(self, action_name: str, params: str) -> Tuple[Optional[bool], Optional[str]]:
        """Execute an action.
        
        Args:
            action_name: Name of the action
            params: Parameter string
        
        Returns:
            (success, message) tuple
        """
        if action_name in self.actions:
            try:
                result = self.actions[action_name](params)
                if isinstance(result, tuple):
                    return result
                else:
                    return True, str(result)
            except Exception as e:
                logger.exception(f"Error executing action {action_name}")
                return False, f"Action failed: {e}"
        else:
            # Unknown action - return for main executor to handle
            return None, None  # Signal to use main executor
    
    @staticmethod
    def _handle_new_tab(params: str) -> Tuple[bool, str]:
        """Handle new_tab action.
        
        Args:
            params: Parameters (browser name optional)
        
        Returns:
            (success, message)
        """
        context = get_context_memory()
        
        # Parse params to get browser if specified
        browser = None
        if 'browser=' in params:
            # Extract browser name
            import re
            m = re.search(r'browser="([^"]+)"', params)
            if m:
                browser = m.group(1)
        
        # Use specified browser or find open one
        if not browser:
            browser = context.get_open_browser()
        
        if browser:
            context.register_new_tab(browser)
            logger.info(f"New tab registered for {browser}")
            return True, f"New tab opened in {browser}"
        else:
            return False, "No browser available for new tab"
    
    @staticmethod
    def _handle_new_window(params: str) -> Tuple[bool, str]:
        """Handle new_window action.
        
        Args:
            params: Parameters (browser name optional)
        
        Returns:
            (success, message)
        """
        context = get_context_memory()
        
        # Parse params to get browser
        browser = None
        if 'browser=' in params:
            import re
            m = re.search(r'browser="([^"]+)"', params)
            if m:
                browser = m.group(1)
        
        if browser:
            context.register_new_window(browser)
            logger.info(f"New window registered for {browser}")
            return True, f"New window opened in {browser}"
        else:
            return False, "Browser name required for new_window"
    
    @staticmethod
    def _handle_search(params: str) -> Tuple[bool, str]:
        """Handle search action.
        
        Args:
            params: Parameters (query required)
        
        Returns:
            (success, message)
        """
        # Extract query from params
        import re
        m = re.search(r'query="([^"]+)"', params)
        if not m:
            return False, "Search query required"
        
        query = m.group(1)
        logger.info(f"Search registered for: {query}")
        return True, f"Searching for: {query}"
    
    @staticmethod
    def _handle_goto(params: str) -> Tuple[bool, str]:
        """Handle go_to action.
        
        Args:
            params: Parameters (url required)
        
        Returns:
            (success, message)
        """
        # Extract URL from params
        import re
        m = re.search(r'url="([^"]+)"', params)
        if not m:
            return False, "URL required"
        
        url = m.group(1)
        logger.info(f"Navigation registered to: {url}")
        return True, f"Navigating to: {url}"


# Global registry instance
_global_registry = None


def get_action_registry() -> ActionRegistry:
    """Get the global action registry."""
    global _global_registry
    if _global_registry is None:
        _global_registry = ActionRegistry()
    return _global_registry
