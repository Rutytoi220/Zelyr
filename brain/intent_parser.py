"""Improved intent parser for Zelyr - converts natural language to multi-step actions."""

import re
from typing import List, Dict, Optional, Tuple
from utils.logger import get_logger
from brain.context_memory import get_context_memory

logger = get_logger(__name__)


class ActionPlan:
    """Represents a multi-step action plan."""
    
    def __init__(self, actions: List[Dict]):
        """Initialize an action plan.
        
        Args:
            actions: List of action dictionaries with 'name' and 'params'
        """
        self.actions = actions
    
    def to_instructions(self) -> Dict:
        """Convert to instruction format for executor.
        
        Returns:
            Instruction dictionary
        """
        if len(self.actions) == 1:
            # Single action - return in old format for compatibility
            action = self.actions[0]
            return {
                'type': 'instruction',
                'action': action['name'],
                'params': self._serialize_params(action.get('params', {}))
            }
        else:
            # Multiple actions - return in new format
            instructions = []
            for action in self.actions:
                instructions.append({
                    'action': action['name'],
                    'params': self._serialize_params(action.get('params', {}))
                })
            return {
                'type': 'instructions',
                'instructions': instructions
            }
    
    @staticmethod
    def _serialize_params(params: dict) -> str:
        """Serialize params dict to command string.
        
        Args:
            params: Dictionary of parameters
        
        Returns:
            Serialized parameter string
        """
        if not params:
            return ''
        parts = []
        for key, value in params.items():
            if isinstance(value, str):
                parts.append(f'{key}="{value}"')
            else:
                parts.append(f'{key}="{value}"')
        return ' '.join(parts)


class IntentParser:
    """Parses natural language into structured action plans."""
    
    def __init__(self):
        """Initialize the intent parser."""
        self.context = get_context_memory()
    
    def parse_intent(self, prompt: str) -> Optional[ActionPlan]:
        """Parse user intent into an action plan.
        
        Args:
            prompt: User's natural language request
        
        Returns:
            ActionPlan if intent recognized, None otherwise
        """
        prompt_lower = prompt.lower().strip()
        
        # Pattern: "search [query]" or "search for [query]"
        match = re.match(r'^(?:search|find)\s+(?:for\s+)?(.+)$', prompt_lower)
        if match:
            query = match.group(1).strip()
            return self._create_search_plan(query)
        
        # Pattern: "open [browser] and search [query]"
        match = re.match(r'^open\s+(\w+)\s+(?:and\s+)?(?:search|find)\s+(?:for\s+)?(.+)$', prompt_lower)
        if match:
            browser = match.group(1).strip()
            query = match.group(2).strip()
            return self._create_browser_search_plan(browser, query)
        
        # Pattern: "open [app] window and search [query]"
        match = re.match(r'^open\s+(?:a\s+)?(?:new\s+)?(\w+)\s+(?:window|tab)\s+(?:and\s+)?(?:search|find)\s+(?:for\s+)?(.+)$', prompt_lower)
        if match:
            browser = match.group(1).strip()
            query = match.group(2).strip()
            return self._create_new_window_search_plan(browser, query)
        
        # Pattern: "open [browser] with [url]"
        match = re.match(r'^open\s+(\w+)\s+(?:with|and go to|and visit)\s+(.+)$', prompt_lower)
        if match:
            browser = match.group(1).strip()
            url = match.group(2).strip()
            return self._create_open_url_plan(browser, url)
        
        # Pattern: "go to [url]"
        match = re.match(r'^go\s+to\s+(.+)$', prompt_lower)
        if match:
            url = match.group(1).strip()
            return self._create_goto_url_plan(url)
        
        # Pattern: "[action] in [app]" (e.g., "search python in firefox")
        match = re.match(r'^(?:search|find)\s+(.+?)\s+in\s+(\w+)$', prompt_lower)
        if match:
            query = match.group(1).strip()
            browser = match.group(2).strip()
            return self._create_browser_search_plan(browser, query)
        
        # Pattern: "open new [browser/app] window" or "open new tab"
        match = re.match(r'^open\s+(?:a\s+)?new\s+(?:(\w+)\s+)?(window|tab)$', prompt_lower)
        if match:
            app_name = match.group(1) or 'browser'
            window_type = match.group(2)
            if window_type == 'window':
                return self._create_new_window_plan(app_name)
            else:
                return self._create_new_tab_plan(app_name)
        
        # Pattern: "find [folder] folder" or "can't find [folder] folder" or "open [folder] folder"
        match = re.search(r"(?:find|open|can't find|cannot find)\s+(?:the\s+)?(\w+)\s+(?:folder|directory)", prompt_lower)
        if match:
            folder_name = match.group(1).strip()
            return self._create_open_folder_plan(folder_name)
        
        # No pattern matched
        return None
    
    def _create_search_plan(self, query: str) -> ActionPlan:
        """Create a plan for generic search (use default browser).
        
        Args:
            query: Search query
        
        Returns:
            ActionPlan
        """
        context = get_context_memory()
        browser = context.get_open_browser()
        
        if not browser:
            # No browser open, use default
            return ActionPlan([
                {'name': 'open_app', 'params': {'name': 'DEFAULT_BROWSER'}},
                {'name': 'new_tab', 'params': {}},
                {'name': 'search', 'params': {'query': query}}
            ])
        else:
            # Browser already open, just search
            return ActionPlan([
                {'name': 'new_tab', 'params': {}},
                {'name': 'search', 'params': {'query': query}}
            ])
    
    def _create_browser_search_plan(self, browser: str, query: str) -> ActionPlan:
        """Create a plan for searching in a specific browser.
        
        Args:
            browser: Browser name (opera, firefox, etc)
            query: Search query
        
        Returns:
            ActionPlan
        """
        context = get_context_memory()
        is_running = context.is_app_running(browser)
        
        if not is_running:
            # Need to open browser first
            return ActionPlan([
                {'name': 'open_app', 'params': {'name': browser}},
                {'name': 'new_tab', 'params': {}},
                {'name': 'search', 'params': {'query': query}}
            ])
        else:
            # Browser already running, just open new tab and search
            return ActionPlan([
                {'name': 'new_tab', 'params': {}},
                {'name': 'search', 'params': {'query': query}}
            ])
    
    def _create_new_window_search_plan(self, browser: str, query: str) -> ActionPlan:
        """Create a plan for opening new window and searching.
        
        Args:
            browser: Browser name
            query: Search query
        
        Returns:
            ActionPlan
        """
        context = get_context_memory()
        is_running = context.is_app_running(browser)
        
        if not is_running:
            # Open new browser with search
            return ActionPlan([
                {'name': 'open_app', 'params': {'name': browser}},
                {'name': 'search', 'params': {'query': query}}
            ])
        else:
            # Browser running, open new window
            return ActionPlan([
                {'name': 'new_window', 'params': {'browser': browser}},
                {'name': 'search', 'params': {'query': query}}
            ])
    
    def _create_new_window_plan(self, app_name: str) -> ActionPlan:
        """Create a plan for opening a new window.
        
        Args:
            app_name: Application name
        
        Returns:
            ActionPlan
        """
        context = get_context_memory()
        is_running = context.is_app_running(app_name)
        
        if not is_running:
            # App not running, just open it
            return ActionPlan([
                {'name': 'open_app', 'params': {'name': app_name}}
            ])
        else:
            # App running, open new window
            return ActionPlan([
                {'name': 'new_window', 'params': {'browser': app_name}}
            ])
    
    def _create_new_tab_plan(self, app_name: str) -> ActionPlan:
        """Create a plan for opening a new tab.
        
        Args:
            app_name: Application name
        
        Returns:
            ActionPlan
        """
        context = get_context_memory()
        
        # Try to find an open browser
        open_browser = context.get_open_browser()
        if open_browser:
            return ActionPlan([
                {'name': 'new_tab', 'params': {'browser': open_browser}}
            ])
        else:
            # No browser open, open default one
            return ActionPlan([
                {'name': 'open_app', 'params': {'name': 'DEFAULT_BROWSER'}},
                {'name': 'new_tab', 'params': {}}
            ])
    
    def _create_open_url_plan(self, browser: str, url: str) -> ActionPlan:
        """Create a plan for opening a URL in a specific browser.
        
        Args:
            browser: Browser name
            url: URL to open
        
        Returns:
            ActionPlan
        """
        context = get_context_memory()
        is_running = context.is_app_running(browser)
        
        if not is_running:
            return ActionPlan([
                {'name': 'open_app', 'params': {'name': browser}},
                {'name': 'go_to', 'params': {'url': url}}
            ])
        else:
            return ActionPlan([
                {'name': 'new_tab', 'params': {}},
                {'name': 'go_to', 'params': {'url': url}}
            ])
    
    def _create_goto_url_plan(self, url: str) -> ActionPlan:
        """Create a plan for going to a URL (use open browser or default).
        
        Args:
            url: URL to navigate to
        
        Returns:
            ActionPlan
        """
        context = get_context_memory()
        open_browser = context.get_open_browser()
        
        if open_browser:
            return ActionPlan([
                {'name': 'new_tab', 'params': {}},
                {'name': 'go_to', 'params': {'url': url}}
            ])
        else:
            return ActionPlan([
                {'name': 'open_app', 'params': {'name': 'DEFAULT_BROWSER'}},
                {'name': 'go_to', 'params': {'url': url}}
            ])
    
    def _create_open_folder_plan(self, folder_name: str) -> ActionPlan:
        """Create a plan for opening a folder.
        
        Args:
            folder_name: Name of folder to open (e.g., 'downloads', 'documents')
        
        Returns:
            ActionPlan
        """
        return ActionPlan([
            {'name': 'open_folder', 'params': {'path': folder_name}}
        ])


def parse_intent(prompt: str) -> Optional[ActionPlan]:
    """Parse natural language into action plan.
    
    Args:
        prompt: User's natural language request
    
    Returns:
        ActionPlan if intent recognized, None otherwise
    """
    parser = IntentParser()
    return parser.parse_intent(prompt)
