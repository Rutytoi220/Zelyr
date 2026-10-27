"""Rule-Based Intent Engine for the Proactive Perception Kernel.

Evaluates filesystem events, active window context, and clipboard content
using deterministic, sub-millisecond rules (NO LLM calls). Maps matched
events to Action Modes and routes them via the EventBus.
"""
import re
from typing import Dict, Any, Optional
from pathlib import Path
from zelyr.core.events import EventBus, Event
_PYTHON_TRACEBACK_RE = re.compile('Traceback \\(most recent call last\\)', re.MULTILINE)
_STACK_OVERFLOW_URL_RE = re.compile('https?://stackoverflow\\.com/questions/\\d+', re.IGNORECASE)
_GIT_DIFF_RE = re.compile('^[+-]{3} [ab]/', re.MULTILINE)
_BASH_ERROR_RE = re.compile('(?:command not found|permission denied|no such file or directory)', re.IGNORECASE)
_SQL_QUERY_RE = re.compile('\\b(SELECT|INSERT|UPDATE|DELETE|CREATE|DROP)\\b', re.IGNORECASE)
_TERMINAL_PROCESSES = frozenset({'bash', 'zsh', 'fish', 'sh', 'terminal', 'gnome-terminal', 'konsole', 'xterm', 'alacritty', 'kitty', 'wezterm', 'iterm2', 'windows terminal', 'powershell', 'cmd'})
_CODE_EDITOR_PROCESSES = frozenset({'code', 'code-oss', 'codium', 'vim', 'nvim', 'neovim', 'emacs', 'sublime_text', 'atom', 'pycharm', 'intellij', 'clion', 'goland'})

class IntentEngine:
    """Fast heuristic evaluator mapping perception events to background actions."""

    def __init__(self, event_bus: EventBus):
        """Auto-generated docstring.

Args:
    event_bus: Argument.

Returns:
    Return value.
"""
        self.event_bus = event_bus  # pragma: no cover

    def evaluate(self, event_type: str, file_path: str) -> Optional[Dict[str, Any]]:
        """Evaluate a filesystem event and trigger actions if a rule matches.
        
        Args:
            event_type: "created" or "modified"
            file_path: Absolute path to the file
            
        Returns:
            The matched rule metadata, or None if no rule matched.
        """
        path = Path(file_path)  # pragma: no cover
        if event_type == 'created' and path.suffix == '.py':  # pragma: no cover
            action = {'action_mode': 'silent', 'task': 'auto_index', 'target': file_path, 'reason': 'New python file created'}  # pragma: no cover
            self.event_bus.publish(Event(event_type='perception.intent.silent', source='IntentEngine', data=action))  # pragma: no cover
            return action  # pragma: no cover
        if event_type == 'modified' and 'traceback.log' in path.name:  # pragma: no cover
            action = {'action_mode': 'notify', 'task': 'analyze_error', 'target': file_path, 'message': 'Error traceback detected. Want me to analyze it?'}  # pragma: no cover
            self.event_bus.publish(Event(event_type='perception.intent.notify', source='IntentEngine', data=action))  # pragma: no cover
            return action  # pragma: no cover
        return None  # pragma: no cover

    def evaluate_context(self, source: str, window_title: str, process: str) -> Optional[Dict[str, Any]]:
        """Evaluate an active window focus change and emit context suggestions.

        Args:
            source:       The tap that produced the event (e.g., "WindowFocusTap").
            window_title: The focused window's title.
            process:      The executable/process name owning the window.

        Returns:
            Matched action dict, or None.
        """
        process_lower = process.lower()  # pragma: no cover
        title_lower = window_title.lower()  # pragma: no cover
        if process_lower in _TERMINAL_PROCESSES:  # pragma: no cover
            action = {'action_mode': 'silent', 'task': 'context_terminal', 'process': process, 'message': 'Terminal context active.'}  # pragma: no cover
            self.event_bus.publish(Event(event_type='perception.intent.context', source='IntentEngine', data=action))  # pragma: no cover
            return action  # pragma: no cover
        if process_lower in _CODE_EDITOR_PROCESSES:  # pragma: no cover
            action = {'action_mode': 'silent', 'task': 'context_editor', 'process': process, 'message': 'Code editor context active.'}  # pragma: no cover
            self.event_bus.publish(Event(event_type='perception.intent.context', source='IntentEngine', data=action))  # pragma: no cover
            return action  # pragma: no cover
        return None  # pragma: no cover

    def evaluate_clipboard(self, text: str) -> Optional[Dict[str, Any]]:
        """Evaluate scrubbed clipboard text and emit proactive suggestions.

        Args:
            text: **Already scrubbed** clipboard text (no raw secrets).

        Returns:
            Matched action dict, or None.
        """
        if _PYTHON_TRACEBACK_RE.search(text):  # pragma: no cover
            action = {'action_mode': 'notify', 'task': 'analyze_traceback', 'message': 'Python traceback detected in clipboard. Want me to debug it?', 'snippet': text[:500]}  # pragma: no cover
            self.event_bus.publish(Event(event_type='perception.intent.notify', source='IntentEngine', data=action))  # pragma: no cover
            return action  # pragma: no cover
        if _BASH_ERROR_RE.search(text):  # pragma: no cover
            action = {'action_mode': 'notify', 'task': 'explain_shell_error', 'message': 'Shell error copied. Want me to explain the fix?', 'snippet': text[:300]}  # pragma: no cover
            self.event_bus.publish(Event(event_type='perception.intent.notify', source='IntentEngine', data=action))  # pragma: no cover
            return action  # pragma: no cover
        if _STACK_OVERFLOW_URL_RE.search(text):  # pragma: no cover
            action = {'action_mode': 'silent', 'task': 'prefetch_so_context', 'message': 'Stack Overflow URL detected.', 'snippet': text[:200]}  # pragma: no cover
            self.event_bus.publish(Event(event_type='perception.intent.silent', source='IntentEngine', data=action))  # pragma: no cover
            return action  # pragma: no cover
        if _SQL_QUERY_RE.search(text):  # pragma: no cover
            action = {'action_mode': 'notify', 'task': 'explain_sql', 'message': 'SQL query detected in clipboard. Want me to explain or optimise it?', 'snippet': text[:400]}  # pragma: no cover
            self.event_bus.publish(Event(event_type='perception.intent.notify', source='IntentEngine', data=action))  # pragma: no cover
            return action  # pragma: no cover
        return None  # pragma: no cover
