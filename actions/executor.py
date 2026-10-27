"""Action executors for Zelyr.

Implements:
- open_folder(path)
- open_app(name)
- copy_to_clipboard(text)
- run_command(cmd)
- Multi-step action execution with retry & stop-on-error

All actions return (ok: bool, message: str).
"""

import os
import shutil
import subprocess
import time
from typing import Tuple, Dict, Optional, List
from utils.logger import get_logger
from utils.config import get_config
from utils.system import get_default_browser, is_browser, KNOWN_BROWSERS
from security.validator import is_app_allowed, find_allowed_app, is_path_allowed, safe_split_command
from utils.folder_search import resolve_folder_path, expand_path

logger = get_logger(__name__)

# Import context memory and action registry
try:
    from brain.context_memory import get_context_memory
    from brain.action_registry import get_action_registry
    CONTEXT_AVAILABLE = True
except ImportError:
    CONTEXT_AVAILABLE = False
    logger.warning("Context memory or action registry not available")


def _get_default_browser() -> str:
    """Get the default browser name (delegates to shared util).
    
    Returns:
        Default browser name (e.g., 'opera', 'firefox')
    """
    return get_default_browser() or 'firefox'


def _replace_placeholders(text: str) -> str:
    """Replace DEFAULT_BROWSER placeholder with actual default browser.
    
    Args:
        text: Text with potential placeholders
    
    Returns:
        Text with placeholders replaced
    """
    if 'DEFAULT_BROWSER' in text:
        default = _get_default_browser()
        return text.replace('DEFAULT_BROWSER', default)
    return text


def _parse_kv_params(params: str) -> Dict[str, str]:
    # basic key="value" parser
    import re
    out = {}
    if not params:
        return out
    for m in re.finditer(r"(\w+)\s*=\s*\"([^\"]+)\"", params):
        out[m.group(1)] = m.group(2)
    for m in re.finditer(r"(\w+)\s*=\s*'([^']+)'", params):
        out[m.group(1)] = m.group(2)
    return out


def open_folder(path: str) -> Tuple[bool, str]:
    path = (path or '').strip()
    if not path:
        return False, "No path provided"
    # allow key=value form
    kv = _parse_kv_params(path)
    if 'path' in kv:
        path = kv['path']
    
    # Try to resolve folder path (handles names like "downloads", paths with $HOME, etc.)
    resolved_path = resolve_folder_path(path)
    if not resolved_path:
        # If not found by name, try direct expansion
        resolved_path = expand_path(path)
    
    if not os.path.isdir(resolved_path):
        return False, f"Not a directory: {resolved_path}"
    if not is_path_allowed(resolved_path):
        return False, f"Path not allowed: {resolved_path}"
    opener = shutil.which('xdg-open') or shutil.which('gio') or shutil.which('gnome-open')
    if not opener:
        return False, 'No opener available (xdg-open/gio/gnome-open)'
    try:
        subprocess.Popen([opener, resolved_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        logger.info("Opened folder: %s", resolved_path)
        return True, f"Opened folder: {resolved_path}"
    except Exception as e:
        logger.exception("Failed to open folder")
        return False, f"Failed to open folder: {e}"


def open_app(name: str) -> Tuple[bool, str]:
    name = (name or '').strip()
    kv = _parse_kv_params(name)
    if 'name' in kv:
        name = kv['name']
    if not name:
        return False, 'No app name provided'
    
    # Replace placeholders
    name = _replace_placeholders(name)
    
    # Prefer the smarter finder that returns the matched allowed key
    found = find_allowed_app(name)
    if found:
        matched_name, bin_path = found
        try:
            subprocess.Popen([bin_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            logger.info("Launched app: %s (%s)", matched_name, bin_path)
            
            # Register in context memory
            if CONTEXT_AVAILABLE:
                context = get_context_memory()
                context.register_app_opened(matched_name)
            
            return True, f"Launched app: {matched_name}"
        except Exception as e:
            logger.exception("Failed to launch app")
            return False, f"Failed to launch app: {e}"

    # fallback to legacy check
    bin_path = is_app_allowed(name)
    if not bin_path:
        return False, f'App not allowed or not found: {name}'
    try:
        subprocess.Popen([bin_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        logger.info("Launched app: %s (%s)", name, bin_path)
        
        # Register in context memory
        if CONTEXT_AVAILABLE:
            context = get_context_memory()
            context.register_app_opened(name)
        
        return True, f"Launched app: {name}"
    except Exception as e:
        logger.exception("Failed to launch app")
        return False, f"Failed to launch app: {e}"


def copy_to_clipboard(text: str) -> Tuple[bool, str]:
    text = (text or '').strip()
    kv = _parse_kv_params(text)
    if 'text' in kv:
        text = kv['text']
    # try external tools
    for cmd in (('wl-copy',), ('xclip', '-selection', 'clipboard'), ('xsel', '--clipboard', '--input')):
        bin_path = shutil.which(cmd[0])
        if bin_path:
            try:
                subprocess.run([bin_path] + list(cmd[1:]), input=text, text=True, check=True)
                return True, 'Copied to clipboard.'
            except Exception as e:
                logger.debug('Clipboard tool %s failed: %s', cmd[0], e)
                return False, f'Clipboard command failed: {e}'
    # fallback to tkinter
    try:
        from tkinter import Tk
        r = Tk()
        r.withdraw()
        r.clipboard_clear()
        r.clipboard_append(text)
        r.update()
        r.destroy()
        return True, 'Copied to clipboard via Tkinter.'
    except Exception as e:
        logger.exception('No clipboard available')
        return False, f'No clipboard tool available: {e}'


def run_command(cmd: str) -> Tuple[bool, str]:
    cmd = (cmd or '').strip()
    kv = _parse_kv_params(cmd)
    if 'cmd' in kv:
        cmd = kv['cmd']
    
    # Replace placeholders
    cmd = _replace_placeholders(cmd)
    
    parts = safe_split_command(cmd)
    if not parts:
        return False, 'Command not allowed or could not parse.'
    
    # Browser commands should be launched in background without waiting
    first_cmd = parts[0].split('/')[-1]  # Handle full paths
    
    if is_browser(first_cmd):
        # Launch browser in background (fire and forget)
        try:
            subprocess.Popen(parts, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            logger.info("Launched browser: %s", first_cmd)
            
            # Register in context
            if CONTEXT_AVAILABLE:
                context = get_context_memory()
                context.register_app_opened(first_cmd)
            
            return True, f'Launched {first_cmd}'
        except Exception as e:
            logger.exception('Browser launch failed')
            return False, f'Failed to launch browser: {e}'
    
    # Regular commands: wait for output
    try:
        r = subprocess.run(parts, capture_output=True, text=True, timeout=10)
        out = r.stdout.strip() or r.stderr.strip() or 'No output.'
        return True, out[:1000]
    except subprocess.TimeoutExpired:
        logger.error('Command timed out: %s', parts)
        return False, f'Command timed out after 10 seconds: {first_cmd}'
    except Exception as e:
        logger.exception('Command execution failed')
        return False, f'Command execution failed: {e}'


def execute_instruction(action: str, params: str) -> Tuple[bool, str]:
    a = (action or '').strip()
    if a == 'open_folder':
        return open_folder(params)
    if a == 'open_app':
        return open_app(params)
    if a == 'copy_to_clipboard':
        return copy_to_clipboard(params)
    if a == 'run_command':
        return run_command(params)
    # Desktop control actions
    if a == 'screenshot':
        from actions.desktop import take_screenshot
        ok, result = take_screenshot(encode_base64=True)
        # For screenshot, return just the path/indicator for now
        return ok, 'Screenshot captured' if ok else result
    if a == 'click':
        from actions.desktop import click_mouse
        # params format: "x=100 y=200" or "x=100 y=200 button=right"
        kv = _parse_kv_params(params)
        x = kv.get('x', '0')
        y = kv.get('y', '0')
        button = kv.get('button', 'left')
        try:
            return click_mouse(int(x), int(y), button=button)
        except:
            return False, 'Invalid click parameters'
    if a == 'type':
        from actions.desktop import type_text
        kv = _parse_kv_params(params)
        if 'text' in kv:
            text = kv['text']
        else:
            text = params.strip()
        return type_text(text)
    if a == 'press':
        from actions.desktop import press_keys
        kv = _parse_kv_params(params)
        if 'keys' in kv:
            keys = kv['keys']
        else:
            keys = params.strip()
        return press_keys(keys)
    if a == 'move':
        from actions.desktop import move_mouse
        kv = _parse_kv_params(params)
        x = kv.get('x', '0')
        y = kv.get('y', '0')
        try:
            return move_mouse(int(x), int(y))
        except:
            return False, 'Invalid move parameters'
    if a == 'screen_size':
        from actions.desktop import get_screen_size
        return get_screen_size()
    
    # Try action registry for extended actions
    if CONTEXT_AVAILABLE:
        registry = get_action_registry()
        result = registry.execute(a, params)
        if result[0] is not None:  # Registry returned a result
            return result
    
    return False, f'Unknown action: {action}'


def execute_and_record(action: str, params: str) -> Tuple[bool, str]:
    """Execute a single instruction and record the result in context memory.

    Thin wrapper around execute_instruction that ensures every action
    is tracked even when called outside of execute_action_sequence.
    """
    ok, msg = execute_instruction(action, params)
    if CONTEXT_AVAILABLE:
        ctx = get_context_memory()
        ctx.record_action(action, params, ok, msg)
    return ok, msg


def _execute_with_retry(action: str, params: str, retries: int = 0,
                        retry_delay: float = 1.0) -> Tuple[bool, str]:
    """Execute a single instruction with optional retry.

    Args:
        action: Action name
        params: Parameter string
        retries: Number of extra attempts on failure (0 = no retry)
        retry_delay: Seconds to wait between retries

    Returns:
        (success, message) tuple
    """
    last_msg = ''
    for attempt in range(1 + retries):
        ok, msg = execute_instruction(action, params)

        # Record in context memory
        if CONTEXT_AVAILABLE:
            ctx = get_context_memory()
            ctx.record_action(action, params, ok, msg)

        if ok:
            return ok, msg

        last_msg = msg
        if attempt < retries:
            logger.info("Retry %d/%d for %s: %s", attempt + 1, retries, action, msg)
            time.sleep(retry_delay)

    return False, last_msg


def execute_action_sequence(actions: List[Dict]) -> Tuple[bool, str]:
    """Execute a sequence of actions.

    Behaviour is controlled by ``config.json``:
    - ``agent.stop_on_error`` (bool, default True): abort remaining steps
      when one fails.
    - ``agent.retry_count`` (int, default 0): how many retries per step.
    - ``agent.retry_delay`` (float, default 1.0): seconds between retries.
    
    Args:
        actions: List of action dicts with 'action' and 'params'
    
    Returns:
        (success, message) tuple
    """
    if not actions:
        return False, "No actions to execute"

    # Read behaviour from config
    cfg = get_config() or {}
    agent_cfg = cfg.get('agent', {})
    stop_on_error = agent_cfg.get('stop_on_error', True)
    retries = agent_cfg.get('retry_count', 0)
    retry_delay = agent_cfg.get('retry_delay', 1.0)
    
    results = []
    logger.info("Executing sequence of %d action(s) (stop_on_error=%s, retries=%d)",
                len(actions), stop_on_error, retries)
    
    for i, action_dict in enumerate(actions):
        action = action_dict.get('action', '')
        params = action_dict.get('params', '')
        
        logger.debug("Step %d/%d: %s %s", i + 1, len(actions), action, params)
        
        # Small delay between actions (let previous action settle)
        if i > 0:
            time.sleep(0.3)
        
        ok, msg = _execute_with_retry(action, params, retries=retries,
                                       retry_delay=retry_delay)
        results.append({
            'action': action,
            'ok': ok,
            'message': msg
        })
        
        logger.info("Step %d result: %s - %s", i + 1, ok, msg)
        
        if not ok and stop_on_error:
            logger.warning("Stopping sequence at step %d due to error", i + 1)
            break
    
    # Aggregate results
    all_ok = all(r['ok'] for r in results)
    messages = [r['message'] for r in results if r['message']]
    final_message = ' → '.join(messages) if messages else 'Action sequence completed'
    
    return all_ok, final_message
