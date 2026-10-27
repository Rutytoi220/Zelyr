"""Temporal Time-Travel Debugger.

Wraps code execution using sys.settrace to maintain a rolling ring-buffer
of the last N local variable state mutations. If an exception occurs, the
buffer is serialized into a JSON trace and returned to the LLM so it can
see *how* the variables mutated leading up to the crash.
"""
import sys
import copy
import json
import logging
import traceback
from collections import deque
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

class TemporalDebuggerService:
    """Provides time-travel debugging capabilities for Python execution."""
    
    def __init__(self, buffer_size: int = 100):
        self.buffer_size = buffer_size
        self._history: deque = deque(maxlen=buffer_size)
        self._is_tracing = False
        
    def _trace_calls(self, frame, event, arg):
        """Global trace function."""
        if event != 'call':
            return None
        
        # Don't trace into standard library or our own debugger code to save overhead
        filename = frame.f_code.co_filename
        if "site-packages" in filename or "temporal_debugger" in filename or filename.startswith("<frozen"):
            return None
            
        return self._trace_lines
        
    def _trace_lines(self, frame, event, arg):
        """Local trace function to capture state at each line."""
        if event not in ('line', 'exception'):
            return self._trace_lines
            
        try:
            filename = frame.f_code.co_filename
            lineno = frame.f_lineno
            
            # Serialize locals safely
            locals_snapshot = {}
            for k, v in frame.f_locals.items():
                if k.startswith("__"): 
                    continue
                try:
                    # Attempt a deepcopy for primitives, fallback to repr for complex objects
                    if isinstance(v, (int, float, str, bool, type(None))):
                        locals_snapshot[k] = v
                    elif isinstance(v, (list, dict, set, tuple)):
                        # A shallow copy of basic containers
                        locals_snapshot[k] = str(v)[:200] # Truncate large collections
                    else:
                        locals_snapshot[k] = repr(v)[:100]
                except Exception:
                    locals_snapshot[k] = "<unserializable>"
                    
            state = {
                "file": filename,
                "line": lineno,
                "event": event,
                "locals": locals_snapshot
            }
            
            if event == 'exception':
                state["exception_arg"] = repr(arg)
                
            self._history.append(state)
        except Exception:
            pass # Never let the debugger crash the target code
            
        return self._trace_lines

    def execute_with_time_travel(self, func, *args, **kwargs) -> Dict[str, Any]:
        """
        Executes a function under the temporal trace.
        Returns a dictionary with 'success', 'result', and 'temporal_trace'.
        """
        self._history.clear()
        self._is_tracing = True
        
        # Install trace
        old_trace = sys.gettrace()
        sys.settrace(self._trace_calls)
        
        success = False
        result = None
        error_msg = None
        
        try:
            result = func(*args, **kwargs)
            success = True
        except Exception as e:
            error_msg = traceback.format_exc()
            logger.warning(f"Temporal Debugger: Caught exception. Generating Time-Travel Trace...")
        finally:
            # Remove trace
            sys.settrace(old_trace)
            self._is_tracing = False
            
        return {
            "success": success,
            "result": result,
            "error": error_msg,
            "temporal_trace": list(self._history) if not success else []
        }
