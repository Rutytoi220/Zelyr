import anthropic
"""Top-level entrypoint preserved for backward compatibility.

Supports both CLI and GUI modes.
"""

import argparse
import sys
import os
import traceback
from pathlib import Path

def _patch_tiktoken_for_pyinstaller():
    """tiktoken uses pkgutil.iter_modules which fails inside PyInstaller's frozen archive.
    We manually populate the registry here before litellm is imported."""
    try:
        import tiktoken.registry
        import tiktoken_ext.openai_public
        if getattr(tiktoken.registry, "ENCODING_CONSTRUCTORS", None) is None:
            tiktoken.registry.ENCODING_CONSTRUCTORS = {}
        tiktoken.registry.ENCODING_CONSTRUCTORS.update(tiktoken_ext.openai_public.ENCODING_CONSTRUCTORS)
    except Exception as e:
        pass # Not critical if tiktoken isn't installed yet, or let it fail normally later

_patch_tiktoken_for_pyinstaller()


def _ensure_directories():
    """Sanity check startup paths."""
    from zelyr.config import initialize_directories
    from zelyr.compat import data_dir
    initialize_directories()
    data_dir().mkdir(parents=True, exist_ok=True)


def daemon_entry_point():
    """Console script entry point for the Zelyr background daemon."""
    from zelyr.server.daemon import main as daemon_main
    daemon_main()


def main():
    parser = argparse.ArgumentParser(description="Zelyr - Local AI Assistant")
    parser.add_argument("--model", default=None, help="Ollama model name (optional)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show raw LLM output (debug)")
    parser.add_argument("--system-prompt", "-s", default=None, help="Persistent system prompt to prepend to each query")
    parser.add_argument("--gui", action="store_true", help="Launch GUI instead of CLI (default: CLI)")
    parser.add_argument("--health-check", action="store_true", help="Boot verify the environment and exit 0 (for OTA rollbacks)")
    args = parser.parse_args()
    
    if args.health_check:
        print("Zelyr Health Check OK")
        sys.exit(0)
        
    _ensure_directories()
    
    if args.gui:
        try:
            # We use the new PySide6 GUI if available
            from zelyr.gui.app import run_gui
            run_gui()
        except ImportError as e:
            print(f"Error: Zelyr PySide6 GUI dependencies not available. {e}", file=sys.stderr)
            try:
                # Fallback to older Tkinter GUI
                from ui.gui import run_gui
                run_gui(model=args.model, verbose=args.verbose, system_prompt=args.system_prompt)
            except ImportError:
                print("Run 'pip install PySide6' or use CLI mode.", file=sys.stderr)
    else:
        # We might have zelyr.cli or ui.cli
        try:
            from zelyr.api.cli import CLI
            CLI().cmdloop()
        except ImportError:
            from ui.cli import run_repl
            run_repl(model=args.model, verbose=args.verbose, system_prompt=args.system_prompt)


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    try:
        main()
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as e:
        from zelyr.compat import data_dir
        crash_log = data_dir() / "zelyr_crash.log"
        # Make sure directory exists even if it failed before _ensure_directories
        crash_log.parent.mkdir(parents=True, exist_ok=True)
        
        trace = traceback.format_exc()
        print(f"\n[Zelyr FATAL CRASH]\n{trace}", file=sys.stderr)
        
        with open(crash_log, "w") as log_file:
            log_file.write(f"Zelyr CRASH REPORT\n")
            log_file.write(f"==================\n")
            log_file.write(trace)
            
        sys.exit(1)
