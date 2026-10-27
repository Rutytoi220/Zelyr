import os
import sys
from pathlib import Path
from typing import Any, Dict

from zelyr.compat import config_file, config_dir, data_file

# tomllib is standard in Python 3.11+
if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

CONFIG_DIR = config_dir()
CONFIG_FILE = config_file("config.toml")

DEFAULT_CONFIG = """[engine]
default_effort = "medium"
memory_db_path = "~/.local/share/zelyr/memory.db"

[models]
low = "laguna-xs-2.1:q4_K_M"
medium = "laguna-xs-2.1:q4_K_M"
balanced = "MiMo-V2.6:latest"
high = "MiMo-V2.6:latest"
max = "MiMo-V2.6:latest"
ultra = "gemini-3.8-flash"

[context_limits]
"laguna-xs-2.1:q4_K_M" = 4
"MiMo-V2.6:latest" = 12
"gemini-3.8-flash" = 100

[cognitive_modes.overthinking]
system_prompt = "COGNITIVE OVERRIDE: Question every assumption, analyze edge cases excessively, and provide deep technical breakdowns."

[cognitive_modes.adhd]
system_prompt = "COGNITIVE OVERRIDE: Think rapidly, make creative conceptual leaps, and hyper-focus on niche details."
"""
DEFAULT_CONFIG = DEFAULT_CONFIG.replace(
    "~/.local/share/zelyr/memory.db", str(data_file("memory.db"))
)

_config_cache: Dict[str, Any] = {}

def load_config() -> Dict[str, Any]:
    global _config_cache
    if _config_cache:
        return _config_cache

    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    
    if not CONFIG_FILE.exists():
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            f.write(DEFAULT_CONFIG)
            
    with open(CONFIG_FILE, "rb") as f:
        _config_cache = tomllib.load(f)
        
    return _config_cache

def get_engine_config() -> Dict[str, Any]:
    return load_config().get("engine", {})

def get_model_routing() -> Dict[str, str]:
    return load_config().get("models", {})

def get_context_limits() -> Dict[str, int]:
    return load_config().get("context_limits", {})

def get_cognitive_modes() -> Dict[str, Dict[str, str]]:
    return load_config().get("cognitive_modes", {})


def reload_config() -> None:
    global _config_cache
    if not CONFIG_FILE.exists():
        return
    with open(CONFIG_FILE, "rb") as f:
        _config_cache = tomllib.load(f)
