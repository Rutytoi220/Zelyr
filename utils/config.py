"""Configuration loader for Zelyr.

Reads `config/config.json` from the project root and returns a dict.
"""

import json
import os
from pathlib import Path

_CONFIG: dict | None = None


def get_config() -> dict:
    global _CONFIG
    if _CONFIG is not None:
        return _CONFIG
    root = Path(__file__).resolve().parents[1]
    cfg_path = root / "config" / "config.json"
    if not cfg_path.exists():
        _CONFIG = {}
        return _CONFIG
    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            _CONFIG = json.load(f)
    except Exception:
        _CONFIG = {}
    return _CONFIG
