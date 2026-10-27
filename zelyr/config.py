"""Zelyr Configuration module."""
from dataclasses import dataclass, field
import logging
import os
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from zelyr.compat import config_dir, config_file, data_dir, get_env

logger = logging.getLogger(__name__)

def get_config_dir() -> Path:
    """Return the config directory, preferring Zelyr and reading legacy AXIOM."""
    return config_dir()

CONFIG_DIR = get_config_dir()

def get_data_dir() -> Path:
    """Return the XDG data directory, preferring Zelyr and reading legacy AXIOM."""
    return data_dir()

DATA_DIR = get_data_dir()

def initialize_directories(config_dir: Optional[Path] = None) -> None:
    """Ensure all required first-run Zelyr directories exist cleanly with parents=True, exist_ok=True."""
    base = config_dir or get_config_dir()
    tools_dir = base / "tools.d"
    plugins_dir = base / "plugins"
    sessions_dir = base / "sessions"
    logs_dir = base / "logs"
    models_dir = base / "models"

    for d in (base, tools_dir, plugins_dir, sessions_dir, logs_dir, models_dir):
        try:
            d.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.warning(f"Could not create directory {d}: {e}")

# Automatically initialize directories on module load
initialize_directories()

def validate_and_clamp_num_ctx(val: Any) -> int:
    """Validate and clamp context length.

    Valid range: 1024 to 65536.
    Invalid values (non-integers, strings, negatives, out-of-range) log a warning and fall back to 8192.
    """
    if val is None:
        return 8192
    try:
        if isinstance(val, bool):
            logger.warning(f"Invalid context length boolean: {val}. Falling back to 8192.")
            return 8192
        int_val = int(val)
    except (ValueError, TypeError):
        logger.warning(f"Invalid context length value: {val}. Falling back to 8192.")
        return 8192

    if int_val < 1024 or int_val > 65536:
        logger.warning(f"Context length {int_val} out of valid range [1024, 65536]. Falling back to 8192.")
        return 8192

    return int_val

class AuthMode(Enum):
    STRICT = 'strict'
    BASIC = 'basic'
    AUTOPILOT = 'autopilot'

@dataclass
class BehaviorConfig:
    """Auto-generated docstring.

"""
    profile: str = 'default'

@dataclass
class ZelyrConfig:
    """Zelyr system configuration."""
    auth_mode: AuthMode = AuthMode.BASIC
    behavior: BehaviorConfig | None = None
    debug: bool = False
    log_level: str = 'INFO'
    proactive_kernel: bool = False
    allow_cloud_fallback: bool = False
    monitor_window_focus: bool = False
    monitor_clipboard: bool = False
    allow_third_party_plugins: bool = False

    def __post_init__(self):
        """Auto-generated docstring.


Returns:
    Return value.
"""
        if self.behavior is None:
            self.behavior = BehaviorConfig()
    ollama_base_url: str = 'http://127.0.0.1:11434'
    ollama_model: str = 'qwen3:8b'
    ollama_num_ctx: int = 8192
    power_saving_model: str = 'qwen2.5:1.5b'
    embedding_model: str = 'nomic-embed-text'
    ollama_temperature: float = 0.7
    db_path: str = 'zelyr.db'
    max_history: int = 1000
    max_agents: int = 10
    max_tools: int = 100
    event_history_limit: int = 1000
    sandbox_mode: bool = True
    allow_system_tools: bool = True
    max_vector_memories: int = 5000
    show_thinking: bool = True
    
    # UI / GUI Settings
    first_launch: bool = True
    oobe_completed: bool = False
    vision_model: str = ''
    theme: str = 'minimalist'  # 'minimalist', 'cyberpunk', 'nothing'
    theme_mode: str = 'dark'  # 'system', 'dark', 'light'
    ui_profile_level: str = 'standard' # 'standard', 'advanced', 'developer'
    persona: dict = field(default_factory=dict)
    persona_key: str = 'default'
    active_persona: str = 'zelyr_core'
    llm_complexity: str = 'detailed'
    auto_ollama_start: bool = True
    model_selection_mode: str = 'auto'  # 'auto', 'manual'
    auto_index_watchdog: bool = False
    monitored_paths: list[str] = field(default_factory=lambda: [str(Path.home() / 'Documents')])

    # Swarm Compute
    swarm_enabled: bool = False
    remote_endpoints: list[str] = field(default_factory=list)
    offload_strategy: str = 'thermal_trigger'

    engine_mode: str = 'local'  # 'local', 'remote', 'fastapi'
    remote_server_ip: str = '127.0.0.1:9412'

    # Plugins
    disabled_plugins: list[str] = field(default_factory=list)
    model_usage_counts: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, config_dict: dict) -> 'ZelyrConfig':
        """Create config from dictionary."""
        filtered = {k: v for k, v in config_dict.items() if k in cls.__dataclass_fields__}
        
        # Migrate legacy loose persona strings to new dict structure
        if 'persona' not in filtered:
            filtered['persona'] = {}
            if 'persona_tone' in config_dict:
                filtered['persona']['communication'] = {'tone': config_dict['persona_tone'], 'verbosity': config_dict.get('persona_complexity', 'standard')}
            if 'special_instructions' in config_dict and config_dict['special_instructions']:
                filtered['persona']['directives'] = [config_dict['special_instructions']]
                
        if 'ollama_base_url' in filtered and isinstance(filtered['ollama_base_url'], str):
            filtered['ollama_base_url'] = filtered['ollama_base_url'].replace("localhost", "127.0.0.1")

        if 'ollama_num_ctx' in filtered:
            filtered['ollama_num_ctx'] = validate_and_clamp_num_ctx(filtered['ollama_num_ctx'])

        if 'auth_mode' in filtered and isinstance(filtered['auth_mode'], str):
            try:
                filtered['auth_mode'] = AuthMode(filtered['auth_mode'])
            except ValueError:
                filtered['auth_mode'] = AuthMode.BASIC
        return cls(**filtered)

    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return {
            'auth_mode': self.auth_mode.value, 
            'debug': self.debug, 
            'log_level': self.log_level, 
            'proactive_kernel': self.proactive_kernel, 
            'ollama_base_url': self.ollama_base_url, 
            'ollama_model': self.ollama_model, 
            'ollama_num_ctx': self.ollama_num_ctx,
            'embedding_model': self.embedding_model, 
            'ollama_temperature': self.ollama_temperature, 
            'db_path': self.db_path, 
            'max_history': self.max_history, 
            'max_agents': self.max_agents, 
            'max_tools': self.max_tools, 
            'event_history_limit': self.event_history_limit, 
            'sandbox_mode': self.sandbox_mode, 
            'allow_system_tools': self.allow_system_tools,
            'max_vector_memories': self.max_vector_memories,
            'allow_cloud_fallback': self.allow_cloud_fallback, 
            'monitor_window_focus': self.monitor_window_focus, 
            'monitor_clipboard': self.monitor_clipboard,
            'allow_third_party_plugins': self.allow_third_party_plugins,
            'first_launch': self.first_launch,
            'oobe_completed': self.oobe_completed,
            'vision_model': self.vision_model,
            'theme': self.theme,
            'theme_mode': self.theme_mode,
            'ui_profile_level': self.ui_profile_level,
            'persona': self.persona,
            'persona_key': self.persona_key,
            'active_persona': self.active_persona,
            'llm_complexity': self.llm_complexity,
            'auto_ollama_start': self.auto_ollama_start,
            'model_selection_mode': self.model_selection_mode,
            'auto_index_watchdog': self.auto_index_watchdog,
            'monitored_paths': self.monitored_paths,
            'swarm_enabled': self.swarm_enabled,
            'remote_endpoints': self.remote_endpoints,
            'offload_strategy': self.offload_strategy,
            'engine_mode': self.engine_mode,
            'remote_server_ip': self.remote_server_ip,
            'disabled_plugins': self.disabled_plugins,
            'model_usage_counts': self.model_usage_counts,
            'show_thinking': self.show_thinking,
        }

    def get_effective_num_ctx(self, caller_override: Optional[Any] = None) -> int:
        """Resolve effective Ollama context length with strict precedence:
        1. Caller/Task override (explicitly passed options in payload)
        2. Environment variable ZELYR_NUM_CTX (parsed as int, clamped)
        3. Config file setting 'ollama_num_ctx' in ~/.config/zelyr/config.json
        4. System default: 8192
        """
        if caller_override is not None:
            return validate_and_clamp_num_ctx(caller_override)

        env_val = get_env("NUM_CTX")
        if env_val is not None:
            return validate_and_clamp_num_ctx(env_val)

        if hasattr(self, "ollama_num_ctx") and self.ollama_num_ctx is not None:
            return validate_and_clamp_num_ctx(self.ollama_num_ctx)

        return 8192

    def save(self) -> None:
        """Save configuration to ~/.config/zelyr/config.json."""
        import json
        from pathlib import Path
        data = self.to_dict()
        base_dir = get_config_dir()
        paths = [
            base_dir / "config.json",
        ]
        for p in paths:
            try:
                p.parent.mkdir(parents=True, exist_ok=True)
                with open(p, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=4)
            except Exception as e:
                print(f"Failed to save Zelyr config to {p}: {e}")

    @classmethod
    def load(cls) -> 'ZelyrConfig':
        """Load configuration from the Zelyr config file or legacy AXIOM paths."""
        import json
        from pathlib import Path
        base_dir = get_config_dir()
        paths = [
            config_file("config.json"),
            base_dir / "config.json",
            config_dir() / "config.json",
            Path.home() / ".config" / "axiom" / "config.json",
            Path.home() / ".config" / "ChienGPT" / "config.json",
        ]
        for config_path in dict.fromkeys(paths):
            if config_path.exists():
                try:
                    with open(config_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    if 'oobe_completed' not in data:
                        legacy_ui_config = config_path.parent / "ui_config.json"
                        data['oobe_completed'] = legacy_ui_config.exists()
                    return cls.from_dict(data)
                except Exception as e:
                    print(f"Failed to load Zelyr config from {config_path}: {e}")
        return cls()

# Backwards-compatible public class name for existing AXIOM integrations.
AxiomConfig = ZelyrConfig

_config = ZelyrConfig.load()

def get_config() -> ZelyrConfig:
    """Get global Zelyr configuration."""
    return _config

def set_config(config: ZelyrConfig) -> None:
    """Set global Zelyr configuration."""
    global _config
    _config = config

def get_effective_num_ctx(caller_override: Optional[Any] = None) -> int:
    """Resolve effective Ollama context length using global config."""
    return get_config().get_effective_num_ctx(caller_override)
try:
    from core.settings_registry import SettingCategory, SettingMetadata, SettingsRegistry, get_settings_registry
except ImportError:
    pass

