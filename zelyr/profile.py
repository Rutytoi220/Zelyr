"""Zelyr Profile — Personality and Behavior configuration.

Defines the user's calibrated preferences as two orthogonal layers:

- **Personality**: How Zelyr communicates (injected into LLM system prompt).
- **Behavior**: How Zelyr acts (injected into ToolExecutor rules).

Profiles are persisted as JSON files at ``~/.zelyr/profile.json`` and loaded
on every Zelyr startup to configure the agent's operating parameters.
"""
from __future__ import annotations
from zelyr.compat import state_dir
import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
logger = logging.getLogger(__name__)
DEFAULT_PROFILE_DIR = state_dir()
DEFAULT_PROFILE_PATH = DEFAULT_PROFILE_DIR / 'profile.json'

@dataclass
class PersonalityConfig:
    """How Zelyr communicates — injected into the LLM system prompt.

    All values are floats in [0.0, 1.0].
    """
    vocabulary: float = 0.5
    verbosity: float = 0.5
    formality: float = 0.5
    humor: float = 0.5
    empathy: float = 0.5

    def __post_init__(self) -> None:
        """Auto-generated docstring.


Returns:
    Return value.
"""
        for fld in ('vocabulary', 'verbosity', 'formality', 'humor', 'empathy'):
            val = getattr(self, fld)
            setattr(self, fld, max(0.0, min(1.0, float(val))))

@dataclass
class BehaviorConfig:
    """How Zelyr acts — injected into ToolExecutor rules.

    All values are floats in [0.0, 1.0].
    """
    autonomy: float = 0.5
    risk_tolerance: float = 0.5
    planning: float = 0.5
    initiative: float = 0.5

    def __post_init__(self) -> None:
        """Auto-generated docstring.


Returns:
    Return value.
"""
        for fld in ('autonomy', 'risk_tolerance', 'planning', 'initiative'):
            val = getattr(self, fld)
            setattr(self, fld, max(0.0, min(1.0, float(val))))

@dataclass
class GuardrailConfig:
    """Hard safety guardrails — always enforced regardless of behavior sliders."""
    destructive_confirmation: bool = True
    scope_enforcement: bool = True
    confidence_threshold: float = 0.4

@dataclass
class EvolutionConfig:
    """Adaptive drift tracking configuration."""
    enabled: bool = True
    last_calibrated: str = ''
    drift_log: List[Dict[str, Any]] = field(default_factory=list)

@dataclass
class ZelyrProfile:
    """Complete Zelyr user profile produced by the Genesis Sequence.

    This is the single source of truth for how Zelyr communicates and acts.
    """
    genesis_version: int = 3
    created: str = ''
    theme: str = 'minimal'
    personality: PersonalityConfig = field(default_factory=PersonalityConfig)
    behavior: BehaviorConfig = field(default_factory=BehaviorConfig)
    guardrails: GuardrailConfig = field(default_factory=GuardrailConfig)
    evolution: EvolutionConfig = field(default_factory=EvolutionConfig)
    workspace_paths: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a plain dictionary."""
        return asdict(self)

    def to_json(self, indent: int=2) -> str:
        """Serialize to a JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ZelyrProfile':
        """Deserialize from a plain dictionary."""
        personality = PersonalityConfig(**data.get('personality', {}))
        behavior = BehaviorConfig(**data.get('behavior', {}))
        guardrails_data = data.get('guardrails', {})
        guardrails = GuardrailConfig(**guardrails_data) if guardrails_data else GuardrailConfig()
        evolution_data = data.get('evolution', {})
        evolution = EvolutionConfig(**evolution_data) if evolution_data else EvolutionConfig()
        return cls(genesis_version=data.get('genesis_version', 3), created=data.get('created', ''), theme=data.get('theme', 'minimal'), personality=personality, behavior=behavior, guardrails=guardrails, evolution=evolution, workspace_paths=data.get('workspace_paths', []))

    @classmethod
    def from_json(cls, json_str: str) -> 'ZelyrProfile':
        """Deserialize from a JSON string."""
        return cls.from_dict(json.loads(json_str))

def profile_exists(path: Optional[Path]=None) -> bool:
    """Check whether a saved profile exists on disk."""
    p = path or DEFAULT_PROFILE_PATH
    return p.is_file()

def load_profile(path: Optional[Path]=None) -> ZelyrProfile:
    """Load a profile from disk, or return defaults if none exists."""
    p = path or DEFAULT_PROFILE_PATH
    if not p.is_file():
        logger.info('No profile found at %s — returning defaults', p)
        return ZelyrProfile()
    try:
        data = json.loads(p.read_text(encoding='utf-8'))
        logger.info('Loaded profile from %s', p)
        return ZelyrProfile.from_dict(data)
    except (json.JSONDecodeError, TypeError, KeyError) as exc:
        logger.warning('Corrupt profile at %s (%s) — returning defaults', p, exc)
        return ZelyrProfile()

def save_profile(profile: ZelyrProfile, path: Optional[Path]=None) -> Path:
    """Persist a profile to disk, creating directories as needed."""
    p = path or DEFAULT_PROFILE_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).isoformat()
    if not profile.created:
        profile.created = now
    if not profile.evolution.last_calibrated:
        profile.evolution.last_calibrated = now
    p.write_text(profile.to_json(), encoding='utf-8')
    logger.info('Saved profile to %s', p)
    return p

# Legacy AXIOM API compatibility aliases
AxiomProfile = ZelyrProfile
