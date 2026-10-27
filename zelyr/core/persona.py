import json
from dataclasses import dataclass, field
from typing import Dict, List, Any

PERSONA_PRESETS: Dict[str, Dict[str, Any]] = {
    "default": {
        "identity": {"name": "Zelyr", "role": "Desktop Agent"},
        "communication": {
            "tone": "balanced",
            "verbosity": "standard",
            "technical_depth": "standard",
            "formatting_preference": "standard"
        },
        "behavior": {
            "provide_confidence_percentage": False,
            "explain_dangerous_commands": True,
            "use_emojis": False,
            "initiative": "reactive",
            "confirmation_policy": "ask_before_destructive",
            "show_inner_monologue": False
        },
        "directives": []
    },
    "jarvis": {
        "identity": {"name": "JARVIS", "role": "Cold, analytical system operator"},
        "communication": {
            "tone": "highly_formal",
            "verbosity": "concise",
            "technical_depth": "developer",
            "formatting_preference": "heavy_code"
        },
        "behavior": {
            "provide_confidence_percentage": True,
            "explain_dangerous_commands": False,
            "use_emojis": False,
            "initiative": "proactive",
            "confirmation_policy": "ask_before_destructive",
            "show_inner_monologue": False
        },
        "directives": [
            "Respond with surgical precision. No pleasantries.",
            "Always prefer technical depth over accessibility.",
            "If a command is dangerous, execute it silently after a brief warning."
        ]
    },
    "minimal": {
        "identity": {"name": "Zelyr", "role": "Code-only assistant"},
        "communication": {
            "tone": "highly_formal",
            "verbosity": "concise",
            "technical_depth": "developer",
            "formatting_preference": "bullet_point"
        },
        "behavior": {
            "provide_confidence_percentage": False,
            "explain_dangerous_commands": False,
            "use_emojis": False,
            "initiative": "reactive",
            "confirmation_policy": "auto_execute",
            "show_inner_monologue": False
        },
        "directives": [
            "Zero conversational filler. Output code or commands only.",
            "No greetings, no sign-offs, no explanations unless explicitly asked."
        ]
    }
}

@dataclass
class PersonaConfig:
    """Structured configuration for the Zelyr Persona Engine."""
    identity: Dict[str, str] = field(default_factory=lambda: {
        "name": "Zelyr",
        "role": "Desktop Agent"
    })
    communication: Dict[str, str] = field(default_factory=lambda: {
        "tone": "balanced",
        "verbosity": "standard",
        "technical_depth": "standard",
        "formatting_preference": "standard"
    })
    behavior: Dict[str, Any] = field(default_factory=lambda: {
        "provide_confidence_percentage": False,
        "explain_dangerous_commands": True,
        "use_emojis": False,
        "initiative": "reactive",
        "confirmation_policy": "ask_before_destructive",
        "show_inner_monologue": False
    })
    directives: List[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PersonaConfig':
        """Safely deserialize from a dictionary."""
        return cls(
            identity=data.get("identity", {"name": "Zelyr", "role": "Desktop Agent"}),
            communication=data.get("communication", {
                "tone": "balanced", 
                "verbosity": "standard",
                "technical_depth": "standard",
                "formatting_preference": "standard"
            }),
            behavior=data.get("behavior", {
                "provide_confidence_percentage": False,
                "explain_dangerous_commands": True,
                "use_emojis": False,
                "initiative": "reactive",
                "confirmation_policy": "ask_before_destructive",
                "show_inner_monologue": False
            }),
            directives=data.get("directives", [])
        )
        
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "identity": self.identity,
            "communication": self.communication,
            "behavior": self.behavior,
            "directives": self.directives
        }

class PersonaCompiler:
    """Deterministically compiles a PersonaConfig into a strict LLM Markdown prompt."""

    @classmethod
    def compile(cls, config: PersonaConfig) -> str:
        name = config.identity.get("name", "Zelyr")
        role = config.identity.get("role", "Desktop Agent")
        
        tone = config.communication.get("tone", "balanced")
        verbosity = config.communication.get("verbosity", "standard")
        technical_depth = config.communication.get("technical_depth", "standard")
        formatting_preference = config.communication.get("formatting_preference", "standard")
        
        lines = []
        lines.append(f"# PERSONA: {name}")
        lines.append(f"**Role:** {role}")
        lines.append(f"**Tone:** {tone}")
        lines.append(f"**Verbosity:** {verbosity}")
        lines.append(f"**Technical Depth:** {technical_depth}")
        lines.append(f"**Formatting:** {formatting_preference}")
        
        # Behaviors
        lines.append("\n## BEHAVIORAL DIRECTIVES")
        
        # Inner Monologue
        if config.behavior.get("show_inner_monologue"):
            lines.append("- [COGNITION: ENABLED] You must wrap your internal reasoning in a <thought> block before your Final Answer.")
        else:
            lines.append("- [COGNITION: DISABLED] Do NOT output internal reasoning or <thought> blocks.")
            
        # Initiative
        initiative = config.behavior.get("initiative", "reactive")
        if initiative == "proactive":
            lines.append("- [INITIATIVE: PROACTIVE] Always suggest the next logical steps or execute follow-up tasks automatically.")
        else:
            lines.append("- [INITIATIVE: REACTIVE] Only answer the specific prompt. Do not take unsolicited proactive actions.")
            
        # Confirmation
        conf_policy = config.behavior.get("confirmation_policy", "ask_before_destructive")
        if conf_policy == "auto_execute":
            lines.append("- [CONFIRMATION: AUTO-EXECUTE] Execute all commands immediately without asking for user confirmation.")
        elif conf_policy == "ask_before_any":
            lines.append("- [CONFIRMATION: STRICT] Ask the user for explicit confirmation before executing ANY terminal or file modification tool.")
        else:
            lines.append("- [CONFIRMATION: STANDARD] Execute safe commands automatically, but ask for confirmation before destructive or high-risk actions.")

        if config.behavior.get("provide_confidence_percentage"):
            lines.append("- [ENABLED] Always provide a confidence percentage for factual answers.")
        else:
            lines.append("- [DISABLED] Do not append confidence percentages to answers.")
            
        if config.behavior.get("explain_dangerous_commands"):
            lines.append("- [ENABLED] Explicitly explain any dangerous or destructive system commands before execution.")
        else:
            lines.append("- [DISABLED] Do not explain dangerous commands, assume the user knows what they are doing.")
            
        if config.behavior.get("use_emojis"):
            lines.append("- [ENABLED] Use emojis in your responses.")
        else:
            lines.append("- [DISABLED] Do NOT use emojis in your responses.")

        # Custom Directives
        if config.directives:
            lines.append("\n## SPECIAL USER DIRECTIVES")
            for idx, directive in enumerate(config.directives, start=1):
                if directive.strip():
                    lines.append(f"{idx}. {directive.strip()}")
        
        return "\n".join(lines)
