"""Specialist sub-agents for the swarm architecture."""
from typing import Optional
from zelyr.agents.swarm.base_swarm import BaseSubagent

class CoderAgent(BaseSubagent):
    """Specialist for coding, shell execution, and file management."""
    def __init__(self, event_bus, tool_registry, llm_client, session_id: Optional[str] = None, blackboard=None):
        super().__init__(
            name="CoderAgent",
            description="Executes code, writes files, runs shell commands.",
            topic="swarm.coder",
            event_bus=event_bus,
            tool_registry=tool_registry,
            llm_client=llm_client,
            blackboard=blackboard,
            session_id=session_id
        )
        self.set_system_prompt(
            "You are CoderAgent. You write clean, bug-free code, execute terminal commands, and manage files. "
            "Use your tools to fulfill your assigned task. Never guess filenames; use shell(ls) or file_search."
        )

class ResearchAgent(BaseSubagent):
    """Specialist for querying memory, searching, and synthesizing facts."""
    def __init__(self, event_bus, tool_registry, llm_client, session_id: Optional[str] = None):
        super().__init__(
            name="ResearchAgent",
            description="Queries semantic memory, reads documents, and summarizes facts.",
            topic="swarm.research",
            event_bus=event_bus,
            tool_registry=tool_registry,
            llm_client=llm_client,
            session_id=session_id
        )
        self.set_system_prompt(
            "You are ResearchAgent. You synthesize facts, search vector memory, and summarize documentation. "
            "Do not modify files or execute code."
        )

class VisionAgent(BaseSubagent):
    """Specialist for screen capture and visual analysis."""
    def __init__(self, event_bus, tool_registry, llm_client, session_id: Optional[str] = None):
        super().__init__(
            name="VisionAgent",
            description="Captures screen and analyzes UI.",
            topic="swarm.vision",
            event_bus=event_bus,
            tool_registry=tool_registry,
            llm_client=llm_client,
            session_id=session_id
        )
        self.set_system_prompt(
            "You are VisionAgent. You use screen capture and visual analysis to answer questions about the current screen state."
        )
from zelyr.engine.cyber_auditor import SecurityAuditorAgent
