"""Swarm dispatcher to concurrently execute agent tasks."""
import asyncio
import logging
from typing import List, Dict, Optional
from zelyr.agents.swarm.supervisor import SwarmTask
from zelyr.agents.swarm.profiles import CoderAgent, ResearchAgent, VisionAgent
from zelyr.agents.swarm.test_runner_agent import TestRunnerAgent

logger = logging.getLogger(__name__)

class SwarmDispatcher:
    """Coordinates and executes concurrent sub-agent tasks."""

    def __init__(self, event_bus, tool_registry, llm_client, session_id: str, blackboard=None):
        self.event_bus = event_bus
        self.tool_registry = tool_registry
        self.llm_client = llm_client
        self.session_id = session_id
        self.blackboard = blackboard
        
        self.agents = {
            "CoderAgent": CoderAgent(self.event_bus, self.tool_registry, self.llm_client, self.session_id),
            "ResearchAgent": ResearchAgent(self.event_bus, self.tool_registry, self.llm_client, self.session_id),
            "VisionAgent": VisionAgent(self.event_bus, self.tool_registry, self.llm_client, self.session_id),
            "TestRunnerAgent": TestRunnerAgent(self.event_bus, self.tool_registry, self.llm_client, blackboard=self.blackboard, session_id=self.session_id),
        }

    async def dispatch(self, tasks: List[SwarmTask], timeout: Optional[float] = None) -> Dict[str, str]:
        """Dispatches tasks concurrently to specialized agents."""
        coroutines = []
        agent_names = []
        
        for task in tasks:
            agent = self.agents.get(task.agent_type)
            if not agent:
                logger.warning(f"Requested unknown agent type: {task.agent_type}")
                agent = self.agents["CoderAgent"]
                
            agent_names.append(agent.name)
            coro = agent.run(task.task_description)
            if timeout is not None:
                coro = asyncio.wait_for(coro, timeout=timeout)
            coroutines.append(coro)
            
        logger.info(f"Dispatching tasks to {len(coroutines)} agents concurrently...")
        
        results = await asyncio.gather(*coroutines, return_exceptions=True)
        
        output_map = {}
        for name, res in zip(agent_names, results):
            if isinstance(res, asyncio.TimeoutError):
                output_map[name] = f"TimeoutError: Agent {name} timed out after {timeout}s"
            elif isinstance(res, Exception):
                output_map[name] = f"Error: {res}"
            else:
                output_map[name] = getattr(res, "output", str(res))
                
        return output_map
