"""Supervisor logic for the Swarm architecture."""
import json
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from zelyr.engine.consensus import SwarmConsensusEngine

logger = logging.getLogger(__name__)

@dataclass
class SwarmTask:
    agent_type: str
    task_description: str

class SwarmSupervisor:
    """Analyzes prompts and coordinates swarm delegation."""

    def __init__(self, llm_client=None, event_bus=None):
        self.llm = llm_client
        self.event_bus = event_bus
        self.consensus_engine = SwarmConsensusEngine(event_bus)
        
        if self.event_bus:
            self.event_bus.subscribe("scheduled.job", self._on_scheduled_job)
            
        self.system_prompt = (
            "You are the Swarm Supervisor for Zelyr. You analyze user prompts to determine if they require "
            "multiple parallel operations, complex multi-step reasoning, or diverse tool usage. "
            "Available specialist agents:\n"
            "- CoderAgent: specialized in writing code, shell execution, and file management.\n"
            "- ResearchAgent: specialized in querying memory, reading documents, and web search.\n"
            "- VisionAgent: specialized in screen capture and visual analysis.\n\n"
            "If the task requires multiple distinct operations that can be parallelized or cleanly separated, "
            "respond with a JSON decomposition:\n"
            "{\n"
            '  "is_swarm_required": true,\n'
            '  "tasks": [\n'
            '    {"agent_type": "ResearchAgent", "task_description": "Query memory for X"},\n'
            '    {"agent_type": "CoderAgent", "task_description": "Run shell command Y"}\n'
            "  ]\n"
            "}\n"
            "If it's a simple chat, single file read, or a single operation, return `{\"is_swarm_required\": false}`."
        )

    def analyze_task(self, user_prompt: str) -> List[SwarmTask]:
        """Analyze task and return a list of SwarmTasks if decomposition is needed, else empty list."""
        if not self.llm:
            return []

        try:
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            response = self.llm.generate(messages, temperature=0.1)
            content = response.get("content", "")
            
            # Extract JSON from response
            import re
            
            # Fix common LLM JSON syntax errors (like missing colons or unquoted keys)
            content_fixed = re.sub(r'("arguments"\s*)\{', r'\1: {', content)
            
            json_match = re.search(r"\{.*\}", content_fixed, re.DOTALL)
            if json_match:
                try:
                    data = json.loads(json_match.group(0))
                    if data.get("is_swarm_required") and data.get("tasks"):
                        return [
                            SwarmTask(agent_type=t["agent_type"], task_description=t["task_description"])
                            for t in data["tasks"]
                        ]
                except json.JSONDecodeError:
                    logger.debug(f"Failed to parse Swarm JSON. Raw: {json_match.group(0)}")
        except Exception as e:
            logger.error(f"Supervisor analysis failed: {e}")
            
        return []

    def synthesize_results(self, user_prompt: str, results: Dict[str, str]) -> str:
        """Combine results from multiple agents into a final coherent answer."""
        if not self.llm:
            return "\n\n".join(f"**{agent}**:\n{res}" for agent, res in results.items())
            
        context = "Here are the findings from the specialized swarm agents:\n"
        for agent, res in results.items():
            context += f"\n--- {agent} ---\n{res}\n"
            
        messages = [
            {"role": "system", "content": "You are Zelyr. Synthesize the findings from your sub-agents into a final, coherent response to the user. Do not mention that you used sub-agents, just present the findings naturally."},
            {"role": "user", "content": f"User's original request: {user_prompt}\n\n{context}"}
        ]
        
        try:
            # Generate the initial synthesis
            response = self.llm.generate(messages)
            content = response.get("content", "")
            
            # Use consensus engine to verify if there's code
            if "```python" in content:
                class DummyCoder:
                    def chat(self, msgs, **kwargs):
                        # The consensus engine passes messages, we just want it to generate using self.llm
                        return self.llm.generate(msgs).get("content", "")
                
                # Re-bind for consensus loop
                dummy = DummyCoder()
                content = self.consensus_engine.run_verification_loop(dummy, f"User's original request: {user_prompt}\n\n{context}\n\nPlease verify and ensure the code is correct.")
                
            return content
        except Exception as e:
            logger.error(f"Synthesis failed: {e}")
            return "\n\n".join(f"**{agent}**:\n{res}" for agent, res in results.items())

    def _on_scheduled_job(self, event) -> None:
        """Handles scheduled job events decoupled from the GUI."""
        prompt = event.data.get("prompt")
        if not prompt:
            return
            
        logger.info(f"Supervisor intercepted scheduled job: {prompt}")
        
        # In a full implementation, we'd spawn agents based on `analyze_task`.
        # Here we emit back a trigger so the agentic loop processes it, OR process it directly.
        # Given this is a background agent swarm, emitting an orchestrator.trigger 
        # routes it cleanly to the orchestrator just like user input!
        if self.event_bus:
            self.event_bus.publish_sync("orchestrator.trigger", data={"prompt": prompt, "source": "scheduler"})
