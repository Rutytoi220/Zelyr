"""Base Subagent class for the Multi-Agent Swarm."""
import logging
from typing import Any, Dict, Optional
from zelyr.agents.base import BaseAgent, AgentResult
from zelyr.agents.swarm.consensus import ConsensusEngine
from zelyr.core.transaction import MUTATING_TOOL_NAMES
logger = logging.getLogger(__name__)

class BaseSubagent(BaseAgent):
    """A specialized lightweight agent for swarm architectures.
    
    Subscribes to specific EventBus topics and enforces consensus
    before executing destructive tool operations. Stores intermediate
    scratchpad state in an ephemeral Blackboard rather than the global
    SQLite memory store.
    """

    def __init__(self, name: str, description: str, topic: str, event_bus=None, tool_registry=None, llm_client=None, blackboard=None, session_id: Optional[str]=None):
        """Auto-generated docstring.

Args:
    name: Argument.
    description: Argument.
    topic: Argument.
    event_bus: Argument.
    tool_registry: Argument.
    llm_client: Argument.
    blackboard: Argument.
    session_id: Argument.

Returns:
    Return value.
"""
        super().__init__(name, description, event_bus, tool_registry)
        self.topic = topic
        self.llm_client = llm_client
        self.consensus = ConsensusEngine(event_bus) if event_bus else None
        self.blackboard = blackboard
        self.session_id = session_id or 'default_session'
        self.system_prompt = ""
        self._memory = []
        if self.event_bus:
            self.event_bus.subscribe(self.topic, self._handle_task_event)

    def _emit_telemetry(self, event_type: str, payload: Dict[str, Any]) -> None:
        """Helper to emit standard swarm telemetry."""
        if self.event_bus:
            full_payload = {'agent_name': self.name, 'session_id': self.session_id}
            full_payload.update(payload)
            self.event_bus.publish(event_type, full_payload)

    def set_system_prompt(self, prompt: str) -> None:
        self.system_prompt = prompt

    def _write_to_blackboard(self, key: str, value: Any) -> None:
        """Write an intermediate artifact to the Blackboard scratchpad.

        Uses the canonical URI: blackboard://<session_id>/<agent_name>/<key>.
        No-ops silently if no Blackboard is attached.
        """
        if self.blackboard is None:
            return
        self.blackboard.write(self.session_id, self.name, key, value)
        logger.debug('[%s] Blackboard WRITE %s', self.name, self.blackboard.uri(self.session_id, self.name, key))

    def _read_from_blackboard(self, key: str, default: Any=None) -> Any:
        """Read a scratchpad artifact from this agent's Blackboard namespace."""
        if self.blackboard is None:
            return default
        return self.blackboard.read(self.session_id, self.name, key, default)

    def _read_agent_artifact(self, agent_id: str, key: str, default: Any=None) -> Any:
        """Read an artifact from another agent's Blackboard namespace (cross-agent read)."""
        if self.blackboard is None:
            return default
        return self.blackboard.read(self.session_id, agent_id, key, default)

    def _handle_task_event(self, event) -> None:
        """Handle incoming tasks from the EventBus asynchronously."""
        task = event.data.get('task')
        logger.info(f"[{self.name}] Received task on topic '{self.topic}': {task}")

    async def _execute_tool_safely(self, tool_name: str, arguments: Dict[str, Any], timeout: float=5.0) -> Any:
        """Execute a tool, requiring consensus if it is destructive.
        
        After execution, the result is automatically written to the Blackboard
        scratchpad under the key ``last_tool_result``.

        Args:
            tool_name: The name of the tool to execute.
            arguments: The arguments for the tool.
            timeout: Consensus timeout in seconds.
            
        Returns:
            The tool's result, or an error dictionary if rejected/failed.
        """
        if not self.tool_registry:
            return {'error': 'No tool registry available'}
        tool = self.tool_registry.get_tool(tool_name)
        if not tool:
            return {'error': f"Tool '{tool_name}' not found."}
        if tool_name in MUTATING_TOOL_NAMES:
            if not self.consensus:
                return {'error': 'Consensus engine required for mutating tools'}
            logger.info(f"[{self.name}] Tool '{tool_name}' is destructive. Proposing to swarm...")
            approved = self.consensus.propose(agent_name=self.name, tool_name=tool_name, arguments=arguments, timeout=timeout)
            if not approved:
                logger.warning(f"[{self.name}] Proposal for '{tool_name}' REJECTED or TIMED OUT.")
                return {'error': 'Execution blocked: Swarm consensus rejected the proposal.'}
            logger.info(f"[{self.name}] Proposal for '{tool_name}' APPROVED. Executing...")
        try:
            result = await tool(**arguments) if getattr(tool, 'is_async', False) else tool(**arguments)
            self._write_to_blackboard(f'{tool_name}_result', result)
            return result
        except Exception as e:
            logger.error(f'[{self.name}] Tool execution failed: {e}')
            error = {'error': str(e)}
            self._write_to_blackboard(f'{tool_name}_error', error)
            return error

    async def run(self, task: str, context: Optional[Dict[str, Any]]=None) -> AgentResult:
        """Run the sub-agent loop."""
        self._emit_telemetry('swarm.agent.started', {'assigned_task': task})
        try:
            # Prepare context
            messages = []
            if self.system_prompt:
                messages.append({'role': 'system', 'content': self.system_prompt})
            if context:
                messages.append({'role': 'system', 'content': f"[Shared Context]: {context}"})
            messages.append({'role': 'user', 'content': task})
            
            # Simple run loop for subagents
            from zelyr.engine.tool_pruner import ToolPruner
            all_schemas = [tool.schema for tool in self.tool_registry.list_tools().values()] if self.tool_registry else []
            
            response = await self._call_llm_async(messages, all_schemas)
            self._emit_telemetry('swarm.agent.completed', {'result_summary': response})
            return AgentResult(success=True, output=response)
        except Exception as e:
            logger.error(f"[{self.name}] Error running task: {e}")
            self._emit_telemetry('swarm.agent.completed', {'result_summary': f"Error: {e}"})
            return AgentResult(success=False, error=str(e))
            
    async def _call_llm_async(self, messages: list, tool_schemas: list) -> str:
        """Execute a simple LLM call that simulates streaming tokens via telemetry."""
        # Note: Since the real self.llm_client.generate is likely sync, we wrap or mock it
        import asyncio
        loop = asyncio.get_event_loop()
        try:
            # We will use the universal LLM client
            def do_generate():
                return self.llm_client.generate(messages, tools=tool_schemas)
            
            response_msg = await loop.run_in_executor(None, do_generate)
            content = response_msg.get('content', '')
            
            # Emit fake token stream for UI flair since real streaming might not be enabled
            if content:
                words = content.split(" ")
                for i in range(0, len(words), 3):
                    chunk = " ".join(words[i:i+3]) + " "
                    self._emit_telemetry('swarm.agent.token', {'chunk': chunk})
                    await asyncio.sleep(0.01)
                    
            # Handle tool calls if any
            tool_calls = response_msg.get('tool_calls', [])
            if tool_calls:
                for tc in tool_calls:
                    name = tc['function']['name']
                    args = tc['function']['arguments']
                    self._emit_telemetry('swarm.agent.token', {'chunk': f"\n[Executing {name}...]\n"})
                    res = await self._execute_tool_safely(name, args)
                    messages.append(response_msg)
                    messages.append({'role': 'tool', 'name': name, 'content': str(res)})
                
                # Recurse
                return await self._call_llm_async(messages, tool_schemas)

            return content
        except Exception as e:
            return f"Agent Error: {e}"
