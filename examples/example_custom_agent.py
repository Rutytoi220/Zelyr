"""Example: Creating a custom agent."""

from zelyr.agents.base_agent import BaseAgent, AgentResponse, AgentState
from zelyr import Engine
from typing import Optional, Dict, Any


class SimpleResponderAgent(BaseAgent):
    """Simple agent that responds with fixed patterns."""
    
    def __init__(self):
        super().__init__(
            agent_id="responder",
            name="Simple Responder",
            description="Agent that responds to simple queries"
        )
        
        self.responses = {
            "hello": "Hello! I'm ready to help.",
            "how are you": "I'm functioning perfectly!",
            "what is your name": "I'm the Simple Responder Agent.",
        }
    
    def process(self, input_text: str, context: Optional[Dict] = None) -> AgentResponse:
        """Process input."""
        lower_input = input_text.lower()
        
        # Find matching response
        response_text = None
        for key, response in self.responses.items():
            if key in lower_input:
                response_text = response
                break
        
        if not response_text:
            response_text = f"I don't have a preset response for '{input_text}'"
        
        return AgentResponse(
            agent_id=self.agent_id,
            success=True,
            output=response_text,
            reasoning="Matched pattern in input"
        )


def main():
    """Register and test custom agent."""
    
    engine = Engine()
    engine.initialize()
    
    # Create and register custom agent
    agent = SimpleResponderAgent()
    agent.set_engine_refs(engine.event_bus, engine.registry)
    engine.registry.register_agent(agent.agent_id, agent)
    
    print("Custom Agent Example")
    print("=" * 60)
    
    # List registered agents
    agents = engine.registry.list_agents()
    print(f"\nRegistered {len(agents)} agent(s):")
    for agent_id, ag in agents.items():
        print(f"  - {ag.name} ({agent_id})")
    
    # Test the agent
    test_inputs = [
        "Hello there!",
        "How are you?",
        "What is your name?",
        "Tell me something interesting",
    ]
    
    print("\nTesting agent:")
    for test_input in test_inputs:
        response = agent(test_input)
        print(f"\n  Input:  {test_input}")
        print(f"  Output: {response.output}")
        print(f"  State:  {agent.get_state().value}")
    
    # Get agent info
    print("\n\nAgent Info:")
    info = agent.get_info()
    print(f"  Name: {info['name']}")
    print(f"  Description: {info['description']}")
    print(f"  State: {info['state']}")
    print(f"  Execution count: {info['execution_count']}")
    
    engine.shutdown()
    print("\nExample complete!")


if __name__ == "__main__":
    main()
