"""Example: Using Zelyr as a library."""

from zelyr import Engine, OrchestratorAgent, MemoryManager, ShellCommandTool


def main():
    """Demonstrate Zelyr library usage."""
    
    # Initialize components
    engine = Engine()
    engine.initialize()
    
    memory = MemoryManager()
    memory.create_conversation("Example Session")
    
    agent = OrchestratorAgent()
    agent.set_engine_refs(engine.event_bus, engine.registry)
    
    # Register a tool
    shell_tool = ShellCommandTool()
    engine.registry.register_tool(shell_tool.tool_id, shell_tool)
    
    # Register the agent
    engine.registry.register_agent(agent.agent_id, agent)
    
    print("Zelyr Library Example")
    print("=" * 60)
    
    # Example interactions
    questions = [
        "What can you do?",
        "How many tools are available?",
        "What agents are registered?",
    ]
    
    for question in questions:
        print(f"\nQuestion: {question}")
        
        # Add to memory
        memory.add_message("user", question)
        
        # Process through agent
        response = agent(question)
        
        # Store response
        memory.add_message("assistant", response.output or "")
        
        # Display result
        print(f"Response:\n{response.output}")
        print("-" * 60)
    
    # Show conversation history
    print("\nConversation History:")
    history = memory.get_conversation_history()
    for msg in history:
        role = msg['role'].upper()
        content = msg['content'][:80] + "..." if len(msg['content']) > 80 else msg['content']
        print(f"  {role}: {content}")
    
    # Shutdown
    engine.shutdown()
    print("\nExample complete!")


if __name__ == "__main__":
    main()
