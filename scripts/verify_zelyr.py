#!/usr/bin/env python3
"""Zelyr System Verification Script.

Verifies that all Zelyr components are working correctly.
"""

import sys
import time
from pathlib import Path

def test_imports():
    """Test that all modules can be imported."""
    print("\n" + "="*60)
    print("TEST 1: Module Imports")
    print("="*60)
    
    try:
        import zelyr
        print(f"✓ zelyr v{zelyr.__version__}")
        
        from zelyr import Engine, EventBus, Registry, ExecutionContext
        print("✓ Core modules")
        
        from zelyr import OllamaClient
        print("✓ LLM module")
        
        from zelyr import MemoryManager
        print("✓ Memory module")
        
        from zelyr import BaseAgent, OrchestratorAgent
        print("✓ Agent module")
        
        from zelyr import BaseTool, ShellCommandTool, ReadFileTool
        print("✓ Tool module")
        
        from zelyr import BasePlugin, NXBTPlugin, AutomationPlugin
        print("✓ Plugin module")
        
        from zelyr import CLI
        print("✓ API module")
        
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False


def test_engine_lifecycle():
    """Test engine initialization and shutdown."""
    print("\n" + "="*60)
    print("TEST 2: Engine Lifecycle")
    print("="*60)
    
    try:
        from zelyr import Engine
        
        engine = Engine()
        print(f"✓ Engine created")
        print(f"  Running: {engine.is_running()}")
        
        engine.initialize()
        print(f"✓ Engine initialized")
        print(f"  Running: {engine.is_running()}")
        
        context = engine.create_context("test input")
        print(f"✓ Context created: {context.context_id[:8]}...")
        
        engine.shutdown()
        print(f"✓ Engine shutdown")
        print(f"  Running: {engine.is_running()}")
        
        return True
    except Exception as e:
        print(f"✗ Engine test failed: {e}")
        return False


def test_tools():
    """Test tool system."""
    print("\n" + "="*60)
    print("TEST 3: Tool System")
    print("="*60)
    
    try:
        from zelyr import ShellCommandTool, ReadFileTool, PythonExecTool
        
        # Test ShellCommandTool
        shell = ShellCommandTool()
        result = shell(command="echo 'test'")
        assert result.success
        print(f"✓ ShellCommandTool works")
        
        # Test PythonExecTool
        python = PythonExecTool()
        result = python(code="print(2 + 2)")
        assert result.success
        print(f"✓ PythonExecTool works")
        
        # Test tool info
        info = shell.get_info()
        assert info["tool_id"] == "shell_command"
        assert len(info["parameters"]) > 0
        print(f"✓ Tool introspection works")
        
        return True
    except Exception as e:
        print(f"✗ Tool test failed: {e}")
        return False


def test_agents():
    """Test agent system."""
    print("\n" + "="*60)
    print("TEST 4: Agent System")
    print("="*60)
    
    try:
        from zelyr import Engine, OrchestratorAgent
        
        engine = Engine()
        engine.initialize()
        
        agent = OrchestratorAgent()
        agent.set_engine_refs(engine.event_bus, engine.registry)
        
        # Test agent processing
        response = agent("test input")
        assert response is not None
        assert response.success
        print(f"✓ Agent processing works")
        
        # Test agent info
        info = agent.get_info()
        assert info["agent_id"] == "orchestrator"
        assert info["execution_count"] > 0
        print(f"✓ Agent introspection works")
        
        engine.shutdown()
        return True
    except Exception as e:
        print(f"✗ Agent test failed: {e}")
        return False


def test_memory():
    """Test memory system."""
    print("\n" + "="*60)
    print("TEST 5: Memory System")
    print("="*60)
    
    try:
        from zelyr import MemoryManager
        import tempfile
        
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = f"{tmpdir}/test.db"
            memory = MemoryManager(db_path)
            
            # Test conversation
            conv_id = memory.create_conversation("Test")
            print(f"✓ Conversation created")
            
            # Test messages
            memory.add_message("user", "Hello")
            memory.add_message("assistant", "Hi there!")
            print(f"✓ Messages added")
            
            # Test history
            history = memory.get_conversation_history()
            assert len(history) == 2
            print(f"✓ History retrieval works ({len(history)} messages)")
            
            # Test tool execution
            memory.save_tool_execution("test_tool", {"input": "x"}, {"output": "y"})
            print(f"✓ Tool execution stored")
        
        return True
    except Exception as e:
        print(f"✗ Memory test failed: {e}")
        return False


def test_events():
    """Test event system."""
    print("\n" + "="*60)
    print("TEST 6: Event System")
    print("="*60)
    
    try:
        from zelyr import EventBus, Event
        
        bus = EventBus()
        received = []
        
        def handler(event):
            received.append(event)
        
        bus.subscribe("test.event", handler)
        print(f"✓ Subscription works")
        
        event = Event(event_type="test.event", source="test", data={"value": 42})
        bus.publish(event)
        
        assert len(received) == 1
        assert received[0].data["value"] == 42
        print(f"✓ Event publishing works")
        
        history = bus.get_history("test.event")
        assert len(history) == 1
        print(f"✓ Event history works")
        
        return True
    except Exception as e:
        print(f"✗ Event test failed: {e}")
        return False


def test_registry():
    """Test registry system."""
    print("\n" + "="*60)
    print("TEST 7: Registry System")
    print("="*60)
    
    try:
        from zelyr import Engine, ShellCommandTool
        
        engine = Engine()
        engine.initialize()
        
        tool = ShellCommandTool()
        engine.registry.register_tool(tool.tool_id, tool)
        
        retrieved = engine.registry.get_tool(tool.tool_id)
        assert retrieved is not None
        assert retrieved.tool_id == tool.tool_id
        print(f"✓ Tool registration works")
        
        tools = engine.registry.list_tools()
        assert len(tools) > 0
        print(f"✓ Tool listing works ({len(tools)} tools)")
        
        engine.shutdown()
        return True
    except Exception as e:
        print(f"✗ Registry test failed: {e}")
        return False


def test_plugins():
    """Test plugin system."""
    print("\n" + "="*60)
    print("TEST 8: Plugin System")
    print("="*60)
    
    try:
        from zelyr import NXBTPlugin, AutomationPlugin
        
        # Test NXBT plugin
        nxbt = NXBTPlugin()
        assert nxbt.initialize()
        print(f"✓ NXBT plugin initializes")
        
        nxbt.enable()
        assert nxbt.is_enabled()
        print(f"✓ NXBT plugin enable/disable works")
        
        info = nxbt.get_info()
        assert info["plugin_id"] == "nxbt"
        assert info["enabled"] == True
        print(f"✓ NXBT plugin info works")
        
        # Test Automation plugin
        auto = AutomationPlugin()
        assert auto.initialize()
        print(f"✓ Automation plugin initializes")
        
        return True
    except Exception as e:
        print(f"✗ Plugin test failed: {e}")
        return False


def test_context():
    """Test execution context."""
    print("\n" + "="*60)
    print("TEST 9: Execution Context")
    print("="*60)
    
    try:
        from zelyr import ExecutionContext
        
        ctx = ExecutionContext(user_input="test")
        print(f"✓ Context created")
        
        ctx.set_variable("key1", "value1")
        assert ctx.get_variable("key1") == "value1"
        print(f"✓ Variable storage works")
        
        ctx.add_tool_result("tool1", {"result": "ok"})
        assert "tool1" in ctx.tool_results
        print(f"✓ Tool result storage works")
        
        ctx.add_agent_output("agent1", "output")
        assert "agent1" in ctx.agent_outputs
        print(f"✓ Agent output storage works")
        
        ctx_dict = ctx.to_dict()
        assert ctx_dict["user_input"] == "test"
        print(f"✓ Context serialization works")
        
        return True
    except Exception as e:
        print(f"✗ Context test failed: {e}")
        return False


def test_configuration():
    """Test configuration system."""
    print("\n" + "="*60)
    print("TEST 10: Configuration System")
    print("="*60)
    
    try:
        from zelyr import ZelyrConfig, get_config
        
        config = ZelyrConfig(debug=True, log_level="DEBUG")
        print(f"✓ Config created")
        
        config_dict = config.to_dict()
        assert config_dict["debug"] == True
        print(f"✓ Config serialization works")
        
        default_config = get_config()
        assert default_config is not None
        print(f"✓ Default config works")
        
        return True
    except Exception as e:
        print(f"✗ Configuration test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("\n" + "█"*60)
    print("█  Zelyr SYSTEM VERIFICATION")
    print("█"*60)
    
    tests = [
        ("Module Imports", test_imports),
        ("Engine Lifecycle", test_engine_lifecycle),
        ("Tool System", test_tools),
        ("Agent System", test_agents),
        ("Memory System", test_memory),
        ("Event System", test_events),
        ("Registry System", test_registry),
        ("Plugin System", test_plugins),
        ("Execution Context", test_context),
        ("Configuration", test_configuration),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n✗ {name} crashed: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")
    
    print("\n" + "="*60)
    print(f"Results: {passed}/{total} tests passed")
    print("="*60 + "\n")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
