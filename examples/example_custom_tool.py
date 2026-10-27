"""Example: Loading a plugin using the Zelyr v2 Plugin Ecosystem (RFC-001).

This script supersedes the v1 ``example_custom_tool.py`` (which registered a
raw ``BaseTool`` directly).  It shows how to load a manifest-driven,
sandboxed plugin via ``PluginManager`` and call its tools.

The v1 API still works for backward compatibility — this just demonstrates
the recommended v2 pattern.
"""

from pathlib import Path

from zelyr.core.engine import Engine
from zelyr.plugins.manager import PluginManager

PLUGIN_DIR = Path(__file__).parent / "example_plugin"


def main() -> None:
    """Load the sandboxed calculator plugin and execute it."""
    engine = Engine()
    engine.initialize()

    # Create the PluginManager and point it at our example plugin directory.
    # In production this would scan ~/.zelyr/plugins/ automatically.
    manager = PluginManager(
        registry=engine.registry,
        event_bus=engine.event_bus,
        plugin_root=PLUGIN_DIR.parent,  # scans examples/
    )

    # Load only the example plugin for this demo
    success = manager.load_from_path(PLUGIN_DIR)

    print("Zelyr v2 Plugin Example")
    print("=" * 60)

    if not success:
        print("ERROR: Plugin failed to load. See logs above.")
        engine.shutdown()
        return

    print(f"\nLoaded plugins : {manager.loaded_plugins}")

    tools = engine.registry.list_tools()
    print(f"Registered tools ({len(tools)}):")
    for tool_id, tool in tools.items():
        print(f"  - {tool.name} [{tool_id}]")

    # Call the sandboxed tool — runs in an isolated child process.
    print("\nTesting sandboxed 'calculate' tool:")
    tool = engine.registry.get_tool("zelyr-calculator::calculate")
    if tool:
        result = tool(expression="2 + sqrt(16)")
        print(f"  2 + sqrt(16) = {result.output}")

        result2 = tool(expression="pi * 5 ** 2")
        print(f"  pi * 5²      = {result2.output}")
    else:
        print("  Tool not found in registry.")

    engine.shutdown()
    print("\nExample complete!")


if __name__ == "__main__":
    main()
