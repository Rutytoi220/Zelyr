"""Unit and integration test suite for ToolCapability matrix and dynamic capability loading."""

import shutil
import tempfile
import unittest
from pathlib import Path

from zelyr.core.plugins import (
    CORE_TOOLS,
    TIER_1_TOOLS,
    TIER_2_TOOLS,
    TIER_3_TOOLS,
    filter_tool_schemas_by_tier,
    get_tool_capabilities,
    get_tool_capability,
    get_tool_schemas,
    load_plugins,
    reload_plugin,
)
from zelyr.tools.browser_cdp import InteractWithBrowserTool
from zelyr.tools.core import ToolCapability
from zelyr.tools.hyprland_ipc import ManageDesktopWindowTool
from zelyr.tools.vision import InteractWithUITool


class TestCapabilityMatrix(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        load_plugins()

    def test_base_tool_capabilities_instantiation(self):
        """Verify built-in BaseTool subclasses declare capability metadata."""
        hypr = ManageDesktopWindowTool()
        self.assertIsInstance(hypr.capability, ToolCapability)
        self.assertEqual(hypr.capability.tier, 1)
        self.assertTrue(hypr.capability.is_core)
        self.assertTrue(hypr.capability.requires_window)
        self.assertFalse(hypr.capability.requires_bridge)
        self.assertEqual(hypr.capability.token_cost, 150)

        browser = InteractWithBrowserTool()
        self.assertIsInstance(browser.capability, ToolCapability)
        self.assertEqual(browser.capability.tier, 2)
        self.assertFalse(browser.capability.is_core)
        self.assertTrue(browser.capability.requires_bridge)
        self.assertTrue(browser.capability.requires_window)
        self.assertEqual(browser.capability.token_cost, 250)

        ui = InteractWithUITool()
        self.assertIsInstance(ui.capability, ToolCapability)
        self.assertEqual(ui.capability.tier, 3)
        self.assertFalse(ui.capability.is_core)
        self.assertFalse(ui.capability.requires_bridge)
        self.assertTrue(ui.capability.requires_window)
        self.assertEqual(ui.capability.token_cost, 600)

    def test_dynamic_capability_extraction(self):
        """Verify dynamic plugins expose capability metadata via get_tool_capability."""
        exec_cap = get_tool_capability("execute_command")
        self.assertTrue(exec_cap.is_core)
        self.assertEqual(exec_cap.tier, 0)

        all_caps = get_tool_capabilities()
        self.assertIn("execute_command", all_caps)
        self.assertIn("manage_desktop_window", all_caps)
        self.assertIn("interact_with_browser", all_caps)

        hypr_cap = get_tool_capability("manage_desktop_window")
        self.assertEqual(hypr_cap.tier, 1)
        self.assertTrue(hypr_cap.is_core)
        self.assertTrue(hypr_cap.requires_window)

    def test_dynamic_set_backwards_compatibility(self):
        """Verify CORE_TOOLS, TIER_1_TOOLS, TIER_2_TOOLS maintain backwards compatibility."""
        self.assertIn("execute_command", CORE_TOOLS)
        self.assertIn("manage_desktop_window", CORE_TOOLS)
        self.assertIn("interact_with_browser", TIER_2_TOOLS)
        self.assertIn("interact_with_ui", TIER_3_TOOLS)

        union_set = CORE_TOOLS | TIER_1_TOOLS
        self.assertTrue("execute_command" in union_set)
        self.assertTrue("manage_desktop_window" in union_set)
        self.assertTrue("manage_system_process" in union_set)

        inter_set = CORE_TOOLS & TIER_1_TOOLS
        self.assertEqual(len(inter_set), 0)

        diff_set = TIER_2_TOOLS - CORE_TOOLS
        self.assertTrue("interact_with_browser" in diff_set)
        self.assertFalse("execute_command" in diff_set)

    def test_filter_tool_schemas_by_tier(self):
        """Verify dynamic schema pruning keeps core tools + target tier tools."""
        schemas = get_tool_schemas()

        tier1_schemas = filter_tool_schemas_by_tier(schemas, tier="1")
        tier1_names = {s.get("function", {}).get("name") or s.get("name") for s in tier1_schemas}
        self.assertIn("execute_command", tier1_names)
        self.assertIn("manage_desktop_window", tier1_names)
        self.assertNotIn("interact_with_browser", tier1_names)
        self.assertNotIn("interact_with_ui", tier1_names)

        tier2_schemas = filter_tool_schemas_by_tier(schemas, tier="2")
        tier2_names = {s.get("function", {}).get("name") or s.get("name") for s in tier2_schemas}
        self.assertIn("execute_command", tier2_names)
        self.assertIn("interact_with_browser", tier2_names)
        self.assertNotIn("interact_with_ui", tier2_names)

        tier3_schemas = filter_tool_schemas_by_tier(schemas, tier="3")
        tier3_names = {s.get("function", {}).get("name") or s.get("name") for s in tier3_schemas}
        self.assertIn("execute_command", tier3_names)
        self.assertIn("interact_with_ui", tier3_names)
        self.assertEqual(len(tier3_schemas), len(schemas))

        filtered_browser = filter_tool_schemas_by_tier(schemas, tier="3", is_browser=True)
        filtered_browser_names = {s.get("function", {}).get("name") or s.get("name") for s in filtered_browser}
        self.assertNotIn("interact_with_ui", filtered_browser_names)

    def test_dynamic_custom_plugin_reload_and_capability(self):
        """Test reload_plugin with a custom plugin declaring TIER and IS_CORE attributes."""
        temp_dir = tempfile.mkdtemp()
        try:
            plugin_path = Path(temp_dir) / "custom_test_tool.py"
            plugin_code = '''
TIER = 2
IS_CORE = False
REQUIRES_BRIDGE = True
REQUIRES_WINDOW = False
TOKEN_COST = 180

TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "custom_test_tool",
        "description": "Test dynamic capability tool",
        "parameters": {
            "type": "object",
            "properties": {"arg": {"type": "string"}},
            "required": ["arg"]
        }
    }
}

REQUIRED_RING = 0
TUI_HINT = "Testing custom capability"

def execute(arg: str):
    return f"Executed with {arg}"
'''
            plugin_path.write_text(plugin_code)
            reload_plugin(plugin_path)

            cap = get_tool_capability("custom_test_tool")
            self.assertEqual(cap.tier, 2)
            self.assertFalse(cap.is_core)
            self.assertTrue(cap.requires_bridge)
            self.assertFalse(cap.requires_window)
            self.assertEqual(cap.token_cost, 180)

            self.assertIn("custom_test_tool", TIER_2_TOOLS)
            self.assertNotIn("custom_test_tool", TIER_1_TOOLS)

            schemas = get_tool_schemas()
            t2_schemas = filter_tool_schemas_by_tier(schemas, tier="2")
            t2_names = {s.get("function", {}).get("name") or s.get("name") for s in t2_schemas}
            self.assertIn("custom_test_tool", t2_names)

            t1_schemas = filter_tool_schemas_by_tier(schemas, tier="1")
            t1_names = {s.get("function", {}).get("name") or s.get("name") for s in t1_schemas}
            self.assertNotIn("custom_test_tool", t1_names)

        finally:
            shutil.rmtree(temp_dir)


if __name__ == "__main__":
    unittest.main()
