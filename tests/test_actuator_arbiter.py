"""Unit and integration test suite for ActuatorArbiter and negative Tier-3 gating.

Covers:
1. Synthetic task routing with mocked browser window active + bridge connected -> Tier 2.
2. Synthetic task routing with terminal/IDE active -> Tier 1.
3. Synthetic task requesting a button click with browser active + bridge connected -> Tier 2
   (verifying Tier 3 interact_with_ui is omitted from tool schemas).
4. Fallback behavior when Hyprland IPC returns no active window.
5. Negative Tier-3 gating: visual grounding requested with browser + bridge active routes to Tier 2
   and purges interact_with_ui.
6. Integration with NativeOrchestrator and schema filtering.
"""

import contextlib
import unittest
from unittest.mock import AsyncMock, patch

from zelyr.agents.actuator_arbiter import (
    ActuatorArbiter,
    is_browser_window,
    is_terminal_or_ide_window,
)
from zelyr.agents.native_orchestrator import NativeOrchestrator
from zelyr.core.plugins import filter_tool_schemas_by_tier, get_tool_schemas, load_plugins


class TestActuatorArbiter(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        load_plugins()
        self.all_schemas = get_tool_schemas(0)

    def test_window_classification(self):
        """Verify window class/title classification helpers."""
        zen_window = {"class": "app.zen_browser.zen", "title": "GitHub — Zen Browser"}
        chrome_window = {"class": "google-chrome", "title": "Google Chrome"}
        firefox_window = {"class": "firefox", "title": "Mozilla Firefox"}
        kitty_window = {"class": "kitty", "title": "zsh"}
        vscode_window = {"class": "code", "title": "main.py - Visual Studio Code"}

        self.assertTrue(is_browser_window(zen_window))
        self.assertTrue(is_browser_window(chrome_window))
        self.assertTrue(is_browser_window(firefox_window))
        self.assertFalse(is_browser_window(kitty_window))
        self.assertFalse(is_browser_window(vscode_window))

        self.assertTrue(is_terminal_or_ide_window(kitty_window))
        self.assertTrue(is_terminal_or_ide_window(vscode_window))
        self.assertFalse(is_terminal_or_ide_window(zen_window))

    def test_browser_active_and_bridge_connected_routes_to_tier2(self):
        """When browser is active and bridge is connected, web interaction tasks route to Tier 2."""
        browser_window = {"class": "app.zen_browser.zen", "title": "Monkeytype"}
        tier = ActuatorArbiter.arbitrate(
            task="Inspect the current page",
            active_window=browser_window,
            bridge_connected=True,
        )
        self.assertEqual(tier, "tier2_browser")

    def test_button_click_with_browser_active_routes_to_tier2_and_omits_tier3(self):
        """A generic button click while in browser + connected bridge routes to Tier 2, omitting Tier 3."""
        browser_window = {"class": "google-chrome", "title": "Checkout"}
        task = "Click the submit button"
        tier = ActuatorArbiter.arbitrate(
            task=task,
            active_window=browser_window,
            bridge_connected=True,
        )
        self.assertEqual(tier, "tier2_browser")

        # Verify tool schemas for this tier omit interact_with_ui
        filtered = filter_tool_schemas_by_tier(self.all_schemas, tier, is_browser=True)
        tool_names = {s.get("function", {}).get("name") or s.get("name") for s in filtered}
        self.assertIn("interact_with_browser", tool_names)
        self.assertNotIn("interact_with_ui", tool_names)

    def test_terminal_or_ide_active_routes_to_tier1(self):
        """When a terminal or IDE is active, tasks route to Tier 1."""
        terminal_window = {"class": "kitty", "title": "terminal"}
        tier = ActuatorArbiter.arbitrate(
            task="Run git status and compile the project",
            active_window=terminal_window,
            bridge_connected=False,
        )
        self.assertEqual(tier, "tier1_ipc")

        ide_window = {"class": "code", "title": "main.py"}
        tier_ide = ActuatorArbiter.arbitrate(
            task="Review modified lines and format code",
            active_window=ide_window,
            bridge_connected=False,
        )
        self.assertEqual(tier_ide, "tier1_ipc")

    def test_desktop_os_action_prioritized_over_browser(self):
        """Desktop/OS operations (shell, media, file, notification) route to Tier 1 even if browser is focused."""
        browser_window = {"class": "zen", "title": "YouTube"}
        tasks_tier1 = [
            "Pause my music playback please",
            "Send a desktop notification saying compilation finished",
            "Switch to workspace 3",
            "Check network connection and Tailscale status",
            "Rollback the change made to main.py",
        ]
        for t in tasks_tier1:
            tier = ActuatorArbiter.arbitrate(
                task=t,
                active_window=browser_window,
                bridge_connected=True,
            )
            self.assertEqual(tier, "tier1_ipc", f"Task '{t}' should route to tier1_ipc")

    def test_visual_grounding_blocked_in_browser_with_bridge(self):
        """Visual screen grounding requested while browser + bridge is active routes to Tier 2 (not Tier 3)."""
        browser_window = {"class": "zen", "title": "Web App"}
        task = "Look at screen and click the button"
        tier = ActuatorArbiter.arbitrate(
            task=task,
            active_window=browser_window,
            bridge_connected=True,
        )
        # Because active window is a browser with functioning DOM bridge, Tier 3 is negative-gated
        self.assertEqual(tier, "tier2_browser")

        filtered = filter_tool_schemas_by_tier(self.all_schemas, tier, is_browser=True)
        tool_names = {s.get("function", {}).get("name") or s.get("name") for s in filtered}
        self.assertNotIn("interact_with_ui", tool_names)

    def test_visual_grounding_allowed_outside_browser(self):
        """Visual grounding on legacy apps or canvas games without DOM bridge routes to Tier 3."""
        game_window = {"class": "retro_game", "title": "OpenGL Canvas"}
        task = "Click the green target in this retro OpenGL canvas game"
        tier = ActuatorArbiter.arbitrate(
            task=task,
            active_window=game_window,
            bridge_connected=False,
        )
        self.assertEqual(tier, "tier3_vision")

    def test_fallback_behavior_when_no_active_window(self):
        """When Hyprland IPC returns no active window, fallback uses deterministic task keyword matching."""
        # 1. OS/IPC query -> Tier 1
        tier_os = ActuatorArbiter.arbitrate(task="List all active processes", active_window=None, bridge_connected=False)
        self.assertEqual(tier_os, "tier1_ipc")

        # 2. Web query -> Tier 2
        tier_web = ActuatorArbiter.arbitrate(task="Navigate current tab to https://news.ycombinator.com", active_window=None, bridge_connected=False)
        self.assertEqual(tier_web, "tier2_browser")

        # 3. Explicit visual grounding -> Tier 3
        tier_vis = ActuatorArbiter.arbitrate(task="Shoot the spaceship on screen", active_window=None, bridge_connected=False)
        self.assertEqual(tier_vis, "tier3_vision")

        # 4. Generic task without browser or screen keywords -> Tier 1 fallback
        tier_generic = ActuatorArbiter.arbitrate(task="What time is it?", active_window=None, bridge_connected=False)
        self.assertEqual(tier_generic, "tier1_ipc")

    async def test_native_orchestrator_generate_stream_purges_tier3_when_browser_active(self):
        """NativeOrchestrator.generate_stream strictly purges interact_with_ui from payload['tools'] when browser is active."""
        orch = NativeOrchestrator()
        browser_window = {"class": "app.zen_browser.zen", "title": "Zen Browser"}

        payload = {
            "messages": [{"role": "user", "content": "Click the submit button"}],
            "tools": list(self.all_schemas),
        }

        @contextlib.asynccontextmanager
        async def mock_stream_client(method, url, **kwargs):
            mock_resp = AsyncMock()
            mock_resp.status_code = 200

            async def aiter_lines():
                yield 'data: {"choices": [{"delta": {"content": "Clicked."}}]}'
                yield 'data: [DONE]'

            mock_resp.aiter_lines = aiter_lines
            yield mock_resp

        # Mock active window as browser, bridge as connected, and Ollama request
        with patch.object(orch, "get_active_window", AsyncMock(return_value=browser_window)), \
             patch("zelyr.tools.browser_extension.get_bridge") as mock_get_bridge, \
             patch("httpx.AsyncClient") as mock_client:

            mock_bridge = mock_get_bridge.return_value
            mock_bridge.is_connected.return_value = True

            mock_client.return_value.__aenter__.return_value.stream = mock_stream_client
            mock_client.return_value.__aenter__.return_value.post = AsyncMock()

            chunks = []
            async for chunk in orch.generate_stream(payload):
                chunks.append(chunk)

            # Check that payload["tools"] had interact_with_ui purged
            tool_names = {t.get("function", {}).get("name") or t.get("name") for t in payload.get("tools", [])}
            self.assertNotIn("interact_with_ui", tool_names)
            self.assertIn("interact_with_browser", tool_names)


if __name__ == "__main__":
    unittest.main()
