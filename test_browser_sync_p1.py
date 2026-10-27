"""Test suite for Tier 2 Browser Hardening (Phase 1).

Covers:
1. test_static_page_settlement_fast_path
2. test_delayed_mutation_settlement_captures_update
3. test_continuous_mutation_hits_hard_cap
4. test_mutation_observer_cleanup
5. test_stale_element_id_recovers_via_unique_fingerprint
6. test_ambiguous_fingerprint_refuses_action
7. test_zero_match_fingerprint_returns_structured_remedy
8. test_tier2_budget_preservation
"""

import asyncio
import json
from pathlib import Path
import subprocess
import time
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from zelyr.tools.browser_cdp import InteractWithBrowserTool
from zelyr.tools.browser_extension import BrowserExtensionBridge, BrowserResult


def _run_node(script: str) -> dict:
    """Helper to run a Node.js script and parse JSON output."""
    res = subprocess.run(
        ["node", "-e", script],
        capture_output=True,
        text=True,
        check=True,
        cwd=str(Path(__file__).parent),
    )
    return json.loads(res.stdout.strip())


class TestBrowserSyncP1(unittest.IsolatedAsyncioTestCase):
    def test_static_page_settlement_fast_path(self):
        """Static DOM action settles within quietWindow (~75ms) without waiting for maxTimeout."""
        js_code = """
class MockMutationObserver {
  static instances = [];
  constructor(cb) { this.cb = cb; this.disconnected = false; MockMutationObserver.instances.push(this); }
  observe(t, o) { this.t = t; }
  disconnect() { this.disconnected = true; }
}
global.MutationObserver = MockMutationObserver;
global.document = { body: {}, documentElement: {} };
global.requestAnimationFrame = (cb) => setTimeout(cb, 5);

const { waitForSettled } = require('./zelyr/tools/extension/background.js');

async function run() {
  const start = Date.now();
  await waitForSettled({ quietWindow: 75, maxTimeout: 800 });
  const duration = Date.now() - start;
  console.log(JSON.stringify({
    duration,
    settled: true,
    observerDisconnected: MockMutationObserver.instances[0].disconnected
  }));
}
run();
"""
        result = _run_node(js_code)
        self.assertTrue(result["settled"])
        self.assertTrue(result["observerDisconnected"])
        # Should resolve around 75ms (+ 2 rAF ticks ~10ms), strictly < 400ms (far before 800ms)
        self.assertGreaterEqual(result["duration"], 50)
        self.assertLess(result["duration"], 400)

    def test_delayed_mutation_settlement_captures_update(self):
        """Simulate a button click triggering a delayed DOM mutation (200ms delay). Verify waitForSettled captures the newly rendered nodes."""
        js_code = """
class MockMutationObserver {
  static instances = [];
  constructor(cb) { this.cb = cb; this.disconnected = false; MockMutationObserver.instances.push(this); }
  observe(t, o) { this.t = t; }
  disconnect() { this.disconnected = true; }
  trigger() { if (!this.disconnected) this.cb(); }
}
global.MutationObserver = MockMutationObserver;
global.document = { body: {}, documentElement: {} };
global.requestAnimationFrame = (cb) => setTimeout(cb, 5);

const { waitForSettled } = require('./zelyr/tools/extension/background.js');

async function run() {
  const start = Date.now();
  let renderedNode = null;
  const iv = setInterval(() => {
    if (MockMutationObserver.instances.length > 0) {
      MockMutationObserver.instances[0].trigger();
    }
  }, 50);

  setTimeout(() => {
    clearInterval(iv);
    renderedNode = '<div id="async-result">Loaded Content</div>';
    if (MockMutationObserver.instances.length > 0) {
      MockMutationObserver.instances[0].trigger();
    }
  }, 200);

  await waitForSettled({ quietWindow: 75, maxTimeout: 800 });
  const duration = Date.now() - start;
  console.log(JSON.stringify({
    duration,
    renderedNode,
    observerDisconnected: MockMutationObserver.instances[0].disconnected
  }));
}
run();
"""
        result = _run_node(js_code)
        self.assertIsNotNone(result["renderedNode"])
        self.assertIn("Loaded Content", result["renderedNode"])
        self.assertTrue(result["observerDisconnected"])
        # Settles at 200ms + ~75ms quietWindow = ~275ms-350ms, strictly < 700ms
        self.assertGreaterEqual(result["duration"], 230)
        self.assertLess(result["duration"], 700)

    def test_continuous_mutation_hits_hard_cap(self):
        """Simulate continuous mutations (e.g. interval ticking every 20ms). Verify waitForSettled terminates cleanly at maxTimeout (~800ms) without hanging."""
        js_code = """
class MockMutationObserver {
  static instances = [];
  constructor(cb) { this.cb = cb; this.disconnected = false; MockMutationObserver.instances.push(this); }
  observe(t, o) { this.t = t; }
  disconnect() { this.disconnected = true; }
  trigger() { if (!this.disconnected) this.cb(); }
}
global.MutationObserver = MockMutationObserver;
global.document = { body: {}, documentElement: {} };
global.requestAnimationFrame = (cb) => setTimeout(cb, 5);

const { waitForSettled } = require('./zelyr/tools/extension/background.js');

async function run() {
  const start = Date.now();
  const iv = setInterval(() => {
    if (MockMutationObserver.instances.length > 0) {
      MockMutationObserver.instances[0].trigger();
    }
  }, 20);

  await waitForSettled({ quietWindow: 75, maxTimeout: 800 });
  clearInterval(iv);
  const duration = Date.now() - start;
  console.log(JSON.stringify({
    duration,
    observerDisconnected: MockMutationObserver.instances[0].disconnected
  }));
}
run();
"""
        result = _run_node(js_code)
        self.assertTrue(result["observerDisconnected"])
        # Should hit maxTimeout cap (~800ms), bounded between 750ms and 1200ms
        self.assertGreaterEqual(result["duration"], 750)
        self.assertLess(result["duration"], 1300)

    def test_mutation_observer_cleanup(self):
        """Verify MutationObserver.disconnect() is invoked under both normal resolution and timeout conditions."""
        js_code = """
class MockMutationObserver {
  static instances = [];
  constructor(cb) { this.cb = cb; this.disconnected = false; MockMutationObserver.instances.push(this); }
  observe(t, o) { this.t = t; }
  disconnect() { this.disconnected = true; }
  trigger() { if (!this.disconnected) this.cb(); }
}
global.MutationObserver = MockMutationObserver;
global.document = { body: {}, documentElement: {} };
global.requestAnimationFrame = (cb) => setTimeout(cb, 5);

const { waitForSettled } = require('./zelyr/tools/extension/background.js');

async function run() {
  // 1. Normal resolution
  await waitForSettled({ quietWindow: 40, maxTimeout: 400 });
  const normalCleaned = MockMutationObserver.instances[0].disconnected;

  // 2. Timeout condition
  const iv = setInterval(() => {
    if (MockMutationObserver.instances.length > 1) {
      MockMutationObserver.instances[1].trigger();
    }
  }, 10);
  await waitForSettled({ quietWindow: 100, maxTimeout: 200 });
  clearInterval(iv);
  const timeoutCleaned = MockMutationObserver.instances[1].disconnected;

  console.log(JSON.stringify({
    normalCleaned,
    timeoutCleaned
  }));
}
run();
"""
        result = _run_node(js_code)
        self.assertTrue(result["normalCleaned"])
        self.assertTrue(result["timeoutCleaned"])

    def test_stale_element_id_recovers_via_unique_fingerprint(self):
        """Simulate a React-style component remount where data-zelyr-id is stripped but the element's semantic fingerprint (tag, text, role) is identical and unique. Verify recovery succeeds."""
        js_code = """
const { resolveElement } = require('./zelyr/tools/extension/background.js');

class MockElement {
  constructor(tag, attrs = {}, text = '') {
    this.tagName = tag.toUpperCase();
    this.attrs = { ...attrs };
    this.textContent = text;
    this.innerText = text;
    this.isConnected = true;
  }
  getAttribute(name) { return this.attrs[name] || null; }
  setAttribute(name, val) { this.attrs[name] = String(val); }
  removeAttribute(name) { delete this.attrs[name]; }
}

global.window = {
  __zelyrFingerprints: {
    7: {
      tag: 'button',
      role: 'button',
      type: 'submit',
      name: 'confirm-pay',
      ariaLabel: 'Confirm Payment',
      textSnippet: 'Pay $25.00'
    }
  }
};

// Remounted replacement button without data-zelyr-id attribute
const remountedButton = new MockElement('button', {
  role: 'button',
  type: 'submit',
  name: 'confirm-pay',
  'aria-label': 'Confirm Payment'
}, 'Pay $25.00');

global.document = {
  querySelector: (sel) => null, // Stale: not found by [data-zelyr-id="7"]
  querySelectorAll: (sel) => [remountedButton]
};

const res = resolveElement(null, 7);
console.log(JSON.stringify({
  success: res.element !== null,
  newZelyrId: remountedButton.getAttribute('data-zelyr-id'),
  errorResult: res.errorResult
}));
"""
        result = _run_node(js_code)
        self.assertTrue(result["success"])
        self.assertEqual(result["newZelyrId"], "7")
        self.assertIsNone(result["errorResult"])

    def test_ambiguous_fingerprint_refuses_action(self):
        """Simulate an element unmount where multiple replacement elements match the fingerprint. Verify refusal with ambiguous error."""
        js_code = """
const { resolveElement } = require('./zelyr/tools/extension/background.js');

class MockElement {
  constructor(tag, attrs = {}, text = '') {
    this.tagName = tag.toUpperCase();
    this.attrs = { ...attrs };
    this.textContent = text;
    this.innerText = text;
    this.isConnected = true;
  }
  getAttribute(name) { return this.attrs[name] || null; }
  setAttribute(name, val) { this.attrs[name] = String(val); }
  removeAttribute(name) { delete this.attrs[name]; }
}

global.window = {
  __zelyrFingerprints: {
    12: {
      tag: 'button',
      role: 'button',
      type: 'button',
      name: 'item-add',
      ariaLabel: 'Add to Cart',
      textSnippet: 'Add'
    }
  }
};

// Two identical buttons match the fingerprint
const btn1 = new MockElement('button', { role: 'button', type: 'button', name: 'item-add', 'aria-label': 'Add to Cart' }, 'Add');
const btn2 = new MockElement('button', { role: 'button', type: 'button', name: 'item-add', 'aria-label': 'Add to Cart' }, 'Add');

global.document = {
  querySelector: (sel) => null,
  querySelectorAll: (sel) => [btn1, btn2]
};

const res = resolveElement(null, 12);
console.log(JSON.stringify({
  success: res.element !== null,
  errorResult: res.errorResult
}));
"""
        result = _run_node(js_code)
        self.assertFalse(result["success"])
        self.assertIsNotNone(result["errorResult"])
        err = result["errorResult"]["error"]
        self.assertIn("Ambiguous element resolution", err)
        self.assertIn("2 elements match fingerprint for ID 12", err)
        self.assertIn("Call get_page_snapshot", result["errorResult"]["remedy_hint"])
        self.assertEqual(result["errorResult"]["allowed_actions"], ["get_page_snapshot"])

    async def test_zero_match_fingerprint_returns_structured_remedy(self):
        """Simulate element removal with 0 matching elements. Verify structured error envelope matches the expected recovery contract."""
        js_code = """
const { resolveElement } = require('./zelyr/tools/extension/background.js');

global.window = {
  __zelyrFingerprints: {
    99: {
      tag: 'button',
      role: 'button',
      type: 'submit',
      name: 'deleted-btn',
      ariaLabel: '',
      textSnippet: 'Deleted Button'
    }
  }
};

global.document = {
  querySelector: (sel) => null,
  querySelectorAll: (sel) => []
};

const res = resolveElement(null, 99);
console.log(JSON.stringify({
  success: res.element !== null,
  errorResult: res.errorResult
}));
"""
        result = _run_node(js_code)
        self.assertFalse(result["success"])
        self.assertIsNotNone(result["errorResult"])
        self.assertEqual(result["errorResult"]["error"], "Element ID 99 not found on page.")
        self.assertIn("Call get_page_snapshot", result["errorResult"]["remedy_hint"])
        self.assertEqual(result["errorResult"]["allowed_actions"], ["get_page_snapshot", "scroll_page"])

        # Also test Python InteractWithBrowserTool preserves this envelope
        tool = InteractWithBrowserTool()
        with patch("zelyr.tools.browser_extension.get_bridge") as mock_get_bridge:
            mock_bridge = MagicMock()
            mock_bridge.is_connected.return_value = True
            mock_bridge.server = object()
            mock_bridge.ensure_server = AsyncMock()
            mock_bridge.send_command = AsyncMock(return_value=BrowserResult({
                "success": False,
                "action": "click_element",
                "error": "Element ID 99 not found on page.",
                "remedy_hint": "Call get_page_snapshot to refresh element IDs, or scroll_page down if the element is below the viewport.",
                "allowed_actions": ["get_page_snapshot", "scroll_page"],
            }))
            mock_get_bridge.return_value = mock_bridge

            res = await tool.execute({"action": "click_element", "element_id": 99})
            self.assertFalse(res.success)
            self.assertEqual(res.error, "Element ID 99 not found on page.")
            self.assertEqual(res.remedy_hint, "Call get_page_snapshot to refresh element IDs, or scroll_page down if the element is below the viewport.")
            self.assertEqual(res.allowed_actions, ["get_page_snapshot", "scroll_page"])

    async def test_tier2_budget_preservation(self):
        """Verify actions executing waitForSettled complete well within the Tier 2 10.0s budget."""
        tool = InteractWithBrowserTool()
        with patch("zelyr.tools.browser_extension.get_bridge") as mock_get_bridge:
            mock_bridge = MagicMock()
            mock_bridge.is_connected.return_value = True
            mock_bridge.server = object()
            mock_bridge.ensure_server = AsyncMock()

            async def simulated_settled_command(action, **kwargs):
                # Simulates tab navigation/click execution + 800ms max settlement duration
                await asyncio.sleep(0.85)
                return BrowserResult({
                    "success": True,
                    "action": action,
                    "clicked_id": kwargs.get("element_id"),
                    "tag": "BUTTON",
                    "text": "Submit",
                })

            mock_bridge.send_command = AsyncMock(side_effect=simulated_settled_command)
            mock_get_bridge.return_value = mock_bridge

            start_time = time.monotonic()
            res = await tool.execute({"action": "click_element", "element_id": 3})
            elapsed = time.monotonic() - start_time

            self.assertTrue(res.success)
            # Must be strictly under the 10.0s Tier 2 budget
            self.assertLess(elapsed, 10.0)
            # In our simulation, should take ~0.85s-1.5s
            self.assertLess(elapsed, 2.5)


if __name__ == "__main__":
    unittest.main()
