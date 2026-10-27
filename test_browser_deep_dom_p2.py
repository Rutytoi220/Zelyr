"""Test suite for Tier 2 Browser Hardening (Phase 2) — Deep DOM Coverage.

Covers:
1. test_open_shadow_dom_discovery_in_snapshot
2. test_click_element_inside_open_shadow_root
3. test_stale_shadow_element_recovers_via_fingerprint
4. test_accessible_iframe_discovery
5. test_cross_origin_iframe_graceful_containment
6. test_recursion_depth_limit
"""

import asyncio
import json
from pathlib import Path
import subprocess
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
    for line in reversed(res.stdout.strip().splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            return json.loads(line)
    return json.loads(res.stdout.strip())


class TestBrowserDeepDomP2(unittest.IsolatedAsyncioTestCase):
    def test_open_shadow_dom_discovery_in_snapshot(self):
        """Simulate a custom web component with an open shadow root containing a button.

        Verify get_page_snapshot discovers the button, stamps a data-zelyr-id,
        and includes it in the snapshot summary and __zelyrFingerprints.
        """
        js_code = """
const { collectInteractiveElements, handleCommand } = require('./zelyr/tools/extension/background.js');

class MockElement {
  constructor(tag, attrs = {}, text = '') {
    this.tagName = tag.toUpperCase();
    this.attrs = { ...attrs };
    this.textContent = text;
    this.innerText = text;
    this.isConnected = true;
    this.shadowRoot = null;
    this.children = [];
  }
  getAttribute(name) { return this.attrs[name] || null; }
  setAttribute(name, val) { this.attrs[name] = String(val); }
  removeAttribute(name) { delete this.attrs[name]; }
  getRootNode() { return global.document; }
  querySelectorAll(sel) {
    const res = [];
    for (const child of this.children) {
      if (child._matches(sel)) res.push(child);
      if (child.querySelectorAll) res.push(...child.querySelectorAll(sel));
    }
    return res;
  }
  _matches(sel) {
    if (sel === '*') return true;
    if (sel.includes('button') && this.tagName === 'BUTTON') return true;
    if (sel.includes('a[href]') && this.tagName === 'A' && this.attrs.href) return true;
    if (sel.includes('[role="button"]') && this.attrs.role === 'button') return true;
    return false;
  }
}

class MockShadowRoot {
  constructor(mode = 'open') {
    this.mode = mode;
    this.children = [];
  }
  querySelectorAll(sel) {
    const res = [];
    for (const child of this.children) {
      if (child._matches(sel)) res.push(child);
      if (child.querySelectorAll) res.push(...child.querySelectorAll(sel));
    }
    return res;
  }
}

// 1. Host element with open shadow root
const host = new MockElement('custom-card');
const shadow = new MockShadowRoot('open');
const shadowBtn = new MockElement('button', { role: 'button', type: 'button' }, 'Shadow Action');
shadow.children.push(shadowBtn);
host.shadowRoot = shadow;

// 2. Main document elements
const mainLink = new MockElement('a', { href: '/dashboard' }, 'Dashboard Link');

global.document = {
  title: 'Deep DOM Page',
  body: { innerText: 'Deep DOM Page Content' },
  children: [mainLink, host],
  querySelectorAll: (sel) => {
    const res = [];
    for (const c of global.document.children) {
      if (c._matches(sel)) res.push(c);
      if (c.querySelectorAll) res.push(...c.querySelectorAll(sel));
    }
    return res;
  },
  querySelector: (sel) => null
};

global.window = {
  location: { href: 'http://localhost/test' },
  __zelyrFingerprints: {}
};

let capturedReply = null;
global.chrome = {
  tabs: {
    query: async () => [{ id: 10, active: true }]
  },
  scripting: {
    executeScript: async ({ func, args }) => {
      const res = await func(...(args || []));
      return [{ result: res }];
    }
  }
};

global.__zelyrSendReplyHook = (reply) => { capturedReply = reply; };

async function run() {
  await handleCommand({ id: 101, action: 'get_page_snapshot' });
  console.log(JSON.stringify({
    success: capturedReply.success,
    count: capturedReply.count,
    elements: capturedReply.elements,
    summary: capturedReply.summary,
    shadowBtnZelyrId: shadowBtn.getAttribute('data-zelyr-id'),
    fingerprints: global.window.__zelyrFingerprints
  }));
}
run();
"""
        result = _run_node(js_code)
        self.assertTrue(result["success"])
        self.assertEqual(result["count"], 2)
        # Verify shadow button discovered and assigned transient data-zelyr-id
        self.assertEqual(result["shadowBtnZelyrId"], "2")
        # Verify shadow button appears in snapshot summary
        self.assertIn("<button", result["summary"])
        self.assertIn('"Shadow Action"', result["summary"])
        # Verify indexed into __zelyrFingerprints
        self.assertIn("2", result["fingerprints"])
        self.assertEqual(result["fingerprints"]["2"]["tag"], "button")
        self.assertEqual(result["fingerprints"]["2"]["textSnippet"], "Shadow Action")

    def test_click_element_inside_open_shadow_root(self):
        """Simulate clicking a button located inside an open shadow root via its data-zelyr-id.

        Verify element resolution locates the shadow button and triggers click cleanly.
        """
        js_code = """
const { handleCommand } = require('./zelyr/tools/extension/background.js');

class MockElement {
  constructor(tag, attrs = {}, text = '') {
    this.tagName = tag.toUpperCase();
    this.attrs = { ...attrs };
    this.textContent = text;
    this.innerText = text;
    this.isConnected = true;
    this.shadowRoot = null;
    this.clicked = false;
    this.children = [];
    this.style = {};
  }
  getAttribute(name) { return this.attrs[name] || null; }
  setAttribute(name, val) { this.attrs[name] = String(val); }
  removeAttribute(name) { delete this.attrs[name]; }
  click() { this.clicked = true; }
  focus() {}
  scrollIntoView() {}
  getRootNode() { return global.shadow; }
  _matches(sel) {
    if (sel === '*') return true;
    if (sel.includes('[data-zelyr-id="3"]') && this.attrs['data-zelyr-id'] === '3') return true;
    return false;
  }
  querySelectorAll(sel) {
    const res = [];
    for (const c of this.children) {
      if (c._matches(sel)) res.push(c);
      if (c.querySelectorAll) res.push(...c.querySelectorAll(sel));
    }
    return res;
  }
  querySelector(sel) {
    const all = this.querySelectorAll(sel);
    return all.length ? all[0] : null;
  }
}

class MockShadowRoot {
  constructor(mode = 'open') {
    this.mode = mode;
    this.children = [];
  }
  querySelectorAll(sel) {
    const res = [];
    for (const c of this.children) {
      if (c._matches(sel)) res.push(c);
      if (c.querySelectorAll) res.push(...c.querySelectorAll(sel));
    }
    return res;
  }
  querySelector(sel) {
    const all = this.querySelectorAll(sel);
    return all.length ? all[0] : null;
  }
}

const host = new MockElement('modal-dialog');
const shadow = new MockShadowRoot('open');
global.shadow = shadow;
const shadowSubmitBtn = new MockElement('button', { 'data-zelyr-id': '3', role: 'button' }, 'Confirm Order');
shadow.children.push(shadowSubmitBtn);
host.shadowRoot = shadow;

global.document = {
  children: [host],
  querySelector: (sel) => {
    // Document top-level does NOT pierce shadow DOM
    return null;
  },
  querySelectorAll: (sel) => {
    if (sel === '*') return [host];
    return [];
  }
};

global.window = {
  __zelyrFingerprints: {
    3: { tag: 'button', role: 'button', textSnippet: 'Confirm Order' }
  }
};

let capturedReply = null;
global.chrome = {
  tabs: { query: async () => [{ id: 11, active: true }] },
  scripting: {
    executeScript: async ({ func, args }) => {
      const res = await func(...(args || []));
      return [{ result: res }];
    }
  }
};
global.__zelyrSendReplyHook = (reply) => { capturedReply = reply; };

async function run() {
  await handleCommand({ id: 102, action: 'click_element', element_id: 3 });
  console.log(JSON.stringify({
    success: capturedReply.success,
    clickedId: capturedReply.clicked_id,
    tag: capturedReply.tag,
    btnClicked: shadowSubmitBtn.clicked
  }));
}
run();
"""
        result = _run_node(js_code)
        self.assertTrue(result["success"])
        self.assertEqual(result["clickedId"], 3)
        self.assertEqual(result["tag"], "BUTTON")
        self.assertTrue(result["btnClicked"])

    def test_stale_shadow_element_recovers_via_fingerprint(self):
        """Remount a shadow DOM component where data-zelyr-id is stripped.

        Verify fingerprint fallback successfully traverses the shadow root
        and recovers the replacement element.
        """
        js_code = """
const { resolveElement } = require('./zelyr/tools/extension/background.js');

class MockElement {
  constructor(tag, attrs = {}, text = '') {
    this.tagName = tag.toUpperCase();
    this.attrs = { ...attrs };
    this.textContent = text;
    this.innerText = text;
    this.isConnected = true;
    this.shadowRoot = null;
    this.children = [];
  }
  getAttribute(name) { return this.attrs[name] || null; }
  setAttribute(name, val) { this.attrs[name] = String(val); }
  removeAttribute(name) { delete this.attrs[name]; }
  getRootNode() { return null; }
  _matches(sel) {
    if (sel === '*') return true;
    if (sel.includes('button') && this.tagName === 'BUTTON') return true;
    return false;
  }
  querySelectorAll(sel) {
    const res = [];
    for (const c of this.children) {
      if (c._matches(sel)) res.push(c);
      if (c.querySelectorAll) res.push(...c.querySelectorAll(sel));
    }
    return res;
  }
}

class MockShadowRoot {
  constructor(mode = 'open') {
    this.mode = mode;
    this.children = [];
  }
  querySelectorAll(sel) {
    const res = [];
    for (const c of this.children) {
      if (c._matches(sel)) res.push(c);
      if (c.querySelectorAll) res.push(...c.querySelectorAll(sel));
    }
    return res;
  }
}

global.window = {
  __zelyrFingerprints: {
    5: {
      tag: 'button',
      role: 'button',
      type: 'submit',
      name: 'shadow-pay',
      ariaLabel: 'Pay Inside Shadow',
      textSnippet: 'Pay Now'
    }
  }
};

// Remounted custom host with new shadow root and button without data-zelyr-id
const host = new MockElement('payment-widget');
const shadow = new MockShadowRoot('open');
const remountedShadowBtn = new MockElement('button', {
  role: 'button',
  type: 'submit',
  name: 'shadow-pay',
  'aria-label': 'Pay Inside Shadow'
}, 'Pay Now');
shadow.children.push(remountedShadowBtn);
host.shadowRoot = shadow;

global.document = {
  querySelector: (sel) => null, // Stale lookup fails
  querySelectorAll: (sel) => {
    if (sel === '*') return [host];
    return [];
  }
};

const res = resolveElement(null, 5);

console.log(JSON.stringify({
  success: res.element !== null,
  newZelyrId: remountedShadowBtn.getAttribute('data-zelyr-id'),
  errorResult: res.errorResult
}));
"""
        result = _run_node(js_code)
        self.assertTrue(result["success"])
        self.assertEqual(result["newZelyrId"], "5")
        self.assertIsNone(result["errorResult"])

    def test_accessible_iframe_discovery(self):
        """Simulate an accessible same-origin iframe with interactive elements.

        Verify iframe interactive elements are recursively discovered and appear
        in the page snapshot with sequential IDs.
        """
        js_code = """
const { handleCommand } = require('./zelyr/tools/extension/background.js');

class MockElement {
  constructor(tag, attrs = {}, text = '') {
    this.tagName = tag.toUpperCase();
    this.attrs = { ...attrs };
    this.textContent = text;
    this.innerText = text;
    this.isConnected = true;
    this.children = [];
    this.contentDocument = null;
  }
  getAttribute(name) { return this.attrs[name] || null; }
  setAttribute(name, val) { this.attrs[name] = String(val); }
  removeAttribute(name) { delete this.attrs[name]; }
  getRootNode() { return global.document; }
  _matches(sel) {
    if (sel === '*') return true;
    if (sel.includes('button') && this.tagName === 'BUTTON') return true;
    if (sel.includes('input') && this.tagName === 'INPUT') return true;
    if ((sel.includes('iframe') || sel.includes('frame')) && this.tagName === 'IFRAME') return true;
    return false;
  }
  querySelectorAll(sel) {
    const res = [];
    for (const c of this.children) {
      if (c._matches(sel)) res.push(c);
      if (c.querySelectorAll) res.push(...c.querySelectorAll(sel));
    }
    return res;
  }
}

// 1. Top-level button
const topBtn = new MockElement('button', {}, 'Top Level Button');

// 2. IFrame with content document
const iframe = new MockElement('iframe');
const frameBtn = new MockElement('button', {}, 'IFrame Button');
const frameInput = new MockElement('input', { type: 'text', name: 'search' }, '');

const frameDoc = {
  body: {},
  children: [frameBtn, frameInput],
  querySelectorAll: (sel) => {
    const res = [];
    for (const c of frameDoc.children) {
      if (c._matches(sel)) res.push(c);
      if (c.querySelectorAll) res.push(...c.querySelectorAll(sel));
    }
    return res;
  }
};
iframe.contentDocument = frameDoc;

global.document = {
  title: 'Frame Test Page',
  children: [topBtn, iframe],
  querySelectorAll: (sel) => {
    const res = [];
    for (const c of global.document.children) {
      if (c._matches(sel)) res.push(c);
      if (c.querySelectorAll) res.push(...c.querySelectorAll(sel));
    }
    return res;
  },
  querySelector: () => null
};

global.window = {
  location: { href: 'http://localhost/frame-test' },
  __zelyrFingerprints: {}
};

let capturedReply = null;
global.chrome = {
  tabs: { query: async () => [{ id: 12, active: true }] },
  scripting: {
    executeScript: async ({ func, args }) => {
      const res = await func(...(args || []));
      return [{ result: res }];
    }
  }
};
global.__zelyrSendReplyHook = (reply) => { capturedReply = reply; };

async function run() {
  await handleCommand({ id: 103, action: 'get_page_snapshot' });
  console.log(JSON.stringify({
    success: capturedReply.success,
    count: capturedReply.count,
    elements: capturedReply.elements,
    summary: capturedReply.summary
  }));
}
run();
"""
        result = _run_node(js_code)
        self.assertTrue(result["success"])
        # Top button + frame button + frame input = 3 elements
        self.assertEqual(result["count"], 3)
        tags = [e["tag"] for e in result["elements"]]
        self.assertEqual(tags, ["button", "button", "input"])
        texts = [e.get("text") for e in result["elements"] if e.get("text")]
        self.assertIn("Top Level Button", texts)
        self.assertIn("IFrame Button", texts)

    def test_cross_origin_iframe_graceful_containment(self):
        """Simulate a cross-origin iframe where accessing contentDocument throws SecurityError.

        Verify snapshot completes cleanly without crashing, excludes the restricted frame,
        and annotates the restricted iframe in the summary.
        """
        js_code = """
const { handleCommand } = require('./zelyr/tools/extension/background.js');

class MockElement {
  constructor(tag, attrs = {}, text = '') {
    this.tagName = tag.toUpperCase();
    this.attrs = { ...attrs };
    this.textContent = text;
    this.innerText = text;
    this.isConnected = true;
    this.children = [];
  }
  getAttribute(name) { return this.attrs[name] || null; }
  setAttribute(name, val) { this.attrs[name] = String(val); }
  removeAttribute(name) { delete this.attrs[name]; }
  getRootNode() { return global.document; }
  _matches(sel) {
    if (sel === '*') return true;
    if (sel.includes('button') && this.tagName === 'BUTTON') return true;
    if ((sel.includes('iframe') || sel.includes('frame')) && this.tagName === 'IFRAME') return true;
    return false;
  }
  querySelectorAll(sel) {
    const res = [];
    for (const c of this.children) {
      if (c._matches(sel)) res.push(c);
      if (c.querySelectorAll) res.push(...c.querySelectorAll(sel));
    }
    return res;
  }
}

const safeBtn = new MockElement('button', {}, 'Safe Top Button');

// Cross-origin iframe whose contentDocument throws SecurityError
const crossIframe = new MockElement('iframe');
Object.defineProperty(crossIframe, 'contentDocument', {
  get() {
    const err = new Error("Blocked a frame with origin from accessing a cross-origin frame.");
    err.name = "SecurityError";
    throw err;
  }
});
Object.defineProperty(crossIframe, 'contentWindow', {
  get() {
    const err = new Error("Blocked a frame with origin from accessing a cross-origin frame.");
    err.name = "SecurityError";
    throw err;
  }
});

global.document = {
  title: 'Cross Origin Page',
  children: [safeBtn, crossIframe],
  querySelectorAll: (sel) => {
    const res = [];
    for (const c of global.document.children) {
      if (c._matches(sel)) res.push(c);
      if (c.querySelectorAll) res.push(...c.querySelectorAll(sel));
    }
    return res;
  },
  querySelector: () => null
};

global.window = {
  location: { href: 'http://localhost/cross' },
  __zelyrFingerprints: {}
};

let capturedReply = null;
global.chrome = {
  tabs: { query: async () => [{ id: 13, active: true }] },
  scripting: {
    executeScript: async ({ func, args }) => {
      const res = await func(...(args || []));
      return [{ result: res }];
    }
  }
};
global.__zelyrSendReplyHook = (reply) => { capturedReply = reply; };

async function run() {
  await handleCommand({ id: 104, action: 'get_page_snapshot' });
  console.log(JSON.stringify({
    success: capturedReply.success,
    count: capturedReply.count,
    elements: capturedReply.elements,
    summary: capturedReply.summary
  }));
}
run();
"""
        result = _run_node(js_code)
        self.assertTrue(result["success"])
        # Only the safe button is collected; cross-origin frame contents excluded
        self.assertEqual(result["count"], 1)
        self.assertEqual(result["elements"][0]["text"], "Safe Top Button")
        # Snapshot contains graceful containment placeholder
        self.assertIn("[iframe: cross-origin restricted]", result["summary"])

    def test_recursion_depth_limit(self):
        """Simulate nested shadow roots exceeding maxDepth (10).

        Verify tree-walker respects depth bounds, terminates without stack overflow,
        and safely handles circular references.
        """
        js_code = """
const { collectInteractiveElements } = require('./zelyr/tools/extension/background.js');

class MockElement {
  constructor(tag, attrs = {}, text = '') {
    this.tagName = tag.toUpperCase();
    this.attrs = { ...attrs };
    this.textContent = text;
    this.innerText = text;
    this.isConnected = true;
    this.shadowRoot = null;
    this.children = [];
  }
  getAttribute(name) { return this.attrs[name] || null; }
  setAttribute(name, val) { this.attrs[name] = String(val); }
  removeAttribute(name) { delete this.attrs[name]; }
  _matches(sel) {
    if (sel === '*') return true;
    if (sel.includes('button') && this.tagName === 'BUTTON') return true;
    return false;
  }
  querySelectorAll(sel) {
    const res = [];
    for (const c of this.children) {
      if (c._matches(sel)) res.push(c);
      if (c.querySelectorAll) res.push(...c.querySelectorAll(sel));
    }
    return res;
  }
}

class MockShadowRoot {
  constructor(mode = 'open') {
    this.mode = mode;
    this.children = [];
  }
  querySelectorAll(sel) {
    const res = [];
    for (const c of this.children) {
      if (c._matches(sel)) res.push(c);
      if (c.querySelectorAll) res.push(...c.querySelectorAll(sel));
    }
    return res;
  }
}

// Build a chain of 15 nested shadow roots
const rootHost = new MockElement('depth-root');
let currentHost = rootHost;
for (let d = 0; d < 15; d++) {
  const shadow = new MockShadowRoot('open');
  const btn = new MockElement('button', {}, `Button Depth ${d}`);
  shadow.children.push(btn);
  currentHost.shadowRoot = shadow;

  const nextHost = new MockElement(`depth-host-${d}`);
  shadow.children.push(nextHost);
  currentHost = nextHost;
}

const mockDoc = {
  querySelectorAll: (sel) => {
    if (sel === '*') return [rootHost];
    return [];
  }
};

// 1. Test deep recursion cutoff at maxDepth = 10
const collected = collectInteractiveElements(mockDoc, 10, 0);

// 2. Test circular reference safety
const circularHost = new MockElement('circular-host');
const circularShadow = new MockShadowRoot('open');
circularHost.shadowRoot = circularShadow;
circularShadow.children.push(circularHost); // cycle

const circularDoc = {
  querySelectorAll: (sel) => {
    if (sel === '*') return [circularHost];
    return [];
  }
};

const circularCollected = collectInteractiveElements(circularDoc, 10, 0);

console.log(JSON.stringify({
  collectedCount: collected.length,
  circularCount: circularCollected.length,
  terminatedCleanly: true
}));
"""
        result = _run_node(js_code)
        self.assertTrue(result["terminatedCleanly"])
        # Max depth 10 limits collection to 10 levels of shadow roots (depths 1..10); depths 11..15 are skipped
        self.assertEqual(result["collectedCount"], 10)
        # Circular structure does not cause infinite loop or stack overflow
        self.assertEqual(result["circularCount"], 0)

    async def test_python_tool_bridge_integration(self):
        """Verify InteractWithBrowserTool routes snapshot and click actions through bridge."""
        tool = InteractWithBrowserTool()
        self.assertEqual(tool.tier, 2)
        self.assertTrue(tool.requires_bridge)

        mock_bridge = MagicMock(spec=BrowserExtensionBridge)
        mock_bridge.server = object()
        mock_bridge.is_connected.return_value = True
        mock_bridge.send_command = AsyncMock(return_value=BrowserResult(
            success=True,
            summary="[1] <button>",
            elements=[{"id": 1, "tag": "button"}]
        ))

        with patch("zelyr.tools.browser_extension.get_bridge", return_value=mock_bridge):
            res = await tool.execute({"action": "get_page_snapshot"})
            self.assertTrue(res.success)
            self.assertEqual(res.output, "[1] <button>")


if __name__ == "__main__":
    unittest.main()
