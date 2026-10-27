#!/usr/bin/env python3
"""Comprehensive test harness for Tier 2 Deep Browser Automation.

Verifies:
1. Static validation of background.js handlers (navigate_url, capture_tab_screenshot, select & checkbox handling).
2. InteractWithBrowserTool schema exposed parameters and actions.
3. Mock WebSocket dispatch for navigate_url, capture_tab_screenshot (with /tmp/zelyr_tab_capture.png save),
   and form control fill_element (<select> and checkbox/radio).
"""

import asyncio
import json
import os
from pathlib import Path
import websockets

from zelyr.tools.browser_cdp import InteractWithBrowserTool
from zelyr.tools.browser_extension import BrowserExtensionBridge


def test_static_background_js():
    print("[CHECK 1] Validating deep browser capabilities in background.js...")
    bg_js = Path("zelyr/tools/extension/background.js")
    assert bg_js.is_file(), "background.js missing!"
    code = bg_js.read_text(encoding="utf-8")

    assert "navigate_url" in code, "navigate_url handler missing in background.js"
    assert "capture_tab_screenshot" in code, "capture_tab_screenshot handler missing in background.js"
    assert "captureVisibleTab" in code, "captureVisibleTab call missing in background.js"
    assert "select" in code, "select handling missing in background.js"
    assert "checkbox" in code, "checkbox handling missing in background.js"
    assert "radio" in code, "radio handling missing in background.js"
    print("✓ Check 1 PASSED: All deep browser handlers verified in background.js.\n")


def test_tool_schema():
    print("[CHECK 2] Validating InteractWithBrowserTool schema for deep browser operations...")
    tool = InteractWithBrowserTool()
    schema = tool.schema
    props = schema.get("properties", {})

    action_desc = props.get("action", {}).get("description", "")
    assert "navigate_url" in action_desc, "navigate_url missing in action description"
    assert "capture_tab_screenshot" in action_desc, "capture_tab_screenshot missing in action description"
    print("✓ Check 2 PASSED: InteractWithBrowserTool schema exposes navigate_url and capture_tab_screenshot.\n")


async def test_live_bridge_deep_browser():
    print("[CHECK 3] Testing live WebSocket bridge dispatch for deep browser actions...")
    test_port = 41147
    BrowserExtensionBridge.reset_instance()
    bridge = BrowserExtensionBridge.get_instance(port=test_port)
    await bridge.start_server()
    assert bridge.server is not None, f"Failed to start WebSocket server on {test_port}"

    tool = InteractWithBrowserTool()

    # 1x1 transparent PNG in base64
    SAMPLE_PNG_B64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
    DATA_URL = f"data:image/png;base64,{SAMPLE_PNG_B64}"

    worker_task = None
    try:
        ws_uri = f"ws://{bridge.host}:{bridge.port}"
        async with websockets.connect(ws_uri) as client_ws:
            await asyncio.sleep(0.1)
            assert bridge.is_connected() is True

            # Mock Extension Responder
            async def mock_extension_worker():
                while True:
                    try:
                        raw = await client_ws.recv()
                        msg = json.loads(raw)
                        req_id = msg.get("id")
                        action = msg.get("action")

                        if not req_id or not action:
                            continue

                        if action == "navigate_url":
                            target_url = msg.get("url", "https://example.com")
                            reply = {
                                "id": req_id,
                                "success": True,
                                "action": "navigate_url",
                                "tab_id": 42,
                                "title": "Example Domain",
                                "url": target_url,
                            }
                        elif action == "capture_tab_screenshot":
                            reply = {
                                "id": req_id,
                                "success": True,
                                "action": "capture_tab_screenshot",
                                "format": "png",
                                "data_url": DATA_URL,
                                "data_url_prefix": DATA_URL[:30],
                                "length": len(DATA_URL),
                            }
                        elif action == "fill_element":
                            elem_id = msg.get("element_id")
                            text_val = msg.get("text", "")
                            # Simulate select or checkbox return
                            if text_val in ("true", "false"):
                                reply = {
                                    "id": req_id,
                                    "success": True,
                                    "action": "fill_element",
                                    "element_id": elem_id,
                                    "element_type": "input",
                                    "value": text_val == "true",
                                    "text": text_val,
                                    "submitted": False,
                                }
                            else:
                                reply = {
                                    "id": req_id,
                                    "success": True,
                                    "action": "fill_element",
                                    "element_id": elem_id,
                                    "element_type": "select",
                                    "value": text_val,
                                    "text": text_val,
                                    "submitted": False,
                                }
                        else:
                            reply = {"id": req_id, "success": True, "action": action}

                        await client_ws.send(json.dumps(reply))
                    except (asyncio.CancelledError, websockets.exceptions.ConnectionClosed):
                        break

            worker_task = asyncio.create_task(mock_extension_worker())

            # 1. Test in-place navigate_url
            res_nav = await tool.execute({
                "action": "navigate_url",
                "url": "https://example.com/deep-page",
            })
            assert res_nav.success, f"navigate_url failed: {res_nav.error}"
            assert res_nav.output.get("url") == "https://example.com/deep-page"
            print(f"  → navigate_url verified: {res_nav.output}")

            # 2. Test capture_tab_screenshot and local file persistence
            capture_path = Path("/tmp/zelyr_tab_capture.png")
            if capture_path.exists():
                capture_path.unlink()

            res_snap = await tool.execute({
                "action": "capture_tab_screenshot",
            })
            assert res_snap.success, f"capture_tab_screenshot failed: {res_snap.error}"
            assert res_snap.output.get("screenshot_path") == "/tmp/zelyr_tab_capture.png"
            assert capture_path.is_file(), "/tmp/zelyr_tab_capture.png was not created!"
            # Verify PNG header
            png_bytes = capture_path.read_bytes()
            assert png_bytes.startswith(b"\x89PNG\r\n\x1a\n"), "Saved file is not a valid PNG header"
            print(f"  → capture_tab_screenshot verified: saved {len(png_bytes)} bytes to /tmp/zelyr_tab_capture.png")

            # 3. Test fill_element on dropdown <select>
            res_select = await tool.execute({
                "action": "fill_element",
                "element_id": 8,
                "text": "Option 2",
            })
            assert res_select.success, f"fill_element select failed: {res_select.error}"
            assert res_select.output.get("element_type") == "select"
            assert res_select.output.get("value") == "Option 2"
            print(f"  → fill_element (<select>) verified: {res_select.output}")

            # 4. Test fill_element on checkbox toggle
            res_chk = await tool.execute({
                "action": "fill_element",
                "element_id": 9,
                "text": "true",
            })
            assert res_chk.success, f"fill_element checkbox failed: {res_chk.error}"
            assert res_chk.output.get("element_type") == "input"
            assert res_chk.output.get("value") is True
            print(f"  → fill_element (checkbox) verified: {res_chk.output}")

    finally:
        if worker_task:
            worker_task.cancel()
        await bridge.stop_server()
        BrowserExtensionBridge.reset_instance()

    print("✓ Check 3 PASSED: Live bridge roundtrip for deep browser suite verified.\n")


async def main():
    print("=" * 60)
    print("Zelyr TIER 2 DEEP BROWSER AUTOMATION SUITE")
    print("=" * 60)
    test_static_background_js()
    test_tool_schema()
    await test_live_bridge_deep_browser()
    print("=" * 60)
    print("✅ ALL TIER 2 DEEP BROWSER AUTOMATION CHECKS PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
