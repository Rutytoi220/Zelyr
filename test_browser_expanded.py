"""Test harness for expanded Zelyr Tier 2 WebExtension automation suite.

Verifies:
1. Static validation of background.js handlers (scroll_page, extract_page_content, duplicate_tab, reload_tab, pin_tab).
2. InteractWithBrowserTool schema enhancements (direction, amount, mode, pinned, and new actions).
3. Live WebSocket bridge roundtrip for scroll_page, extract_page_content with 4000-char truncation, and tab pinning.
4. InteractWithBrowserTool dispatch through extension bridge.
"""

import asyncio
import json
from pathlib import Path
import websockets

from zelyr.tools.browser_cdp import InteractWithBrowserTool
from zelyr.tools.browser_extension import BrowserExtensionBridge


def test_static_extension_handlers():
    print("[CHECK 1] Validating expanded background.js handlers...")
    ext_dir = Path("zelyr/tools/extension")
    bg_js = ext_dir / "background.js"
    assert bg_js.is_file(), "background.js missing!"

    code = bg_js.read_text(encoding="utf-8")
    assert "scroll_page" in code, "scroll_page handler missing in background.js"
    assert "extract_page_content" in code, "extract_page_content handler missing in background.js"
    assert "duplicate_tab" in code, "duplicate_tab handler missing in background.js"
    assert "reload_tab" in code, "reload_tab handler missing in background.js"
    assert "pin_tab" in code, "pin_tab handler missing in background.js"
    assert "manage_browser_tabs" in code, "manage_browser_tabs consolidation missing in background.js"
    assert "4000" in code, "4000 character limit budget missing in background.js"
    assert "content truncated for context budget" in code, "Truncation notice missing in background.js"
    print("✓ Check 1 PASSED: All expanded handlers and context budget rules present in background.js.\n")


def test_tool_schema():
    print("[CHECK 2] Validating InteractWithBrowserTool schema updates...")
    tool = InteractWithBrowserTool()
    schema = tool.schema
    props = schema.get("properties", {})

    assert "port" not in props, "'port' must remain purged from schema"
    assert "direction" in props, "'direction' parameter missing from schema"
    assert "amount" in props, "'amount' parameter missing from schema"
    assert "mode" in props, "'mode' parameter missing from schema"
    assert "pinned" in props, "'pinned' parameter missing from schema"

    action_desc = props.get("action", {}).get("description", "")
    assert "scroll_page" in action_desc
    assert "extract_page_content" in action_desc
    assert "duplicate_tab" in action_desc
    assert "reload_tab" in action_desc
    assert "pin_tab" in action_desc
    print("✓ Check 2 PASSED: InteractWithBrowserTool schema properly exposes expanded actions and arguments.\n")


async def test_live_bridge_expanded_actions():
    print("[CHECK 3] Testing live WebSocket bridge dispatch for expanded browser suite...")
    test_port = 41146
    BrowserExtensionBridge.reset_instance()
    bridge = BrowserExtensionBridge.get_instance(port=test_port)
    await bridge.start_server()
    assert bridge.server is not None, f"Failed to start WebSocket server on {test_port}"

    tool = InteractWithBrowserTool()

    try:
        ws_uri = f"ws://{bridge.host}:{bridge.port}"
        async with websockets.connect(ws_uri) as client_ws:
            await asyncio.sleep(0.1)
            assert bridge.is_connected() is True

            # Background mock responder
            async def mock_worker():
                while True:
                    try:
                        raw = await client_ws.recv()
                        msg = json.loads(raw)
                        req_id = msg.get("id")
                        action = msg.get("action")

                        if not req_id or not action:
                            continue

                        if action == "scroll_page":
                            amt = msg.get("amount", 600)
                            dir_ = msg.get("direction", "down")
                            reply = {
                                "id": req_id,
                                "success": True,
                                "action": "scroll_page",
                                "scrollY": amt if dir_ == "down" else 0,
                                "innerHeight": 1080,
                                "scrollHeight": 4500,
                                "direction": dir_,
                            }
                        elif action == "extract_page_content":
                            mode = msg.get("mode", "readable")
                            # Simulate long article exceeding 4000 characters
                            base_text = "# High Performance Computing Article\n\nThis is paragraph content. " * 150
                            max_budget = 4000
                            truncated = False
                            if len(base_text) > max_budget:
                                base_text = base_text[:max_budget] + "\n\n[... content truncated for context budget]"
                                truncated = True
                            reply = {
                                "id": req_id,
                                "success": True,
                                "action": "extract_page_content",
                                "title": "HPC Article",
                                "url": "https://example.com/hpc",
                                "content": base_text,
                                "length": len(base_text),
                                "truncated": truncated,
                            }
                        elif action == "duplicate_tab":
                            reply = {
                                "id": req_id,
                                "success": True,
                                "action": "duplicate_tab",
                                "tab_id": 202,
                                "title": "Google Gemini (Copy)",
                                "url": "https://gemini.google.com",
                            }
                        elif action == "reload_tab":
                            reply = {
                                "id": req_id,
                                "success": True,
                                "action": "reload_tab",
                                "tab_id": 101,
                                "title": "Google Gemini",
                                "url": "https://gemini.google.com",
                            }
                        elif action == "pin_tab":
                            pin_val = msg.get("pinned", True)
                            reply = {
                                "id": req_id,
                                "success": True,
                                "action": "pin_tab",
                                "tab_id": 101,
                                "pinned": bool(pin_val),
                                "title": "Google Gemini",
                            }
                        else:
                            reply = {"id": req_id, "success": True, "action": action}

                        await client_ws.send(json.dumps(reply))
                    except (asyncio.CancelledError, websockets.ConnectionClosed):
                        break

            worker_task = asyncio.create_task(mock_worker())

            try:
                # 1. Test scroll_page
                scroll_res = await bridge.send_command("scroll_page", direction="down", amount=800)
                assert scroll_res.success is True, f"scroll_page failed: {scroll_res.error}"
                assert scroll_res["scrollY"] == 800
                assert scroll_res["direction"] == "down"
                print(f"  → scroll_page roundtrip verified: scrolled {scroll_res['scrollY']}px / {scroll_res['scrollHeight']}px total.")

                # 2. Test extract_page_content with truncation
                extract_res = await bridge.send_command("extract_page_content", mode="readable")
                assert extract_res.success is True, f"extract_page_content failed: {extract_res.error}"
                content = extract_res["content"]
                assert extract_res["truncated"] is True
                assert "[... content truncated for context budget]" in content
                assert len(content) <= 4100
                print(f"  → extract_page_content roundtrip verified: extracted {len(content)} chars (budget capped at 4,000 + notice).")

                # 3. Test tab management
                dup_res = await bridge.send_command("duplicate_tab")
                assert dup_res.success is True
                assert dup_res["tab_id"] == 202
                print("  → duplicate_tab roundtrip verified: new tab_id 202.")

                reload_res = await bridge.send_command("reload_tab")
                assert reload_res.success is True
                print("  → reload_tab roundtrip verified: reloaded tab 101.")

                pin_res = await bridge.send_command("pin_tab", pinned=True)
                assert pin_res.success is True
                assert pin_res["pinned"] is True
                print("  → pin_tab roundtrip verified: tab 101 pinned.")

                # 4. Test tool dispatch via InteractWithBrowserTool.execute()
                tool_scroll = await tool.execute({"action": "scroll_page", "direction": "down", "amount": 600})
                assert tool_scroll.success is True
                print(f"  → InteractWithBrowserTool scroll_page dispatch verified: {tool_scroll.output}")

                tool_extract = await tool.execute({"action": "extract_page_content"})
                assert tool_extract.success is True
                assert "[... content truncated for context budget]" in str(tool_extract.output)
                print("  → InteractWithBrowserTool extract_page_content dispatch verified.")

                tool_pin = await tool.execute({"action": "pin_tab", "pinned": True})
                assert tool_pin.success is True
                print("  → InteractWithBrowserTool pin_tab dispatch verified.")

            finally:
                worker_task.cancel()
                try:
                    await worker_task
                except asyncio.CancelledError:
                    pass

        print("✓ Check 3 PASSED: Live bridge roundtrip for expanded browser suite verified.\n")

    finally:
        await bridge.stop_server()


async def main():
    print("=" * 60)
    print("Zelyr EXPANDED BROWSER SUITE VERIFICATION")
    print("=" * 60)
    test_static_extension_handlers()
    test_tool_schema()
    await test_live_bridge_expanded_actions()
    print("=" * 60)
    print("✅ ALL EXPANDED BROWSER SUITE TESTS PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
