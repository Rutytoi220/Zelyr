"""Verification Harness for Zelyr Browser Extension & Local WebSocket Bridge."""

import asyncio
import contextlib
import io
import json
from pathlib import Path
import sys
import websockets

from zelyr.tools.browser_extension import (
    BrowserExtensionBridge,
    get_bridge,
    interact_with_browser,
)


def test_extension_scaffold():
    print("[CHECK 1] Testing WebExtension Scaffold Files...")
    ext_dir = Path(__file__).parent / "zelyr" / "tools" / "extension"
    assert ext_dir.is_dir(), f"Extension directory missing: {ext_dir}"

    manifest_file = ext_dir / "manifest.json"
    assert manifest_file.is_file(), "manifest.json missing"
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))

    assert manifest.get("manifest_version") == 3, "Expected manifest_version 3"
    perms = manifest.get("permissions", [])
    assert "tabs" in perms, "Missing 'tabs' permission"
    assert "activeTab" in perms, "Missing 'activeTab' permission"
    assert "scripting" in perms, "Missing 'scripting' permission"
    assert "alarms" in perms, "Missing 'alarms' permission"
    assert "<all_urls>" in manifest.get("host_permissions", []), "Missing '<all_urls>' host permission"

    bg_file = ext_dir / "background.js"
    assert bg_file.is_file(), "background.js missing"
    bg_code = bg_file.read_text(encoding="utf-8")
    assert "41144" in bg_code, "Port 41144 not configured in background.js"
    assert "2000" in bg_code, "2000ms initial reconnect delay not configured in background.js"
    assert "5000" in bg_code, "5000ms max reconnect delay cap not configured in background.js"
    assert "zelyr_keep_alive" in bg_code, "zelyr_keep_alive alarm missing in background.js"
    assert "ping" in bg_code, "ping keep-alive loop missing in background.js"
    assert "switch_tab" in bg_code, "switch_tab handler missing in background.js"
    assert "open_tab" in bg_code, "open_tab handler missing in background.js"
    assert "get_active_tab" in bg_code, "get_active_tab handler missing in background.js"
    assert "close_tab" in bg_code, "close_tab handler missing in background.js"
    assert "get_page_snapshot" in bg_code, "get_page_snapshot handler missing in background.js"
    assert "click_element" in bg_code, "click_element handler missing in background.js"
    assert "fill_element" in bg_code, "fill_element handler missing in background.js"
    assert "click" in bg_code, "click handler missing in background.js"
    assert "type" in bg_code, "type handler missing in background.js"
    assert "get_dom" in bg_code, "get_dom handler missing in background.js"

    # Verify icons
    icons_dir = ext_dir / "icons"
    assert (icons_dir / "icon.svg").is_file(), "icon.svg missing"
    assert (icons_dir / "icon-16.png").is_file(), "icon-16.png missing"
    assert (icons_dir / "icon-48.png").is_file(), "icon-48.png missing"
    assert (icons_dir / "icon-128.png").is_file(), "icon-128.png missing"
    print("✓ Check 1 PASSED: WebExtension Manifest V3, background worker, and icons validated.\n")


async def run_bridge_lifecycle_tests():
    print("[CHECK 2] Testing Local WebSocket Bridge Server Lifecycle & Commands...")
    from zelyr.tools.browser_cdp import InteractWithBrowserTool
    tool = InteractWithBrowserTool()

    # 0. Verify InteractWithBrowserTool schema does NOT contain 'port' and contains required parameters
    schema = tool.schema
    properties = schema.get("properties", {})
    assert "port" not in properties, "InteractWithBrowserTool.schema must NOT contain 'port'!"
    assert "query" in properties, "InteractWithBrowserTool.schema should contain 'query'!"
    assert "url" in properties, "InteractWithBrowserTool.schema should contain 'url'!"
    assert "element_id" in properties, "InteractWithBrowserTool.schema should contain 'element_id'!"
    assert "submit" in properties, "InteractWithBrowserTool.schema should contain 'submit'!"
    print("  → InteractWithBrowserTool.schema verified: 'port' purged, 'query', 'url', 'element_id', 'submit' present.")

    import socket
    test_port = 41144
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind(("127.0.0.1", 41144))
        except OSError:
            test_port = 41145

    BrowserExtensionBridge.reset_instance()
    bridge = BrowserExtensionBridge.get_instance(port=test_port)
    await bridge.start_server()
    assert bridge.server is not None, f"Failed to start WebSocket server on {test_port}"

    try:
        # 1. Test offline error reporting via bridge and tool
        offline_res = await interact_with_browser("list_tabs")
        assert offline_res.success is False
        expected_msg = (
            f"No browser extension connected on ws://127.0.0.1:{bridge.port}. "
            "Ensure the Zelyr extension is loaded in Zen Browser (about:debugging) or Chromium. "
            "Do not ask for --remote-debugging-port flags."
        )
        assert offline_res.error == expected_msg, f"Expected '{expected_msg}', got '{offline_res.error}'"
        print(f"  → Offline error handled cleanly: '{offline_res.error}'")

        # Test InteractWithBrowserTool offline behavior
        tool_offline = await tool.execute({"action": "list_tabs"})
        assert tool_offline.success is False
        assert tool_offline.error == expected_msg
        print("  → InteractWithBrowserTool offline explicit error confirmed without CDP fallback.")

        # 2. Connect mock WebExtension client
        ws_uri = f"ws://{bridge.host}:{bridge.port}"
        async with websockets.connect(ws_uri) as client_ws:
            # Allow server to register client
            await asyncio.sleep(0.1)
            assert bridge.is_connected() is True
            print(f"  → Mock extension connected to {ws_uri} (active clients: {bridge.client_count})")

            # 2a. Verify heartbeat ping frames produce ZERO sys.stdout writes
            captured_stdout = io.StringIO()
            with contextlib.redirect_stdout(captured_stdout):
                await client_ws.send(json.dumps({"type": "ping", "id": "test_hb_1"}))
                pong_raw = await asyncio.wait_for(client_ws.recv(), timeout=2.0)
                pong = json.loads(pong_raw)
                assert pong.get("type") == "pong"
                assert pong.get("id") == "test_hb_1"
            assert captured_stdout.getvalue() == "", f"Heartbeat ping produced unexpected stdout: {captured_stdout.getvalue()}"
            print("  → Heartbeat ping verified: Replied with pong and produced ZERO stdout writes.")

            # Background task to emulate background.js responses
            async def mock_extension_worker():
                try:
                    while True:
                        msg_raw = await client_ws.recv()
                        msg = json.loads(msg_raw)
                        req_id = msg.get("id")
                        action = msg.get("action")

                        if action == "switch_tab":
                            query = (msg.get("query") or "").lower()
                            if query == "nonexistent":
                                reply = {
                                    "id": req_id,
                                    "success": False,
                                    "action": "switch_tab",
                                    "error": f"No open tab matching query: '{msg.get('query')}'",
                                    "open_tabs": [
                                        {"id": 101, "title": "Google Gemini", "url": "https://gemini.google.com"},
                                        {"id": 102, "title": "Monkeytype", "url": "https://monkeytype.com"},
                                    ],
                                }
                            else:
                                reply = {
                                    "id": req_id,
                                    "success": True,
                                    "action": "switch_tab",
                                    "tab": {
                                        "id": 101,
                                        "title": "Google Gemini - AI Chat",
                                        "url": "https://gemini.google.com/app",
                                    },
                                }
                        elif action == "open_tab":
                            reply = {
                                "id": req_id,
                                "success": True,
                                "action": "open_tab",
                                "tab_id": 103,
                                "title": "Hacker News",
                                "url": msg.get("url") or "https://news.ycombinator.com",
                            }
                        elif action == "get_active_tab":
                            reply = {
                                "id": req_id,
                                "success": True,
                                "action": "get_active_tab",
                                "tab_id": 101,
                                "title": "Google Gemini - AI Chat",
                                "url": "https://gemini.google.com/app",
                                "active": True,
                            }
                        elif action == "close_tab":
                            reply = {
                                "id": req_id,
                                "success": True,
                                "action": "close_tab",
                                "closed_tab_id": 102,
                                "title": "Monkeytype",
                                "url": "https://monkeytype.com",
                            }
                        elif action == "get_page_snapshot":
                            reply = {
                                "id": req_id,
                                "success": True,
                                "action": "get_page_snapshot",
                                "title": "GitHub - Repository",
                                "url": "https://github.com/project",
                                "elements": [
                                    {"id": 1, "tag": "input", "type": "text", "placeholder": "Search or jump to...", "value": ""},
                                    {"id": 2, "tag": "button", "text": "New"},
                                    {"id": 3, "tag": "a", "text": "Pull requests", "href": "/pulls"},
                                ],
                                "summary": (
                                    '[1] <input type="text" placeholder="Search or jump to...">\n'
                                    '[2] <button> "New"\n'
                                    '[3] <a (href: /pulls)> "Pull requests"'
                                ),
                                "count": 3,
                            }
                        elif action == "click_element":
                            elem_id = msg.get("element_id") or 2
                            reply = {
                                "id": req_id,
                                "success": True,
                                "action": "click_element",
                                "clicked_id": elem_id,
                                "tag": "BUTTON",
                                "text": "New",
                            }
                        elif action == "fill_element":
                            elem_id = msg.get("element_id") or 1
                            reply = {
                                "id": req_id,
                                "success": True,
                                "action": "fill_element",
                                "element_id": elem_id,
                                "text": msg.get("text") or "",
                                "submitted": bool(msg.get("submit", False)),
                            }
                        elif action == "list_tabs":
                            reply = {
                                "id": req_id,
                                "success": True,
                                "action": "list_tabs",
                                "tabs": [
                                    {"id": 101, "title": "Google Gemini", "url": "https://gemini.google.com", "active": True},
                                    {"id": 102, "title": "Monkeytype", "url": "https://monkeytype.com", "active": False},
                                ],
                            }
                        elif action == "click":
                            reply = {
                                "id": req_id,
                                "success": True,
                                "action": "click",
                                "result": f"Clicked <button> matching '{msg.get('selector')}'",
                            }
                        elif action == "type":
                            reply = {
                                "id": req_id,
                                "success": True,
                                "action": "type",
                                "result": f"Typed '{msg.get('text')}' into '{msg.get('selector')}'",
                            }
                        elif action == "get_dom":
                            reply = {
                                "id": req_id,
                                "success": True,
                                "action": "get_dom",
                                "title": "Google Gemini",
                                "url": "https://gemini.google.com",
                                "content": "Welcome to Gemini. How can I help you today?",
                            }
                        else:
                            reply = {"id": req_id, "success": False, "error": f"Unknown {action}"}

                        await client_ws.send(json.dumps(reply))
                except (asyncio.CancelledError, websockets.exceptions.ConnectionClosed):
                    pass

            worker_task = asyncio.create_task(mock_extension_worker())

            try:
                # 3. Test switch_tab (Successful match)
                res_switch = await interact_with_browser("switch_tab", query="gemini")
                assert res_switch.success is True, f"switch_tab failed: {res_switch.error}"
                assert res_switch["tab"]["title"] == "Google Gemini - AI Chat"
                print(f"  → switch_tab roundtrip verified: Focused '{res_switch['tab']['title']}'")

                # 4. Test switch_tab (Unmatched fallback with open_tabs)
                res_unmatched = await interact_with_browser("switch_tab", query="nonexistent")
                assert res_unmatched.success is False
                assert "No open tab matching query" in res_unmatched.error
                assert "open_tabs" in res_unmatched
                print(f"  → switch_tab unmatched fallback verified: {res_unmatched.error}")

                # 5. Test open_tab
                res_open = await interact_with_browser("open_tab", url="https://news.ycombinator.com")
                assert res_open.success is True, f"open_tab failed: {res_open.error}"
                assert res_open.get("tab_id") == 103
                assert "news.ycombinator.com" in res_open.get("url")
                print(f"  → open_tab roundtrip verified: Tab {res_open.get('tab_id')} at {res_open.get('url')}")

                # 6. Test get_active_tab
                res_active = await interact_with_browser("get_active_tab")
                assert res_active.success is True, f"get_active_tab failed: {res_active.error}"
                assert res_active.get("tab_id") == 101
                assert res_active.get("title") == "Google Gemini - AI Chat"
                print(f"  → get_active_tab roundtrip verified: Tab {res_active.get('tab_id')} ('{res_active.get('title')}')")

                # 7. Test close_tab
                res_close = await interact_with_browser("close_tab", query="monkeytype")
                assert res_close.success is True, f"close_tab failed: {res_close.error}"
                assert res_close.get("closed_tab_id") == 102
                print(f"  → close_tab roundtrip verified: Closed tab {res_close.get('closed_tab_id')}")

                # 8. Test get_page_snapshot
                res_snap = await interact_with_browser("get_page_snapshot")
                assert res_snap.success is True, f"get_page_snapshot failed: {res_snap.error}"
                assert len(res_snap.get("elements", [])) == 3
                assert "[1] <input" in res_snap.get("summary", "")
                print(f"  → get_page_snapshot roundtrip verified: {res_snap.get('count')} elements scanned.")

                # 9. Test click_element
                res_click_elem = await interact_with_browser("click_element", element_id=2)
                assert res_click_elem.success is True, f"click_element failed: {res_click_elem.error}"
                assert res_click_elem.get("clicked_id") == 2
                assert res_click_elem.get("tag") == "BUTTON"
                print(f"  → click_element roundtrip verified: Clicked #{res_click_elem.get('clicked_id')}")

                # 10. Test fill_element
                res_fill = await interact_with_browser("fill_element", element_id=1, text="search query", submit=True)
                assert res_fill.success is True, f"fill_element failed: {res_fill.error}"
                assert res_fill.get("element_id") == 1
                assert res_fill.get("text") == "search query"
                assert res_fill.get("submitted") is True
                print(f"  → fill_element roundtrip verified: Filled #{res_fill.get('element_id')} (submitted: {res_fill.get('submitted')})")

                # 11. Test list_tabs
                res_tabs = await interact_with_browser("list_tabs")
                assert res_tabs.success is True
                assert len(res_tabs["tabs"]) == 2
                print(f"  → list_tabs roundtrip verified: Retrieved {len(res_tabs['tabs'])} tabs")

                # 12. Test click
                res_click = await interact_with_browser("click", selector="button.submit")
                assert res_click.success is True
                assert "Clicked" in res_click["result"]
                print(f"  → click roundtrip verified: {res_click['result']}")

                # 13. Test type
                res_type = await interact_with_browser("type", selector="input#search", text="hello zelyr")
                assert res_type.success is True
                assert "hello zelyr" in res_type["result"]
                print(f"  → type roundtrip verified: {res_type['result']}")

                # 14. Test get_dom
                res_dom = await interact_with_browser("get_dom")
                assert res_dom.success is True
                assert "Gemini" in res_dom["content"]
                print(f"  → get_dom roundtrip verified: Content length {len(res_dom['content'])} chars")

                # 15. Test InteractWithBrowserTool routes cleanly through extension when connected
                tool_res = await tool.execute({"action": "click", "selector": "button.submit"})
                assert tool_res.success is True
                assert "Clicked" in str(tool_res.output)
                print("  → InteractWithBrowserTool routed to extension bridge seamlessly when connected.")

                # 16. Verify calling execute for tab and element actions via tool
                tool_open = await tool.execute({"action": "open_tab", "url": "https://news.ycombinator.com"})
                assert tool_open.success is True
                assert tool_open.output.get("tab_id") == 103

                tool_active = await tool.execute({"action": "get_active_tab"})
                assert tool_active.success is True
                assert tool_active.output.get("tab_id") == 101

                tool_close = await tool.execute({"action": "close_tab", "query": "monkeytype"})
                assert tool_close.success is True
                assert tool_close.output.get("closed_tab_id") == 102

                tool_switch_fail = await tool.execute({"action": "switch_tab", "query": "nonexistent"})
                assert tool_switch_fail.success is False
                assert "Open tabs:" in tool_switch_fail.error
                assert "Hint: Use action='open_tab'" in tool_switch_fail.error

                tool_snap = await tool.execute({"action": "get_page_snapshot"})
                assert tool_snap.success is True
                assert "[1] <input" in str(tool_snap.output)

                tool_click_elem = await tool.execute({"action": "click_element", "element_id": 2})
                assert tool_click_elem.success is True

                tool_fill_elem = await tool.execute({"action": "fill_element", "element_id": 1, "text": "search query", "submit": True})
                assert tool_fill_elem.success is True
                print("  → InteractWithBrowserTool page snapshot & element interaction verified.")

            finally:
                worker_task.cancel()
                try:
                    await worker_task
                except asyncio.CancelledError:
                    pass

    finally:
        await bridge.stop_server()

    # 17. Verify starting and stopping bridge sequentially does not raise [Errno 98]
    print("  → Testing sequential start/stop cycles for [Errno 98] re-bind resilience...")
    for cycle in range(3):
        await bridge.start_server()
        assert bridge.server is not None, f"Failed to restart bridge server on cycle {cycle + 1}"
        await bridge.stop_server()
        assert bridge.server is None
    print("  → Sequential start/stop cycles passed without [Errno 98].")
    BrowserExtensionBridge.reset_instance()

    print("✓ Check 2 PASSED: End-to-end WebSocket bridge roundtrip and lifecycle fully operational.\n")


def test_main():
    test_extension_scaffold()
    asyncio.run(run_bridge_lifecycle_tests())
    print("============================================================")
    print("✅ ALL BROWSER EXTENSION & WEBSOCKET BRIDGE CHECKS PASSED!")
    print("============================================================")


if __name__ == "__main__":
    test_main()
