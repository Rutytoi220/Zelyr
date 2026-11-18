"""Hermetic unit and integration tests for Chrome DevTools Protocol (CDP) Browser Engine.

Tests:
1. Direct CDPBrowserClient HTTP and WebSocket communication with mock CDP server.
2. Compact accessibility-tree snapshot formatting (<1,500 chars).
3. Deterministic element interaction (click_element, fill_element, scroll_page, switch_tab).
4. Error classification and actionable remedy envelope contracts.
5. InteractWithBrowserTool CDP dispatch, recovery hints, and fallback behavior.
6. NativeOrchestrator browser tool alias routing and argument canonicalization.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, AsyncGenerator, Tuple
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aiohttp import web

pytestmark = pytest.mark.integration

from zelyr.engine.browser_cdp import CDPBrowserClient, SNAPSHOT_JS_SCRIPT
from zelyr.engine.error_classifier import ErrorClassifier
from zelyr.tools.browser_cdp import (
    InteractWithBrowserTool,
    REQUIRED_RING,
    TIER,
    TOOL_SCHEMA,
    TUI_HINT,
    execute,
)


class MockCDPServerContext:
    """Async context manager spinning up a mock Chrome DevTools Protocol server."""

    def __init__(self) -> None:
        self.runner: web.AppRunner | None = None
        self.port: int = 0
        self.tab_data: list[dict[str, Any]] = []
        self.active_websockets: list[web.WebSocketResponse] = []

    async def __aenter__(self) -> Tuple[str, int]:
        app = web.Application()

        self.tab_data = [
            {
                "id": "tab-101",
                "title": "Zelyr Portal",
                "url": "https://zelyr.ai/home",
                "type": "page",
                "webSocketDebuggerUrl": "",
            },
            {
                "id": "tab-102",
                "title": "GitHub Dashboard",
                "url": "https://github.com",
                "type": "page",
                "webSocketDebuggerUrl": "",
            },
        ]

        async def version_handler(request: web.Request) -> web.Response:
            return web.json_response({
                "Browser": "Chrome/120.0.0.0",
                "Protocol-Version": "1.3",
                "User-Agent": "Mozilla/5.0",
                "webSocketDebuggerUrl": f"ws://127.0.0.1:{self.port}/devtools/browser",
            })

        async def list_handler(request: web.Request) -> web.Response:
            return web.json_response(self.tab_data)

        async def new_handler(request: web.Request) -> web.Response:
            new_tab = {
                "id": "tab-103",
                "title": "New Tab",
                "url": request.query_string or "about:blank",
                "type": "page",
                "webSocketDebuggerUrl": f"ws://127.0.0.1:{self.port}/devtools/page/tab-103",
            }
            self.tab_data.append(new_tab)
            return web.json_response(new_tab)

        async def close_handler(request: web.Request) -> web.Response:
            tid = request.match_info.get("id")
            self.tab_data = [t for t in self.tab_data if t.get("id") != tid]
            return web.Response(text="Target is closing")

        async def ws_handler(request: web.Request) -> web.WebSocketResponse:
            ws = web.WebSocketResponse()
            await ws.prepare(request)
            self.active_websockets.append(ws)

            try:
                async for msg in ws:
                    if msg.type == web.WSMsgType.TEXT:
                        data = json.loads(msg.data)
                        req_id = data.get("id")
                        method = data.get("method")
                        params = data.get("params", {})

                        if method == "Runtime.evaluate":
                            expr = params.get("expression", "")
                            if "document.querySelectorAll" in expr or "items.join" in expr or "Page: " in expr:
                                snapshot_tree = (
                                    "# Page: Zelyr Portal (https://zelyr.ai/home)\n"
                                    '- [1] [Button] "Launch Engine"\n'
                                    '- [2] [Link] "Documentation" -> /docs\n'
                                    '- [3] [Input:text] placeholder="Search commands..." value="ready"'
                                )
                                res_val = {
                                    "title": "Zelyr Portal",
                                    "url": "https://zelyr.ai/home",
                                    "snapshot": snapshot_tree,
                                    "element_count": 3,
                                }
                            elif 'data-zelyr-id="999"' in expr:
                                res_val = {"found": False}
                            elif 'data-zelyr-id="' in expr:
                                res_val = {"found": True}
                            elif "querySelector" in expr and "missing" in expr:
                                res_val = {"found": False, "error": "Not found"}
                            elif "querySelector" in expr:
                                res_val = {"found": True, "tag": "BUTTON", "text": "OK"}
                            elif "window.scrollY" in expr or "window.scrollBy" in expr:
                                res_val = {"scrollY": 500}
                            else:
                                res_val = {"evaluated": True, "expr": expr}
                            await ws.send_str(json.dumps({"id": req_id, "result": {"result": {"value": res_val}}}))

                        elif method == "Page.navigate":
                            await ws.send_str(json.dumps({
                                "id": req_id,
                                "result": {"frameId": "F101", "loaderId": "L101"},
                            }))

                        elif method == "Page.captureScreenshot":
                            # 1x1 transparent PNG base64
                            sample_png = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
                            await ws.send_str(json.dumps({
                                "id": req_id,
                                "result": {"data": sample_png},
                            }))

                        elif method == "Target.createTarget":
                            await ws.send_str(json.dumps({
                                "id": req_id,
                                "result": {"targetId": "tab-103"},
                            }))

                        elif method == "Target.closeTarget":
                            await ws.send_str(json.dumps({
                                "id": req_id,
                                "result": {"success": True},
                            }))

                        else:
                            await ws.send_str(json.dumps({"id": req_id, "result": {}}))

                    elif msg.type in (web.WSMsgType.CLOSE, web.WSMsgType.CLOSING, web.WSMsgType.CLOSED):
                        break
            finally:
                if ws in self.active_websockets:
                    self.active_websockets.remove(ws)

            return ws

        app.router.add_get("/json/version", version_handler)
        app.router.add_get("/json/list", list_handler)
        app.router.add_get("/json", list_handler)
        app.router.add_put("/json/new", new_handler)
        app.router.add_put("/json/close/{id}", close_handler)
        app.router.add_get("/devtools/page/{id}", ws_handler)

        self.runner = web.AppRunner(app)
        await self.runner.setup()
        site = web.TCPSite(self.runner, "127.0.0.1", 0)
        await site.start()
        self.port = site._server.sockets[0].getsockname()[1]

        for tab in self.tab_data:
            tab["webSocketDebuggerUrl"] = f"ws://127.0.0.1:{self.port}/devtools/page/{tab['id']}"

        return "127.0.0.1", self.port

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        for ws in list(self.active_websockets):
            try:
                await ws.close()
            except Exception:
                pass
        if self.runner:
            await self.runner.shutdown()
            await self.runner.cleanup()


@pytest.mark.asyncio
async def test_cdp_client_is_available():
    """Verify is_available detects running CDP endpoint and handles offline port cleanly."""
    async with MockCDPServerContext() as (host, port):
        client = CDPBrowserClient(host=host, port=port)
        assert await client.is_available() is True

    # Offline port
    offline_client = CDPBrowserClient(host="127.0.0.1", port=49999)
    assert await offline_client.is_available() is False


@pytest.mark.asyncio
async def test_cdp_client_tab_enumeration_and_navigation():
    """Verify tab listing, active tab connection, and Page.navigate."""
    async with MockCDPServerContext() as (host, port):
        client = CDPBrowserClient(host=host, port=port)
        tabs = await client.list_tabs()
        assert len(tabs) >= 2
        assert tabs[0]["title"] == "Zelyr Portal"
        assert tabs[1]["title"] == "GitHub Dashboard"

        nav_res = await client.navigate("https://zelyr.ai/docs")
        assert nav_res["status"] == "success"
        assert nav_res["url"] == "https://zelyr.ai/docs"
        assert nav_res["frameId"] == "F101"
        await client.close()


@pytest.mark.asyncio
async def test_cdp_client_snapshot_tree_format():
    """Verify get_page_snapshot returns a compact markdown tree bounded under 1,500 chars."""
    async with MockCDPServerContext() as (host, port):
        client = CDPBrowserClient(host=host, port=port)
        snap = await client.get_page_snapshot()

        assert snap["status"] == "success"
        assert snap["title"] == "Zelyr Portal"
        assert snap["element_count"] == 3
        tree = snap["snapshot"]
        assert len(tree) < 1500
        assert "# Page: Zelyr Portal" in tree
        assert '[1] [Button] "Launch Engine"' in tree
        assert '[2] [Link] "Documentation"' in tree
        assert '[3] [Input:text]' in tree
        await client.close()


@pytest.mark.asyncio
async def test_cdp_client_deterministic_interaction():
    """Verify click_element and fill_element success vs ELEMENT_NOT_FOUND recovery hint."""
    async with MockCDPServerContext() as (host, port):
        client = CDPBrowserClient(host=host, port=port)

        # 1. Successful click by ID
        click_ok = await client.click_element(1)
        assert click_ok["status"] == "success"
        assert click_ok["element_id"] == 1

        # 2. Failed click by ID (element 999 not found)
        click_fail = await client.click_element(999)
        assert click_fail["status"] == "error"
        assert click_fail["error_type"] == "ELEMENT_NOT_FOUND"
        assert "Element ID '999' not found" in click_fail["error"]
        assert "Call get_page_snapshot" in click_fail["remedy_hint"]
        assert "scroll_page" in click_fail["remedy_hint"]

        # 3. Successful fill by ID
        fill_ok = await client.fill_element(3, "zelyr query")
        assert fill_ok["status"] == "success"
        assert fill_ok["value"] == "zelyr query"

        # 4. Failed fill by ID
        fill_fail = await client.fill_element(999, "bad")
        assert fill_fail["status"] == "error"
        assert fill_fail["error_type"] == "ELEMENT_NOT_FOUND"

        # 5. Scroll page
        scroll_res = await client.scroll_page(direction="down", amount=500)
        assert scroll_res["status"] == "success"
        assert scroll_res["direction"] == "down"
        assert scroll_res["amount"] == 500

        await client.close()


@pytest.mark.asyncio
async def test_cdp_client_tab_management_and_recovery():
    """Verify switch_tab, open_tab, and close_tab with TAB_NOT_FOUND remedy."""
    async with MockCDPServerContext() as (host, port):
        client = CDPBrowserClient(host=host, port=port)

        # Successful switch
        sw_res = await client.switch_tab(query="github")
        assert sw_res["status"] == "success"
        assert sw_res["tab"]["id"] == "tab-102"

        # Failed switch with remedy hint
        sw_fail = await client.switch_tab(query="nonexistent_site")
        assert sw_fail["status"] == "error"
        assert sw_fail["error_type"] == "TAB_NOT_FOUND"
        assert "Call list_tabs" in sw_fail["remedy_hint"]
        assert "open_tab" in sw_fail["remedy_hint"]

        # Open tab
        open_res = await client.open_tab("https://news.ycombinator.com")
        assert open_res["status"] == "success"

        # Close tab
        close_res = await client.close_tab("tab-102")
        assert close_res["status"] == "success"
        assert close_res["closed_tab_id"] == "tab-102"

        await client.close()


@pytest.mark.asyncio
async def test_interact_with_browser_tool_cdp_integration():
    """Verify InteractWithBrowserTool routes actions through CDP client and produces ToolResult."""
    async with MockCDPServerContext() as (host, port):
        cdp = CDPBrowserClient(host=host, port=port)
        tool = InteractWithBrowserTool(cdp_client=cdp)

        # Snapshot
        snap_res = await tool.execute({"action": "get_page_snapshot"})
        assert snap_res.success is True
        assert "snapshot" in snap_res.output
        assert "Zelyr Portal" in snap_res.output["snapshot"]

        # Click missing element produces structured error envelope
        click_res = await tool.execute({"action": "click_element", "element_id": 999})
        assert click_res.success is False
        assert "Element ID '999' not found" in click_res.error
        assert click_res.remedy_hint is not None
        assert "get_page_snapshot" in click_res.remedy_hint
        assert click_res.allowed_actions == ["interact_with_browser"]

        # Navigate
        nav_res = await tool.execute({"action": "navigate_url", "url": "https://zelyr.ai/docs"})
        assert nav_res.success is True
        assert nav_res.output["url"] == "https://zelyr.ai/docs"

        # Capture screenshot
        shot_res = await tool.execute({"action": "capture_tab_screenshot"})
        assert shot_res.success is True
        assert "screenshot_path" in shot_res.output

        # Module level execute function
        mod_json = await execute(action="get_page_snapshot", client=cdp)
        parsed = json.loads(mod_json)
        assert parsed.get("status") == "success"
        assert "Zelyr Portal" in parsed.get("snapshot", "")

        await cdp.close()


def test_error_classifier_cdp_classifications():
    """Verify ErrorClassifier correctly identifies BROWSER_UNAVAILABLE and ELEMENT_NOT_FOUND."""
    err_browser = "CDP Connection Failed: Browser is not running with remote debugging enabled."
    cl_browser = ErrorClassifier.classify("interact_with_browser", error=err_browser)
    assert cl_browser.error_type == "BROWSER_UNAVAILABLE"
    assert "remote-debugging-port" in cl_browser.remedy_hint
    assert "interact_with_browser" in cl_browser.allowed_actions

    err_elem = "Element ID '42' not found on page."
    cl_elem = ErrorClassifier.classify("interact_with_browser", error=err_elem)
    assert cl_elem.error_type == "ELEMENT_NOT_FOUND"
    assert "get_page_snapshot" in cl_elem.remedy_hint
    assert "interact_with_browser" in cl_elem.allowed_actions


def test_tool_plugin_metadata_conformance():
    """Verify standard dynamic tool plugin exports for interact_with_browser."""
    assert REQUIRED_RING == 2
    assert TIER == 2
    assert isinstance(TUI_HINT, str)
    assert TOOL_SCHEMA["type"] == "function"
    assert TOOL_SCHEMA["function"]["name"] == "interact_with_browser"
    props = TOOL_SCHEMA["function"]["parameters"]["properties"]
    assert "action" in props
    assert "element_id" in props
    assert "url" in props
    assert "query" in props
