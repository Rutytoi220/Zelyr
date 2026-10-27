"""Semantic Browser CDP Bridge Tool for Zelyr.

Provides deterministic sub-10ms browser automation via Chrome DevTools Protocol (CDP) WebSocket.
Compatible with Chrome, Chromium, Brave, Zen Browser, and Electron apps.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional

import httpx
import websockets

from zelyr.tools.core import BaseTool, ToolParameter, ToolResult

logger = logging.getLogger(__name__)

DEFAULT_CDP_PORTS = [9222, 9223, 9224]


class InteractWithBrowserTool(BaseTool):
    """Semantic browser automation using the Zelyr WebExtension bridge & Chrome DevTools Protocol."""

    def __init__(self):
        super().__init__(
            tool_id="interact_with_browser",
            name="interact_with_browser",
            description=(
                "Tier 2 Semantic UI Automation: Browser and tab automation via Zelyr WebExtension bridge. "
                "Use 'click_element' with 'element_id' from the latest get_page_snapshot to click buttons, links, or controls on the page. "
                "Actions: 'navigate_url' (navigate active tab in-place), 'capture_tab_screenshot' (capture PNG of viewport), "
                "'evaluate_script' (execute JS expression in active tab), 'get_page_snapshot' (list numbered interactive elements), "
                "'click_element' (click by element_id), 'fill_element' (type text into element_id or select dropdown/checkbox), "
                "'scroll_page' (scroll down/up/top/bottom), 'extract_page_content' (extract clean markdown content up to 4000 chars), "
                "'switch_tab' (focus tab by query), 'open_tab' (open url), 'close_tab' (close tab), "
                "'duplicate_tab', 'reload_tab', 'pin_tab', 'get_active_tab', 'list_tabs', 'click' (DOM selector), 'type', 'get_dom'."
            ),
            tier=2,
            is_core=False,
            requires_bridge=True,
            requires_window=True,
            token_cost=250,
        )
        self.parameters = [
            ToolParameter(
                name="action",
                type="string",
                description=(
                    "The browser action: 'navigate_url', 'capture_tab_screenshot', 'evaluate_script', 'get_page_snapshot', 'click_element', "
                    "'fill_element', 'scroll_page', 'extract_page_content', 'switch_tab', 'open_tab', 'close_tab', 'duplicate_tab', "
                    "'reload_tab', 'pin_tab', 'get_active_tab', 'list_tabs', 'click', 'type', 'get_dom'."
                ),
                required=True,
            ),
            ToolParameter(
                name="element_id",
                type="integer",
                description="Numeric snapshot ID of the interactive element from get_page_snapshot (used with click_element or fill_element).",
                required=False,
                default=None,
            ),
            ToolParameter(
                name="submit",
                type="boolean",
                description="Whether to submit the form after filling the element (used with fill_element).",
                required=False,
                default=False,
            ),
            ToolParameter(
                name="direction",
                type="string",
                description="Scroll direction when action is 'scroll_page': 'down', 'up', 'top', or 'bottom' (default: 'down').",
                required=False,
                default="down",
            ),
            ToolParameter(
                name="amount",
                type="integer",
                description="Scroll offset in pixels when action is 'scroll_page' (default: 600).",
                required=False,
                default=600,
            ),
            ToolParameter(
                name="mode",
                type="string",
                description="Content extraction mode when action is 'extract_page_content': 'readable' or 'markdown' (default: 'readable').",
                required=False,
                default="readable",
            ),
            ToolParameter(
                name="pinned",
                type="boolean",
                description="Pin state when action is 'pin_tab': true to pin, false to unpin.",
                required=False,
                default=None,
            ),
            ToolParameter(
                name="query",
                type="string",
                description="Tab title, domain, or URL query to search and switch to when action is 'switch_tab' or 'close_tab'.",
                required=False,
                default="",
            ),
            ToolParameter(
                name="url",
                type="string",
                description="Target URL to open when action is 'open_tab'.",
                required=False,
                default="",
            ),
            ToolParameter(
                name="selector",
                type="string",
                description="CSS selector of the DOM element to interact with inside a web page (e.g. 'button.submit', '#search-input').",
                required=False,
                default="",
            ),
            ToolParameter(
                name="text",
                type="string",
                description="Text to type into the targeted DOM element.",
                required=False,
                default="",
            ),
            ToolParameter(
                name="expression",
                type="string",
                description="JavaScript code expression to evaluate in the page context.",
                required=False,
                default="",
            ),
            ToolParameter(
                name="tab_id",
                type="string",
                description="Optional tab ID or URL filter to select a specific browser tab.",
                required=False,
                default="",
            ),
        ]

    async def _discover_tab(self, requested_port: int, tab_filter: str = "") -> tuple[Optional[str], Optional[dict], Optional[str]]:
        """Finds active CDP WebSocket debugger URL across candidate ports."""
        ports_to_try = [requested_port] + [p for p in DEFAULT_CDP_PORTS if p != requested_port]
        last_error = ""

        async with httpx.AsyncClient(timeout=1.5) as client:
            for port in ports_to_try:
                url = f"http://127.0.0.1:{port}/json"
                try:
                    resp = await client.get(url)
                    if resp.status_code == 200:
                        targets: List[dict] = resp.json()
                        page_targets = [t for t in targets if t.get("type") == "page"]
                        if not page_targets:
                            page_targets = targets

                        if tab_filter:
                            matching = [
                                t for t in page_targets
                                if tab_filter.lower() in t.get("url", "").lower()
                                or tab_filter.lower() in t.get("title", "").lower()
                                or tab_filter == t.get("id")
                            ]
                            if matching:
                                chosen = matching[0]
                                return chosen.get("webSocketDebuggerUrl"), chosen, None

                        if page_targets:
                            chosen = page_targets[0]
                            return chosen.get("webSocketDebuggerUrl"), chosen, None
                except Exception as e:
                    last_error = str(e)
                    continue

        return None, None, f"Could not connect to browser CDP port ({', '.join(str(p) for p in ports_to_try)}). {last_error}"

    async def _send_cdp(self, ws_url: str, method: str, params: Optional[dict] = None) -> dict:
        """Connects to CDP WebSocket and dispatches a JSON-RPC command."""
        req_id = 1
        payload = {
            "id": req_id,
            "method": method,
            "params": params or {},
        }
        async with websockets.connect(ws_url, ping_interval=None, close_timeout=2.0) as ws:
            await ws.send(json.dumps(payload))
            while True:
                raw_resp = await asyncio.wait_for(ws.recv(), timeout=5.0)
                resp = json.loads(raw_resp)
                if resp.get("id") == req_id:
                    return resp

    async def execute(self, params: Optional[Dict[str, Any]] = None, **kwargs: Any) -> ToolResult:
        if params is None:
            params = {}
        if kwargs:
            params = {**params, **kwargs}
        action = params.get("action", "").strip().lower()
        selector = str(params.get("selector", "") or "").strip()
        text = str(params.get("text", "") or "")
        expression = str(params.get("expression", "") or "").strip()
        tab_id = str(params.get("tab_id", "") or "").strip()

        # Only route to legacy CDP if explicitly requested by unit test fixtures via internal keyword argument _force_cdp=True
        force_cdp = bool(params.get("_force_cdp", False))

        # Tier 2 WebExtension Bridge is primary for browser automation (100% default).
        if not force_cdp:
            try:
                from zelyr.tools.browser_extension import get_bridge
                bridge = get_bridge()
                if bridge.server is None:
                    try:
                        await bridge.ensure_server()
                    except Exception:
                        pass

                if bridge.is_connected():
                    ext_params = {k: v for k, v in params.items() if k not in ("action", "_force_cdp")}
                    ext_res = await bridge.send_command(action, **ext_params)
                    if ext_res.success:
                        if action == "capture_tab_screenshot":
                            import base64
                            from pathlib import Path
                            raw_data = ext_res.get("data_url") or ext_res.get("data") or ""
                            if "," in raw_data:
                                b64_bytes = raw_data.split(",", 1)[1]
                            else:
                                b64_bytes = raw_data
                            screenshot_path = "/tmp/zelyr_tab_capture.png"
                            try:
                                if b64_bytes:
                                    img_bytes = base64.b64decode(b64_bytes)
                                    Path(screenshot_path).write_bytes(img_bytes)
                                return ToolResult(
                                    True,
                                    output={
                                        "screenshot_path": screenshot_path,
                                        "format": ext_res.get("format", "png"),
                                        "length": ext_res.get("length", len(raw_data)),
                                    },
                                )
                            except Exception as save_err:
                                return ToolResult(False, error=f"Failed saving screenshot: {save_err}")
                        return ToolResult(True, output=ext_res.output or ext_res)
                    error_msg = ext_res.error or "Extension command failed"
                    remedy_hint = ext_res.remedy_hint if hasattr(ext_res, "remedy_hint") else ext_res.get("remedy_hint")
                    allowed_actions = ext_res.allowed_actions if hasattr(ext_res, "allowed_actions") else (ext_res.get("allowed_actions") or [])

                    if action in ("click_element", "fill_element"):
                        elem_id = params.get("element_id")
                        if "ambiguous" in error_msg.lower():
                            if not remedy_hint:
                                remedy_hint = "Call get_page_snapshot to re-index unique element IDs."
                            if not allowed_actions:
                                allowed_actions = ["get_page_snapshot"]
                        else:
                            if elem_id is not None and ("not found" in error_msg.lower() or not ext_res.error or "Element not found" in error_msg):
                                error_msg = f"Element ID {elem_id} not found on page."
                            if not remedy_hint:
                                remedy_hint = "Call get_page_snapshot to refresh element IDs, or scroll_page down if the element is below the viewport."
                            if not allowed_actions:
                                allowed_actions = ["get_page_snapshot", "scroll_page"]
                    elif action == "switch_tab":
                        target_url = params.get("query") or params.get("url") or "<url>"
                        if "open_tabs" in ext_res:
                            tabs_summary = ", ".join([f"'{t.get('title') or t.get('url')}'" for t in ext_res["open_tabs"]])
                            error_msg = f"{error_msg}. Open tabs: [{tabs_summary}]."
                        if not remedy_hint:
                            remedy_hint = f"Call list_tabs to see valid targets, or open_tab with url='{target_url}' to launch it."
                        if not allowed_actions:
                            allowed_actions = ["list_tabs", "open_tab"]

                    return ToolResult(
                        False,
                        error=error_msg,
                        remedy_hint=remedy_hint,
                        allowed_actions=allowed_actions,
                        metadata={"open_tabs": ext_res.get("open_tabs", [])} if "open_tabs" in ext_res else None,
                    )
                else:
                    elem_id = params.get("element_id")
                    target_url = params.get("query") or params.get("url") or "<url>"
                    base_err = (
                        f"No browser extension connected on ws://{bridge.host}:{bridge.port}. "
                        "Ensure the Zelyr extension is loaded in Zen Browser (about:debugging) or Chromium. "
                        "Do not ask for --remote-debugging-port flags."
                    )
                    if action in ("click_element", "fill_element"):
                        return ToolResult(
                            False,
                            error=base_err,
                            remedy_hint="Call get_page_snapshot to refresh element IDs, or scroll_page down if the element is below the viewport.",
                            allowed_actions=["get_page_snapshot", "scroll_page"],
                        )
                    elif action == "switch_tab":
                        return ToolResult(
                            False,
                            error=base_err,
                            remedy_hint=f"Call list_tabs to see valid targets, or open_tab with url='{target_url}' to launch it.",
                            allowed_actions=["list_tabs", "open_tab"],
                        )
                    return ToolResult(
                        False,
                        error=base_err,
                        remedy_hint="Ensure the browser extension is running or use Tier 1 execute_command or manage_desktop_window.",
                        allowed_actions=["execute_command", "manage_desktop_window"],
                    )
            except ConnectionResetError:
                return ToolResult(
                    False,
                    error="Browser bridge connection lost mid-command. The browser or tab may have closed or crashed.",
                    remedy_hint="Verify browser window is running and focused, then retry get_page_snapshot.",
                    allowed_actions=["get_page_snapshot"],
                )
            except Exception as exc:
                elem_id = params.get("element_id")
                target_url = params.get("query") or params.get("url") or "<url>"
                base_err = (
                    f"Browser extension bridge error: {exc}. "
                    f"No browser extension connected on ws://{bridge.host}:{bridge.port}. "
                    "Ensure the Zelyr extension is loaded in Zen Browser (about:debugging) or Chromium. "
                    "Do not ask for --remote-debugging-port flags."
                )
                if action in ("click_element", "fill_element"):
                    return ToolResult(
                        False,
                        error=base_err,
                        remedy_hint="Call get_page_snapshot to refresh element IDs, or scroll_page down if the element is below the viewport.",
                        allowed_actions=["get_page_snapshot", "scroll_page"],
                    )
                elif action == "switch_tab":
                    return ToolResult(
                        False,
                        error=base_err,
                        remedy_hint=f"Call list_tabs to see valid targets, or open_tab with url='{target_url}' to launch it.",
                        allowed_actions=["list_tabs", "open_tab"],
                    )
                return ToolResult(
                    False,
                    error=base_err,
                    remedy_hint="Ensure the browser extension is running or use Tier 1 execute_command or manage_desktop_window.",
                    allowed_actions=["execute_command", "manage_desktop_window"],
                )

        # Legacy CDP path (strictly exercised only when _force_cdp=True)
        port = int(params.get("port") or 9222)

        # Handle list_tabs discovery
        if action in ("list_tabs", "tabs"):
            ports_to_try = [port] + [p for p in DEFAULT_CDP_PORTS if p != port]
            async with httpx.AsyncClient(timeout=1.5) as client:
                for p in ports_to_try:
                    try:
                        resp = await client.get(f"http://127.0.0.1:{p}/json")
                        if resp.status_code == 200:
                            targets = resp.json()
                            summary = [
                                {
                                    "id": t.get("id"),
                                    "title": t.get("title"),
                                    "url": t.get("url"),
                                    "type": t.get("type"),
                                }
                                for t in targets
                            ]
                            return ToolResult(True, output={"port": p, "tabs": summary})
                    except Exception:
                        continue
            return ToolResult(
                False,
                error="No running browser found on CDP debug ports (9222, 9223). Start browser with '--remote-debugging-port=9222'.",
            )

        # Discover active page WebSocket
        ws_url, page_info, err = await self._discover_tab(port, tab_id)
        if not ws_url or err:
            return ToolResult(
                False,
                error=(
                    "CDP Connection Failed: Browser is not running with remote debugging enabled. "
                    "Ensure Zen Browser, Chrome, or Brave is launched with '--remote-debugging-port=9222'."
                ),
            )

        try:
            if action == "click":
                if not selector:
                    return ToolResult(False, error="CSS selector is required for 'click' action.")

                js_click = f"""
                (() => {{
                    const el = document.querySelector({json.dumps(selector)});
                    if (!el) return {{ success: false, error: "Element not found for selector: " + {json.dumps(selector)} }};
                    el.scrollIntoView({{ block: "center", inline: "center", behavior: "instant" }});
                    el.focus();
                    el.click();
                    el.dispatchEvent(new MouseEvent('mousedown', {{ bubbles: true, cancelable: true, view: window }}));
                    el.dispatchEvent(new MouseEvent('mouseup', {{ bubbles: true, cancelable: true, view: window }}));
                    return {{
                        success: true,
                        tag: el.tagName,
                        text: (el.innerText || el.value || "").slice(0, 100),
                        selector: {json.dumps(selector)}
                    }};
                }})()
                """
                resp = await self._send_cdp(ws_url, "Runtime.evaluate", {
                    "expression": js_click,
                    "returnByValue": True,
                    "awaitPromise": True,
                })
                result = resp.get("result", {}).get("result", {}).get("value", {})
                if isinstance(result, dict) and not result.get("success", False):
                    return ToolResult(False, error=result.get("error", "Click failed."))
                return ToolResult(True, output=result)

            elif action == "type":
                if not selector:
                    return ToolResult(False, error="CSS selector is required for 'type' action.")

                js_type = f"""
                (() => {{
                    const el = document.querySelector({json.dumps(selector)});
                    if (!el) return {{ success: false, error: "Element not found for selector: " + {json.dumps(selector)} }};
                    el.scrollIntoView({{ block: "center", inline: "center", behavior: "instant" }});
                    el.focus();
                    if ('value' in el) {{
                        el.value = {json.dumps(text)};
                        el.dispatchEvent(new Event('input', {{ bubbles: true }}));
                        el.dispatchEvent(new Event('change', {{ bubbles: true }}));
                    }} else {{
                        el.textContent = {json.dumps(text)};
                        el.dispatchEvent(new InputEvent('input', {{ bubbles: true, data: {json.dumps(text)} }}));
                    }}
                    return {{
                        success: true,
                        selector: {json.dumps(selector)},
                        typed: {json.dumps(text)}
                    }};
                }})()
                """
                resp = await self._send_cdp(ws_url, "Runtime.evaluate", {
                    "expression": js_type,
                    "returnByValue": True,
                    "awaitPromise": True,
                })
                result = resp.get("result", {}).get("result", {}).get("value", {})
                if isinstance(result, dict) and not result.get("success", False):
                    return ToolResult(False, error=result.get("error", "Type failed."))
                return ToolResult(True, output=result)

            elif action in ("evaluate", "js", "exec"):
                if not expression:
                    return ToolResult(False, error="JavaScript expression is required for 'evaluate' action.")

                resp = await self._send_cdp(ws_url, "Runtime.evaluate", {
                    "expression": expression,
                    "returnByValue": True,
                    "awaitPromise": True,
                })
                res_obj = resp.get("result", {}).get("result", {})
                if "value" in res_obj:
                    return ToolResult(True, output=res_obj["value"])
                return ToolResult(True, output=res_obj)

            elif action in ("get_content", "read_page", "content"):
                js_content = """
                (() => {
                    return {
                        title: document.title,
                        url: location.href,
                        body_text: (document.body ? document.body.innerText : "").slice(0, 3000)
                    };
                })()
                """
                resp = await self._send_cdp(ws_url, "Runtime.evaluate", {
                    "expression": js_content,
                    "returnByValue": True,
                    "awaitPromise": True,
                })
                result = resp.get("result", {}).get("result", {}).get("value", {})
                return ToolResult(True, output=result)

            else:
                return ToolResult(
                    False,
                    error=f"Unknown browser action '{action}'. Supported actions: click, type, evaluate, get_content, list_tabs.",
                )

        except Exception as exc:
            logger.exception("Error executing CDP command")
            return ToolResult(False, error=f"CDP command execution error: {exc}")
