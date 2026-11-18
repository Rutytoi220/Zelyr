"""Direct Chrome DevTools Protocol (CDP) Asynchronous Browser Client & DOM Extractor.

Provides direct WebSocket and HTTP-based interaction with Chromium/Chrome instances via CDP:
- Tab enumeration and WebSocket debugger connection.
- Compact accessibility-tree snapshot extraction (<1,500 chars).
- Deterministic element interaction (click, fill, scroll, navigation).
- Fail-closed error handling with structured recovery hints.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
from typing import Any, Dict, List, Optional

import httpx
import websockets

logger = logging.getLogger(__name__)


def _is_ws_open(ws: Any) -> bool:
    """Check if a WebSocket connection is alive across modern websockets versions."""
    if ws is None:
        return False
    state = getattr(ws, "state", None)
    if state is not None:
        name = getattr(state, "name", str(state))
        return name == "OPEN"
    return bool(getattr(ws, "open", True) and not getattr(ws, "closed", False))


# Compact JavaScript to extract interactive accessibility-tree snapshot
SNAPSHOT_JS_SCRIPT = """
(() => {
  const elements = Array.from(document.querySelectorAll(
    'a[href], button, input, textarea, select, [role="button"], [role="link"], [onclick]'
  ));
  let count = 0;
  const items = [];
  for (const el of elements) {
    const style = window.getComputedStyle(el);
    if (style.display === 'none' || style.visibility === 'hidden') continue;
    const rect = el.getBoundingClientRect();
    if (rect.width === 0 || rect.height === 0) continue;
    if (el.offsetParent === null && style.position !== 'fixed') continue;

    count++;
    el.setAttribute('data-zelyr-id', String(count));

    let tag = el.tagName.toLowerCase();
    let role = el.getAttribute('role');
    let type = el.getAttribute('type') || '';
    let label = (
      el.getAttribute('aria-label') ||
      el.getAttribute('placeholder') ||
      el.innerText ||
      el.value ||
      ''
    ).trim().replace(/\\s+/g, ' ').slice(0, 50);

    let roleStr = 'Element';
    if (tag === 'button' || role === 'button') roleStr = 'Button';
    else if (tag === 'a' || role === 'link') roleStr = 'Link';
    else if (tag === 'input') roleStr = type ? `Input:${type}` : 'Input';
    else if (tag === 'textarea') roleStr = 'Textarea';
    else if (tag === 'select') roleStr = 'Select';

    let itemStr = `- [${count}] [${roleStr}] "${label}"`;
    if (tag === 'a' && el.getAttribute('href')) {
      itemStr += ` -> ${el.getAttribute('href')}`;
    }
    if (el.getAttribute('placeholder')) {
      itemStr += ` placeholder="${el.getAttribute('placeholder')}"`;
    }
    if (el.value !== undefined && el.value !== '') {
      itemStr += ` value="${String(el.value).slice(0, 30)}"`;
    }
    items.push(itemStr);
    if (items.length >= 40) break;
  }
  let tree = `# Page: ${document.title} (${window.location.href})\\n` + items.join('\\n');
  if (tree.length > 1450) {
    tree = tree.slice(0, 1450) + '\\n... [truncated]';
  }
  return {
    title: document.title,
    url: window.location.href,
    snapshot: tree,
    element_count: count
  };
})()
"""


class CDPBrowserClient:
    """Asynchronous client communicating directly with Chrome DevTools Protocol."""

    def __init__(self, host: str = "127.0.0.1", port: int = 9222, timeout: float = 3.0):
        self.host = host
        self.port = port
        self.timeout = timeout
        self._ws: Any = None
        self._msg_id = 0
        self._active_tab: Optional[Dict[str, Any]] = None

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def is_connected(self) -> bool:
        """Check if active WebSocket client connection is currently open."""
        return _is_ws_open(self._ws)

    async def is_available(self) -> bool:
        """Check if CDP HTTP endpoint is reachable."""
        try:
            async with httpx.AsyncClient(timeout=0.5) as client:
                resp = await client.get(f"{self.base_url}/json/version")
                return resp.status_code == 200
        except Exception:
            return False

    async def list_tabs(self) -> List[Dict[str, Any]]:
        """Query http://host:port/json/list, returning page targets."""
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.get(f"{self.base_url}/json/list")
                if resp.status_code == 200:
                    targets = resp.json()
                    pages = [t for t in targets if t.get("type") in ("page", "app")]
                    return pages if pages else targets
                return []
        except Exception as exc:
            logger.debug("Failed to list CDP tabs: %s", exc)
            return []

    async def connect_tab(self, target_id: Optional[str] = None) -> Any:
        """Establish async WebSocket connection to target page tab."""
        tabs = await self.list_tabs()
        chosen = None
        if target_id:
            for t in tabs:
                if t.get("id") == target_id:
                    chosen = t
                    break

        if not chosen and tabs:
            chosen = tabs[0]

        if not chosen:
            # Attempt to create a new tab if none exist
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    resp = await client.put(f"{self.base_url}/json/new")
                    if resp.status_code == 200:
                        chosen = resp.json()
            except Exception:
                pass

        if not chosen or not chosen.get("webSocketDebuggerUrl"):
            raise RuntimeError(f"No active browser page tabs found at {self.base_url}.")

        if self._ws:
            try:
                await self._ws.close()
            except Exception:
                pass

        self._active_tab = chosen
        ws_url = chosen["webSocketDebuggerUrl"]
        self._ws = await asyncio.wait_for(
            websockets.connect(ws_url),
            timeout=self.timeout,
        )
        return self._ws

    async def send_cdp_command(
        self,
        method: str,
        params: Optional[Dict[str, Any]] = None,
        timeout: float = 5.0,
    ) -> Dict[str, Any]:
        """Send a JSON-RPC CDP command over the WebSocket connection and await result."""
        if not self.is_connected():
            await self.connect_tab(self._active_tab.get("id") if self._active_tab else None)

        self._msg_id += 1
        req_id = self._msg_id
        payload = {
            "id": req_id,
            "method": method,
            "params": params or {},
        }
        await self._ws.send(json.dumps(payload))

        end_time = asyncio.get_running_loop().time() + timeout
        while True:
            remaining = end_time - asyncio.get_running_loop().time()
            if remaining <= 0:
                raise asyncio.TimeoutError(f"CDP command '{method}' timed out after {timeout}s.")
            raw_msg = await asyncio.wait_for(self._ws.recv(), timeout=remaining)
            data = json.loads(raw_msg)
            if data.get("id") == req_id:
                if "error" in data:
                    raise RuntimeError(f"CDP command '{method}' failed: {data['error']}")
                return data.get("result", {})

    async def evaluate_script(self, expression: str, timeout: float = 5.0) -> Any:
        """Evaluate a JavaScript expression via Runtime.evaluate."""
        res = await self.send_cdp_command(
            "Runtime.evaluate",
            {
                "expression": expression,
                "returnByValue": True,
                "awaitPromise": True,
            },
            timeout=timeout,
        )
        result_obj = res.get("result", {})
        if res.get("exceptionDetails"):
            raise RuntimeError(f"JavaScript evaluation exception: {res.get('exceptionDetails')}")
        return result_obj.get("value")

    async def navigate(self, url: str) -> Dict[str, Any]:
        """Navigate the active tab to a URL."""
        if not self.is_connected():
            await self.connect_tab()

        target_url = url if "://" in url else f"https://{url}"
        res = await self.send_cdp_command("Page.navigate", {"url": target_url})
        return {
            "status": "success",
            "url": target_url,
            "frameId": res.get("frameId"),
        }

    async def capture_screenshot(self, format: str = "png") -> str:
        """Capture screenshot of the viewport and return base64 string."""
        if not self.is_connected():
            await self.connect_tab()

        res = await self.send_cdp_command("Page.captureScreenshot", {"format": format})
        return res.get("data", "")

    async def get_page_snapshot(self) -> Dict[str, Any]:
        """Extract interactive accessibility tree snapshot (<1,500 chars)."""
        if not self.is_connected():
            await self.connect_tab()

        val = await self.evaluate_script(SNAPSHOT_JS_SCRIPT)
        if isinstance(val, dict):
            return {
                "status": "success",
                "title": val.get("title", ""),
                "url": val.get("url", ""),
                "snapshot": val.get("snapshot", ""),
                "element_count": val.get("element_count", 0),
            }
        return {
            "status": "success",
            "title": "",
            "url": "",
            "snapshot": str(val or ""),
            "element_count": 0,
        }

    async def click_element(self, element_id: int | str) -> Dict[str, Any]:
        """Scroll element into view, focus, and click by numerical snapshot ID."""
        if not self.is_connected():
            await self.connect_tab()

        script = f"""
        (() => {{
            const el = document.querySelector('[data-zelyr-id="{element_id}"]');
            if (!el) return {{ found: false }};
            el.scrollIntoView({{ block: 'center', behavior: 'instant' }});
            if (typeof el.focus === 'function') el.focus();
            if (typeof el.click === 'function') el.click();
            return {{ found: true }};
        }})()
        """
        res = await self.evaluate_script(script)
        if isinstance(res, dict) and res.get("found"):
            return {
                "status": "success",
                "action": "click_element",
                "element_id": element_id,
                "message": f"Successfully clicked element [{element_id}].",
            }

        return {
            "status": "error",
            "error_type": "ELEMENT_NOT_FOUND",
            "error": f"Element ID '{element_id}' not found on page.",
            "remedy_hint": "Call get_page_snapshot to refresh element IDs, or scroll_page down if the element is below the viewport.",
            "allowed_actions": ["interact_with_browser"],
        }

    async def fill_element(self, element_id: int | str, text: str) -> Dict[str, Any]:
        """Fill an interactive element by numerical snapshot ID."""
        if not self.is_connected():
            await self.connect_tab()

        text_json = json.dumps(str(text))
        script = f"""
        (() => {{
            const el = document.querySelector('[data-zelyr-id="{element_id}"]');
            if (!el) return {{ found: false }};
            el.scrollIntoView({{ block: 'center', behavior: 'instant' }});
            if (typeof el.focus === 'function') el.focus();
            el.value = {text_json};
            el.dispatchEvent(new Event('input', {{ bubbles: true }}));
            el.dispatchEvent(new Event('change', {{ bubbles: true }}));
            return {{ found: true }};
        }})()
        """
        res = await self.evaluate_script(script)
        if isinstance(res, dict) and res.get("found"):
            return {
                "status": "success",
                "action": "fill_element",
                "element_id": element_id,
                "value": text,
                "message": f"Successfully filled element [{element_id}] with text.",
            }

        return {
            "status": "error",
            "error_type": "ELEMENT_NOT_FOUND",
            "error": f"Element ID '{element_id}' not found on page.",
            "remedy_hint": "Call get_page_snapshot to refresh element IDs, or scroll_page down if the element is below the viewport.",
            "allowed_actions": ["interact_with_browser"],
        }

    async def scroll_page(self, direction: str = "down", amount: int = 500) -> Dict[str, Any]:
        """Scroll the active page viewport."""
        if not self.is_connected():
            await self.connect_tab()

        dir_json = json.dumps(direction.lower())
        script = f"""
        (() => {{
            const dir = {dir_json};
            const amt = {int(amount)};
            if (dir === 'up') window.scrollBy({{ top: -amt, behavior: 'instant' }});
            else if (dir === 'top') window.scrollTo({{ top: 0, behavior: 'instant' }});
            else if (dir === 'bottom') window.scrollTo({{ top: document.body.scrollHeight, behavior: 'instant' }});
            else window.scrollBy({{ top: amt, behavior: 'instant' }});
            return {{ scrollY: window.scrollY }};
        }})()
        """
        res = await self.evaluate_script(script)
        return {
            "status": "success",
            "action": "scroll_page",
            "direction": direction,
            "amount": amount,
            "scroll": res,
        }

    async def switch_tab(
        self,
        query: Optional[str] = None,
        target_id: Optional[str] = None,
        url: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Switch active tab focus."""
        tabs = await self.list_tabs()
        target = None
        search_query = (query or url or target_id or "").lower()

        if target_id:
            for t in tabs:
                if t.get("id") == target_id:
                    target = t
                    break

        if not target and search_query:
            for t in tabs:
                if (
                    search_query in (t.get("url") or "").lower()
                    or search_query in (t.get("title") or "").lower()
                    or search_query == (t.get("id") or "").lower()
                ):
                    target = t
                    break

        if target:
            await self.connect_tab(target_id=target.get("id"))
            return {
                "status": "success",
                "action": "switch_tab",
                "tab": target,
                "message": f"Switched to tab '{target.get('title')}' ({target.get('id')}).",
            }

        target_url = query or url or target_id or "<url>"
        return {
            "status": "error",
            "error_type": "TAB_NOT_FOUND",
            "error": f"Tab matching query '{target_url}' not found.",
            "remedy_hint": f"Call list_tabs to see valid targets, or open_tab with url='{target_url}' to launch it.",
            "allowed_actions": ["interact_with_browser", "list_tabs", "open_tab"],
            "open_tabs": tabs,
        }

    async def click_selector(self, selector: str) -> Dict[str, Any]:
        """Click element by CSS selector."""
        if not self.is_connected():
            await self.connect_tab()

        selector_json = json.dumps(selector)
        script = f"""
        (() => {{
            const el = document.querySelector({selector_json});
            if (!el) return {{ found: false, error: "Element not found for selector: " + {selector_json} }};
            el.scrollIntoView({{ block: "center", inline: "center", behavior: "instant" }});
            el.focus();
            el.click();
            el.dispatchEvent(new MouseEvent('mousedown', {{ bubbles: true, cancelable: true, view: window }}));
            el.dispatchEvent(new MouseEvent('mouseup', {{ bubbles: true, cancelable: true, view: window }}));
            return {{
                found: true,
                tag: el.tagName,
                text: (el.innerText || el.value || "").slice(0, 100),
                selector: {selector_json}
            }};
        }})()
        """
        res = await self.evaluate_script(script)
        if isinstance(res, dict) and res.get("found"):
            return {
                "status": "success",
                "action": "click",
                "result": f"Clicked <{res.get('tag')}> matching '{selector}'",
                **res,
            }

        return {
            "status": "error",
            "error_type": "ELEMENT_NOT_FOUND",
            "error": f"Element not found for selector: '{selector}'",
            "remedy_hint": "Call get_page_snapshot to inspect current DOM elements and interactive IDs.",
            "allowed_actions": ["interact_with_browser", "get_page_snapshot"],
        }

    async def type_selector(self, selector: str, text: str) -> Dict[str, Any]:
        """Type text into element by CSS selector."""
        if not self.is_connected():
            await self.connect_tab()

        selector_json = json.dumps(selector)
        text_json = json.dumps(text)
        script = f"""
        (() => {{
            const el = document.querySelector({selector_json});
            if (!el) return {{ found: false, error: "Element not found for selector: " + {selector_json} }};
            el.scrollIntoView({{ block: "center", inline: "center", behavior: "instant" }});
            el.focus();
            if ('value' in el) {{
                el.value = {text_json};
                el.dispatchEvent(new Event('input', {{ bubbles: true }}));
                el.dispatchEvent(new Event('change', {{ bubbles: true }}));
            }} else {{
                el.textContent = {text_json};
                el.dispatchEvent(new InputEvent('input', {{ bubbles: true, data: {text_json} }}));
            }}
            return {{
                found: true,
                selector: {selector_json},
                typed: {text_json}
            }};
        }})()
        """
        res = await self.evaluate_script(script)
        if isinstance(res, dict) and res.get("found"):
            return {
                "status": "success",
                "action": "type",
                "result": f"Typed '{text}' into '{selector}'",
                **res,
            }

        return {
            "status": "error",
            "error_type": "ELEMENT_NOT_FOUND",
            "error": f"Element not found for selector: '{selector}'",
            "remedy_hint": "Call get_page_snapshot to inspect current DOM elements and interactive IDs.",
            "allowed_actions": ["interact_with_browser", "get_page_snapshot"],
        }

    async def open_tab(self, url: str = "about:blank") -> Dict[str, Any]:
        """Open a new browser tab."""
        target_url = url if "://" in url or url.startswith("about:") else f"https://{url}"
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.put(f"{self.base_url}/json/new?{target_url}")
                if resp.status_code == 200:
                    tab = resp.json()
                    await self.connect_tab(target_id=tab.get("id"))
                    return {
                        "status": "success",
                        "action": "open_tab",
                        "tab": tab,
                    }
        except Exception as exc:
            logger.debug("Failed to open tab via HTTP: %s", exc)

        res = await self.send_cdp_command("Target.createTarget", {"url": target_url})
        return {
            "status": "success",
            "action": "open_tab",
            "targetId": res.get("targetId"),
        }

    async def close_tab(self, target_id: Optional[str] = None) -> Dict[str, Any]:
        """Close specified tab or active tab."""
        tid = target_id or (self._active_tab.get("id") if self._active_tab else None)
        if not tid:
            return {"status": "error", "error": "No tab target to close."}

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.put(f"{self.base_url}/json/close/{tid}")
                if resp.status_code == 200:
                    if self._active_tab and self._active_tab.get("id") == tid:
                        self._active_tab = None
                    return {"status": "success", "action": "close_tab", "closed_tab_id": tid}
        except Exception:
            pass

        await self.send_cdp_command("Target.closeTarget", {"targetId": tid})
        return {"status": "success", "action": "close_tab", "closed_tab_id": tid}

    async def close(self) -> None:
        """Close active WebSocket connection."""
        if self._ws:
            try:
                await self._ws.close()
            except Exception:
                pass
            self._ws = None


__all__ = ["CDPBrowserClient", "SNAPSHOT_JS_SCRIPT"]
