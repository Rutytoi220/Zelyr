import logging
import asyncio
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field
from zelyr.tools.core import BaseTool

logger = logging.getLogger(__name__)

class PlaywrightWebSchema(BaseModel):
    action: str = Field(..., description="Action to perform: 'navigate', 'inspect', 'click_and_fill', 'screenshot'")
    url: Optional[str] = Field(None, description="URL to navigate to (required for 'navigate')")
    selector: Optional[str] = Field(None, description="CSS or XPath selector for interaction or inspection")
    text: Optional[str] = Field(None, description="Text to fill in a form field")

class PlaywrightWebTool(BaseTool):
    """Autonomous Web Engine for interacting with DOM environments via Playwright."""
    
    name = "web_engine"
    description = "Navigate web pages, inspect DOM structures, and interact with web elements."
    schema = PlaywrightWebSchema
    is_async = True
    
    def __init__(self):
        super().__init__()
        self._browser = None
        self._context = None
        self._page = None
        self._playwright = None
        
    async def _ensure_browser(self):
        if not self._page:
            try:
                from playwright.async_api import async_playwright
                self._playwright = await async_playwright().start()
                self._browser = await self._playwright.chromium.launch(headless=True)
                self._context = await self._browser.new_context()
                self._page = await self._context.new_page()
            except ImportError as e:
                from zelyr.utils.optional_deps import OptionalDependencyError
                msg = "'playwright' is required for web browser automation. Install it with: pip install 'zelyr-core[browser]'"
                logger.error(msg)
                raise OptionalDependencyError(
                    package_name="playwright",
                    extra_name="browser",
                    feature_description="web browser automation",
                ) from e

    async def execute(self, params: Optional[Dict[str, Any]] = None, **kwargs) -> Dict[str, Any]:
        args = dict(params or {})
        args.update(kwargs)
        return await self(**args)

    async def __call__(self, action: str, url: str = None, selector: str = None, text: str = None) -> Dict[str, Any]:
        await self._ensure_browser()
        
        try:
            if action == 'navigate':
                if not url:
                    return {"status": "FAILED", "error": "URL required for navigate."}
                await self._page.goto(url, wait_until='networkidle')
                title = await self._page.title()
                return {"status": "SUCCESS", "title": title, "url": self._page.url}
                
            elif action == 'inspect':
                # Return basic innerText for simplified token-optimized DOM structure
                if selector:
                    element = await self._page.query_selector(selector)
                    if element:
                        content = await element.inner_text()
                        return {"status": "SUCCESS", "content": content}
                    return {"status": "FAILED", "error": f"Selector {selector} not found."}
                else:
                    # Full page text
                    content = await self._page.evaluate("document.body.innerText")
                    return {"status": "SUCCESS", "content": content[:5000] + ("..." if len(content) > 5000 else "")}
                    
            elif action == 'click_and_fill':
                if not selector:
                    return {"status": "FAILED", "error": "Selector required for click_and_fill."}
                if text:
                    await self._page.fill(selector, text)
                    return {"status": "SUCCESS", "message": f"Filled {selector} with '{text}'"}
                else:
                    await self._page.click(selector)
                    return {"status": "SUCCESS", "message": f"Clicked {selector}"}
                    
            elif action == 'screenshot':
                screenshot_bytes = await self._page.screenshot(full_page=True, type='jpeg', quality=50)
                import base64
                b64 = base64.b64encode(screenshot_bytes).decode('utf-8')
                return {"status": "SUCCESS", "screenshot_b64": b64}
                
            else:
                return {"status": "FAILED", "error": f"Unknown action: {action}"}
                
        except Exception as e:
            logger.error(f"Playwright error: {e}")
            return {"status": "FAILED", "error": str(e)}

    async def cleanup(self):
        if self._browser:
            await self._browser.close()
        if self._playwright:
            await self._playwright.stop()
