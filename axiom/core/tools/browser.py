"""Browser automation tool using Playwright.

Provides full browser control: navigate, click, type, screenshot,
extract text, run JavaScript, and capture accessibility snapshots.
"""

from __future__ import annotations

import asyncio
import base64
import logging
from typing import Any, Optional

from axiom.core.tools.base import AxiomTool, ToolError

logger = logging.getLogger(__name__)

# Module-level persistent browser state
_browser_state: dict[str, Any] = {
    "browser": None,
    "context": None,
    "page": None,
    "playwright": None,
}


async def _ensure_browser(headless: bool = True) -> Any:
    """Lazy-init a persistent Playwright browser + page.

    Returns the active page. Reuses the same browser session across
    multiple tool calls within a CLI session.
    """
    if _browser_state["page"] is not None:
        try:
            # Check if page is still alive
            await _browser_state["page"].title()
            return _browser_state["page"]
        except Exception:
            # Page/browser died — restart
            _browser_state["page"] = None
            _browser_state["context"] = None
            _browser_state["browser"] = None

    try:
        from playwright.async_api import async_playwright
    except ImportError:
        raise ToolError(
            "Playwright not installed. Run: pip install playwright && python -m playwright install chromium",
            tool_name="browser",
        )

    pw = await async_playwright().start()
    _browser_state["playwright"] = pw

    try:
        browser = await pw.chromium.launch(
            headless=headless,
            args=[
                "--no-sandbox",
                "--disable-blink-features=AutomationControlled",
            ],
        )
    except Exception as e:
        raise ToolError(
            f"Failed to launch browser. Run: python -m playwright install chromium\nError: {e}",
            tool_name="browser",
        )

    context = await browser.new_context(
        viewport={"width": 1280, "height": 800},
        user_agent=(
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
    )
    page = await context.new_page()

    _browser_state["browser"] = browser
    _browser_state["context"] = context
    _browser_state["page"] = page

    return page


async def _close_browser() -> None:
    """Shut down the persistent browser."""
    for key in ("page", "context", "browser"):
        obj = _browser_state.get(key)
        if obj is not None:
            try:
                await obj.close()
            except Exception:
                pass
            _browser_state[key] = None

    pw = _browser_state.get("playwright")
    if pw is not None:
        try:
            await pw.stop()
        except Exception:
            pass
        _browser_state["playwright"] = None


class BrowserTool(AxiomTool):
    """Full browser automation via Playwright.

    Supports navigate, screenshot, click, type, scroll,
    get_text, evaluate JS, wait_for, snapshot (a11y tree), and close.
    """

    name = "browser"
    description = (
        "Control a web browser: navigate to URLs, click elements, fill forms, "
        "take screenshots, extract text, run JavaScript, and get accessibility snapshots. "
        "The browser session persists between calls."
    )
    risk_level = "medium"
    parameters_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "description": (
                    "Browser action: navigate, screenshot, click, type, scroll, "
                    "get_text, evaluate, wait_for, snapshot, close"
                ),
                "enum": [
                    "navigate", "screenshot", "click", "type", "scroll",
                    "get_text", "evaluate", "wait_for", "snapshot", "close",
                ],
            },
            "url": {
                "type": "string",
                "description": "URL to navigate to (for 'navigate' action)",
            },
            "selector": {
                "type": "string",
                "description": "CSS selector for element interactions",
            },
            "text": {
                "type": "string",
                "description": "Text to type or JS code to evaluate",
            },
            "direction": {
                "type": "string",
                "description": "Scroll direction: up or down",
                "enum": ["up", "down"],
            },
            "timeout": {
                "type": "integer",
                "description": "Timeout in milliseconds (default 30000)",
                "default": 30000,
            },
        },
        "required": ["action"],
    }

    def __init__(self, headless: bool = True):
        self._headless = headless

    async def execute(self, **kwargs: Any) -> str:
        action = kwargs.get("action", "")
        if not action:
            raise ToolError("No action specified", tool_name=self.name)

        try:
            handler = getattr(self, f"_action_{action}", None)
            if handler is None:
                raise ToolError(
                    f"Unknown browser action: {action}. "
                    f"Available: navigate, screenshot, click, type, scroll, "
                    f"get_text, evaluate, wait_for, snapshot, close",
                    tool_name=self.name,
                )
            return await handler(**kwargs)
        except ToolError:
            raise
        except Exception as e:
            raise ToolError(f"Browser error: {e}", tool_name=self.name)

    async def _action_navigate(self, **kwargs: Any) -> str:
        url = kwargs.get("url", "")
        if not url:
            raise ToolError("No URL provided for navigate", tool_name=self.name)
        if not url.startswith(("http://", "https://")):
            url = "https://" + url

        page = await _ensure_browser(self._headless)
        timeout = kwargs.get("timeout", 30000)

        await page.goto(url, timeout=timeout, wait_until="domcontentloaded")
        title = await page.title()
        current_url = page.url
        return f"Navigated to: {current_url}\nTitle: {title}"

    async def _action_screenshot(self, **kwargs: Any) -> str:
        page = await _ensure_browser(self._headless)
        screenshot_bytes = await page.screenshot(type="png")
        b64 = base64.b64encode(screenshot_bytes).decode("ascii")
        title = await page.title()
        url = page.url
        return (
            f"Screenshot captured ({len(screenshot_bytes)} bytes)\n"
            f"Page: {title} ({url})\n"
            f"[base64_image:{b64[:100]}...]"
        )

    async def _action_click(self, **kwargs: Any) -> str:
        selector = kwargs.get("selector", "")
        if not selector:
            raise ToolError("No selector provided for click", tool_name=self.name)

        page = await _ensure_browser(self._headless)
        timeout = kwargs.get("timeout", 10000)

        await page.click(selector, timeout=timeout)
        await page.wait_for_load_state("domcontentloaded")
        return f"Clicked element: {selector}"

    async def _action_type(self, **kwargs: Any) -> str:
        selector = kwargs.get("selector", "")
        text = kwargs.get("text", "")
        if not selector:
            raise ToolError("No selector provided for type", tool_name=self.name)
        if not text:
            raise ToolError("No text provided for type", tool_name=self.name)

        page = await _ensure_browser(self._headless)
        timeout = kwargs.get("timeout", 10000)

        await page.fill(selector, text, timeout=timeout)
        return f"Typed '{text[:50]}' into {selector}"

    async def _action_scroll(self, **kwargs: Any) -> str:
        direction = kwargs.get("direction", "down")
        page = await _ensure_browser(self._headless)

        delta = 500 if direction == "down" else -500
        await page.mouse.wheel(0, delta)
        await asyncio.sleep(0.5)
        return f"Scrolled {direction}"

    async def _action_get_text(self, **kwargs: Any) -> str:
        selector = kwargs.get("selector", "body")
        page = await _ensure_browser(self._headless)
        timeout = kwargs.get("timeout", 10000)

        element = await page.query_selector(selector)
        if element is None:
            return f"No element found for selector: {selector}"

        text = await element.text_content()
        text = (text or "").strip()

        if len(text) > 5000:
            text = text[:5000] + f"\n... [truncated, {len(text)} chars total]"
        return text or "(empty)"

    async def _action_evaluate(self, **kwargs: Any) -> str:
        text = kwargs.get("text", "")
        if not text:
            raise ToolError("No JavaScript code provided", tool_name=self.name)

        page = await _ensure_browser(self._headless)
        result = await page.evaluate(text)
        return str(result)[:5000] if result is not None else "(undefined)"

    async def _action_wait_for(self, **kwargs: Any) -> str:
        selector = kwargs.get("selector", "")
        if not selector:
            raise ToolError("No selector provided for wait_for", tool_name=self.name)

        page = await _ensure_browser(self._headless)
        timeout = kwargs.get("timeout", 30000)

        await page.wait_for_selector(selector, timeout=timeout)
        return f"Element found: {selector}"

    async def _action_snapshot(self, **kwargs: Any) -> str:
        """Return an accessibility tree snapshot of the page."""
        page = await _ensure_browser(self._headless)
        snapshot = await page.accessibility.snapshot()

        if snapshot is None:
            return "(no accessibility tree available)"

        lines: list[str] = []
        _walk_a11y(snapshot, lines, depth=0)
        result = "\n".join(lines)

        if len(result) > 10000:
            result = result[:10000] + "\n... [truncated]"
        return result

    async def _action_close(self, **kwargs: Any) -> str:
        await _close_browser()
        return "Browser closed."


def _walk_a11y(node: dict[str, Any], lines: list[str], depth: int = 0) -> None:
    """Recursively walk the accessibility tree to build a text representation."""
    indent = "  " * depth
    role = node.get("role", "")
    name = node.get("name", "")
    value = node.get("value", "")

    label_parts = [role]
    if name:
        label_parts.append(f'"{name}"')
    if value:
        label_parts.append(f"value={value}")

    lines.append(f"{indent}{' '.join(label_parts)}")

    for child in node.get("children", []):
        _walk_a11y(child, lines, depth + 1)
