"""Desktop automation tool — full GUI control via pyautogui.

GOD MODE: Click, type, screenshot, OCR, hotkeys, drag, window management.
Combined with the vision tool, enables controlling ANY desktop application.
"""

from __future__ import annotations

import asyncio
import base64
import io
import logging
import sys
from typing import Any

from axiom.core.tools.base import AxiomTool, ToolError

logger = logging.getLogger(__name__)


class DesktopTool(AxiomTool):
    """Full desktop automation — screenshot, click, type, hotkey, OCR.

    Enables the agent to interact with ANY desktop application by
    combining screenshot capture with mouse/keyboard control.
    """

    name = "desktop"
    description = (
        "Control the desktop: take screenshots, click at coordinates, "
        "type text, press hotkeys, drag, find text on screen via OCR, "
        "and manage windows. Enables controlling ANY application."
    )
    risk_level = "high"
    parameters_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "description": (
                    "Desktop action: screenshot, screenshot_region, click, "
                    "double_click, right_click, type_text, hotkey, move_mouse, "
                    "drag, ocr, get_windows, focus_window, get_mouse_position"
                ),
                "enum": [
                    "screenshot", "screenshot_region", "click", "double_click",
                    "right_click", "type_text", "hotkey", "move_mouse",
                    "drag", "ocr", "get_windows", "focus_window",
                    "get_mouse_position",
                ],
            },
            "x": {"type": "integer", "description": "X coordinate"},
            "y": {"type": "integer", "description": "Y coordinate"},
            "x2": {"type": "integer", "description": "End X for drag / region width"},
            "y2": {"type": "integer", "description": "End Y for drag / region height"},
            "text": {"type": "string", "description": "Text to type or hotkey combo"},
            "title": {"type": "string", "description": "Window title for focus_window"},
        },
        "required": ["action"],
    }

    async def execute(self, **kwargs: Any) -> str:
        action = kwargs.get("action", "")
        if not action:
            raise ToolError("No desktop action specified", tool_name=self.name)

        # Run GUI operations in thread pool (pyautogui is sync)
        loop = asyncio.get_event_loop()
        handler = getattr(self, f"_action_{action}", None)
        if handler is None:
            raise ToolError(f"Unknown desktop action: {action}", tool_name=self.name)

        try:
            return await loop.run_in_executor(None, handler, kwargs)
        except ToolError:
            raise
        except Exception as e:
            raise ToolError(f"Desktop error: {e}", tool_name=self.name)

    def _action_screenshot(self, kwargs: dict) -> str:
        """Capture full screen screenshot."""
        try:
            import pyautogui
            from PIL import Image
        except ImportError:
            raise ToolError(
                "pyautogui/Pillow not installed. Run: pip install pyautogui Pillow",
                tool_name=self.name,
            )

        img = pyautogui.screenshot()
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")

        w, h = img.size
        return (
            f"Screenshot captured: {w}x{h} pixels\n"
            f"[base64_image:{b64[:100]}...]"
        )

    def _action_screenshot_region(self, kwargs: dict) -> str:
        """Capture a specific region of the screen."""
        try:
            import pyautogui
        except ImportError:
            raise ToolError("pyautogui not installed", tool_name=self.name)

        x = kwargs.get("x", 0)
        y = kwargs.get("y", 0)
        w = kwargs.get("x2", 400)
        h = kwargs.get("y2", 300)

        img = pyautogui.screenshot(region=(x, y, w, h))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")

        return (
            f"Region screenshot: ({x},{y}) {w}x{h}\n"
            f"[base64_image:{b64[:100]}...]"
        )

    def _action_click(self, kwargs: dict) -> str:
        """Click at coordinates."""
        try:
            import pyautogui
        except ImportError:
            raise ToolError("pyautogui not installed", tool_name=self.name)

        x = kwargs.get("x")
        y = kwargs.get("y")
        if x is None or y is None:
            raise ToolError("x and y coordinates required for click", tool_name=self.name)

        pyautogui.click(x, y)
        return f"Clicked at ({x}, {y})"

    def _action_double_click(self, kwargs: dict) -> str:
        """Double-click at coordinates."""
        try:
            import pyautogui
        except ImportError:
            raise ToolError("pyautogui not installed", tool_name=self.name)

        x = kwargs.get("x")
        y = kwargs.get("y")
        if x is None or y is None:
            raise ToolError("x and y coordinates required", tool_name=self.name)

        pyautogui.doubleClick(x, y)
        return f"Double-clicked at ({x}, {y})"

    def _action_right_click(self, kwargs: dict) -> str:
        """Right-click at coordinates."""
        try:
            import pyautogui
        except ImportError:
            raise ToolError("pyautogui not installed", tool_name=self.name)

        x = kwargs.get("x")
        y = kwargs.get("y")
        if x is None or y is None:
            raise ToolError("x and y coordinates required", tool_name=self.name)

        pyautogui.rightClick(x, y)
        return f"Right-clicked at ({x}, {y})"

    def _action_type_text(self, kwargs: dict) -> str:
        """Type text at current cursor position."""
        try:
            import pyautogui
        except ImportError:
            raise ToolError("pyautogui not installed", tool_name=self.name)

        text = kwargs.get("text", "")
        if not text:
            raise ToolError("No text provided for type_text", tool_name=self.name)

        pyautogui.write(text, interval=0.02)
        return f"Typed: '{text[:50]}{'...' if len(text) > 50 else ''}'"

    def _action_hotkey(self, kwargs: dict) -> str:
        """Press a keyboard shortcut (e.g. 'ctrl+c', 'alt+tab')."""
        try:
            import pyautogui
        except ImportError:
            raise ToolError("pyautogui not installed", tool_name=self.name)

        text = kwargs.get("text", "")
        if not text:
            raise ToolError("No hotkey specified (e.g. 'ctrl+c')", tool_name=self.name)

        keys = [k.strip() for k in text.split("+")]
        pyautogui.hotkey(*keys)
        return f"Pressed hotkey: {text}"

    def _action_move_mouse(self, kwargs: dict) -> str:
        """Move mouse to coordinates."""
        try:
            import pyautogui
        except ImportError:
            raise ToolError("pyautogui not installed", tool_name=self.name)

        x = kwargs.get("x")
        y = kwargs.get("y")
        if x is None or y is None:
            raise ToolError("x and y coordinates required", tool_name=self.name)

        pyautogui.moveTo(x, y, duration=0.3)
        return f"Mouse moved to ({x}, {y})"

    def _action_drag(self, kwargs: dict) -> str:
        """Drag from (x,y) to (x2,y2)."""
        try:
            import pyautogui
        except ImportError:
            raise ToolError("pyautogui not installed", tool_name=self.name)

        x = kwargs.get("x")
        y = kwargs.get("y")
        x2 = kwargs.get("x2")
        y2 = kwargs.get("y2")
        if any(v is None for v in (x, y, x2, y2)):
            raise ToolError("x, y, x2, y2 all required for drag", tool_name=self.name)

        pyautogui.moveTo(x, y)
        pyautogui.drag(x2 - x, y2 - y, duration=0.5)
        return f"Dragged from ({x},{y}) to ({x2},{y2})"

    def _action_ocr(self, kwargs: dict) -> str:
        """Extract text from screen using OCR (pytesseract)."""
        try:
            import pyautogui
            import pytesseract
        except ImportError:
            raise ToolError(
                "pytesseract not installed. Run: pip install pytesseract\n"
                "Also install Tesseract OCR: https://github.com/tesseract-ocr/tesseract",
                tool_name=self.name,
            )

        x = kwargs.get("x")
        y = kwargs.get("y")
        w = kwargs.get("x2")
        h = kwargs.get("y2")

        if all(v is not None for v in (x, y, w, h)):
            img = pyautogui.screenshot(region=(x, y, w, h))
        else:
            img = pyautogui.screenshot()

        text = pytesseract.image_to_string(img)
        text = text.strip()

        if not text:
            return "(no text detected by OCR)"
        return f"OCR Result:\n{text[:3000]}"

    def _action_get_windows(self, kwargs: dict) -> str:
        """List all open windows."""
        try:
            import pyautogui
        except ImportError:
            raise ToolError("pyautogui not installed", tool_name=self.name)

        if sys.platform == "win32":
            try:
                import pygetwindow as gw
                windows = gw.getAllWindows()
                lines = [f"Open windows ({len(windows)}):"]
                for w in windows:
                    if w.title.strip():
                        lines.append(f"  - '{w.title}' ({w.width}x{w.height} at {w.left},{w.top})")
                return "\n".join(lines)
            except ImportError:
                return "pygetwindow not installed. Run: pip install pygetwindow"
        else:
            return "Window listing is only supported on Windows with pygetwindow"

    def _action_focus_window(self, kwargs: dict) -> str:
        """Bring a window to the foreground by title."""
        title = kwargs.get("title", "")
        if not title:
            raise ToolError("No window title specified", tool_name=self.name)

        if sys.platform == "win32":
            try:
                import pygetwindow as gw
                windows = gw.getWindowsWithTitle(title)
                if not windows:
                    return f"No window found with title containing: '{title}'"
                windows[0].activate()
                return f"Focused window: '{windows[0].title}'"
            except ImportError:
                return "pygetwindow not installed"
        else:
            return "Window focus only supported on Windows"

    def _action_get_mouse_position(self, kwargs: dict) -> str:
        """Get current mouse cursor position."""
        try:
            import pyautogui
        except ImportError:
            raise ToolError("pyautogui not installed", tool_name=self.name)

        x, y = pyautogui.position()
        screen_w, screen_h = pyautogui.size()
        return f"Mouse position: ({x}, {y})\nScreen size: {screen_w}x{screen_h}"
