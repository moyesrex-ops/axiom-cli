"""Vision tool — send screenshots to multimodal LLM for analysis.

Enables the agent to understand GUI state by sending screenshots
to Claude Vision, Gemini Vision, or other multimodal models.
"""

from __future__ import annotations

import asyncio
import base64
import io
import logging
from typing import Any, Optional

from axiom.core.tools.base import AxiomTool, ToolError

logger = logging.getLogger(__name__)


class VisionTool(AxiomTool):
    """Analyze screenshots and images using multimodal LLM vision.

    Can analyze the current screen, a specific image file, or
    describe UI elements and find coordinates of described elements.
    """

    name = "vision"
    description = (
        "Send a screenshot or image to a vision-capable LLM for analysis. "
        "Can analyze the screen, describe UI elements, find buttons/icons, "
        "read text from images, or compare two screenshots."
    )
    risk_level = "low"
    parameters_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "description": (
                    "Vision action: analyze_screen (screenshot + describe), "
                    "analyze_image (load file + describe), "
                    "find_element (locate described UI element), "
                    "read_text (extract all text from screen)"
                ),
                "enum": ["analyze_screen", "analyze_image", "find_element", "read_text"],
            },
            "prompt": {
                "type": "string",
                "description": "What to look for or analyze (e.g. 'Find the login button')",
            },
            "image_path": {
                "type": "string",
                "description": "Path to image file (for analyze_image action)",
            },
        },
        "required": ["action"],
    }

    def __init__(self, router: Any = None):
        self._router = router

    async def execute(self, **kwargs: Any) -> str:
        action = kwargs.get("action", "")
        prompt = kwargs.get("prompt", "Describe what you see in this image in detail.")

        if not action:
            raise ToolError("No vision action specified", tool_name=self.name)

        if self._router is None:
            raise ToolError(
                "Vision tool requires an LLM router for multimodal analysis",
                tool_name=self.name,
            )

        if action == "analyze_screen":
            return await self._analyze_screen(prompt)
        elif action == "analyze_image":
            image_path = kwargs.get("image_path", "")
            if not image_path:
                raise ToolError("No image_path provided", tool_name=self.name)
            return await self._analyze_image(image_path, prompt)
        elif action == "find_element":
            if not prompt:
                raise ToolError("Provide a description of what to find", tool_name=self.name)
            return await self._find_element(prompt)
        elif action == "read_text":
            return await self._analyze_screen(
                "Extract and list ALL text visible on this screen, organized by area."
            )
        else:
            raise ToolError(f"Unknown vision action: {action}", tool_name=self.name)

    async def _capture_screen_b64(self) -> str:
        """Capture the screen and return base64-encoded PNG."""
        loop = asyncio.get_event_loop()

        def _capture() -> bytes:
            try:
                import pyautogui
            except ImportError:
                raise ToolError(
                    "pyautogui not installed. Run: pip install pyautogui Pillow",
                    tool_name=self.name,
                )
            img = pyautogui.screenshot()
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            return buf.getvalue()

        img_bytes = await loop.run_in_executor(None, _capture)
        return base64.b64encode(img_bytes).decode("ascii")

    async def _analyze_screen(self, prompt: str) -> str:
        """Screenshot + send to vision LLM."""
        b64_image = await self._capture_screen_b64()
        return await self._send_vision_request(b64_image, prompt)

    async def _analyze_image(self, image_path: str, prompt: str) -> str:
        """Load image file + send to vision LLM."""
        try:
            with open(image_path, "rb") as f:
                img_bytes = f.read()
        except FileNotFoundError:
            raise ToolError(f"Image not found: {image_path}", tool_name=self.name)
        except Exception as e:
            raise ToolError(f"Failed to read image: {e}", tool_name=self.name)

        b64_image = base64.b64encode(img_bytes).decode("ascii")
        return await self._send_vision_request(b64_image, prompt)

    async def _find_element(self, description: str) -> str:
        """Screenshot + ask vision to locate UI element."""
        prompt = (
            f"Find the UI element described as: '{description}'\n"
            f"Return the approximate coordinates (x, y) of the center of this element. "
            f"Format: 'FOUND: (x, y) - description of element'\n"
            f"If not found, say 'NOT FOUND: reason'"
        )
        b64_image = await self._capture_screen_b64()
        return await self._send_vision_request(b64_image, prompt)

    async def _send_vision_request(self, b64_image: str, prompt: str) -> str:
        """Send a vision request to the multimodal LLM."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{b64_image}",
                        },
                    },
                ],
            }
        ]

        # Use a vision-capable model
        # Claude, GPT-4o, Gemini all support vision
        response_text = ""
        try:
            async for chunk in self._router.complete(
                messages=messages,
                stream=True,
            ):
                if hasattr(chunk, "choices") and chunk.choices:
                    delta = chunk.choices[0].delta
                    if hasattr(delta, "content") and delta.content:
                        response_text += delta.content
        except Exception as e:
            raise ToolError(
                f"Vision analysis failed: {e}\n"
                f"Make sure you're using a vision-capable model (Claude, GPT-4o, Gemini)",
                tool_name=self.name,
            )

        return response_text or "(no response from vision model)"
