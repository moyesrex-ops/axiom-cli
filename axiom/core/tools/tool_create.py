"""Dynamic tool creation — generate new tools at runtime.

The agent can create new tools by describing what they should do.
Generated tools are saved to disk and persist across sessions.
"""

from __future__ import annotations

import importlib
import importlib.util
import logging
import os
import re
import textwrap
from pathlib import Path
from typing import Any

from axiom.core.tools.base import AxiomTool, ToolError

logger = logging.getLogger(__name__)

# Disallowed imports for safety
_DANGEROUS_IMPORTS = {
    "subprocess", "shutil.rmtree", "os.remove", "os.rmdir",
    "ctypes", "multiprocessing",
}

_TOOLS_DIR = Path.home() / ".axiom" / "tools"


class ToolCreateTool(AxiomTool):
    """Dynamically generate new tools at runtime.

    The agent describes what tool it needs, provides the Python code,
    and this tool validates, saves, and registers it immediately.
    """

    name = "create_tool"
    description = (
        "Create a new tool at runtime by providing Python code. "
        "The tool must subclass AxiomTool with name, description, "
        "parameters_schema, and an async execute() method. "
        "Created tools persist across sessions."
    )
    risk_level = "high"
    parameters_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "description": "Action: create (new tool), list (show custom tools), delete (remove tool)",
                "enum": ["create", "list", "delete"],
            },
            "tool_name": {
                "type": "string",
                "description": "Snake_case name for the tool",
            },
            "code": {
                "type": "string",
                "description": (
                    "Python source code for the tool class. Must subclass AxiomTool "
                    "and implement async execute(**kwargs). Import AxiomTool from "
                    "axiom.core.tools.base."
                ),
            },
            "description": {
                "type": "string",
                "description": "Human-readable description of what the tool does",
            },
        },
        "required": ["action"],
    }

    def __init__(self, registry: Any = None):
        self._registry = registry

    async def execute(self, **kwargs: Any) -> str:
        action = kwargs.get("action", "")

        if action == "create":
            return await self._create(kwargs)
        elif action == "list":
            return self._list()
        elif action == "delete":
            return self._delete(kwargs)
        else:
            raise ToolError(f"Unknown action: {action}", tool_name=self.name)

    async def _create(self, kwargs: dict) -> str:
        """Create and register a new tool."""
        tool_name = kwargs.get("tool_name", "")
        code = kwargs.get("code", "")

        if not tool_name:
            raise ToolError("tool_name is required", tool_name=self.name)
        if not code:
            raise ToolError("code is required (Python class inheriting AxiomTool)", tool_name=self.name)

        # Validate the code
        validation_error = self._validate_code(code, tool_name)
        if validation_error:
            raise ToolError(validation_error, tool_name=self.name)

        # Save to disk
        _TOOLS_DIR.mkdir(parents=True, exist_ok=True)
        file_path = _TOOLS_DIR / f"{tool_name}.py"
        file_path.write_text(code, encoding="utf-8")

        # Try to import and register
        try:
            tool_instance = self._load_tool_from_file(file_path, tool_name)
            if self._registry:
                self._registry.register(tool_instance)
            return (
                f"Tool '{tool_name}' created and registered successfully!\n"
                f"Saved to: {file_path}\n"
                f"The tool is now available for use."
            )
        except Exception as e:
            # Clean up on failure
            file_path.unlink(missing_ok=True)
            raise ToolError(
                f"Tool creation failed during import: {e}\n"
                f"Fix the code and try again.",
                tool_name=self.name,
            )

    def _list(self) -> str:
        """List all custom-created tools."""
        if not _TOOLS_DIR.exists():
            return "No custom tools created yet."

        tools = list(_TOOLS_DIR.glob("*.py"))
        if not tools:
            return "No custom tools created yet."

        lines = [f"Custom tools ({len(tools)}):"]
        for f in sorted(tools):
            name = f.stem
            # Try to read description from file
            try:
                content = f.read_text()
                desc_match = re.search(r'description\s*=\s*["\'](.+?)["\']', content)
                desc = desc_match.group(1)[:60] if desc_match else "No description"
            except Exception:
                desc = "Error reading"
            lines.append(f"  • {name}: {desc}")

        return "\n".join(lines)

    def _delete(self, kwargs: dict) -> str:
        """Delete a custom tool."""
        tool_name = kwargs.get("tool_name", "")
        if not tool_name:
            raise ToolError("tool_name required for delete", tool_name=self.name)

        file_path = _TOOLS_DIR / f"{tool_name}.py"
        if not file_path.exists():
            return f"Tool '{tool_name}' not found in custom tools."

        file_path.unlink()

        # Unregister if possible
        if self._registry:
            try:
                self._registry._tools.pop(tool_name, None)
            except Exception:
                pass

        return f"Tool '{tool_name}' deleted."

    def _validate_code(self, code: str, tool_name: str) -> str | None:
        """Validate tool code for safety and correctness.

        Returns error message if invalid, None if OK.
        """
        # Check for dangerous imports
        for dangerous in _DANGEROUS_IMPORTS:
            if dangerous in code:
                return f"Dangerous import/call detected: {dangerous}"

        # Must contain AxiomTool subclass
        if "AxiomTool" not in code:
            return "Tool must subclass AxiomTool (from axiom.core.tools.base)"

        # Must have execute method
        if "async def execute" not in code:
            return "Tool must implement 'async def execute(self, **kwargs)'"

        # Must have name attribute
        if "name" not in code:
            return "Tool class must define a 'name' attribute"

        # Try to compile
        try:
            compile(code, f"<tool:{tool_name}>", "exec")
        except SyntaxError as e:
            return f"Syntax error in tool code: {e}"

        return None

    def _load_tool_from_file(self, file_path: Path, tool_name: str) -> AxiomTool:
        """Dynamically import a tool from a file and return an instance."""
        spec = importlib.util.spec_from_file_location(
            f"axiom_custom_tools.{tool_name}", str(file_path)
        )
        if spec is None or spec.loader is None:
            raise ToolError(f"Could not load module from {file_path}", tool_name=self.name)

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Find the AxiomTool subclass
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if (
                isinstance(attr, type)
                and issubclass(attr, AxiomTool)
                and attr is not AxiomTool
            ):
                return attr()

        raise ToolError(
            f"No AxiomTool subclass found in {file_path}",
            tool_name=self.name,
        )


def load_custom_tools(registry: Any) -> int:
    """Load all custom tools from the tools directory.

    Called during startup to restore persisted custom tools.
    Returns the number of tools loaded.
    """
    if not _TOOLS_DIR.exists():
        return 0

    loaded = 0
    for file_path in _TOOLS_DIR.glob("*.py"):
        try:
            tool_name = file_path.stem
            creator = ToolCreateTool(registry=registry)
            tool = creator._load_tool_from_file(file_path, tool_name)
            registry.register(tool)
            loaded += 1
            logger.info("Loaded custom tool: %s", tool_name)
        except Exception as e:
            logger.warning("Failed to load custom tool %s: %s", file_path.stem, e)

    return loaded
