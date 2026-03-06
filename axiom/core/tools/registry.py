"""Singleton tool registry -- register, discover, and invoke Axiom tools.

Usage:
    from axiom.core.tools.registry import ToolRegistry

    registry = ToolRegistry()
    registry.register(MyTool())
    result = await registry.invoke("my_tool", arg1="value")
"""

from __future__ import annotations

import logging
import time
from typing import Any, Optional

from axiom.core.tools.base import AxiomTool, ToolError
from axiom.core.agent.state import ToolCallRecord

logger = logging.getLogger(__name__)


class ToolRegistry:
    """Central registry for all available tools.

    Implements the singleton pattern -- every call to ``ToolRegistry()``
    returns the same instance.
    """

    _instance: Optional[ToolRegistry] = None
    _tools: dict[str, AxiomTool]

    def __new__(cls) -> ToolRegistry:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._tools = {}
        return cls._instance

    # ── Registration ──────────────────────────────────────────────

    def register(self, tool: AxiomTool) -> None:
        """Register a tool instance.  Overwrites if name already exists."""
        if not tool.name:
            raise ValueError(f"Tool {tool.__class__.__name__} has no name set")
        self._tools[tool.name] = tool
        logger.debug("Registered tool: %s (risk=%s)", tool.name, tool.risk_level)

    def register_many(self, tools: list[AxiomTool]) -> None:
        """Convenience -- register a list of tools at once."""
        for tool in tools:
            self.register(tool)

    # ── Lookup ────────────────────────────────────────────────────

    def get(self, name: str) -> Optional[AxiomTool]:
        """Return a tool by name, or None if not found."""
        return self._tools.get(name)

    def list_tools(self) -> list[AxiomTool]:
        """Return all registered tool instances, sorted by name."""
        return sorted(self._tools.values(), key=lambda t: t.name)

    def list_names(self) -> list[str]:
        """Return sorted list of all registered tool names."""
        return sorted(self._tools.keys())

    def get_by_risk(self, risk: str) -> list[AxiomTool]:
        """Return all tools matching the given risk level.

        Args:
            risk: One of ``"low"``, ``"medium"``, ``"high"``.

        Returns:
            List of matching tools, sorted by name.
        """
        return sorted(
            [t for t in self._tools.values() if t.risk_level == risk],
            key=lambda t: t.name,
        )

    @property
    def count(self) -> int:
        """Total number of registered tools."""
        return len(self._tools)

    # ── LLM Schemas ───────────────────────────────────────────────

    def to_llm_schemas(self) -> list[dict[str, Any]]:
        """Return OpenAI-compatible tool schemas for all registered tools."""
        return [tool.to_llm_schema() for tool in self.list_tools()]

    # ── Invocation ────────────────────────────────────────────────

    async def invoke(self, tool_name: str, **kwargs: Any) -> ToolCallRecord:
        """Look up a tool by name and execute it, returning a ToolCallRecord.

        Args:
            tool_name: The registered name of the tool to invoke.
            **kwargs: Arguments forwarded to ``tool.execute()``.

        Returns:
            A ``ToolCallRecord`` with the result, timing, and success flag.
        """
        tool = self.get(tool_name)
        if tool is None:
            return ToolCallRecord(
                tool_name=tool_name,
                args=kwargs,
                result=f"Error: tool '{tool_name}' not found. "
                       f"Available: {', '.join(self.list_names())}",
                success=False,
                duration_ms=0.0,
            )

        start = time.perf_counter()
        try:
            result = await tool.execute(**kwargs)
            duration_ms = (time.perf_counter() - start) * 1000
            logger.debug(
                "Tool %s completed in %.1fms (success)", tool_name, duration_ms
            )
            return ToolCallRecord(
                tool_name=tool_name,
                args=kwargs,
                result=result,
                success=True,
                duration_ms=round(duration_ms, 1),
            )
        except ToolError as exc:
            duration_ms = (time.perf_counter() - start) * 1000
            logger.warning(
                "Tool %s failed in %.1fms: %s", tool_name, duration_ms, exc
            )
            return ToolCallRecord(
                tool_name=tool_name,
                args=kwargs,
                result=f"ToolError: {exc}",
                success=False,
                duration_ms=round(duration_ms, 1),
            )
        except Exception as exc:
            duration_ms = (time.perf_counter() - start) * 1000
            logger.error(
                "Tool %s crashed in %.1fms: %s", tool_name, duration_ms, exc,
                exc_info=True,
            )
            return ToolCallRecord(
                tool_name=tool_name,
                args=kwargs,
                result=f"Unexpected error: {type(exc).__name__}: {exc}",
                success=False,
                duration_ms=round(duration_ms, 1),
            )

    # ── Reset (useful for testing) ────────────────────────────────

    def clear(self) -> None:
        """Remove all registered tools."""
        self._tools.clear()

    @classmethod
    def reset_singleton(cls) -> None:
        """Destroy the singleton instance (for tests only)."""
        cls._instance = None

    # ── Display ───────────────────────────────────────────────────

    def __repr__(self) -> str:
        return f"<ToolRegistry tools={self.count}>"

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return self.count


# ── Module-level helpers ──────────────────────────────────────────────


def get_registry() -> ToolRegistry:
    """Return the singleton ToolRegistry, populating it with P0 tools on first call.

    Safe to call multiple times -- the singleton ensures tools are only
    registered once.
    """
    registry = ToolRegistry()
    if registry.count == 0:
        register_default_tools(registry)
    return registry


def register_default_tools(registry: ToolRegistry) -> None:
    """Register all built-in P0 tools into the given registry.

    Tools registered:
      - bash (shell execution)
      - read_file, write_file, edit_file, glob, grep (file operations)
      - git (version control)
      - web_fetch (HTTP fetching)
      - code_exec (isolated code execution)
    """
    from axiom.core.tools.bash import BashTool
    from axiom.core.tools.code_exec import CodeExecTool
    from axiom.core.tools.files import (
        EditFileTool,
        GlobTool,
        GrepTool,
        ReadFileTool,
        WriteFileTool,
    )
    from axiom.core.tools.git import GitTool
    from axiom.core.tools.web_fetch import WebFetchTool
    from axiom.core.tools.http import HTTPTool
    from axiom.core.tools.think import ThinkTool

    registry.register_many([
        ThinkTool(),       # Must be first -- reasoning before action
        BashTool(),
        ReadFileTool(),
        WriteFileTool(),
        EditFileTool(),
        GlobTool(),
        GrepTool(),
        GitTool(),
        WebFetchTool(),
        HTTPTool(),
        CodeExecTool(),
    ])
