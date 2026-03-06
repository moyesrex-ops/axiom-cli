"""Executor node -- invokes tools from plan steps or react actions."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


async def execute_tool(
    registry: Any,
    tool_name: str,
    tool_args: dict[str, Any],
) -> dict[str, Any]:
    """Execute a single tool call.

    Returns dict with: tool_name, args, result, success, duration_ms
    """
    if tool_name in ("think", "finish", "reasoning"):
        return {
            "tool_name": tool_name,
            "args": tool_args,
            "result": tool_args.get("answer", tool_args.get("description", "")),
            "success": True,
            "duration_ms": 0,
        }

    tool = registry.get(tool_name)
    if tool is None:
        return {
            "tool_name": tool_name,
            "args": tool_args,
            "result": f"Error: Tool '{tool_name}' not found. Available: {', '.join(registry.list_names())}",
            "success": False,
            "duration_ms": 0,
        }

    record = await registry.invoke(tool_name, **tool_args)
    return {
        "tool_name": tool_name,
        "args": tool_args,
        "result": record.result,
        "success": record.success,
        "duration_ms": int(record.duration_ms),
    }
