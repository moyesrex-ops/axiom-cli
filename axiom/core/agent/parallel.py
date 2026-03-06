"""Parallel execution engine -- run independent agent steps concurrently.

When a plan has independent steps (no data dependencies between them),
this engine executes them in parallel using asyncio.gather().
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

logger = logging.getLogger(__name__)


async def execute_parallel(
    registry: Any,
    steps: list[dict[str, Any]],
    max_concurrent: int = 5,
) -> list[dict[str, Any]]:
    """Execute multiple tool calls in parallel.

    Args:
        registry: Tool registry for executing tools.
        steps: List of step dicts with ``tool_name`` and ``tool_args`` keys.
        max_concurrent: Maximum concurrent executions.

    Returns:
        List of result dicts with ``tool_name``, ``result``, ``success``,
        ``duration_ms``.
    """
    semaphore = asyncio.Semaphore(max_concurrent)

    async def _execute_one(step: dict[str, Any]) -> dict[str, Any]:
        tool_name = step.get("tool_name", step.get("action", ""))
        tool_args = step.get("tool_args", step.get("args", {}))

        async with semaphore:
            start = time.time()
            try:
                record = await registry.invoke(tool_name, **tool_args)
                duration_ms = (time.time() - start) * 1000
                return {
                    "tool_name": tool_name,
                    "result": record.result,
                    "success": record.success,
                    "duration_ms": duration_ms,
                }
            except Exception as exc:
                duration_ms = (time.time() - start) * 1000
                return {
                    "tool_name": tool_name,
                    "result": f"Error: {exc}",
                    "success": False,
                    "duration_ms": duration_ms,
                }

    results = await asyncio.gather(
        *[_execute_one(step) for step in steps],
        return_exceptions=False,
    )
    return list(results)


def find_parallel_groups(
    steps: list[dict[str, Any]],
) -> list[list[dict[str, Any]]]:
    """Identify groups of steps that can run in parallel.

    Simple heuristic: steps that don't reference previous step outputs
    can run together. Steps with data dependencies must be sequential.

    Returns:
        List of groups, where each group is a list of steps that
        can execute in parallel.
    """
    groups: list[list[dict[str, Any]]] = []
    current_group: list[dict[str, Any]] = []

    for step in steps:
        tool_name = step.get("tool_name", step.get("action", ""))
        args = step.get("tool_args", step.get("args", {}))

        # Check if this step depends on previous results
        args_str = str(args).lower()
        has_dependency = any(
            marker in args_str
            for marker in ["$prev", "{{result}}", "previous", "step_result"]
        )

        # Sequential tools that must not be parallelized
        sequential_tools = {"write_file", "edit_file", "git", "bash"}
        is_sequential = tool_name in sequential_tools

        if has_dependency or is_sequential:
            # Flush current parallel group
            if current_group:
                groups.append(current_group)
                current_group = []
            groups.append([step])  # This step runs alone
        else:
            current_group.append(step)

    # Don't forget the last group
    if current_group:
        groups.append(current_group)

    return groups
