"""Observer node -- LLM-based reflection on agent progress (Reflexion pattern)."""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)

OBSERVER_SYSTEM = """You are an expert observer analyzing an AI agent's progress.

Review the agent's work and decide:
1. Is the task COMPLETE?
2. Should the agent CONTINUE with more steps?
3. Should the agent REPLAN (current approach isn't working)?

Consider:
- Has the user's request been fully addressed?
- Were there any tool errors that need retry?
- Is the agent going in circles?

Respond with JSON:
{{"status": "complete" | "continue" | "replan", "confidence": 0.0-1.0, "reason": "brief explanation"}}"""


async def observe_progress(
    router: Any,
    messages: list[dict[str, str]],
    tool_results: list[dict[str, Any]],
    iteration: int,
    max_iterations: int,
) -> dict[str, Any]:
    """Evaluate agent progress and decide next action.

    Returns dict with: status ("complete"/"continue"/"replan"), confidence, reason
    """
    # Build a summary of what happened
    results_summary = []
    for r in tool_results[-5:]:  # Last 5 results
        status = "OK" if r.get("success") else "FAILED"
        result_preview = str(r.get("result", ""))[:500]
        results_summary.append(
            f"  [{status}] {r.get('tool_name', 'unknown')}: {result_preview}"
        )

    context = f"""Iteration {iteration}/{max_iterations}

Recent tool results:
{chr(10).join(results_summary) if results_summary else '  No tools executed yet.'}

Original request: {messages[0]['content'] if messages else 'Unknown'}"""

    obs_messages = [
        {"role": "system", "content": OBSERVER_SYSTEM},
        {"role": "user", "content": context},
    ]

    response_text = ""
    async for chunk in router.complete(messages=obs_messages, stream=True):
        if hasattr(chunk, "choices") and chunk.choices:
            delta = chunk.choices[0].delta
            if hasattr(delta, "content") and delta.content:
                response_text += delta.content

    try:
        start = response_text.find("{")
        end = response_text.rfind("}") + 1
        if start >= 0 and end > start:
            result = json.loads(response_text[start:end])
            return {
                "status": result.get("status", "continue"),
                "confidence": float(result.get("confidence", 0.5)),
                "reason": result.get("reason", ""),
            }
    except (json.JSONDecodeError, ValueError) as e:
        logger.warning("Observer parse failed: %s", e)

    # Default: continue if under iteration limit
    if iteration >= max_iterations:
        return {"status": "complete", "confidence": 0.5, "reason": "Max iterations reached"}
    return {"status": "continue", "confidence": 0.5, "reason": "Could not parse observer response"}
