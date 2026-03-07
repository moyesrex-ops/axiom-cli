"""Observer node -- LLM-based reflection on agent progress (Reflexion pattern)."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

OBSERVER_SYSTEM = """You evaluate whether an AI agent has completed its task.

EVALUATION CRITERIA (check each one):
1. TASK MATCH: Does the work done actually address what was asked?
2. TOOL SUCCESS: Did the critical tools succeed (not just run)?
3. COMPLETENESS: Are there obvious missing pieces?
4. STUCK DETECTION: Is the agent repeating the same actions?

DECISION RULES:
- "complete": Task is genuinely done. Tools succeeded. Output matches request.
- "continue": Making progress but not done. More steps needed.
- "replan": Current approach failed. Need fundamentally different strategy.
  Only use replan if: multiple tool failures, wrong approach, or going in circles.

Respond with JSON ONLY:
{"status": "complete" | "continue" | "replan", "confidence": 0.0-1.0, "reason": "1 sentence"}"""


async def observe_progress(
    router: Any,
    messages: list[dict[str, str]],
    tool_results: list[dict[str, Any]],
    iteration: int,
    max_iterations: int,
) -> dict[str, Any]:
    """Evaluate agent progress and decide next action."""
    # Build a structured summary
    results_summary = []
    success_count = 0
    fail_count = 0
    for r in tool_results[-5:]:
        success = r.get("success", False)
        if success:
            success_count += 1
        else:
            fail_count += 1
        status = "OK" if success else "FAILED"
        result_preview = str(r.get("result", ""))[:500]
        tool_name = r.get("tool_name", r.get("tool", "unknown"))
        results_summary.append(f"  [{status}] {tool_name}: {result_preview}")

    # Detect repetition (same tool called 3+ times in last 5)
    recent_tools = [r.get("tool_name", r.get("tool", "")) for r in tool_results[-5:]]
    is_repeating = any(recent_tools.count(t) >= 3 for t in set(recent_tools) if t)

    # Extract original request
    original_request = "Unknown"
    for msg in messages:
        if msg.get("role") == "user":
            original_request = msg.get("content", "Unknown")[:200]
            break

    context = f"""Iteration {iteration}/{max_iterations}
Success rate: {success_count}/{success_count + fail_count} tools succeeded
Repeating: {"YES — agent may be stuck" if is_repeating else "no"}

Original request: {original_request}

Recent tool results:
{chr(10).join(results_summary) if results_summary else '  No tools executed yet.'}"""

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
        result = _extract_observer_json(response_text)
        if result:
            status = result.get("status", "continue")
            if status not in ("complete", "continue", "replan"):
                status = "continue"
            return {
                "status": status,
                "confidence": min(1.0, max(0.0, float(result.get("confidence", 0.5)))),
                "reason": result.get("reason", ""),
            }
    except Exception as e:
        logger.warning("Observer parse failed: %s", e)

    # Smart fallback based on metrics (NOT just "continue")
    if iteration >= max_iterations:
        return {"status": "complete", "confidence": 0.3, "reason": "Max iterations reached"}
    if is_repeating:
        return {"status": "replan", "confidence": 0.7, "reason": "Agent repeating same actions"}
    if fail_count > 0 and success_count == 0:
        return {"status": "replan", "confidence": 0.6, "reason": "All recent tools failed"}
    if success_count > 0 and fail_count == 0:
        return {"status": "complete", "confidence": 0.6, "reason": "All recent tools succeeded"}

    return {"status": "continue", "confidence": 0.5, "reason": "Parse failed, continuing cautiously"}


def _extract_observer_json(text: str) -> dict | None:
    """Extract JSON object from observer response."""
    # Try code fence first
    fence_match = re.search(r'```(?:json)?\s*(\{[\s\S]*?\})\s*```', text)
    if fence_match:
        try:
            return json.loads(fence_match.group(1))
        except json.JSONDecodeError:
            pass
    # Try finding { ... }
    start = text.find("{")
    end = text.rfind("}") + 1
    if start >= 0 and end > start:
        try:
            return json.loads(text[start:end])
        except json.JSONDecodeError:
            pass
    return None
