"""Planner node -- generates multi-step plans or ReAct thoughts."""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Optional

logger = logging.getLogger(__name__)

PLAN_SYSTEM = """You are Axiom, an expert AI agent that creates precise execution plans.

Given a user task, create a step-by-step plan. Each step should:
1. Have a clear action (use a specific tool or reasoning)
2. Be independently executable
3. Build on previous steps

Available tools: {tool_names}

Respond with a JSON array of steps:
[
  {{"step": 1, "action": "tool_name", "description": "What to do", "args": {{}}}},
  {{"step": 2, "action": "tool_name", "description": "What to do", "args": {{}}}}
]

If no tools are needed (pure reasoning), use "action": "think".
Keep plans concise -- 2-6 steps for most tasks."""

REACT_THOUGHT_SYSTEM = """You are Axiom, an expert AI agent using ReAct (Reasoning + Acting).

For each turn:
1. **Thought**: Analyze the current state and decide what to do next
2. **Action**: Choose a tool to execute (or "finish" if done)
3. **Observation**: You'll receive the tool result

Think step by step. Be concise but thorough.

Available tools: {tool_names}

Respond with JSON:
{{"thought": "your reasoning", "action": "tool_name", "args": {{}}}}

Use "action": "finish" with "args": {{"answer": "..."}} when the task is complete."""


async def generate_plan(
    router: Any,
    messages: list[dict[str, str]],
    tool_names: list[str],
) -> list[dict[str, Any]]:
    """Generate a multi-step plan for PLAN mode."""
    system_prompt = PLAN_SYSTEM.format(tool_names=", ".join(tool_names))

    plan_messages = [
        {"role": "system", "content": system_prompt},
        *messages,
        {"role": "user", "content": "Create an execution plan for the above task. Return ONLY a JSON array."},
    ]

    response_text = ""
    async for chunk in router.complete(messages=plan_messages, stream=True):
        if hasattr(chunk, "choices") and chunk.choices:
            delta = chunk.choices[0].delta
            if hasattr(delta, "content") and delta.content:
                response_text += delta.content

    plan = _extract_json_array(response_text)
    if plan is not None:
        # Validate each step has required fields
        validated = []
        for i, step in enumerate(plan):
            if not isinstance(step, dict):
                continue
            validated.append({
                "step": step.get("step", i + 1),
                "action": step.get("action", "think"),
                "description": step.get("description", ""),
                "args": step.get("args", {}),
            })
        if validated:
            return validated

    # Fallback: wrap the response as a single think step
    logger.warning("Plan parse failed, falling back to single think step")
    return [{"step": 1, "action": "think", "description": response_text[:500], "args": {}}]


async def generate_react_thought(
    router: Any,
    messages: list[dict[str, str]],
    react_trace: list[dict[str, Any]],
    tool_names: list[str],
) -> dict[str, Any]:
    """Generate a single ReAct thought+action."""
    system_prompt = REACT_THOUGHT_SYSTEM.format(tool_names=", ".join(tool_names))

    trace_context = ""
    for entry in react_trace[-5:]:
        trace_context += f"\nThought: {entry.get('thought', '')}"
        trace_context += f"\nAction: {entry.get('action', '')}"
        if 'observation' in entry:
            obs = str(entry['observation'])[:1000]
            trace_context += f"\nObservation: {obs}"

    react_messages = [
        {"role": "system", "content": system_prompt},
        *messages,
    ]

    if trace_context:
        react_messages.append({
            "role": "user",
            "content": f"Previous steps:\n{trace_context}\n\nWhat's your next thought and action?"
        })

    response_text = ""
    async for chunk in router.complete(messages=react_messages, stream=True):
        if hasattr(chunk, "choices") and chunk.choices:
            delta = chunk.choices[0].delta
            if hasattr(delta, "content") and delta.content:
                response_text += delta.content

    result = _extract_json_object(response_text)
    if result is not None:
        return {
            "thought": result.get("thought", ""),
            "action": result.get("action", "finish"),
            "args": result.get("args", {}),
        }

    return {"thought": response_text, "action": "finish", "args": {"answer": response_text}}


# ── Robust JSON Extraction ─────────────────────────────────────────────────────


def _extract_json_array(text: str) -> list | None:
    """Extract a JSON array from LLM response text.

    Tries multiple strategies:
    1. Code fence extraction (```json ... ```)
    2. Balanced bracket matching (skipping brackets inside strings)
    3. Raw parse of entire text
    """
    # Strategy 1: Extract from code fence
    fence_match = re.search(r'```(?:json)?\s*(\[[\s\S]*?\])\s*```', text)
    if fence_match:
        try:
            return json.loads(fence_match.group(1))
        except json.JSONDecodeError:
            pass

    # Strategy 2: Find the first [ that starts a valid JSON array
    for i, ch in enumerate(text):
        if ch == '[':
            end = _find_matching_bracket(text, i, '[', ']')
            if end is not None:
                try:
                    return json.loads(text[i:end + 1])
                except json.JSONDecodeError:
                    continue  # This [ wasn't the start, try next

    # Strategy 3: Try parsing the whole text
    try:
        result = json.loads(text.strip())
        if isinstance(result, list):
            return result
    except json.JSONDecodeError:
        pass

    return None


def _extract_json_object(text: str) -> dict | None:
    """Extract a JSON object from LLM response text."""
    # Strategy 1: Code fence
    fence_match = re.search(r'```(?:json)?\s*(\{[\s\S]*?\})\s*```', text)
    if fence_match:
        try:
            return json.loads(fence_match.group(1))
        except json.JSONDecodeError:
            pass

    # Strategy 2: Balanced brace matching
    for i, ch in enumerate(text):
        if ch == '{':
            end = _find_matching_bracket(text, i, '{', '}')
            if end is not None:
                try:
                    return json.loads(text[i:end + 1])
                except json.JSONDecodeError:
                    continue

    # Strategy 3: Parse whole text
    try:
        result = json.loads(text.strip())
        if isinstance(result, dict):
            return result
    except json.JSONDecodeError:
        pass

    return None


def _find_matching_bracket(
    text: str, start: int, open_ch: str, close_ch: str
) -> int | None:
    """Find the matching closing bracket, respecting string context.

    Returns the index of the closing bracket, or None if not found.
    """
    depth = 0
    in_string = False
    escape = False

    for j in range(start, len(text)):
        c = text[j]
        if escape:
            escape = False
            continue
        if c == '\\' and in_string:
            escape = True
            continue
        if c == '"' and not escape:
            in_string = not in_string
            continue
        if not in_string:
            if c == open_ch:
                depth += 1
            elif c == close_ch:
                depth -= 1
                if depth == 0:
                    return j

    return None
