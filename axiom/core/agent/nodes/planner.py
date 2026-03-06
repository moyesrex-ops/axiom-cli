"""Planner node -- generates multi-step plans or ReAct thoughts."""

from __future__ import annotations

import json
import logging
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
    """Generate a multi-step plan for PLAN mode.

    Returns a list of plan steps, each with:
    - step: int
    - action: str (tool name or "think")
    - description: str
    - args: dict
    """
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

    # Parse the plan from the response
    try:
        # Try to extract JSON from the response
        start = response_text.find("[")
        end = response_text.rfind("]") + 1
        if start >= 0 and end > start:
            plan = json.loads(response_text[start:end])
            return plan
    except (json.JSONDecodeError, ValueError) as e:
        logger.warning("Failed to parse plan: %s", e)

    # Fallback: single-step plan
    return [{"step": 1, "action": "think", "description": response_text, "args": {}}]


async def generate_react_thought(
    router: Any,
    messages: list[dict[str, str]],
    react_trace: list[dict[str, Any]],
    tool_names: list[str],
) -> dict[str, Any]:
    """Generate a single ReAct thought+action for REACT mode.

    Returns dict with: thought, action, args
    """
    system_prompt = REACT_THOUGHT_SYSTEM.format(tool_names=", ".join(tool_names))

    # Build context from react trace
    trace_context = ""
    for entry in react_trace[-5:]:  # Keep last 5 steps
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

    # Parse the thought from the response
    try:
        start = response_text.find("{")
        end = response_text.rfind("}") + 1
        if start >= 0 and end > start:
            thought = json.loads(response_text[start:end])
            return thought
    except (json.JSONDecodeError, ValueError) as e:
        logger.warning("Failed to parse react thought: %s", e)

    return {"thought": response_text, "action": "finish", "args": {"answer": response_text}}
