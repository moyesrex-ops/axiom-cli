"""Learner node -- extract reusable patterns from successful agent runs.

After a successful task completion, the learner analyzes the execution
trace and extracts patterns, skills, and strategies that can be
reused in future tasks. Implements the self-improvement loop.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

LEARNER_SYSTEM = """You are analyzing a successful agent execution trace to extract reusable patterns.

From the trace below, extract:
1. **Strategy**: What approach worked well? (2-3 sentences)
2. **Tools Used**: Which tools were most effective and in what sequence?
3. **Pitfalls Avoided**: Any errors that were recovered from?
4. **Reusable Pattern**: A concise description of the pattern that could help with similar future tasks.

Format your response as a Markdown skill file:

```
---
name: <skill-name>
description: <one-line description>
---
# <Skill Title>

## Strategy
<what worked>

## Tool Sequence
<effective tool ordering>

## Key Insights
<lessons learned>
```

Keep it concise (under 500 words). Focus on ACTIONABLE patterns."""


async def extract_learnings(
    router: Any,
    task: str,
    tool_results: list[dict[str, Any]],
    final_answer: str,
    save_dir: Path | None = None,
) -> str:
    """Analyze a successful run and extract reusable patterns.

    Args:
        router: LLM router for analysis.
        task: The original user task.
        tool_results: List of tool execution records.
        final_answer: The synthesized answer.
        save_dir: Directory to save extracted skill files.

    Returns:
        Extracted pattern as markdown string.
    """
    # Build trace summary
    trace_parts = [f"**Task:** {task}\n"]

    for i, r in enumerate(tool_results, 1):
        status = "OK" if r.get("success") else "FAILED"
        tool = r.get("tool_name", "unknown")
        result_preview = str(r.get("result", ""))[:300]
        duration = r.get("duration_ms", 0)
        trace_parts.append(
            f"{i}. [{status}] **{tool}** ({duration}ms): {result_preview}"
        )

    trace_parts.append(f"\n**Final Answer (preview):** {final_answer[:500]}")
    trace_text = "\n".join(trace_parts)

    messages = [
        {"role": "system", "content": LEARNER_SYSTEM},
        {"role": "user", "content": trace_text},
    ]

    response_text = ""
    try:
        async for chunk in router.complete(messages=messages, stream=True):
            if hasattr(chunk, "choices") and chunk.choices:
                delta = chunk.choices[0].delta
                if hasattr(delta, "content") and delta.content:
                    response_text += delta.content
    except Exception as exc:
        logger.debug("Learner extraction failed: %s", exc)
        return ""

    # Save to skills directory if path provided
    if response_text and save_dir:
        try:
            save_dir.mkdir(parents=True, exist_ok=True)

            # Generate a filename from the task
            safe_name = re.sub(r"[^\w\s-]", "", task[:50]).strip().replace(" ", "-").lower()
            if not safe_name:
                safe_name = "skill"
            timestamp = datetime.now().strftime("%Y%m%d-%H%M")
            filename = f"{safe_name}-{timestamp}.md"

            skill_path = save_dir / filename
            skill_path.write_text(response_text, encoding="utf-8")
            logger.info("Saved learned skill to %s", skill_path)
        except Exception as exc:
            logger.debug("Failed to save skill: %s", exc)

    return response_text


async def should_learn(
    tool_results: list[dict[str, Any]],
    min_tools: int = 3,
    min_success_rate: float = 0.7,
) -> bool:
    """Determine if a run is worth learning from.

    Only extract patterns from runs that:
    - Used at least ``min_tools`` tool calls
    - Had a success rate above ``min_success_rate``
    """
    if len(tool_results) < min_tools:
        return False

    successes = sum(1 for r in tool_results if r.get("success"))
    success_rate = successes / len(tool_results) if tool_results else 0

    return success_rate >= min_success_rate
