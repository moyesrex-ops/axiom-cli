"""Synthesizer node -- generates the final answer from agent work."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

SYNTH_SYSTEM = """You are Axiom, synthesizing the final answer from completed work.

Based on the tool results and reasoning below, provide a clear, comprehensive answer
to the user's original request. Include:
- What was accomplished
- Key findings or results
- Any files created/modified
- Next steps if applicable

Be concise but thorough. Use markdown formatting."""


async def synthesize_answer(
    router: Any,
    messages: list[dict[str, str]],
    tool_results: list[dict[str, Any]],
) -> str:
    """Generate a final synthesized answer from agent work.

    Returns the synthesized answer text.
    """
    # Build context from tool results
    results_context = []
    for r in tool_results:
        status = "SUCCESS" if r.get("success") else "FAILED"
        result_text = str(r.get("result", ""))[:2000]
        results_context.append(
            f"**{r.get('tool_name', 'unknown')}** [{status}]:\n{result_text}\n"
        )

    synth_messages = [
        {"role": "system", "content": SYNTH_SYSTEM},
        *messages,
        {
            "role": "user",
            "content": f"Tool execution results:\n\n{''.join(results_context)}\n\nSynthesize a final answer.",
        },
    ]

    response_text = ""
    async for chunk in router.complete(messages=synth_messages, stream=True):
        if hasattr(chunk, "choices") and chunk.choices:
            delta = chunk.choices[0].delta
            if hasattr(delta, "content") and delta.content:
                response_text += delta.content

    return response_text or "Task completed but no summary could be generated."
