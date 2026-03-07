"""Synthesizer node -- generates the final answer from completed agent work."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

SYNTH_SYSTEM = """You are Axiom, reporting the results of work you have ALREADY completed.

CRITICAL RULES:
1. These tools have ALREADY BEEN EXECUTED. Report what was ACCOMPLISHED (past tense).
2. DO NOT give advice, suggestions, or "next steps". The work is DONE.
3. DO NOT say "you should", "you could", "consider", or "I recommend".
4. DO say "I created", "I installed", "I configured", "I found", "The results show".
5. If tools failed, explain what went wrong and what you tried differently.
6. Include exact file paths, URLs, and outputs from the tool results.
7. Be concise. No filler. No preamble like "Sure!" or "Great question!".

FORMAT:
- Start with a 1-line summary of what was accomplished
- List specific actions taken with results
- Include file paths and command outputs where relevant
- If files were created, list them with their purpose
- End with the final state (what exists now, not what to do next)"""


async def synthesize_answer(
    router: Any,
    messages: list[dict[str, str]],
    tool_results: list[dict[str, Any]],
) -> str:
    """Generate a final answer reporting completed work.

    Returns the synthesized answer text.
    """
    # Extract the original user request
    original_request = ""
    for msg in messages:
        if msg.get("role") == "user":
            original_request = msg.get("content", "")
            break

    # Build context from tool results — keep more detail for important results
    results_context = []
    for r in tool_results:
        status = "SUCCESS" if r.get("success") else "FAILED"
        tool_name = r.get("tool_name", r.get("tool", "unknown"))
        result_text = str(r.get("result", ""))

        # Smart truncation: keep first 1500 + last 500 for long results
        if len(result_text) > 2500:
            result_text = (
                result_text[:1500]
                + "\n... [truncated] ...\n"
                + result_text[-500:]
            )

        results_context.append(
            f"**{tool_name}** [{status}]:\n{result_text}\n"
        )

    synth_messages = [
        {"role": "system", "content": SYNTH_SYSTEM},
        {
            "role": "user",
            "content": (
                f"ORIGINAL USER REQUEST:\n{original_request}\n\n"
                f"COMPLETED TOOL RESULTS:\n\n{''.join(results_context)}\n\n"
                f"Report what was accomplished. Past tense only. No advice."
            ),
        },
    ]

    response_text = ""
    async for chunk in router.complete(messages=synth_messages, stream=True):
        if hasattr(chunk, "choices") and chunk.choices:
            delta = chunk.choices[0].delta
            if hasattr(delta, "content") and delta.content:
                response_text += delta.content

    return response_text or "Task completed but no summary could be generated."
