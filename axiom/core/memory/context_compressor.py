"""Context compressor — summarize old messages to save token space.

When the conversation grows long, this module summarizes older messages
into a compact form while preserving key information, achieving ~70-80%
token reduction while maintaining continuity.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Default thresholds
COMPRESS_AFTER_MESSAGES = 15
KEEP_RECENT_MESSAGES = 5
MAX_SUMMARY_CHARS = 2000


COMPRESS_PROMPT = """Summarize the following conversation into structured bullet points.
Preserve:
- Key decisions made
- Important facts learned
- Tool results and file changes
- User preferences expressed
- Any errors encountered and how they were resolved

Be concise but complete. Focus on information that would be needed to continue the conversation.

Conversation to summarize:
{conversation}

Respond with a structured summary using bullet points."""


async def should_compress(
    messages: list[dict[str, Any]],
    max_messages: int = COMPRESS_AFTER_MESSAGES,
) -> bool:
    """Check if the conversation should be compressed.

    Returns True if the message count exceeds the threshold.
    """
    # Only count user/assistant messages (not system)
    conversation_messages = [
        m for m in messages if m.get("role") in ("user", "assistant")
    ]
    return len(conversation_messages) >= max_messages


async def compress_context(
    router: Any,
    messages: list[dict[str, Any]],
    keep_recent: int = KEEP_RECENT_MESSAGES,
) -> list[dict[str, Any]]:
    """Compress older messages into a summary while keeping recent ones.

    Args:
        router: UniversalRouter for LLM calls.
        messages: Full conversation messages.
        keep_recent: Number of recent messages to keep uncompressed.

    Returns:
        New message list with old messages replaced by a summary.
    """
    # Separate system messages from conversation
    system_messages = [m for m in messages if m.get("role") == "system"]
    conversation = [m for m in messages if m.get("role") != "system"]

    if len(conversation) <= keep_recent:
        return messages  # Nothing to compress

    # Split into old (to compress) and recent (to keep)
    old_messages = conversation[:-keep_recent]
    recent_messages = conversation[-keep_recent:]

    # Format old messages for summarization
    formatted_old = []
    for msg in old_messages:
        role = msg.get("role", "unknown")
        content = str(msg.get("content", ""))[:500]
        formatted_old.append(f"[{role}]: {content}")

    conversation_text = "\n".join(formatted_old)

    # Generate summary using LLM
    summary = await _generate_summary(router, conversation_text)

    # Build compressed message list
    compressed: list[dict[str, Any]] = list(system_messages)

    # Add summary as a system message
    if summary:
        compressed.append({
            "role": "system",
            "content": (
                f"[Context Summary — {len(old_messages)} messages compressed]\n"
                f"{summary}"
            ),
        })

    # Add recent uncompressed messages
    compressed.extend(recent_messages)

    old_count = len(messages)
    new_count = len(compressed)
    reduction = ((old_count - new_count) / old_count * 100) if old_count > 0 else 0

    logger.info(
        "Context compressed: %d -> %d messages (%.0f%% reduction)",
        old_count, new_count, reduction,
    )

    return compressed


async def _generate_summary(router: Any, conversation_text: str) -> str:
    """Use the LLM to generate a conversation summary."""
    if not conversation_text.strip():
        return ""

    # Truncate if extremely long
    if len(conversation_text) > 10000:
        conversation_text = conversation_text[:10000] + "\n... [truncated]"

    prompt = COMPRESS_PROMPT.format(conversation=conversation_text)

    summary_messages = [
        {"role": "system", "content": "You are a precise summarizer. Be concise."},
        {"role": "user", "content": prompt},
    ]

    response_text = ""
    try:
        async for chunk in router.complete(messages=summary_messages, stream=True):
            if hasattr(chunk, "choices") and chunk.choices:
                delta = chunk.choices[0].delta
                if hasattr(delta, "content") and delta.content:
                    response_text += delta.content
    except Exception as e:
        logger.warning("Failed to generate summary: %s", e)
        # Fallback: create a basic summary from the text
        response_text = _fallback_summary(conversation_text)

    # Truncate if too long
    if len(response_text) > MAX_SUMMARY_CHARS:
        response_text = response_text[:MAX_SUMMARY_CHARS] + "\n... [summary truncated]"

    return response_text


def _fallback_summary(conversation_text: str) -> str:
    """Create a basic extractive summary without LLM."""
    lines = conversation_text.split("\n")
    # Keep first and last few lines of each role
    summary_parts: list[str] = []
    for line in lines:
        line = line.strip()
        if line.startswith("[user]"):
            summary_parts.append(line[:200])
        elif "error" in line.lower() or "success" in line.lower():
            summary_parts.append(line[:200])

    return "\n".join(summary_parts[:20]) if summary_parts else "(no summary available)"


def estimate_tokens(messages: list[dict[str, Any]]) -> int:
    """Rough token estimate for a message list.

    Uses the ~4 chars per token heuristic.
    """
    total_chars = sum(len(str(m.get("content", ""))) for m in messages)
    return total_chars // 4
