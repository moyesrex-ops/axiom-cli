"""Context compressor — summarize old messages to save token space.

When the conversation grows long, this module summarizes older messages
into a compact form while preserving key information, achieving ~70-80%
token reduction while maintaining continuity.

**Critical:** Preserves tool_call → tool_result message pairing across
the compression boundary to avoid litellm ``BadRequestError``.
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

    Preserves tool_call → tool_result pairing across the split boundary.
    """
    system_messages = [m for m in messages if m.get("role") == "system"]
    conversation = [m for m in messages if m.get("role") != "system"]

    if len(conversation) <= keep_recent:
        return messages

    # Find safe split point that doesn't break tool chains
    split_idx = len(conversation) - keep_recent
    split_idx = _find_safe_split(conversation, split_idx)

    if split_idx <= 0:
        return messages  # Can't compress safely

    old_messages = conversation[:split_idx]
    recent_messages = conversation[split_idx:]

    # Validate: recent_messages must not start with a tool response
    # that references a tool_call in old_messages
    if not _validate_message_pairing(recent_messages):
        logger.warning("Cannot compress: tool pairing would break")
        return messages

    # Format old messages for summarization (preserve tool metadata)
    formatted_old = []
    for msg in old_messages:
        role = msg.get("role", "unknown")
        content = str(msg.get("content", ""))[:500]
        if role == "assistant" and "tool_calls" in msg:
            tool_names = [
                tc.get("function", {}).get("name", "?")
                for tc in msg.get("tool_calls", [])
                if isinstance(tc, dict)
            ]
            formatted_old.append(f"[{role}]: Called tools: {', '.join(tool_names)}")
        elif role == "tool":
            tool_name = msg.get("name", "unknown")
            formatted_old.append(f"[tool:{tool_name}]: {content}")
        else:
            formatted_old.append(f"[{role}]: {content}")

    conversation_text = "\n".join(formatted_old)
    summary = await _generate_summary(router, conversation_text)

    compressed: list[dict[str, Any]] = list(system_messages)
    if summary:
        compressed.append({
            "role": "system",
            "content": (
                f"[Context Summary — {len(old_messages)} messages compressed]\n"
                f"{summary}"
            ),
        })
    compressed.extend(recent_messages)

    logger.info(
        "Context compressed: %d -> %d messages (%.0f%% reduction)",
        len(messages), len(compressed),
        ((len(messages) - len(compressed)) / len(messages) * 100),
    )
    return compressed


def _find_safe_split(conversation: list[dict[str, Any]], target_idx: int) -> int:
    """Find the nearest split point that doesn't break a tool_call → tool chain.

    Scans backward from target_idx to find a position where:
    - The message at split_idx is NOT a 'tool' role message
    - The message at split_idx-1 does NOT have 'tool_calls'
    """
    idx = target_idx
    # Scan backward up to 10 positions to find a safe split
    for _ in range(min(10, idx)):
        if idx <= 0:
            return 0
        msg_at_split = conversation[idx]
        msg_before_split = conversation[idx - 1]

        # Bad: recent starts with tool response
        if msg_at_split.get("role") == "tool":
            idx -= 1
            continue
        # Bad: last old message has tool_calls (its results are in recent)
        if "tool_calls" in msg_before_split:
            idx -= 1
            continue
        # Safe split found
        return idx

    # If we couldn't find a safe point, try scanning forward
    idx = target_idx
    for _ in range(min(10, len(conversation) - idx)):
        if idx >= len(conversation):
            return len(conversation)
        msg_at_split = conversation[idx]
        if msg_at_split.get("role") != "tool" and (
            idx == 0 or "tool_calls" not in conversation[idx - 1]
        ):
            return idx
        idx += 1

    return target_idx  # Give up, use original


def _validate_message_pairing(messages: list[dict[str, Any]]) -> bool:
    """Validate that tool responses have matching tool_calls in the same list."""
    pending_tool_call_ids: set[str] = set()

    for msg in messages:
        if msg.get("role") == "assistant" and "tool_calls" in msg:
            for tc in msg.get("tool_calls", []):
                if isinstance(tc, dict) and "id" in tc:
                    pending_tool_call_ids.add(tc["id"])
        elif msg.get("role") == "tool" and "tool_call_id" in msg:
            tc_id = msg["tool_call_id"]
            if tc_id not in pending_tool_call_ids:
                return False  # Orphaned tool response
            pending_tool_call_ids.discard(tc_id)

    return True


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
