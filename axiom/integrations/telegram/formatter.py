"""Rich → Telegram MarkdownV2 formatting.

Telegram's MarkdownV2 requires escaping of 18 special characters.
This module converts Axiom's Rich-formatted output into safe Telegram messages.
"""

from __future__ import annotations

import re
from typing import Optional

# ── Characters that MUST be escaped in Telegram MarkdownV2 ───────────
_MD2_SPECIAL = r"_*[]()~`>#+-=|{}.!"


def escape_md2(text: str) -> str:
    """Escape all MarkdownV2 special characters in *text*.

    This is the core safety function — every piece of dynamic text that
    goes into a Telegram message MUST pass through this first.
    """
    return re.sub(r"([" + re.escape(_MD2_SPECIAL) + r"])", r"\\\1", text)


def format_tool_call(tool_name: str, args: dict) -> str:
    """Format a tool invocation for Telegram display.

    Example output (MarkdownV2):
        🔧 *bash*
        ```
        command: ls -la
        ```
    """
    escaped_name = escape_md2(tool_name)
    args_text = "\n".join(f"{k}: {v}" for k, v in args.items())
    # Code blocks don't need internal escaping in MD2
    return f"🔧 *{escaped_name}*\n```\n{args_text}\n```"


def format_tool_result(tool_name: str, success: bool, result: str, duration_ms: float = 0) -> str:
    """Format a tool result for Telegram display.

    Example:
        ✅ bash (120ms)
        ```
        file1.py  file2.py
        ```
    """
    icon = "✅" if success else "❌"
    dur = f" \\({int(duration_ms)}ms\\)" if duration_ms else ""
    truncated = truncate_for_telegram(result, max_len=1500)
    escaped_name = escape_md2(tool_name)
    return f"{icon} *{escaped_name}*{dur}\n```\n{truncated}\n```"


def format_agent_response(text: str) -> str:
    """Convert an agent response to Telegram-safe MarkdownV2.

    Preserves code blocks, bold, and italic while escaping everything else.
    Handles the tricky case where code blocks should NOT have their
    content escaped, but surrounding text should.
    """
    # Split on code blocks (``` ... ```)
    parts = re.split(r"(```[\s\S]*?```)", text)
    formatted_parts = []

    for part in parts:
        if part.startswith("```"):
            # Code blocks are sent as-is (Telegram handles them natively)
            formatted_parts.append(part)
        else:
            # For regular text, escape special chars but preserve
            # common markdown that Telegram supports
            converted = _convert_inline_formatting(part)
            formatted_parts.append(converted)

    return "".join(formatted_parts)


def _convert_inline_formatting(text: str) -> str:
    """Convert standard markdown inline formatting to Telegram MarkdownV2.

    Handles: **bold** → *bold*, *italic* → _italic_
    Escapes everything else.
    """
    # Extract bold and italic patterns first
    bold_parts = []
    italic_parts = []

    # Replace **bold** markers temporarily
    text = re.sub(
        r"\*\*(.*?)\*\*",
        lambda m: f"__BOLD_START__{m.group(1)}__BOLD_END__",
        text,
    )
    # Replace *italic* markers temporarily
    text = re.sub(
        r"\*(.*?)\*",
        lambda m: f"__ITALIC_START__{m.group(1)}__ITALIC_END__",
        text,
    )

    # Escape all special characters
    text = escape_md2(text)

    # Restore formatting markers (now with escaped content)
    text = text.replace("__BOLD_START__", "*").replace("__BOLD_END__", "*")
    text = text.replace("__ITALIC_START__", "_").replace("__ITALIC_END__", "_")

    return text


def truncate_for_telegram(text: str, max_len: int = 4000) -> str:
    """Truncate text to fit Telegram's message limit (4096 chars).

    Leaves room for formatting wrappers. Adds truncation indicator.
    """
    if len(text) <= max_len:
        return text
    return text[:max_len] + "\n... (truncated)"


def format_status_message(
    model: str,
    mode: str,
    tool_count: int,
    memory_count: int = 0,
) -> str:
    """Format a status/info message for Telegram.

    Used when user sends /status to the bot.
    """
    return (
        f"🤖 *Axiom Status*\n\n"
        f"Model: `{escape_md2(model)}`\n"
        f"Mode: `{escape_md2(mode)}`\n"
        f"Tools: `{tool_count}`\n"
        f"Memories: `{memory_count}`\n"
    )
