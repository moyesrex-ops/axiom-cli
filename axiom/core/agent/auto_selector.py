"""Auto-selector -- choose the best agent mode for a task.

Analyzes the user's message to determine whether PLAN, REACT,
or COUNCIL mode is most appropriate.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Keywords that suggest structured, multi-step work (-> PLAN mode)
PLAN_KEYWORDS = {
    "create", "build", "implement", "add", "write", "generate",
    "set up", "setup", "configure", "install", "deploy", "scaffold",
    "refactor", "migrate", "update", "upgrade", "fix", "debug",
    "modify", "delete", "remove", "replace", "restructure",
    "design", "architect", "plan", "organize", "automate",
}

# Keywords that suggest exploratory work (-> REACT mode)
REACT_KEYWORDS = {
    "what", "how", "why", "explain", "describe", "show",
    "find", "search", "look", "investigate", "analyze", "check",
    "tell", "list", "help", "where", "which", "compare",
    "summarize", "review", "read", "explore",
}

# Keywords that suggest high-stakes decisions (-> COUNCIL mode)
COUNCIL_KEYWORDS = {
    "best approach", "pros and cons", "trade-offs", "tradeoffs",
    "architecture decision", "which framework", "compare options",
    "should i use", "recommend", "evaluate alternatives",
    "critical decision", "important choice", "which is better",
}


def auto_select_mode(
    message: str,
    available_modes: list[str] | None = None,
) -> str:
    """Determine the best agent mode for a given user message.

    Args:
        message: The user's input message.
        available_modes: List of available modes. Defaults to all.

    Returns:
        One of ``"plan"``, ``"react"``, ``"council"``, or ``"chat"``.
    """
    available = set(available_modes or ["plan", "react", "council", "chat"])
    message_lower = message.lower().strip()
    words = set(message_lower.split())

    # Check for explicit mode requests
    if message_lower.startswith("/plan"):
        return "plan" if "plan" in available else "react"
    if message_lower.startswith("/react"):
        return "react" if "react" in available else "plan"
    if message_lower.startswith("/council"):
        return "council" if "council" in available else "react"

    # Score each mode
    plan_score = sum(1 for kw in PLAN_KEYWORDS if kw in message_lower)
    react_score = sum(1 for kw in REACT_KEYWORDS if kw in message_lower)
    council_score = sum(1 for kw in COUNCIL_KEYWORDS if kw in message_lower)

    # Short messages (< 10 words) that are questions -> chat or react
    if len(words) < 10 and message_lower.endswith("?"):
        if council_score > 0 and "council" in available:
            return "council"
        return "react" if "react" in available else "chat"

    # Very short messages -> chat
    if len(words) < 5 and plan_score == 0:
        return "chat" if "chat" in available else "react"

    # Council wins if council keywords found
    if council_score > 0 and "council" in available:
        return "council"

    # Plan wins for action-oriented tasks
    if plan_score > react_score and "plan" in available:
        return "plan"

    # React wins for exploratory tasks
    if react_score > 0 and "react" in available:
        return "react"

    # Default: plan for longer messages, chat for shorter
    if len(words) >= 10:
        return "plan" if "plan" in available else "react"

    return "chat" if "chat" in available else "react"
