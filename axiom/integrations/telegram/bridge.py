"""Telegram ↔ Axiom bridge.

Wraps AxiomApp.chat_headless() so Telegram messages get processed through
the full agent pipeline and results come back formatted for Telegram.

The bridge uses the shared ConversationStore — so CLI and Telegram see
the same conversation thread (mirrored experience).
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from axiom.integrations.telegram.formatter import (
    escape_md2,
    format_agent_response,
)

logger = logging.getLogger(__name__)

# Keywords that trigger automatic task creation
_TASK_KEYWORDS = (
    "remind me",
    "task:",
    "todo:",
    "remember to",
    "don't forget",
    "dont forget",
    "schedule",
)


class TelegramBridge:
    """Bridge between Telegram input and the Axiom agent pipeline.

    Uses the shared ConversationStore (via ``app.conversation_store``)
    instead of maintaining its own per-user dict — this is what makes
    CLI ↔ Telegram mirroring work.
    """

    def __init__(self, app: Any) -> None:
        """
        Args:
            app: An AxiomApp instance (from axiom.cli.app).
        """
        self.app = app

    # ── Convenience properties ─────────────────────────────────

    @property
    def conversation_store(self):
        """Shared conversation store (same as CLI uses)."""
        return self.app.conversation_store

    @property
    def task_store(self):
        """Shared task store (same as CLI uses)."""
        return self.app.task_store

    # ── Message processing ─────────────────────────────────────

    async def process_message(self, user_id: int, text: str) -> str:
        """Process a text message from Telegram through the Axiom agent.

        Args:
            user_id: Telegram user ID.
            text: The user's message text.

        Returns:
            Formatted response string (MarkdownV2).
        """
        # Handle slash commands first
        if text.startswith("/"):
            return await self._handle_command(user_id, text)

        # Auto-detect task intent and store
        await self._maybe_create_task(text)

        try:
            # Use the headless pipeline — shared conversation + no Rich rendering
            response = await self.app.chat_headless(
                text, channel="telegram", user_id=user_id,
            )
            return format_agent_response(response)

        except Exception as exc:
            logger.error("Bridge error for user %s: %s", user_id, exc)
            return f"❌ Error: `{escape_md2(str(exc)[:500])}`"

    async def process_agent_task(self, user_id: int, text: str) -> list[str]:
        """Process a message and return multiple chunks for long responses.

        Returns a list of formatted message chunks (Telegram max 4096 chars).
        """
        # Auto-detect task intent and store
        await self._maybe_create_task(text)

        try:
            response = await self.app.chat_headless(
                text, channel="telegram", user_id=user_id,
            )
            formatted = format_agent_response(response)

            # Chunk the response if it's too long for a single Telegram message
            if len(formatted) > 4000:
                return [formatted[i : i + 4000] for i in range(0, len(formatted), 4000)]
            return [formatted]

        except Exception as exc:
            logger.error("Agent task error for user %s: %s", user_id, exc)
            return [f"❌ Error: `{escape_md2(str(exc)[:500])}`"]

    # ── Task auto-detection ────────────────────────────────────

    async def _maybe_create_task(self, text: str) -> None:
        """If the message looks like a task request, auto-create one."""
        lower = text.lower().strip()
        if any(lower.startswith(kw) for kw in _TASK_KEYWORDS):
            try:
                await self.task_store.add(
                    text, source="telegram", priority="normal",
                )
            except Exception as exc:
                logger.debug("Task auto-creation failed: %s", exc)

    # ── Slash commands ─────────────────────────────────────────

    async def _handle_command(self, user_id: int, text: str) -> str:
        """Handle Telegram slash commands.

        Supports:
            /start — Welcome message
            /status — Current model, tools, memory info
            /clear — Clear conversation history
            /tasks — Show pending tasks
            /model <name> — Switch model
            /help — List commands
        """
        parts = text.strip().split(maxsplit=1)
        cmd = parts[0].lower()
        arg = parts[1] if len(parts) > 1 else ""

        if cmd == "/start":
            return (
                "🤖 *Axiom* is online\\!\n\n"
                "Send me any message and I'll process it through the "
                "full Axiom agent pipeline\\.\n\n"
                "Your conversation is *mirrored* with CLI \\— "
                "everything you say here shows up there too\\.\n\n"
                "Commands:\n"
                "/status \\- Current status\n"
                "/tasks \\- Pending tasks\n"
                "/model \\<name\\> \\- Switch model\n"
                "/clear \\- Reset conversation\n"
                "/help \\- Show this help"
            )

        if cmd == "/help":
            return (
                "📋 *Axiom Telegram Commands*\n\n"
                "/status \\- Model, tools, memory info\n"
                "/tasks \\- Show pending tasks\n"
                "/model \\<name\\> \\- Switch LLM\n"
                "/models \\- List all available models\n"
                "/clear \\- Clear conversation\n"
                "/help \\- This help message\n\n"
                "Just type any message to chat with Axiom\\!"
            )

        if cmd == "/clear":
            try:
                await self.conversation_store.clear()
            except Exception:
                pass
            return "🗑️ Conversation cleared and archived\\."

        if cmd == "/tasks":
            try:
                pending = await self.task_store.get_pending()
                if not pending:
                    return "✅ No pending tasks\\."
                lines = ["📋 *Pending Tasks*\n"]
                for t in pending[:10]:
                    desc = escape_md2(t["description"][:60])
                    lines.append(f"  • \\#{t['id']}: {desc}")
                return "\n".join(lines)
            except Exception as exc:
                return f"❌ Error loading tasks: `{escape_md2(str(exc))}`"

        if cmd == "/status":
            model = "unknown"
            if hasattr(self.app, "router") and self.app.router:
                model = getattr(self.app.router, "active_model", "unknown")
            tool_count = 0
            if hasattr(self.app, "registry") and self.app.registry:
                tool_count = self.app.registry.count
            try:
                conv_stats = await self.conversation_store.get_stats()
                msg_count = conv_stats.get("messages", 0)
                channels = ", ".join(conv_stats.get("channels", set())) or "none"
            except Exception:
                msg_count = 0
                channels = "unknown"
            return (
                f"🤖 *Axiom Status*\n\n"
                f"Model: `{escape_md2(model)}`\n"
                f"Tools: `{tool_count}`\n"
                f"Messages: `{msg_count}`\n"
                f"Channels: `{escape_md2(channels)}`\n"
            )

        if cmd in ("/model", "/models"):
            if cmd == "/models" or not arg:
                try:
                    available = self.app.settings.available_providers()
                    lines = [f"📦 *Available Providers*\n"]
                    for p in available:
                        lines.append(f"  • `{escape_md2(p)}`")
                    return "\n".join(lines)
                except Exception:
                    return "Could not list models\\."

            # Switch model
            try:
                if hasattr(self.app, "_handle_model_command"):
                    self.app._handle_model_command(arg)
                    return f"✅ Switched to `{escape_md2(arg)}`"
                return f"Model switching not available\\."
            except Exception as exc:
                return f"❌ Error switching model: `{escape_md2(str(exc))}`"

        return f"Unknown command: `{escape_md2(cmd)}`\\. Try /help"
