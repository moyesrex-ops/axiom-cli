"""Telegram bot handler — receives messages and routes through TelegramBridge.

Supports: text messages, voice notes (Whisper transcription),
photos (vision analysis), and documents (file processing).
"""

from __future__ import annotations

import io
import logging
import tempfile
from pathlib import Path
from typing import Any, Optional, Set

from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from axiom.integrations.telegram.bridge import TelegramBridge
from axiom.integrations.telegram.formatter import escape_md2

logger = logging.getLogger(__name__)


class TelegramBot:
    """Full-featured Telegram bot for Axiom.

    Handles text, voice, photo, and document messages.
    Routes everything through TelegramBridge for agent processing.
    """

    def __init__(
        self,
        token: str,
        bridge: TelegramBridge,
        allowed_users: Optional[Set[int]] = None,
    ) -> None:
        """
        Args:
            token: Telegram Bot API token.
            bridge: TelegramBridge instance connected to AxiomApp.
            allowed_users: Set of Telegram user IDs allowed to interact.
                           If None, all users are allowed (use with caution).
        """
        self.token = token
        self.bridge = bridge
        self.allowed_users = allowed_users
        self._app: Optional[Application] = None

    def _is_authorized(self, user_id: int) -> bool:
        """Check if a user is authorized to use the bot."""
        if self.allowed_users is None:
            return True
        return user_id in self.allowed_users

    async def _check_auth(self, update: Update) -> bool:
        """Check authorization and send rejection if unauthorized."""
        user_id = update.effective_user.id if update.effective_user else 0
        if not self._is_authorized(user_id):
            if update.message:
                await update.message.reply_text(
                    "⛔ Unauthorized. Your user ID is not in the allowed list."
                )
            return False
        return True

    # ── Command handlers ────────────────────────────────────────

    async def _handle_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await self._check_auth(update):
            return
        user_id = update.effective_user.id
        response = await self.bridge._handle_command(user_id, "/start")
        await update.message.reply_text(response, parse_mode="MarkdownV2")

    async def _handle_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await self._check_auth(update):
            return
        user_id = update.effective_user.id
        response = await self.bridge._handle_command(user_id, "/help")
        await update.message.reply_text(response, parse_mode="MarkdownV2")

    async def _handle_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await self._check_auth(update):
            return
        user_id = update.effective_user.id
        response = await self.bridge._handle_command(user_id, "/status")
        await update.message.reply_text(response, parse_mode="MarkdownV2")

    async def _handle_clear(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await self._check_auth(update):
            return
        user_id = update.effective_user.id
        response = await self.bridge._handle_command(user_id, "/clear")
        await update.message.reply_text(response, parse_mode="MarkdownV2")

    async def _handle_model(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await self._check_auth(update):
            return
        user_id = update.effective_user.id
        arg = " ".join(context.args) if context.args else ""
        response = await self.bridge._handle_command(user_id, f"/model {arg}")
        await update.message.reply_text(response, parse_mode="MarkdownV2")

    # ── Message handlers ────────────────────────────────────────

    async def _handle_text(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle regular text messages."""
        if not await self._check_auth(update):
            return

        user_id = update.effective_user.id
        text = update.message.text

        # Show typing indicator
        await update.message.chat.send_action("typing")

        try:
            response = await self.bridge.process_message(user_id, text)
            await self._send_long_message(update, response)
        except Exception as exc:
            logger.error("Text handler error: %s", exc)
            await update.message.reply_text(
                f"❌ Error: `{escape_md2(str(exc)[:200])}`",
                parse_mode="MarkdownV2",
            )

    async def _handle_voice(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle voice messages — transcribe with Whisper, then process."""
        if not await self._check_auth(update):
            return

        user_id = update.effective_user.id
        await update.message.chat.send_action("typing")

        try:
            # Download voice file
            voice = update.message.voice or update.message.audio
            if not voice:
                await update.message.reply_text("Could not read voice message.")
                return

            file = await voice.get_file()
            voice_bytes = await file.download_as_bytearray()

            # Save to temp file for Whisper
            with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as tmp:
                tmp.write(voice_bytes)
                tmp_path = tmp.name

            # Transcribe with Whisper
            transcribed = await self._transcribe_voice(tmp_path)
            if not transcribed:
                await update.message.reply_text("Could not transcribe voice message.")
                return

            # Send transcription back as confirmation
            await update.message.reply_text(
                f"🎙️ _{escape_md2(transcribed)}_",
                parse_mode="MarkdownV2",
            )

            # Process through agent
            response = await self.bridge.process_message(user_id, transcribed)
            await self._send_long_message(update, response)

        except Exception as exc:
            logger.error("Voice handler error: %s", exc)
            await update.message.reply_text(f"❌ Voice error: {str(exc)[:200]}")
        finally:
            # Cleanup temp file
            try:
                Path(tmp_path).unlink(missing_ok=True)
            except Exception:
                pass

    async def _handle_photo(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle photo messages — describe via vision, then process with caption."""
        if not await self._check_auth(update):
            return

        user_id = update.effective_user.id
        await update.message.chat.send_action("typing")

        try:
            photo = update.message.photo[-1]  # Highest resolution
            file = await photo.get_file()
            photo_bytes = await file.download_as_bytearray()

            caption = update.message.caption or "Analyze this image."

            # Save photo and process
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                tmp.write(photo_bytes)
                tmp_path = tmp.name

            # Process as text message with image context
            prompt = f"[User sent a photo with caption: {caption}. Photo saved at: {tmp_path}]"
            response = await self.bridge.process_message(user_id, prompt)
            await self._send_long_message(update, response)

        except Exception as exc:
            logger.error("Photo handler error: %s", exc)
            await update.message.reply_text(f"❌ Photo error: {str(exc)[:200]}")

    async def _handle_document(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle document uploads — save and process."""
        if not await self._check_auth(update):
            return

        user_id = update.effective_user.id
        await update.message.chat.send_action("typing")

        try:
            doc = update.message.document
            file = await doc.get_file()
            doc_bytes = await file.download_as_bytearray()

            # Save to workspace
            save_dir = Path("workspace/telegram_uploads")
            save_dir.mkdir(parents=True, exist_ok=True)
            save_path = save_dir / doc.file_name
            save_path.write_bytes(doc_bytes)

            caption = update.message.caption or f"Process this file: {doc.file_name}"
            prompt = f"[User uploaded file: {save_path}. Caption: {caption}]"

            response = await self.bridge.process_message(user_id, prompt)
            await self._send_long_message(update, response)

        except Exception as exc:
            logger.error("Document handler error: %s", exc)
            await update.message.reply_text(f"❌ Document error: {str(exc)[:200]}")

    # ── Helpers ──────────────────────────────────────────────────

    async def _send_long_message(self, update: Update, text: str) -> None:
        """Send a message, splitting into chunks if needed (Telegram 4096 char limit)."""
        if len(text) <= 4000:
            try:
                await update.message.reply_text(text, parse_mode="MarkdownV2")
            except Exception:
                # Fallback to plain text if MarkdownV2 fails
                plain = text.replace("\\", "")
                await update.message.reply_text(plain[:4000])
        else:
            # Split into chunks
            for i in range(0, len(text), 4000):
                chunk = text[i : i + 4000]
                try:
                    await update.message.reply_text(chunk, parse_mode="MarkdownV2")
                except Exception:
                    plain = chunk.replace("\\", "")
                    await update.message.reply_text(plain[:4000])

    async def _transcribe_voice(self, audio_path: str) -> Optional[str]:
        """Transcribe audio using OpenAI Whisper (local or API).

        Tries local whisper first, falls back to OpenAI Whisper API.
        """
        # Try local whisper first (free, private)
        try:
            import whisper

            model = whisper.load_model("base")
            result = model.transcribe(audio_path)
            return result.get("text", "").strip()
        except ImportError:
            pass
        except Exception as exc:
            logger.debug("Local whisper failed: %s", exc)

        # Fallback to OpenAI Whisper API
        try:
            from openai import OpenAI

            client = OpenAI()
            with open(audio_path, "rb") as f:
                transcript = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=f,
                )
                return transcript.text.strip()
        except Exception as exc:
            logger.warning("Whisper API failed: %s", exc)

        return None

    # ── Lifecycle ────────────────────────────────────────────────

    def build(self) -> Application:
        """Build and configure the Telegram Application with all handlers."""
        builder = Application.builder().token(self.token)
        self._app = builder.build()

        # Command handlers
        self._app.add_handler(CommandHandler("start", self._handle_start))
        self._app.add_handler(CommandHandler("help", self._handle_help))
        self._app.add_handler(CommandHandler("status", self._handle_status))
        self._app.add_handler(CommandHandler("clear", self._handle_clear))
        self._app.add_handler(CommandHandler("model", self._handle_model))
        self._app.add_handler(CommandHandler("models", self._handle_model))

        # Message handlers (order matters — most specific first)
        self._app.add_handler(MessageHandler(filters.VOICE | filters.AUDIO, self._handle_voice))
        self._app.add_handler(MessageHandler(filters.PHOTO, self._handle_photo))
        self._app.add_handler(MessageHandler(filters.Document.ALL, self._handle_document))
        self._app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self._handle_text))

        return self._app

    async def start(self) -> None:
        """Start the bot polling loop."""
        if self._app is None:
            self.build()

        logger.info("Telegram bot starting (polling)...")
        await self._app.initialize()
        await self._app.start()
        await self._app.updater.start_polling(drop_pending_updates=True)

    async def stop(self) -> None:
        """Gracefully stop the bot."""
        if self._app:
            logger.info("Telegram bot stopping...")
            await self._app.updater.stop()
            await self._app.stop()
            await self._app.shutdown()
