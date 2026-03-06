"""Append-only JSONL conversation store — crash-safe, zero dependencies.

Both CLI and Telegram write to the same JSONL file, creating a shared
conversation thread that persists across restarts.
"""

import json
import time
import asyncio
import aiofiles
from pathlib import Path
from typing import Optional


class ConversationStore:
    """Persistent conversation storage using JSONL (one JSON object per line).

    Why JSONL over SQLite:
    - Append-only = crash-safe (partial writes only lose last line)
    - Human-readable (can inspect with any text editor)
    - No extra dependencies
    - Git-friendly diffs
    - Perfect for single-user system
    """

    def __init__(self, base_dir: str = "memory/conversations"):
        self._base = Path(base_dir)
        self._base.mkdir(parents=True, exist_ok=True)
        self._lock = asyncio.Lock()

    def _path(self, conversation_id: str = "current") -> Path:
        return self._base / f"{conversation_id}.jsonl"

    async def append(
        self,
        role: str,
        content: str,
        *,
        channel: str = "cli",
        conversation_id: str = "current",
        metadata: Optional[dict] = None,
    ) -> dict:
        """Append a message to the conversation log.

        Args:
            role: Message role ("user" or "assistant")
            content: Message text
            channel: Source channel ("cli", "telegram")
            conversation_id: Conversation identifier (default "current")
            metadata: Optional extra data (tool calls, etc.)

        Returns:
            The stored message dict
        """
        msg = {
            "role": role,
            "content": content,
            "channel": channel,
            "ts": time.time(),
            "metadata": metadata or {},
        }
        async with self._lock:
            async with aiofiles.open(
                self._path(conversation_id), "a", encoding="utf-8"
            ) as f:
                await f.write(json.dumps(msg, ensure_ascii=False) + "\n")
        return msg

    async def load(
        self,
        conversation_id: str = "current",
        last_n: Optional[int] = None,
    ) -> list[dict]:
        """Load conversation messages from JSONL file.

        Args:
            conversation_id: Which conversation to load
            last_n: Only return the last N messages (None = all)

        Returns:
            List of message dicts in chronological order
        """
        path = self._path(conversation_id)
        if not path.exists():
            return []
        messages: list[dict] = []
        async with aiofiles.open(path, "r", encoding="utf-8") as f:
            async for line in f:
                line = line.strip()
                if line:
                    try:
                        messages.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue  # Skip corrupted lines gracefully
        if last_n is not None:
            messages = messages[-last_n:]
        return messages

    async def to_llm_messages(
        self,
        conversation_id: str = "current",
        last_n: int = 40,
    ) -> list[dict]:
        """Convert stored messages to LLM-compatible format.

        Returns:
            List of {role, content} dicts suitable for LLM message arrays
        """
        raw = await self.load(conversation_id, last_n=last_n)
        return [{"role": m["role"], "content": m["content"]} for m in raw]

    async def clear(self, conversation_id: str = "current") -> None:
        """Archive current conversation and start fresh.

        Renames the current file with a timestamp suffix so history
        is preserved but a new conversation begins.
        """
        path = self._path(conversation_id)
        if path.exists():
            archive_name = f"{conversation_id}_{int(time.time())}"
            path.rename(self._base / f"{archive_name}.jsonl")

    async def get_stats(self) -> dict:
        """Return conversation statistics.

        Returns:
            Dict with 'messages' count and 'channels' set
        """
        path = self._path("current")
        if not path.exists():
            return {"messages": 0, "channels": set()}
        messages = await self.load("current")
        channels = set(m.get("channel", "unknown") for m in messages)
        return {"messages": len(messages), "channels": channels}
