"""Unified memory manager that combines vector store + file memory.

This is the single interface the agent uses for all memory operations.
"""

import logging
import time
from typing import Any, Optional

from axiom.core.memory.vector_store import VectorStore
from axiom.core.memory.file_memory import FileMemory

logger = logging.getLogger(__name__)

class MemoryManager:
    """Unified memory interface combining vector search + file storage.

    Provides:
    - Semantic search across all past conversations and facts
    - Persistent file-based memory for structured knowledge
    - Session management (save/load summaries)
    - Context injection for agent system prompts
    """

    def __init__(
        self,
        memory_dir: str = "memory",
        chroma_dir: Optional[str] = None,
    ):
        self.file_memory = FileMemory(memory_dir=memory_dir)
        chroma_path = chroma_dir or f"{memory_dir}/chroma"
        self.vector_store = VectorStore(persist_dir=chroma_path)

    # ── Search ──────────────────────────────────────────────────────

    def search(self, query: str, k: int = 5) -> list[dict[str, Any]]:
        """Semantic search across all stored memories."""
        return self.vector_store.search(query, k=k)

    # ── Store ───────────────────────────────────────────────────────

    def store_message(
        self,
        role: str,
        content: str,
        session_id: str = "",
        topic: str = "",
    ) -> str:
        """Store a conversation message in vector memory."""
        return self.vector_store.add(
            text=content,
            metadata={
                "role": role,
                "session_id": session_id,
                "topic": topic,
                "type": "message",
            },
        )

    def store_fact(self, fact: str, tags: Optional[list[str]] = None) -> None:
        """Store a fact in both vector store and file memory."""
        self.vector_store.add(
            text=fact,
            metadata={
                "type": "fact",
                "tags": ",".join(tags) if tags else "",
            },
        )
        self.file_memory.save_fact(fact, tags=tags)

    # ── Session Management ──────────────────────────────────────────

    def save_session(self, summary: str, session_id: Optional[str] = None) -> None:
        """Save a session summary to both stores."""
        self.file_memory.save_session_summary(summary, session_id=session_id)
        self.vector_store.add(
            text=summary,
            metadata={
                "type": "session_summary",
                "session_id": session_id or time.strftime("%Y-%m-%d_%H-%M-%S"),
            },
        )

    def load_core_memory(self) -> str:
        """Load core persistent memory (axiom.md)."""
        return self.file_memory.load_core_memory()

    def save_core_memory(self, content: str) -> None:
        """Update core persistent memory."""
        self.file_memory.save_core_memory(content)

    # ── Context Building ────────────────────────────────────────────

    def build_context(self, query: str = "", k: int = 3) -> str:
        """Build a memory context string for injection into system prompts.

        Combines core memory + relevant vector search results.
        """
        parts = []

        # Core memory (always included)
        core = self.load_core_memory()
        if core:
            parts.append("## Core Memory\n" + core[:2000])

        # Relevant past memories (if query provided)
        if query:
            results = self.search(query, k=k)
            if results:
                parts.append("## Relevant Past Memories")
                for r in results:
                    meta = r.get("metadata", {})
                    role = meta.get("role", "unknown")
                    parts.append(f"- [{role}] {r['text'][:300]}")

        return "\n\n".join(parts) if parts else ""

    # ── Stats ───────────────────────────────────────────────────────

    def get_stats(self) -> dict[str, Any]:
        """Get combined memory statistics."""
        file_stats = self.file_memory.get_stats()
        return {
            "vector_entries": self.vector_store.count,
            "core_exists": file_stats["core_exists"],
            "sessions": file_stats["sessions"],
            "facts": file_stats["facts"],
        }
