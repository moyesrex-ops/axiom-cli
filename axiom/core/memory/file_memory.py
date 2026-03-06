"""Markdown file-based persistent memory for Axiom.

Manages:
- memory/axiom.md — Core knowledge file (user prefs, project states)
- memory/sessions/ — Session summaries (auto on exit)
- memory/facts/ — Extracted learnings from tasks
"""

import time
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

class FileMemory:
    """File-based persistent memory using Markdown files.

    This is the human-readable, git-friendly layer of memory.
    The vector store handles semantic search; this handles
    structured persistent knowledge.
    """

    def __init__(self, memory_dir: str = "memory"):
        self.memory_dir = Path(memory_dir)
        self.core_file = self.memory_dir / "axiom.md"
        self.sessions_dir = self.memory_dir / "sessions"
        self.facts_dir = self.memory_dir / "facts"

        # Ensure directories exist
        for d in [self.memory_dir, self.sessions_dir, self.facts_dir]:
            d.mkdir(parents=True, exist_ok=True)

    def load_core_memory(self) -> str:
        """Load the core memory file (axiom.md).

        Returns empty string if file doesn't exist.
        """
        if self.core_file.exists():
            try:
                return self.core_file.read_text(encoding="utf-8")
            except Exception as e:
                logger.warning("Failed to read core memory: %s", e)
        return ""

    def save_core_memory(self, content: str) -> None:
        """Save content to the core memory file."""
        try:
            self.core_file.write_text(content, encoding="utf-8")
        except Exception as e:
            logger.warning("Failed to save core memory: %s", e)

    def append_to_core(self, section: str, content: str) -> None:
        """Append content under a section header in core memory."""
        current = self.load_core_memory()
        timestamp = time.strftime("%Y-%m-%d %H:%M")
        entry = f"\n## {section}\n*{timestamp}*\n{content}\n"
        self.save_core_memory(current + entry)

    def save_session_summary(self, summary: str, session_id: Optional[str] = None) -> Path:
        """Save a session summary to the sessions directory.

        Returns the path of the saved file.
        """
        if session_id is None:
            session_id = time.strftime("%Y-%m-%d_%H-%M-%S")

        filename = f"{session_id}.md"
        filepath = self.sessions_dir / filename

        header = f"# Session: {session_id}\n\n"
        try:
            filepath.write_text(header + summary, encoding="utf-8")
            logger.debug("Session summary saved to %s", filepath)
        except Exception as e:
            logger.warning("Failed to save session summary: %s", e)

        return filepath

    def save_fact(self, fact: str, tags: Optional[list[str]] = None) -> Path:
        """Save an extracted fact/learning."""
        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        tag_str = "_".join(tags) if tags else "general"
        filename = f"{tag_str}_{timestamp}.md"
        filepath = self.facts_dir / filename

        content = f"# Fact: {tag_str}\n*{time.strftime('%Y-%m-%d %H:%M')}*\n\n{fact}\n"
        try:
            filepath.write_text(content, encoding="utf-8")
        except Exception as e:
            logger.warning("Failed to save fact: %s", e)

        return filepath

    def list_sessions(self, limit: int = 10) -> list[dict[str, str]]:
        """List recent session summaries."""
        sessions = []
        files = sorted(self.sessions_dir.glob("*.md"), reverse=True)[:limit]
        for f in files:
            try:
                content = f.read_text(encoding="utf-8")
                preview = content[:200].replace("\n", " ")
                sessions.append({
                    "id": f.stem,
                    "path": str(f),
                    "preview": preview,
                })
            except Exception:
                continue
        return sessions

    def get_stats(self) -> dict[str, int]:
        """Get memory statistics."""
        return {
            "core_exists": self.core_file.exists(),
            "sessions": len(list(self.sessions_dir.glob("*.md"))),
            "facts": len(list(self.facts_dir.glob("*.md"))),
        }
