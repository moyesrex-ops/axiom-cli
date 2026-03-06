"""Persistent task store — remembers tasks across restarts.

Tasks survive process restarts. The agent can add tasks from user
requests, mark them complete/failed, and resume pending tasks on startup.
"""

import json
import time
import asyncio
import aiofiles
from pathlib import Path
from typing import Optional
from enum import Enum


class TaskStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class TaskStore:
    """JSON-file-backed persistent task queue.

    Tasks survive restarts. Agent can:
    - Add tasks from user requests
    - Mark tasks complete/failed
    - Resume pending tasks on startup
    - Query task history
    """

    def __init__(self, path: str | Path | None = None):
        if path is None:
            from axiom.config.settings import get_settings
            path = get_settings().AXIOM_HOME / "tasks" / "tasks.json"
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = asyncio.Lock()
        self._tasks: list[dict] = []
        self._loaded = False

    async def _ensure_loaded(self) -> None:
        """Lazy-load tasks from disk on first access."""
        if self._loaded:
            return
        if self._path.exists():
            try:
                async with aiofiles.open(self._path, "r", encoding="utf-8") as f:
                    content = await f.read()
                    self._tasks = json.loads(content) if content.strip() else []
            except (json.JSONDecodeError, OSError):
                self._tasks = []
        self._loaded = True

    async def _save(self) -> None:
        """Persist tasks to disk."""
        async with aiofiles.open(self._path, "w", encoding="utf-8") as f:
            await f.write(json.dumps(self._tasks, indent=2, ensure_ascii=False))

    async def add(
        self,
        description: str,
        *,
        source: str = "user",
        priority: str = "normal",
        metadata: Optional[dict] = None,
    ) -> dict:
        """Add a new task.

        Args:
            description: What needs to be done
            source: Where the task came from ("user", "telegram", "agent")
            priority: Task priority ("low", "normal", "high")
            metadata: Optional extra data

        Returns:
            The created task dict
        """
        async with self._lock:
            await self._ensure_loaded()
            task = {
                "id": len(self._tasks) + 1,
                "description": description,
                "status": TaskStatus.PENDING.value,
                "source": source,
                "priority": priority,
                "created_at": time.time(),
                "updated_at": time.time(),
                "result": None,
                "metadata": metadata or {},
            }
            self._tasks.append(task)
            await self._save()
            return task

    async def update(
        self,
        task_id: int,
        *,
        status: Optional[str] = None,
        result: Optional[str] = None,
    ) -> Optional[dict]:
        """Update a task's status and/or result.

        Args:
            task_id: The task ID to update
            status: New status (pending/in_progress/completed/failed)
            result: Result text or error message

        Returns:
            Updated task dict, or None if not found
        """
        async with self._lock:
            await self._ensure_loaded()
            for task in self._tasks:
                if task["id"] == task_id:
                    if status:
                        task["status"] = status
                    if result is not None:
                        task["result"] = result
                    task["updated_at"] = time.time()
                    await self._save()
                    return task
            return None

    async def get_pending(self) -> list[dict]:
        """Get all pending and in-progress tasks."""
        async with self._lock:
            await self._ensure_loaded()
            return [
                t
                for t in self._tasks
                if t["status"]
                in (TaskStatus.PENDING.value, TaskStatus.IN_PROGRESS.value)
            ]

    async def get_all(self, limit: int = 50) -> list[dict]:
        """Get all tasks (most recent first)."""
        async with self._lock:
            await self._ensure_loaded()
            return list(reversed(self._tasks[-limit:]))

    def format_for_prompt(self, tasks: list[dict]) -> str:
        """Format tasks as text for injection into system prompt.

        Args:
            tasks: List of task dicts

        Returns:
            Human-readable task list with status icons
        """
        if not tasks:
            return "No pending tasks."
        lines = []
        for t in tasks:
            status_icon = {
                "pending": "⏳",
                "in_progress": "🔄",
                "completed": "✅",
                "failed": "❌",
            }.get(t["status"], "❓")
            lines.append(
                f"{status_icon} #{t['id']}: {t['description']} [{t['status']}]"
            )
        return "\n".join(lines)
