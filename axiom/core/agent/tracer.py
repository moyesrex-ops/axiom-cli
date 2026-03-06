"""Agent tracer -- observability for agent execution.

Logs every decision, tool call, and state transition during
agent execution. Can output to console, file, or both.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class TraceEntry:
    """A single entry in the execution trace."""

    timestamp: float
    category: str  # PLAN, TOOL, OBSERVE, MODEL, COST, ERROR, etc.
    message: str
    data: dict[str, Any] = field(default_factory=dict)
    duration_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "time": datetime.fromtimestamp(self.timestamp).strftime(
                "%H:%M:%S.%f"
            )[:-3],
            "category": self.category,
            "message": self.message,
            "data": self.data,
            "duration_ms": self.duration_ms,
        }


class AgentTracer:
    """Records and manages agent execution traces.

    Usage::

        tracer = AgentTracer()
        tracer.log("PLAN", "Selected PLAN mode", {"task_type": "structured"})
        tracer.log("TOOL", "bash: ls -la", {"duration_ms": 120})
        tracer.save("memory/traces/")
    """

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        self._entries: list[TraceEntry] = []
        self._start_time = time.time()
        self._session_id = datetime.now().strftime("%Y%m%d-%H%M%S")

    def log(
        self,
        category: str,
        message: str,
        data: dict[str, Any] | None = None,
        duration_ms: float = 0.0,
    ) -> None:
        """Add a trace entry."""
        if not self.enabled:
            return

        entry = TraceEntry(
            timestamp=time.time(),
            category=category,
            message=message,
            data=data or {},
            duration_ms=duration_ms,
        )
        self._entries.append(entry)
        logger.debug("[TRACE] [%s] %s", category, message)

    def log_tool_call(
        self,
        tool_name: str,
        args: dict[str, Any],
        result: str = "",
        success: bool = True,
        duration_ms: float = 0.0,
    ) -> None:
        """Convenience method for logging tool calls."""
        status = "OK" if success else "FAIL"
        self.log(
            "TOOL",
            f"{status} {tool_name} ({duration_ms:.0f}ms)",
            {
                "tool": tool_name,
                "args": {k: str(v)[:100] for k, v in args.items()},
                "result_preview": str(result)[:200],
                "success": success,
            },
            duration_ms=duration_ms,
        )

    def log_model_call(
        self,
        model: str,
        tokens_in: int = 0,
        tokens_out: int = 0,
        cost: float = 0.0,
        duration_ms: float = 0.0,
    ) -> None:
        """Log an LLM API call."""
        self.log(
            "MODEL",
            f"{model} ({tokens_in}->{tokens_out} tokens, ${cost:.4f})",
            {
                "model": model,
                "tokens_in": tokens_in,
                "tokens_out": tokens_out,
                "cost": cost,
            },
            duration_ms=duration_ms,
        )

    def log_mode_selection(self, mode: str, reason: str = "") -> None:
        """Log agent mode selection."""
        self.log("PLAN", f"Mode: {mode}", {"mode": mode, "reason": reason})

    def log_observation(
        self,
        status: str,
        confidence: float,
        reason: str = "",
    ) -> None:
        """Log an observer reflection."""
        self.log(
            "OBSERVE",
            f"{status} (confidence: {confidence:.0%})",
            {"status": status, "confidence": confidence, "reason": reason},
        )

    # -- Output Methods ---------------------------------------------------

    def format_console(self, last_n: int = 0) -> str:
        """Format trace entries for console display."""
        entries = self._entries[-last_n:] if last_n else self._entries
        lines = []
        for entry in entries:
            ts = datetime.fromtimestamp(entry.timestamp).strftime("%H:%M:%S")
            lines.append(f"[{ts}] [{entry.category}] {entry.message}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Export full trace as a dictionary."""
        elapsed = time.time() - self._start_time
        return {
            "session_id": self._session_id,
            "duration_seconds": round(elapsed, 2),
            "entry_count": len(self._entries),
            "entries": [e.to_dict() for e in self._entries],
        }

    def save(self, directory: str | Path) -> Optional[Path]:
        """Save trace to a JSON file."""
        if not self._entries:
            return None

        dir_path = Path(directory)
        dir_path.mkdir(parents=True, exist_ok=True)

        filepath = dir_path / f"trace-{self._session_id}.json"
        try:
            filepath.write_text(
                json.dumps(self.to_dict(), indent=2),
                encoding="utf-8",
            )
            logger.info("Saved trace to %s", filepath)
            return filepath
        except Exception as exc:
            logger.debug("Failed to save trace: %s", exc)
            return None

    def clear(self) -> None:
        """Clear all trace entries."""
        self._entries.clear()

    @property
    def entry_count(self) -> int:
        return len(self._entries)

    @property
    def elapsed_seconds(self) -> float:
        return time.time() - self._start_time
