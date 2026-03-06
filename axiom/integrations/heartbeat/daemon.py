"""Heartbeat daemon — reads workspace/HEARTBEAT.md and runs checks on schedule.

The HEARTBEAT.md file uses a simple markdown format:
```
## Every 30 minutes
- Check if nginx is running: `systemctl status nginx`
- Check disk usage: `df -h /`

## Every 1 hour
- Pull latest git changes: `cd /project && git pull`

## Every 6 hours
- Run test suite: `pytest tests/ -q`
```

The daemon parses these sections, runs the commands at the specified
intervals, and sends alerts via Telegram if any check fails.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Coroutine, Optional

logger = logging.getLogger(__name__)

# ── Interval parsing ────────────────────────────────────────────────

_INTERVAL_PATTERN = re.compile(
    r"every\s+(\d+)\s+(minute|hour|day)s?",
    re.IGNORECASE,
)

_COMMAND_PATTERN = re.compile(r"`([^`]+)`")


def _parse_interval_seconds(text: str) -> int:
    """Parse 'Every N minutes/hours/days' into seconds."""
    match = _INTERVAL_PATTERN.search(text)
    if not match:
        return 1800  # Default: 30 minutes
    amount = int(match.group(1))
    unit = match.group(2).lower()
    multipliers = {"minute": 60, "hour": 3600, "day": 86400}
    return amount * multipliers.get(unit, 60)


@dataclass
class HeartbeatCheck:
    """A single scheduled check parsed from HEARTBEAT.md."""

    description: str
    command: str
    interval_seconds: int
    last_run: float = 0.0
    last_result: Optional[str] = None
    last_success: bool = True

    @property
    def is_due(self) -> bool:
        return (time.time() - self.last_run) >= self.interval_seconds


def parse_heartbeat_file(path: Path) -> list[HeartbeatCheck]:
    """Parse workspace/HEARTBEAT.md into a list of HeartbeatCheck objects.

    Format:
        ## Every N minutes/hours
        - Description: `command`
        - Another check: `another_command`
    """
    if not path.exists():
        logger.info("No HEARTBEAT.md found at %s", path)
        return []

    text = path.read_text(encoding="utf-8")
    checks: list[HeartbeatCheck] = []
    current_interval = 1800  # Default 30 min

    for line in text.splitlines():
        line = line.strip()

        # Section header: ## Every N minutes
        if line.startswith("##"):
            current_interval = _parse_interval_seconds(line)
            continue

        # Check line: - Description: `command`
        if line.startswith("-") or line.startswith("*"):
            line = line.lstrip("-* ").strip()
            cmd_match = _COMMAND_PATTERN.search(line)
            if cmd_match:
                command = cmd_match.group(1)
                # Description is everything before the backtick
                desc = line[: cmd_match.start()].strip().rstrip(":")
                if not desc:
                    desc = command[:60]
                checks.append(
                    HeartbeatCheck(
                        description=desc,
                        command=command,
                        interval_seconds=current_interval,
                    )
                )

    logger.info("Parsed %d heartbeat checks from %s", len(checks), path)
    return checks


class HeartbeatDaemon:
    """Background daemon that runs scheduled checks.

    Reads checks from HEARTBEAT.md, executes them on schedule,
    and calls an alert callback (e.g., Telegram message) on failure.
    """

    def __init__(
        self,
        heartbeat_path: Path,
        interval_minutes: int = 30,
        alert_callback: Optional[Callable[[str], Coroutine]] = None,
    ) -> None:
        """
        Args:
            heartbeat_path: Path to HEARTBEAT.md file.
            interval_minutes: Default check interval (overridden by file sections).
            alert_callback: Async function called with alert message on failures.
        """
        self.heartbeat_path = heartbeat_path
        self.default_interval = interval_minutes * 60
        self.alert_callback = alert_callback
        self._checks: list[HeartbeatCheck] = []
        self._running = False
        self._task: Optional[asyncio.Task] = None

    def reload(self) -> int:
        """Reload checks from HEARTBEAT.md. Returns count of loaded checks."""
        self._checks = parse_heartbeat_file(self.heartbeat_path)
        return len(self._checks)

    async def _run_check(self, check: HeartbeatCheck) -> bool:
        """Execute a single heartbeat check via subprocess.

        Returns True if the command exited with code 0.
        """
        try:
            proc = await asyncio.create_subprocess_shell(
                check.command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=60)

            check.last_run = time.time()
            check.last_success = proc.returncode == 0
            check.last_result = (stdout or stderr or b"").decode("utf-8", errors="replace")[:500]

            if not check.last_success:
                logger.warning(
                    "Heartbeat FAILED: %s (exit %s): %s",
                    check.description,
                    proc.returncode,
                    check.last_result[:200],
                )
            else:
                logger.debug("Heartbeat OK: %s", check.description)

            return check.last_success

        except asyncio.TimeoutError:
            check.last_run = time.time()
            check.last_success = False
            check.last_result = "TIMEOUT (60s)"
            logger.warning("Heartbeat TIMEOUT: %s", check.description)
            return False

        except Exception as exc:
            check.last_run = time.time()
            check.last_success = False
            check.last_result = str(exc)[:500]
            logger.error("Heartbeat ERROR: %s: %s", check.description, exc)
            return False

    async def _alert(self, message: str) -> None:
        """Send an alert via the configured callback."""
        if self.alert_callback:
            try:
                await self.alert_callback(message)
            except Exception as exc:
                logger.error("Alert callback failed: %s", exc)

    async def _loop(self) -> None:
        """Main daemon loop — checks all due items every 30 seconds."""
        logger.info("Heartbeat daemon started (%d checks)", len(self._checks))

        while self._running:
            for check in self._checks:
                if check.is_due:
                    success = await self._run_check(check)
                    if not success:
                        alert_msg = (
                            f"🚨 *Heartbeat Alert*\n\n"
                            f"Check: {check.description}\n"
                            f"Command: `{check.command}`\n"
                            f"Result: {check.last_result[:300]}"
                        )
                        await self._alert(alert_msg)

            # Sleep 30 seconds between sweeps
            await asyncio.sleep(30)

    async def start(self) -> None:
        """Start the heartbeat daemon as a background task."""
        self.reload()
        if not self._checks:
            logger.info("No heartbeat checks configured, daemon not starting")
            return

        self._running = True
        self._task = asyncio.create_task(self._loop())
        logger.info("Heartbeat daemon started with %d checks", len(self._checks))

    async def stop(self) -> None:
        """Stop the heartbeat daemon."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("Heartbeat daemon stopped")

    @property
    def status(self) -> dict[str, Any]:
        """Return current daemon status."""
        return {
            "running": self._running,
            "checks": len(self._checks),
            "failed": sum(1 for c in self._checks if not c.last_success),
            "details": [
                {
                    "description": c.description,
                    "command": c.command,
                    "interval_s": c.interval_seconds,
                    "last_success": c.last_success,
                    "last_result": (c.last_result or "")[:100],
                }
                for c in self._checks
            ],
        }
