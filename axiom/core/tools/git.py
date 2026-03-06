"""Git tool for Axiom CLI.

Runs git subcommands (status, diff, log, add, commit, push, etc.)
by shelling out to the git binary via asyncio subprocess.
"""

from __future__ import annotations

import asyncio
import os
import sys
from typing import Any

from axiom.core.tools.base import AxiomTool, ToolError

_MAX_OUTPUT = 50_000

# Dangerous commands that should be blocked
_DANGEROUS_PATTERNS = frozenset({
    "push --force", "push -f",
    "reset --hard",
    "clean -f", "clean -fd",
    "branch -D",
    "checkout .",
    "restore .",
    "rebase -i",
})


class GitTool(AxiomTool):
    """Run git commands in a repository."""

    name = "git"
    description = (
        "Execute git commands (status, diff, log, add, commit, push, pull, "
        "branch, checkout, stash, etc.). Provide the subcommand and arguments "
        "as a single string. Blocks destructive commands like push --force."
    )
    risk_level = "medium"
    parameters_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": (
                    "Git subcommand and arguments "
                    "(e.g., 'status', 'diff HEAD~1', 'log --oneline -10', "
                    "'add -A', 'commit -m \"message\"')"
                ),
            },
            "cwd": {
                "type": "string",
                "description": "Repository directory (default: current directory)",
            },
        },
        "required": ["command"],
    }

    async def execute(self, **kwargs: Any) -> str:
        command: str = kwargs["command"]
        cwd: str | None = kwargs.get("cwd")

        if not command.strip():
            raise ToolError("Empty git command", tool_name=self.name)

        work_dir = cwd if cwd else os.getcwd()

        # Safety check: block dangerous commands
        cmd_lower = command.strip().lower()
        for pattern in _DANGEROUS_PATTERNS:
            if pattern in cmd_lower:
                raise ToolError(
                    f"Blocked dangerous git command: 'git {command}'. "
                    f"Pattern '{pattern}' is potentially destructive. "
                    "Use the bash tool directly if you really need this.",
                    tool_name=self.name,
                )

        # Build command -- use create_subprocess_exec to avoid shell injection
        full_cmd = f"git {command}"

        if sys.platform == "win32":
            shell_args = [
                "powershell.exe", "-NoProfile", "-NonInteractive",
                "-Command", full_cmd,
            ]
        else:
            shell_args = ["/bin/bash", "-c", full_cmd]

        try:
            proc = await asyncio.create_subprocess_exec(
                *shell_args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=work_dir,
                **(_win_flags() if sys.platform == "win32" else {}),
            )
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate(), timeout=60
            )
        except asyncio.TimeoutError:
            raise ToolError(
                f"Git command timed out (60s): git {command}",
                tool_name=self.name,
            )
        except FileNotFoundError:
            raise ToolError(
                "git is not installed or not on PATH",
                tool_name=self.name,
            )
        except OSError as e:
            raise ToolError(f"Failed to run git: {e}", tool_name=self.name)

        stdout = _safe_decode(stdout_bytes)
        stderr = _safe_decode(stderr_bytes)

        parts: list[str] = []
        if stdout.strip():
            parts.append(stdout.strip())
        if stderr.strip():
            parts.append(f"[stderr]\n{stderr.strip()}")

        output = "\n".join(parts) if parts else "(no output)"
        exit_code = proc.returncode

        if exit_code and exit_code != 0:
            output = f"[exit code: {exit_code}]\n{output}"

        # Truncate
        if len(output) > _MAX_OUTPUT:
            half = _MAX_OUTPUT // 2
            output = (
                output[:half]
                + f"\n\n... [truncated {len(output) - _MAX_OUTPUT} chars] ...\n\n"
                + output[-half:]
            )

        return output


def _safe_decode(data: bytes) -> str:
    if not data:
        return ""
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data.decode("latin-1", errors="replace")


def _win_flags() -> dict[str, Any]:
    import subprocess as _sp
    return {"creationflags": _sp.CREATE_NO_WINDOW}  # type: ignore[attr-defined]
