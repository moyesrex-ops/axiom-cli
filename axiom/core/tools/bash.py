"""Bash/shell execution tool for Axiom CLI.

Runs shell commands via asyncio subprocess. Automatically selects
PowerShell on Windows and /bin/bash on Linux/Mac.
"""

from __future__ import annotations

import asyncio
import os
import sys
from typing import Any

from axiom.core.tools.base import AxiomTool, ToolError

# Maximum characters to return from command output
_MAX_OUTPUT = 50_000


class BashTool(AxiomTool):
    """Execute a shell command and return its output."""

    name = "bash"
    description = (
        "Execute a shell command and return its output. "
        "Uses PowerShell on Windows, bash on Linux/Mac. "
        "Output is truncated to 50 000 characters."
    )
    risk_level = "medium"
    parameters_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "The shell command to execute",
            },
            "timeout": {
                "type": "integer",
                "description": "Timeout in seconds (default 120)",
                "default": 120,
            },
            "cwd": {
                "type": "string",
                "description": "Working directory (optional, defaults to current directory)",
            },
        },
        "required": ["command"],
    }

    async def execute(self, **kwargs: Any) -> str:
        command: str = kwargs["command"]
        timeout: int = kwargs.get("timeout", 120)
        cwd: str | None = kwargs.get("cwd")

        if not command.strip():
            raise ToolError("Empty command", tool_name=self.name)

        # Resolve working directory
        work_dir = cwd if cwd else os.getcwd()

        # Build the shell invocation using create_subprocess_exec
        # (not create_subprocess_shell) to avoid double-shell injection
        if sys.platform == "win32":
            # PowerShell for Windows
            shell_cmd = [
                "powershell.exe",
                "-NoProfile",
                "-NonInteractive",
                "-Command",
                command,
            ]
        else:
            shell_cmd = ["/bin/bash", "-c", command]

        try:
            proc = await asyncio.create_subprocess_exec(
                *shell_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=work_dir,
                **(_win_creation_flags() if sys.platform == "win32" else {}),
            )

            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    proc.communicate(), timeout=timeout
                )
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                raise ToolError(
                    f"Command timed out after {timeout}s: {command[:200]}",
                    tool_name=self.name,
                )

        except FileNotFoundError as e:
            raise ToolError(
                f"Shell not found: {e}",
                tool_name=self.name,
            )
        except OSError as e:
            raise ToolError(
                f"Failed to start process: {e}",
                tool_name=self.name,
            )

        # Decode output
        stdout = _safe_decode(stdout_bytes)
        stderr = _safe_decode(stderr_bytes)

        # Combine stdout and stderr
        parts: list[str] = []
        if stdout.strip():
            parts.append(stdout.strip())
        if stderr.strip():
            parts.append(f"[stderr]\n{stderr.strip()}")

        output = "\n".join(parts) if parts else "(no output)"
        exit_code = proc.returncode

        # Prepend exit code if non-zero
        if exit_code and exit_code != 0:
            output = f"[exit code: {exit_code}]\n{output}"

        # Truncate if too long
        if len(output) > _MAX_OUTPUT:
            half = _MAX_OUTPUT // 2
            output = (
                output[:half]
                + f"\n\n... [truncated {len(output) - _MAX_OUTPUT} chars] ...\n\n"
                + output[-half:]
            )

        return output


def _safe_decode(data: bytes) -> str:
    """Decode bytes to string, trying utf-8 first then latin-1 as fallback."""
    if not data:
        return ""
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data.decode("latin-1", errors="replace")


def _win_creation_flags() -> dict[str, Any]:
    """Return subprocess creation flags for Windows to suppress console windows."""
    import subprocess as _sp

    return {"creationflags": _sp.CREATE_NO_WINDOW}  # type: ignore[attr-defined]
