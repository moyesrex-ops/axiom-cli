"""Code execution tool for Axiom CLI.

Runs Python, JavaScript (Node.js), or Bash code in an isolated
subprocess with timeout protection.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

from axiom.core.tools.base import AxiomTool, ToolError

_MAX_OUTPUT = 50_000
_DEFAULT_TIMEOUT = 30

# Language config: file extension, interpreter command list
_LANG_CONFIG: dict[str, tuple[str, list[str]]] = {
    "python": (".py", [sys.executable]),
    "javascript": (".js", ["node"]),
    "bash": (
        ".sh",
        ["/bin/bash"] if sys.platform != "win32"
        else ["powershell.exe", "-NoProfile", "-File"],
    ),
}


class CodeExecTool(AxiomTool):
    """Execute code in an isolated subprocess and return the output."""

    name = "code_exec"
    description = (
        "Execute Python, JavaScript, or Bash code in an isolated subprocess. "
        "Returns stdout and stderr. Code runs in a temporary file with a timeout. "
        "Use for calculations, data processing, or testing code snippets."
    )
    risk_level = "high"
    parameters_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "The code to execute",
            },
            "language": {
                "type": "string",
                "description": "Language: 'python', 'javascript', or 'bash' (default: python)",
                "enum": ["python", "javascript", "bash"],
                "default": "python",
            },
            "timeout": {
                "type": "integer",
                "description": "Timeout in seconds (default: 30, max: 300)",
                "default": 30,
            },
        },
        "required": ["code"],
    }

    async def execute(self, **kwargs: Any) -> str:
        code: str = kwargs["code"]
        language: str = kwargs.get("language", "python")
        timeout: int = min(kwargs.get("timeout", _DEFAULT_TIMEOUT), 300)

        if not code.strip():
            raise ToolError("Empty code", tool_name=self.name)

        language = language.lower()
        if language not in _LANG_CONFIG:
            raise ToolError(
                f"Unsupported language: {language}. "
                f"Choose from: {', '.join(_LANG_CONFIG)}",
                tool_name=self.name,
            )

        ext, interpreter = _LANG_CONFIG[language]

        # Verify interpreter exists (skip for python -- we use sys.executable)
        if language != "python" and not shutil.which(interpreter[0]):
            raise ToolError(
                f"Interpreter not found: {interpreter[0]}. "
                f"Make sure {language} runtime is installed and on PATH.",
                tool_name=self.name,
            )

        # Write code to a temporary file
        tmp_dir = tempfile.mkdtemp(prefix="axiom_exec_")
        tmp_file = Path(tmp_dir) / f"script{ext}"

        try:
            tmp_file.write_text(code, encoding="utf-8")

            # Build command: [interpreter..., script_path]
            cmd = interpreter + [str(tmp_file)]

            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=tmp_dir,
                env=_sandboxed_env(),
                **(_win_flags() if sys.platform == "win32" else {}),
            )

            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    proc.communicate(), timeout=timeout
                )
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                raise ToolError(
                    f"Code execution timed out after {timeout}s",
                    tool_name=self.name,
                )

        except ToolError:
            raise
        except FileNotFoundError as e:
            raise ToolError(
                f"Interpreter not found: {e}",
                tool_name=self.name,
            )
        except OSError as e:
            raise ToolError(
                f"Execution failed: {e}",
                tool_name=self.name,
            )
        finally:
            # Clean up temp files
            try:
                tmp_file.unlink(missing_ok=True)
                Path(tmp_dir).rmdir()
            except OSError:
                pass

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

        return f"[{language}] {output}"


def _safe_decode(data: bytes) -> str:
    if not data:
        return ""
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data.decode("latin-1", errors="replace")


def _sandboxed_env() -> dict[str, str]:
    """Return a minimal environment for subprocess execution.

    Inherits PATH and basic locale settings but drops other vars
    to reduce accidental leakage of secrets from the parent env.
    """
    env: dict[str, str] = {}
    for key in (
        "PATH", "HOME", "USERPROFILE", "SYSTEMROOT", "TEMP", "TMP",
        "LANG", "LC_ALL", "PYTHONPATH", "NODE_PATH",
        "PYTHONUTF8", "PYTHONIOENCODING",
    ):
        val = os.environ.get(key)
        if val:
            env[key] = val

    # Force UTF-8 output
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"

    return env


def _win_flags() -> dict[str, Any]:
    import subprocess as _sp
    return {"creationflags": _sp.CREATE_NO_WINDOW}  # type: ignore[attr-defined]
