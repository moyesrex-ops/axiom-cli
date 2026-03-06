"""File-system tools for Axiom CLI.

Provides ReadFileTool, WriteFileTool, EditFileTool, GlobTool, and GrepTool
for comprehensive file manipulation and search.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

from axiom.core.tools.base import AxiomTool, ToolError

# ── Limits ───────────────────────────────────────────────────────────
_MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB read limit
_MAX_OUTPUT = 50_000  # max chars returned to LLM
_MAX_GREP_MATCHES = 200  # max grep results


# =====================================================================
#  ReadFileTool
# =====================================================================
class ReadFileTool(AxiomTool):
    """Read the contents of a file and return it with line numbers."""

    name = "read_file"
    description = (
        "Read the contents of a file. Returns file contents with line numbers. "
        "Supports offset and limit to read specific line ranges."
    )
    risk_level = "low"
    parameters_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Absolute or relative file path",
            },
            "offset": {
                "type": "integer",
                "description": "Start line (0-based, default 0)",
                "default": 0,
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of lines to read (default 2000)",
                "default": 2000,
            },
        },
        "required": ["path"],
    }

    async def execute(self, **kwargs: Any) -> str:
        path_str: str = kwargs["path"]
        offset: int = kwargs.get("offset", 0)
        limit: int = kwargs.get("limit", 2000)

        fpath = Path(path_str).resolve()

        if not fpath.exists():
            raise ToolError(f"File not found: {fpath}", tool_name=self.name)
        if not fpath.is_file():
            raise ToolError(f"Not a file: {fpath}", tool_name=self.name)
        if fpath.stat().st_size > _MAX_FILE_SIZE:
            raise ToolError(
                f"File too large ({fpath.stat().st_size:,} bytes, limit {_MAX_FILE_SIZE:,})",
                tool_name=self.name,
            )

        try:
            raw = fpath.read_bytes()
        except PermissionError:
            raise ToolError(f"Permission denied: {fpath}", tool_name=self.name)
        except OSError as exc:
            raise ToolError(f"Read error: {exc}", tool_name=self.name)

        # Decode with fallback
        text = _decode(raw)
        lines = text.splitlines()

        # Apply offset and limit
        selected = lines[offset : offset + limit]
        if not selected and lines:
            raise ToolError(
                f"Offset {offset} is beyond end of file ({len(lines)} lines)",
                tool_name=self.name,
            )

        # Format with line numbers
        numbered = []
        for i, line in enumerate(selected, start=offset + 1):
            # Truncate very long lines
            display = line[:2000] + "..." if len(line) > 2000 else line
            numbered.append(f"{i:>6}\t{display}")

        output = "\n".join(numbered) if numbered else "(empty file)"

        # Metadata header
        total = len(lines)
        shown = len(selected)
        header = f"[{fpath}] {total} lines total, showing {offset + 1}-{offset + shown}"
        if total > offset + shown:
            header += f" ({total - offset - shown} more lines)"

        return f"{header}\n{output}"


# =====================================================================
#  WriteFileTool
# =====================================================================
class WriteFileTool(AxiomTool):
    """Create or overwrite a file with the given content."""

    name = "write_file"
    description = (
        "Create or overwrite a file with the given content. "
        "Parent directories are created automatically."
    )
    risk_level = "medium"
    parameters_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "File path to write",
            },
            "content": {
                "type": "string",
                "description": "Content to write to the file",
            },
        },
        "required": ["path", "content"],
    }

    async def execute(self, **kwargs: Any) -> str:
        path_str: str = kwargs["path"]
        content: str = kwargs["content"]

        fpath = Path(path_str).resolve()

        try:
            # Create parent directories
            fpath.parent.mkdir(parents=True, exist_ok=True)
            fpath.write_text(content, encoding="utf-8")
        except PermissionError:
            raise ToolError(f"Permission denied: {fpath}", tool_name=self.name)
        except OSError as exc:
            raise ToolError(f"Write error: {exc}", tool_name=self.name)

        lines = content.count("\n") + (1 if content and not content.endswith("\n") else 0)
        return f"Wrote {len(content):,} chars ({lines} lines) to {fpath}"


# =====================================================================
#  EditFileTool
# =====================================================================
class EditFileTool(AxiomTool):
    """Replace a specific string in a file with new content."""

    name = "edit_file"
    description = (
        "Replace a specific string in a file with new content. "
        "The old_string must appear exactly once in the file to avoid ambiguity. "
        "Use read_file first to see the exact content to replace."
    )
    risk_level = "medium"
    parameters_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "File path to edit",
            },
            "old_string": {
                "type": "string",
                "description": "Exact string to find and replace (must be unique in the file)",
            },
            "new_string": {
                "type": "string",
                "description": "Replacement string",
            },
        },
        "required": ["path", "old_string", "new_string"],
    }

    async def execute(self, **kwargs: Any) -> str:
        path_str: str = kwargs["path"]
        old_string: str = kwargs["old_string"]
        new_string: str = kwargs["new_string"]

        if old_string == new_string:
            raise ToolError(
                "old_string and new_string are identical -- nothing to change",
                tool_name=self.name,
            )

        fpath = Path(path_str).resolve()

        if not fpath.exists():
            raise ToolError(f"File not found: {fpath}", tool_name=self.name)
        if not fpath.is_file():
            raise ToolError(f"Not a file: {fpath}", tool_name=self.name)

        try:
            content = fpath.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            content = fpath.read_bytes().decode("latin-1", errors="replace")
        except OSError as exc:
            raise ToolError(f"Read error: {exc}", tool_name=self.name)

        # Check uniqueness
        count = content.count(old_string)
        if count == 0:
            # Show a preview to help the LLM debug
            preview = content[:500] + "..." if len(content) > 500 else content
            raise ToolError(
                f"old_string not found in {fpath.name}. "
                f"File starts with:\n{preview}",
                tool_name=self.name,
            )
        if count > 1:
            raise ToolError(
                f"old_string appears {count} times in {fpath.name} -- must be unique. "
                "Provide more surrounding context to make the match unique.",
                tool_name=self.name,
            )

        # Perform replacement
        new_content = content.replace(old_string, new_string, 1)

        try:
            fpath.write_text(new_content, encoding="utf-8")
        except OSError as exc:
            raise ToolError(f"Write error: {exc}", tool_name=self.name)

        return f"Edited {fpath} -- replaced 1 occurrence ({len(old_string)} -> {len(new_string)} chars)"


# =====================================================================
#  GlobTool
# =====================================================================
class GlobTool(AxiomTool):
    """Find files matching a glob pattern."""

    name = "glob"
    description = (
        "Find files matching a glob pattern (e.g., '**/*.py', 'src/**/*.ts'). "
        "Returns a sorted list of matching file paths."
    )
    risk_level = "low"
    parameters_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "pattern": {
                "type": "string",
                "description": "Glob pattern (e.g., '**/*.py')",
            },
            "path": {
                "type": "string",
                "description": "Base directory to search in (default: current directory)",
                "default": ".",
            },
        },
        "required": ["pattern"],
    }

    async def execute(self, **kwargs: Any) -> str:
        pattern: str = kwargs["pattern"]
        base: str = kwargs.get("path", ".")

        base_path = Path(base).resolve()
        if not base_path.is_dir():
            raise ToolError(f"Not a directory: {base_path}", tool_name=self.name)

        try:
            matches = sorted(base_path.glob(pattern))
        except ValueError as exc:
            raise ToolError(f"Invalid glob pattern: {exc}", tool_name=self.name)

        # Filter to files only, skip hidden/vcs dirs
        files = [
            m for m in matches
            if m.is_file() and not _is_hidden_or_vcs(m, base_path)
        ]

        if not files:
            return f"No files matching '{pattern}' in {base_path}"

        # Format output -- show relative paths
        lines = []
        for f in files[:1000]:  # cap at 1000 results
            try:
                rel = f.relative_to(base_path)
            except ValueError:
                rel = f
            lines.append(str(rel))

        result = "\n".join(lines)
        if len(files) > 1000:
            result += f"\n\n... and {len(files) - 1000} more files"

        return f"Found {len(files)} files matching '{pattern}':\n{result}"


# =====================================================================
#  GrepTool
# =====================================================================
class GrepTool(AxiomTool):
    """Search file contents for a regex pattern."""

    name = "grep"
    description = (
        "Search file contents for a regex pattern. "
        "Returns matching lines with file paths and line numbers. "
        "Searches recursively through directories."
    )
    risk_level = "low"
    parameters_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "pattern": {
                "type": "string",
                "description": "Regex pattern to search for",
            },
            "path": {
                "type": "string",
                "description": "File or directory to search (default: current directory)",
                "default": ".",
            },
            "include": {
                "type": "string",
                "description": "File glob to filter (e.g., '*.py', '*.ts')",
            },
        },
        "required": ["pattern"],
    }

    async def execute(self, **kwargs: Any) -> str:
        pattern_str: str = kwargs["pattern"]
        search_path: str = kwargs.get("path", ".")
        include_glob: str | None = kwargs.get("include")

        try:
            regex = re.compile(pattern_str)
        except re.error as exc:
            raise ToolError(f"Invalid regex: {exc}", tool_name=self.name)

        root = Path(search_path).resolve()
        if not root.exists():
            raise ToolError(f"Path not found: {root}", tool_name=self.name)

        results: list[str] = []
        match_count = 0

        # Collect files to search
        if root.is_file():
            files_to_search = [root]
        else:
            files_to_search = _walk_files(root, include_glob)

        for fpath in files_to_search:
            if match_count >= _MAX_GREP_MATCHES:
                break

            try:
                content = fpath.read_text(encoding="utf-8", errors="replace")
            except (OSError, PermissionError):
                continue

            for lineno, line in enumerate(content.splitlines(), start=1):
                if match_count >= _MAX_GREP_MATCHES:
                    break
                if regex.search(line):
                    try:
                        rel = fpath.relative_to(root if root.is_dir() else root.parent)
                    except ValueError:
                        rel = fpath
                    # Truncate very long lines
                    display = line.rstrip()
                    if len(display) > 500:
                        display = display[:500] + "..."
                    results.append(f"{rel}:{lineno}: {display}")
                    match_count += 1

        if not results:
            return f"No matches for pattern '{pattern_str}' in {root}"

        output = "\n".join(results)
        header = f"Found {match_count} matches for '{pattern_str}'"
        if match_count >= _MAX_GREP_MATCHES:
            header += f" (showing first {_MAX_GREP_MATCHES}, more exist)"
        return f"{header}:\n{output}"


# ── Helpers ──────────────────────────────────────────────────────────

def _decode(raw: bytes) -> str:
    """Decode bytes, trying utf-8 first then latin-1."""
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("latin-1", errors="replace")


def _is_hidden_or_vcs(path: Path, base: Path) -> bool:
    """Check if a path is inside a hidden or VCS directory."""
    _skip = {".git", ".hg", ".svn", "__pycache__", "node_modules", ".venv", "venv"}
    try:
        rel = path.relative_to(base)
    except ValueError:
        return False
    return any(part in _skip or part.startswith(".") for part in rel.parts[:-1])


def _walk_files(root: Path, include_glob: str | None = None) -> list[Path]:
    """Recursively collect files, optionally filtered by glob, skipping hidden/VCS dirs."""
    _skip_dirs = {".git", ".hg", ".svn", "__pycache__", "node_modules", ".venv", "venv", ".tox", ".mypy_cache"}
    _binary_exts = {
        ".pyc", ".pyo", ".so", ".dll", ".exe", ".bin", ".o", ".a",
        ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".ico", ".svg",
        ".woff", ".woff2", ".ttf", ".eot",
        ".zip", ".tar", ".gz", ".bz2", ".xz", ".7z",
        ".pdf", ".doc", ".docx", ".xls", ".xlsx",
        ".mp3", ".mp4", ".avi", ".mov", ".mkv",
        ".db", ".sqlite", ".sqlite3",
    }

    files: list[Path] = []
    try:
        for entry in root.rglob("*"):
            # Skip directories matching skip list
            if any(part in _skip_dirs for part in entry.parts):
                continue
            if not entry.is_file():
                continue
            # Skip binary files
            if entry.suffix.lower() in _binary_exts:
                continue
            # Apply include glob filter
            if include_glob and not entry.match(include_glob):
                continue
            # Skip very large files
            try:
                if entry.stat().st_size > _MAX_FILE_SIZE:
                    continue
            except OSError:
                continue
            files.append(entry)
    except PermissionError:
        pass

    return sorted(files)
