"""
Axiom CLI -- Streaming output renderer.

Provides ``StreamRenderer``, the single entry point for all visual output
during an agent turn: thinking spinners, streamed Markdown, tool-call
panels, plan displays, errors, and trace lines.

All rendering goes through a shared :class:`rich.console.Console` with the
Axiom theme applied so that colours are consistent everywhere.
"""

from __future__ import annotations

import json
import time
from contextlib import suppress
from typing import Any

from rich.console import Console, Group
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.spinner import Spinner
from rich.style import Style
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

from axiom.cli.theme import (
    AXIOM_CYAN,
    AXIOM_DIM,
    AXIOM_GREEN,
    AXIOM_PURPLE,
    AXIOM_RED,
    AXIOM_YELLOW,
    RISK_BORDER_COLORS,
    RISK_STYLES,
    STYLE_DIM,
    STYLE_ERROR,
    STYLE_INFO,
    STYLE_SUCCESS,
    STYLE_THINKING,
    make_console,
)


class StreamRenderer:
    """Rich-powered renderer for the Axiom interactive loop.

    Manages a :class:`~rich.live.Live` context for streaming tokens and a
    separate spinner for the "thinking" state.  All public methods are
    safe to call outside of a Live context -- they degrade gracefully to
    plain ``console.print`` calls.
    """

    _FILE_TOOLS = frozenset({"read_file"})
    _LIST_TOOLS = frozenset({"glob", "grep"})

    def __init__(self, console: Console | None = None) -> None:
        self._console = console or make_console()
        self._live: Live | None = None
        self._spinner_live: Live | None = None
        self._buffer: str = ""
        self._stream_start: float = 0.0

    # ── Properties ────────────────────────────────────────────────────────

    @property
    def console(self) -> Console:
        """The underlying Rich console."""
        return self._console

    # ── Thinking Spinner ──────────────────────────────────────────────────

    def start_thinking(self) -> None:
        """Display a pulsing spinner while the model is generating."""
        if self._spinner_live is not None:
            return  # already spinning
        spinner = Spinner(
            "dots",
            text=Text.from_markup(f"[{AXIOM_PURPLE}]Axiom is thinking...[/]"),
            style=STYLE_THINKING,
        )
        self._spinner_live = Live(
            spinner,
            console=self._console,
            refresh_per_second=12,
            transient=True,
        )
        try:
            self._spinner_live.start()
        except Exception:
            self._spinner_live = None

    def stop_thinking(self) -> None:
        """Stop the thinking spinner (if active)."""
        if self._spinner_live is not None:
            with suppress(Exception):
                self._spinner_live.stop()
            self._spinner_live = None

    # ── Streaming Tokens ──────────────────────────────────────────────────

    def stream_token(self, token: str) -> None:
        """Append *token* to the live Markdown display.

        On the first token the Live context is automatically created; each
        subsequent token re-renders the accumulated buffer as Markdown so
        that formatting (headings, lists, code blocks) appears correctly
        as it streams in.
        """
        if not token:
            return

        # First token -- print speaker label + open the Live display
        if self._live is None:
            # Show "Axiom" speaker label before the response
            self._console.print(
                Text.from_markup(f"\n[bold {AXIOM_PURPLE}]Axiom[/]"),
            )
            self._buffer = ""
            self._stream_start = time.monotonic()
            self._live = Live(
                Markdown(""),
                console=self._console,
                refresh_per_second=10,
                vertical_overflow="visible",
            )
            try:
                self._live.start()
            except Exception:
                self._live = None

        self._buffer += token

        if self._live is not None:
            try:
                self._live.update(Markdown(self._buffer))
            except Exception:
                # If Live update fails, just accumulate -- finish will print.
                pass

    def finish_stream(self) -> None:
        """Finalize the streaming display.

        If the ``Live`` context was active, the last update is already
        rendered to the terminal (Rich keeps it on-screen when
        ``transient=False``).  We only fall back to a full
        ``console.print`` when ``Live`` was never started (e.g. in a
        non-TTY pipe or when the Live constructor raised).
        """
        live_was_active = self._live is not None
        if self._live is not None:
            with suppress(Exception):
                self._live.stop()
            self._live = None

        if self._buffer:
            if not live_was_active:
                # Live display failed / never started — print the
                # accumulated text so the user still sees it.
                self._console.print(Markdown(self._buffer))

            elapsed = time.monotonic() - self._stream_start
            if elapsed > 0.5:
                self._console.print(
                    Text.from_markup(
                        f"  [{AXIOM_DIM}]{elapsed:.1f}s[/]"
                    ),
                )
            self._buffer = ""

    # ── Tool Calls ────────────────────────────────────────────────────────

    def show_tool_call(
        self,
        tool_name: str,
        args: dict[str, Any],
        risk: str = "low",
    ) -> None:
        """Render a tool invocation panel.

        Parameters
        ----------
        tool_name:
            Name of the tool being invoked.
        args:
            Arguments dictionary (will be pretty-printed as JSON).
        risk:
            ``"low"`` / ``"read"`` (green), ``"medium"`` / ``"write"``
            (yellow), ``"high"`` / ``"destructive"`` (red).
        """
        risk_key = risk.lower()
        border_color = RISK_BORDER_COLORS.get(risk_key, AXIOM_GREEN)
        label_style = RISK_STYLES.get(risk_key, Style(color=AXIOM_GREEN))

        # Format arguments -- truncate very long values
        formatted_args = _format_args(args)

        # Build the body
        body_parts: list[Any] = []
        if formatted_args:
            body_parts.append(
                Syntax(
                    formatted_args,
                    "json",
                    theme="monokai",
                    line_numbers=False,
                    word_wrap=True,
                )
            )

        risk_label = risk_key.upper() if risk_key in ("high", "destructive") else ""
        title_text = Text()
        title_text.append("\u2699 ", style=label_style)
        title_text.append(tool_name, style=Style(bold=True, color=border_color))
        if risk_label:
            title_text.append(f"  [{risk_label}]", style=Style(color=AXIOM_RED, bold=True))

        panel = Panel(
            Group(*body_parts) if body_parts else Text("(no arguments)", style=STYLE_DIM),
            title=title_text,
            title_align="left",
            border_style=Style(color=border_color),
            padding=(0, 1),
            expand=False,
        )
        self._console.print(panel)

    def show_tool_result(
        self,
        tool_name: str,
        result: str,
        success: bool = True,
        duration_ms: int = 0,
    ) -> None:
        """Render the result of a completed tool call."""
        icon = "\u2713" if success else "\u2717"
        color = AXIOM_GREEN if success else AXIOM_RED
        status_style = STYLE_SUCCESS if success else STYLE_ERROR

        title_text = Text()
        title_text.append(f" {icon} ", style=status_style)
        title_text.append(tool_name, style=Style(bold=True, color=color))
        if duration_ms > 0:
            title_text.append(f"  {duration_ms}ms", style=STYLE_DIM)

        body = self._render_tool_body(tool_name, result, success)

        panel = Panel(
            body,
            title=title_text,
            title_align="left",
            border_style=Style(color=color, dim=True),
            padding=(0, 1),
            expand=False,
        )
        self._console.print(panel)

    # ── Smart Tool Body Renderers ──────────────────────────────────────

    def _render_tool_body(self, tool_name: str, result: str, success: bool) -> Any:
        """Choose the best Rich renderable for a tool result."""
        if not success:
            return Text(result[:2000], style=STYLE_ERROR, overflow="fold")

        if tool_name in self._FILE_TOOLS:
            return self._render_file_content(result)

        if tool_name in self._LIST_TOOLS:
            return self._render_file_list(result)

        return self._render_truncated(result)

    def _render_file_content(self, result: str) -> Any:
        """Render read_file output with syntax highlighting."""
        lines = result.split("\n", 1)
        header = lines[0] if lines else ""
        body = lines[1] if len(lines) > 1 else ""

        # Extract filepath from header: [/path/to/file.py] N lines...
        filepath = ""
        if header.startswith("[") and "]" in header:
            filepath = header[1 : header.index("]")]

        lexer = _guess_lexer(filepath)

        # Strip line numbers (format: "     1\tcontent")
        stripped: list[str] = []
        for line in body.split("\n"):
            if "\t" in line:
                stripped.append(line.split("\t", 1)[1])
            else:
                stripped.append(line)

        code = "\n".join(stripped)

        max_display = 3000
        truncated = len(code) > max_display
        if truncated:
            code = code[:max_display]
            last_nl = code.rfind("\n")
            if last_nl > max_display * 0.8:
                code = code[:last_nl]

        parts: list[Any] = [
            Text.from_markup(f"[{AXIOM_DIM}]{header}[/]"),
        ]

        if code.strip():
            parts.append(
                Syntax(
                    code,
                    lexer,
                    theme="monokai",
                    line_numbers=True,
                    word_wrap=True,
                )
            )

        if truncated:
            remaining = len(result) - max_display
            parts.append(
                Text.from_markup(f"\n[{AXIOM_DIM}]... {remaining:,} more chars[/]")
            )

        return Group(*parts)

    def _render_file_list(self, result: str) -> Any:
        """Render glob/grep output compactly."""
        lines = result.strip().split("\n")
        summary = lines[0] if lines else result
        display_lines = lines[1:21]
        remaining = len(lines) - 21 if len(lines) > 21 else 0

        parts: list[Any] = [
            Text.from_markup(f"[bold {AXIOM_CYAN}]{summary}[/]"),
        ]
        if display_lines:
            parts.append(
                Text("\n".join(display_lines), style=STYLE_DIM, overflow="fold")
            )
        if remaining > 0:
            parts.append(
                Text.from_markup(f"\n[{AXIOM_DIM}]... {remaining} more entries[/]")
            )

        return Group(*parts)

    def _render_truncated(self, result: str) -> Any:
        """Render generic tool output with smart truncation."""
        max_display = 2000
        if len(result) <= max_display:
            return Text(result, overflow="fold")

        truncated = result[:max_display]
        last_nl = truncated.rfind("\n")
        if last_nl > max_display * 0.7:
            truncated = truncated[:last_nl]

        remaining = len(result) - len(truncated)
        return Group(
            Text(truncated, overflow="fold"),
            Text.from_markup(f"\n[{AXIOM_DIM}]... {remaining:,} more chars[/]"),
        )

    # ── Plan Display ──────────────────────────────────────────────────────

    def show_plan(self, steps: list[dict[str, Any]]) -> None:
        """Render a numbered execution plan.

        Each *step* dict should have at least a ``"description"`` key.
        Optional keys: ``"tool"``, ``"status"`` (pending/running/done/failed).
        """
        if not steps:
            return

        table = Table(
            title=Text.from_markup(f"[bold {AXIOM_CYAN}]Execution Plan[/]"),
            show_header=True,
            header_style=Style(bold=True, color=AXIOM_CYAN),
            border_style=Style(color=AXIOM_DIM),
            padding=(0, 1),
            expand=False,
        )
        table.add_column("#", justify="right", width=4, style=STYLE_DIM)
        table.add_column("Step", min_width=30)
        table.add_column("Tool", min_width=12, style=Style(color=AXIOM_PURPLE))
        table.add_column("Status", width=10, justify="center")

        status_icons = {
            "pending": f"[{AXIOM_DIM}]\u2500[/]",
            "running": f"[{AXIOM_YELLOW}]\u25b6[/]",
            "done": f"[{AXIOM_GREEN}]\u2713[/]",
            "failed": f"[{AXIOM_RED}]\u2717[/]",
        }

        for idx, step in enumerate(steps, 1):
            description = step.get("description", step.get("text", ""))
            tool = step.get("tool", "")
            status = step.get("status", "pending").lower()
            icon = status_icons.get(status, status_icons["pending"])

            table.add_row(
                str(idx),
                description,
                tool,
                Text.from_markup(icon),
            )

        self._console.print()
        self._console.print(table)
        self._console.print()

    # ── Info / Error Panels ───────────────────────────────────────────────

    def show_error(self, message: str) -> None:
        """Render a red error panel."""
        panel = Panel(
            Text(message, style=STYLE_ERROR),
            title=Text.from_markup(f"[bold {AXIOM_RED}]Error[/]"),
            title_align="left",
            border_style=Style(color=AXIOM_RED),
            padding=(0, 1),
            expand=False,
        )
        self._console.print(panel)

    def show_info(self, message: str) -> None:
        """Render a blue/cyan info panel."""
        panel = Panel(
            Text(message, style=STYLE_INFO),
            title=Text.from_markup(f"[bold {AXIOM_CYAN}]Info[/]"),
            title_align="left",
            border_style=Style(color=AXIOM_CYAN),
            padding=(0, 1),
            expand=False,
        )
        self._console.print(panel)

    # ── Compact Tool Display (default) ─────────────────────────────────

    def show_tool_call_compact(
        self,
        tool_name: str,
        args: dict[str, Any],
    ) -> None:
        """Render a tool invocation as a single quiet line.

        Example output::

            ⚡ bash  mkdir calorie_counter
            ⚡ grep  "Flask" in *.py
            ⚡ memory_search  calorie counter website deploy
        """
        preview = _compact_args_preview(tool_name, args)
        self._console.print(
            Text.from_markup(
                f"  [{AXIOM_PURPLE}]\u26a1 {tool_name}[/]"
                + (f"  [{AXIOM_DIM}]{preview}[/]" if preview else "")
            )
        )

    def show_tool_result_compact(
        self,
        tool_name: str,
        result: str,
        success: bool = True,
        duration_ms: int = 0,
    ) -> None:
        """Render a tool result as a single quiet line.

        Shows full error details when ``success=False``.

        Example output::

            ✓ bash (42ms)
            ✓ glob — 169 files (254ms)
            ✗ bash — command not found
        """
        icon = "\u2713" if success else "\u2717"
        color = AXIOM_GREEN if success else AXIOM_RED
        dur = f" ({duration_ms}ms)" if duration_ms > 0 else ""
        summary = _compact_result_summary(tool_name, result, success)

        self._console.print(
            Text.from_markup(
                f"  [{color}]{icon}[/] [{AXIOM_DIM}]{tool_name}{dur}[/]"
                + (f"  {summary}" if summary else "")
            )
        )

        # Errors always get details — but compact, no panel
        if not success and result:
            err_preview = result[:300].split("\n")[0]
            self._console.print(
                Text.from_markup(f"    [{AXIOM_RED}]{err_preview}[/]")
            )

    # ── Agent Trace ───────────────────────────────────────────────────────

    def show_agent_trace(self, trace: dict[str, Any]) -> None:
        """Render a single trace line in dim text.

        Useful for showing the internal agent loop state transitions
        (thought, action, observation, etc.) without being too noisy.
        The *trace* dict is serialized to a compact one-liner.
        """
        parts: list[str] = []
        for key in ("type", "mode", "step", "tool", "status", "message"):
            if key in trace:
                val = trace[key]
                if isinstance(val, str) and len(val) > 80:
                    val = val[:77] + "..."
                parts.append(f"{key}={val}")

        if not parts:
            parts.append(json.dumps(trace, default=str)[:120])

        line = "  \u2502 " + "  ".join(parts)
        self._console.print(Text(line, style=STYLE_DIM))


# ── Helpers ───────────────────────────────────────────────────────────────────


_EXTENSION_LEXER_MAP: dict[str, str] = {
    ".py": "python", ".js": "javascript", ".ts": "typescript",
    ".tsx": "tsx", ".jsx": "jsx", ".json": "json",
    ".yaml": "yaml", ".yml": "yaml", ".toml": "toml",
    ".md": "markdown", ".html": "html", ".css": "css",
    ".sql": "sql", ".sh": "bash", ".bash": "bash",
    ".rs": "rust", ".go": "go", ".java": "java",
    ".c": "c", ".cpp": "cpp", ".h": "c", ".rb": "ruby",
    ".xml": "xml", ".ini": "ini", ".env": "bash",
    ".mq5": "c", ".mq4": "c",
}


def _guess_lexer(filepath: str) -> str:
    """Guess the Pygments lexer name from a file path extension."""
    if not filepath:
        return "text"
    dot_idx = filepath.rfind(".")
    if dot_idx >= 0:
        ext = filepath[dot_idx:].lower()
        return _EXTENSION_LEXER_MAP.get(ext, "text")
    return "text"


def _format_args(args: dict[str, Any], max_value_len: int = 300) -> str:
    """Pretty-print tool arguments as JSON, truncating long values."""
    if not args:
        return ""

    truncated: dict[str, Any] = {}
    for key, value in args.items():
        if isinstance(value, str) and len(value) > max_value_len:
            truncated[key] = value[:max_value_len] + "..."
        else:
            truncated[key] = value

    try:
        return json.dumps(truncated, indent=2, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return str(truncated)


# ── Compact display helpers ──────────────────────────────────────────────────

# Tools whose single most-important arg we know how to extract.
_ARG_PRIORITY: dict[str, tuple[str, ...]] = {
    "bash": ("command",),
    "grep": ("pattern", "path"),
    "glob": ("pattern", "path"),
    "read_file": ("path", "file_path"),
    "write_file": ("path", "file_path"),
    "create_file": ("path", "file_path"),
    "memory_search": ("query",),
    "memory_save": ("content",),
    "web_search": ("query",),
    "http_request": ("url", "method"),
}


def _compact_args_preview(tool_name: str, args: dict[str, Any]) -> str:
    """Extract a short, human-readable preview of a tool's arguments.

    Returns at most ~100 chars — enough for context, never overwhelming.
    """
    if not args:
        return ""

    # Try known priority keys first
    priority_keys = _ARG_PRIORITY.get(tool_name, ())
    for key in priority_keys:
        if key in args:
            val = str(args[key])
            return val[:100] + ("..." if len(val) > 100 else "")

    # Fallback: first string-valued arg
    for key, val in args.items():
        if isinstance(val, str) and val:
            preview = val[:80]
            return f"{key}={preview}" + ("..." if len(val) > 80 else "")

    return ""


def _compact_result_summary(
    tool_name: str, result: str, success: bool
) -> str:
    """Extract a short summary from tool output for the compact display.

    Returns a dim-styled string like ``"— 5 results"`` or ``""`` if
    there is nothing worth summarizing.
    """
    if not success:
        return ""  # error details shown separately

    if not result:
        return ""

    lower = tool_name.lower()

    # glob / grep — count matches
    if lower in ("glob", "grep"):
        first_line = result.split("\n", 1)[0]
        if "no match" in first_line.lower() or "0 " in first_line:
            return "-- no matches"
        # Try to extract count from first line like "Found 12 files..."
        for word in first_line.split():
            if word.isdigit():
                label = "files" if "glob" in lower else "matches"
                return f"-- {word} {label}"
        lines = result.strip().split("\n")
        return f"-- {len(lines)} items"

    # memory_search — count results
    if "memory" in lower and "search" in lower:
        for word in result.split():
            if word.isdigit():
                return f"-- {word} results"
        return ""

    # bash — show first non-empty line of output
    if lower == "bash":
        for line in result.split("\n"):
            stripped = line.strip()
            if stripped:
                if len(stripped) > 80:
                    return f"-- {stripped[:77]}..."
                return f"-- {stripped}"
        return ""

    # read_file — show line count
    if lower in ("read_file",):
        lines = result.count("\n")
        return f"-- {lines} lines"

    # Default: nothing (keep it clean)
    return ""
