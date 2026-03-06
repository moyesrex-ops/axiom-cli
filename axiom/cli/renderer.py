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
        """Render the result of a completed tool call.

        Parameters
        ----------
        tool_name:
            Name of the tool that ran.
        result:
            Textual output / return value of the tool.
        success:
            Whether the execution succeeded.
        duration_ms:
            Wall-clock time in milliseconds.
        """
        icon = "\u2713" if success else "\u2717"
        color = AXIOM_GREEN if success else AXIOM_RED
        status_style = STYLE_SUCCESS if success else STYLE_ERROR

        # Truncate very long results for display
        display_result = result
        if len(display_result) > 2000:
            display_result = display_result[:1997] + "..."

        title_text = Text()
        title_text.append(f" {icon} ", style=status_style)
        title_text.append(tool_name, style=Style(bold=True, color=color))
        if duration_ms > 0:
            title_text.append(f"  {duration_ms}ms", style=STYLE_DIM)

        panel = Panel(
            Text(display_result, overflow="fold"),
            title=title_text,
            title_align="left",
            border_style=Style(color=color, dim=True),
            padding=(0, 1),
            expand=False,
        )
        self._console.print(panel)

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
