"""
Axiom CLI -- Theme constants and Rich style definitions.

Central source of truth for all colors, styles, and visual tokens
used across the CLI rendering layer.
"""

from __future__ import annotations

import io
import os
import sys

from rich.console import Console
from rich.style import Style
from rich.theme import Theme

# ── Force UTF-8 on Windows ────────────────────────────────────────────────────
# Without this, Rich falls back to legacy Windows rendering (cp1252) which
# chokes on Unicode block-drawing and symbol characters.

if sys.platform == "win32":
    os.environ.setdefault("PYTHONUTF8", "1")
    if hasattr(sys.stdout, "buffer"):
        sys.stdout = io.TextIOWrapper(
            sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True,
        )
    if hasattr(sys.stderr, "buffer"):
        sys.stderr = io.TextIOWrapper(
            sys.stderr.buffer, encoding="utf-8", errors="replace", line_buffering=True,
        )

# ── Brand Colors ──────────────────────────────────────────────────────────────

AXIOM_CYAN = "#00d4ff"
AXIOM_PURPLE = "#a855f7"
AXIOM_GREEN = "#22c55e"
AXIOM_YELLOW = "#eab308"
AXIOM_RED = "#ef4444"
AXIOM_DIM = "#6b7280"

# ── Semantic Aliases ──────────────────────────────────────────────────────────

COLOR_PRIMARY = AXIOM_CYAN
COLOR_ACCENT = AXIOM_PURPLE
COLOR_SUCCESS = AXIOM_GREEN
COLOR_WARNING = AXIOM_YELLOW
COLOR_ERROR = AXIOM_RED
COLOR_MUTED = AXIOM_DIM

# ── Rich Styles ───────────────────────────────────────────────────────────────

STYLE_PROMPT = Style(color=AXIOM_CYAN, bold=True)
STYLE_BANNER_BORDER = Style(color=AXIOM_CYAN)
STYLE_BANNER_TITLE = Style(color=AXIOM_CYAN, bold=True)
STYLE_BANNER_SUBTITLE = Style(color=AXIOM_DIM)

STYLE_THINKING = Style(color=AXIOM_PURPLE, italic=True)

STYLE_TOOL_READ = Style(color=AXIOM_GREEN)
STYLE_TOOL_WRITE = Style(color=AXIOM_YELLOW)
STYLE_TOOL_DESTRUCTIVE = Style(color=AXIOM_RED, bold=True)

STYLE_ERROR = Style(color=AXIOM_RED, bold=True)
STYLE_INFO = Style(color=AXIOM_CYAN)
STYLE_SUCCESS = Style(color=AXIOM_GREEN, bold=True)
STYLE_DIM = Style(color=AXIOM_DIM, dim=True)

# ── Risk-Level Style Map ──────────────────────────────────────────────────────

RISK_STYLES: dict[str, Style] = {
    "low": STYLE_TOOL_READ,
    "read": STYLE_TOOL_READ,
    "medium": STYLE_TOOL_WRITE,
    "write": STYLE_TOOL_WRITE,
    "high": STYLE_TOOL_DESTRUCTIVE,
    "destructive": STYLE_TOOL_DESTRUCTIVE,
}

RISK_BORDER_COLORS: dict[str, str] = {
    "low": AXIOM_GREEN,
    "read": AXIOM_GREEN,
    "medium": AXIOM_YELLOW,
    "write": AXIOM_YELLOW,
    "high": AXIOM_RED,
    "destructive": AXIOM_RED,
}

# ── Rich Theme (for Console) ─────────────────────────────────────────────────

AXIOM_THEME = Theme(
    {
        "axiom.primary": AXIOM_CYAN,
        "axiom.accent": AXIOM_PURPLE,
        "axiom.success": AXIOM_GREEN,
        "axiom.warning": AXIOM_YELLOW,
        "axiom.error": AXIOM_RED,
        "axiom.dim": AXIOM_DIM,
        "axiom.prompt": f"bold {AXIOM_CYAN}",
    }
)

# ── prompt_toolkit Style (for input) ─────────────────────────────────────────

PROMPT_TOOLKIT_STYLE: list[tuple[str, str]] = [
    ("prompt", f"bold {AXIOM_CYAN}"),
    ("", ""),  # default text
]


# ── Console Factory ──────────────────────────────────────────────────────────


def make_console(**kwargs: object) -> Console:
    """Create a :class:`~rich.console.Console` with the Axiom theme.

    On Windows, forces ``force_terminal=True`` so Rich uses ANSI escape
    sequences instead of the legacy Win32 renderer (which can't handle
    the full Unicode range we need).
    """
    defaults: dict[str, object] = {"theme": AXIOM_THEME}
    if sys.platform == "win32":
        defaults["force_terminal"] = True
    defaults.update(kwargs)
    return Console(**defaults)  # type: ignore[arg-type]
