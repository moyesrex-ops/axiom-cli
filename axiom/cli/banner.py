"""
Axiom CLI -- Startup banner.

Renders the branded ASCII art banner with system/session information
using Rich panels and styled text.
"""

from __future__ import annotations

import platform
import sys
from datetime import datetime

from rich.panel import Panel
from rich.text import Text

from axiom.cli.theme import (
    AXIOM_CYAN,
    AXIOM_DIM,
    AXIOM_GREEN,
    AXIOM_PURPLE,
    STYLE_BANNER_BORDER,
    make_console,
)

# ── ASCII Art ─────────────────────────────────────────────────────────────────

_ASCII_ART = (
    "\n"
    " [bold {cyan}] \u2584\u2580\u2588 \u2580\u2584\u2580 \u2588 \u2588\u2580\u2588 \u2588\u2580\u2584\u2580\u2588[/]\n"
    " [bold {cyan}] \u2588\u2580\u2588 \u2588\u2591\u2588 \u2588 \u2588\u2584\u2588 \u2588\u2591\u2580\u2591\u2588[/]\n"
).format(cyan=AXIOM_CYAN)

_console = make_console()


def print_banner(
    model_name: str = "claude-opus-4",
    tool_count: int = 0,
    memory_count: int = 0,
    skill_count: int = 0,
    mcp_count: int = 0,
    telegram_active: bool = False,
) -> None:
    """Print the Axiom startup banner with session metadata.

    Parameters
    ----------
    model_name:
        Display name of the active LLM (e.g. "Claude Opus 4").
    tool_count:
        Number of tools currently loaded / registered.
    memory_count:
        Number of memory entries currently available.
    skill_count:
        Number of domain-knowledge skills loaded.
    mcp_count:
        Number of MCP tools bridged from external servers.
    """
    # ── Build the banner body ─────────────────────────────────────────────
    body = Text()

    # ASCII art (already has Rich markup -- use from_markup)
    art = Text.from_markup(_ASCII_ART)
    body.append_text(art)
    body.append("\n")

    # Version / model / tools tagline
    cap_parts = [f"{tool_count} tools"]
    if skill_count > 0:
        cap_parts.append(f"{skill_count} skills")
    if mcp_count > 0:
        cap_parts.append(f"{mcp_count} MCP")

    tagline = Text.from_markup(
        f"  [{AXIOM_DIM}]v1.0.0[/]  [{AXIOM_CYAN}]\u2502[/]  "
        f"[bold {AXIOM_CYAN}]{model_name}[/]  [{AXIOM_CYAN}]\u2502[/]  "
        f"[{AXIOM_DIM}]{' \u00b7 '.join(cap_parts)}[/]"
    )
    body.append_text(tagline)
    body.append("\n\n")

    # System info line
    py_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    os_label = f"{platform.system()} {platform.release()}"

    sys_line = Text.from_markup(
        f"  [{AXIOM_DIM}]Python {py_version}  |  {os_label}  |  Model: {model_name}[/]"
    )
    body.append_text(sys_line)
    body.append("\n")

    # Memory status
    if memory_count > 0:
        mem_color = AXIOM_GREEN
        mem_text = f"{memory_count} memories loaded"
    else:
        mem_color = AXIOM_DIM
        mem_text = "no persistent memories"

    mem_line = Text.from_markup(
        f"  [{mem_color}]\u25cf[/] [{AXIOM_DIM}]{mem_text}[/]  "
        f"[{AXIOM_DIM}]|  {datetime.now().strftime('%Y-%m-%d %H:%M')}[/]"
    )
    body.append_text(mem_line)

    # Integration status
    integrations: list[str] = []
    if telegram_active:
        integrations.append(f"[{AXIOM_GREEN}]\u25cf Telegram[/]")
    else:
        integrations.append(f"[{AXIOM_DIM}]\u25cb Telegram[/]")
    if mcp_count > 0:
        integrations.append(f"[{AXIOM_GREEN}]\u25cf MCP ({mcp_count})[/]")

    if integrations:
        int_line = Text.from_markup(f"  {'  '.join(integrations)}")
        body.append("\n")
        body.append_text(int_line)

    # ── Render ────────────────────────────────────────────────────────────
    panel = Panel(
        body,
        border_style=STYLE_BANNER_BORDER,
        subtitle=Text.from_markup(
            f"[{AXIOM_DIM}]The Unstoppable AI Agent[/]"
        ),
        subtitle_align="center",
        padding=(0, 1),
        expand=False,
    )

    _console.print()
    _console.print(panel)
    _console.print()
