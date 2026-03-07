"""Tool approval flow — interactive approve/deny/edit for tool calls.

Provides a Rich-based approval interface that displays tool calls with
risk-level coloring and allows the user to approve, deny, edit args,
or skip the tool call.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.text import Text

from axiom.cli.theme import (
    AXIOM_CYAN,
    AXIOM_DIM,
    AXIOM_GREEN,
    AXIOM_PURPLE,
    AXIOM_RED,
    AXIOM_YELLOW,
)

logger = logging.getLogger(__name__)

# Tools that are always safe to auto-approve
SAFE_TOOLS = frozenset({
    "read_file",
    "glob",
    "grep",
    "memory_search",
    "vision",
})


class ToolApproval:
    """Interactive tool approval handler.

    Displays tool calls with risk-level coloring and provides
    an approve/deny/edit interface.
    """

    def __init__(
        self,
        console: Console,
        yolo_mode: bool = False,
        trusted_tools: set[str] | None = None,
    ):
        self.console = console
        self.yolo_mode = yolo_mode
        self.trusted_tools = trusted_tools or set()

    def should_auto_approve(self, tool_name: str, risk_level: str) -> bool:
        """Check if a tool call should be auto-approved."""
        if self.yolo_mode:
            return True
        if tool_name in SAFE_TOOLS:
            return True
        if tool_name in self.trusted_tools:
            return True
        if risk_level == "low":
            return True
        return False

    def request_approval(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
        risk_level: str = "medium",
        description: str = "",
    ) -> tuple[bool, dict[str, Any]]:
        """Request user approval for a tool call.

        Args:
            tool_name: Name of the tool.
            tool_args: Arguments to pass to the tool.
            risk_level: Risk level (low, medium, high).
            description: Human-readable description of what the tool does.

        Returns:
            Tuple of (approved: bool, possibly_edited_args: dict).
        """
        if self.should_auto_approve(tool_name, risk_level):
            return True, tool_args

        # Display the tool call panel
        self._show_approval_panel(tool_name, tool_args, risk_level, description)

        # Get user decision
        while True:
            try:
                response = input(
                    "  [A]ccept  [D]eny  [E]dit  [T]rust always > "
                ).strip().lower()
            except (EOFError, KeyboardInterrupt):
                self.console.print(f"  [{AXIOM_DIM}]Denied.[/]")
                return False, tool_args

            if response in ("", "a", "accept", "y", "yes"):
                return True, tool_args

            elif response in ("d", "deny", "n", "no"):
                self.console.print(f"  [{AXIOM_DIM}]Denied.[/]")
                return False, tool_args

            elif response in ("e", "edit"):
                edited = self._edit_args(tool_args)
                if edited is not None:
                    return True, edited
                # Edit cancelled, re-show prompt
                continue

            elif response in ("t", "trust"):
                self.trusted_tools.add(tool_name)
                self.console.print(
                    f"  [{AXIOM_GREEN}]Added '{tool_name}' to trusted tools.[/]"
                )
                return True, tool_args

            else:
                self.console.print(
                    f"  [{AXIOM_DIM}]Enter A (accept), D (deny), "
                    f"E (edit args), or T (trust always).[/]"
                )

    def _show_approval_panel(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
        risk_level: str,
        description: str,
    ) -> None:
        """Display a compact approval prompt — NOT a massive bordered panel.

        Shows the tool name, risk badge, and a short preview of the key
        argument on 2-3 lines max.  Full args are only shown if the user
        presses [E]dit.
        """
        risk_color = {
            "low": AXIOM_GREEN,
            "medium": AXIOM_YELLOW,
            "high": AXIOM_RED,
        }.get(risk_level, AXIOM_YELLOW)

        risk_tag = risk_level.upper()

        # Extract the most relevant arg for preview
        preview = _compact_arg_preview(tool_name, tool_args)

        self.console.print()
        self.console.print(
            f"  [{AXIOM_PURPLE}]\u26a1 {tool_name}[/]"
            f"  [{risk_color}][{risk_tag}][/]"
        )
        if preview:
            # Truncate long previews (e.g. code blocks) to 3 lines
            preview_lines = preview.split("\n")
            if len(preview_lines) > 3:
                shown = "\n".join(preview_lines[:3])
                self.console.print(
                    f"  [{AXIOM_DIM}]{shown}\n  ... ({len(preview_lines) - 3} more lines)[/]"
                )
            else:
                self.console.print(f"  [{AXIOM_DIM}]{preview}[/]")
        if description:
            self.console.print(f"  [{AXIOM_DIM}]{description}[/]")

    def _edit_args(self, tool_args: dict[str, Any]) -> dict[str, Any] | None:
        """Allow the user to edit tool arguments interactively.

        Returns edited args dict, or None if edit was cancelled.
        """
        self.console.print(
            f"\n  [{AXIOM_CYAN}]Edit arguments (JSON). "
            f"Press Enter on empty line to finish, 'cancel' to abort:[/]"
        )

        current_json = json.dumps(tool_args, indent=2)
        self.console.print(f"  [{AXIOM_DIM}]Current:[/]")
        for line in current_json.split("\n"):
            self.console.print(f"    {line}")

        self.console.print(f"\n  [{AXIOM_CYAN}]Enter new JSON:[/]")

        lines: list[str] = []
        try:
            while True:
                line = input("  > ")
                if line.strip().lower() == "cancel":
                    self.console.print(f"  [{AXIOM_DIM}]Edit cancelled.[/]")
                    return None
                if line.strip() == "" and lines:
                    break
                lines.append(line)
        except (EOFError, KeyboardInterrupt):
            self.console.print(f"  [{AXIOM_DIM}]Edit cancelled.[/]")
            return None

        if not lines:
            return tool_args  # No changes

        try:
            edited = json.loads("\n".join(lines))
            self.console.print(f"  [{AXIOM_GREEN}]Arguments updated.[/]")
            return edited
        except json.JSONDecodeError as e:
            self.console.print(
                f"  [{AXIOM_RED}]Invalid JSON: {e}. Using original args.[/]"
            )
            return tool_args


# ── Helpers ──────────────────────────────────────────────────────────────────


# Map of tool → which arg key to preview
_PREVIEW_KEYS: dict[str, tuple[str, ...]] = {
    "bash": ("command",),
    "write_file": ("path", "file_path"),
    "edit_file": ("path", "file_path"),
    "create_file": ("path", "file_path"),
    "code_exec": ("code",),
    "git": ("command",),
    "http_request": ("url",),
    "web_fetch": ("url",),
}


def _compact_arg_preview(tool_name: str, args: dict[str, Any]) -> str:
    """Return a short preview of the tool's key argument."""
    if not args:
        return ""

    # Try known keys
    for key in _PREVIEW_KEYS.get(tool_name, ()):
        if key in args:
            val = str(args[key])
            if len(val) > 200:
                return val[:200] + "..."
            return val

    # Fallback: first string arg
    for key, val in args.items():
        if isinstance(val, str) and val:
            preview = val[:150]
            return f"{key}: {preview}" + ("..." if len(val) > 150 else "")

    return ""
