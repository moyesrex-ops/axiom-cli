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
        """Display a Rich panel showing the tool call details."""
        # Color based on risk
        border_color = {
            "low": AXIOM_GREEN,
            "medium": AXIOM_YELLOW,
            "high": AXIOM_RED,
        }.get(risk_level, AXIOM_YELLOW)

        risk_label = {
            "low": f"[{AXIOM_GREEN}]LOW[/]",
            "medium": f"[{AXIOM_YELLOW}]MEDIUM[/]",
            "high": f"[{AXIOM_RED}]HIGH[/]",
        }.get(risk_level, f"[{AXIOM_YELLOW}]UNKNOWN[/]")

        # Format args as JSON
        args_str = json.dumps(tool_args, indent=2)

        # Build panel content
        lines = [f"[bold]Risk:[/] {risk_label}"]
        if description:
            lines.append(f"[bold]Description:[/] {description}")
        lines.append("")

        # Show args with syntax highlighting for readability
        if len(args_str) > 500:
            lines.append(f"[bold]Arguments:[/]\n{args_str[:500]}...")
        else:
            lines.append(f"[bold]Arguments:[/]\n{args_str}")

        content = "\n".join(lines)

        self.console.print(
            Panel(
                content,
                title=f"[bold {AXIOM_PURPLE}]Tool Call: {tool_name}[/]",
                border_style=border_color,
                expand=False,
                padding=(1, 2),
            )
        )

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
