"""
Axiom CLI -- Input handling.

Provides ``InputHandler``, which wraps :mod:`prompt_toolkit` to give the
user a polished interactive prompt with:

* Persistent file-based history (``~/.axiom/history``).
* Multi-line editing (Shift+Enter = newline, Enter = submit).
* Slash-command autocompletion.
* A styled ``you > `` prompt string.
* A tool-approval helper for the human-in-the-loop gate.

Both sync (``get_input``) and async (``get_input_async``) interfaces are
provided.  When Axiom runs inside an ``asyncio`` event loop, the *async*
variants must be used to avoid the ``asyncio.run() cannot be called from a
running event loop`` crash that prompt_toolkit's synchronous API triggers.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any

from prompt_toolkit import PromptSession
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.keys import Keys
from prompt_toolkit.styles import Style as PTStyle

from axiom.cli.theme import AXIOM_CYAN, AXIOM_DIM

# ── Constants ─────────────────────────────────────────────────────────────────

_AXIOM_HOME = Path.home() / ".axiom"
_HISTORY_FILE = _AXIOM_HOME / "history"

_SLASH_COMMANDS: list[str] = [
    "/help",
    "/tools",
    "/model",
    "/model list",
    "/model cost",
    "/model auto",
    "/memory",
    "/memory search",
    "/memory save",
    "/memory stats",
    "/agent",
    "/agents",
    "/council",
    "/skills",
    "/skills list",
    "/skills search",
    "/skills show",
    "/mcp",
    "/mcp list",
    "/mcp connect",
    "/clear",
    "/compact",
    "/reset",
    "/history",
    "/think",
    "/trace",
    "/yolo",
    "/voice",
    "/quit",
    "/exit",
]

_PROMPT_STYLE = PTStyle.from_dict(
    {
        "prompt": f"bold {AXIOM_CYAN}",
        "": "",  # default text color
    }
)

_PROMPT_TEXT: list[tuple[str, str]] = [
    ("class:prompt", "you > "),
]


# ── Key Bindings ──────────────────────────────────────────────────────────────

def _make_bindings() -> KeyBindings:
    """Create key bindings for the prompt session.

    * **Enter** submits the current buffer.
    * **Shift+Enter** (or Escape then Enter) inserts a newline, enabling
      multi-line input.
    """
    kb = KeyBindings()

    @kb.add(Keys.Enter)
    def _submit(event: Any) -> None:
        """Submit the buffer on plain Enter."""
        event.current_buffer.validate_and_handle()

    # Shift+Enter inserts a literal newline.  Some terminals send
    # escape-then-enter for Shift+Enter, so we bind that as well.
    @kb.add(Keys.Escape, Keys.Enter)
    def _newline_escape(event: Any) -> None:
        event.current_buffer.insert_text("\n")

    return kb


# ── InputHandler ──────────────────────────────────────────────────────────────


class InputHandler:
    """Interactive prompt with history, autocompletion, and multi-line support.

    In non-interactive environments (piped stdin, CI runners, certain
    Windows terminal wrappers) the handler falls back to plain
    ``input()`` calls so the rest of the CLI can still operate.

    **Important:** When running inside an ``asyncio`` event loop (which is
    the normal case for Axiom's ``run_interactive`` method), always use the
    ``*_async`` variants to avoid nested-loop crashes.
    """

    def __init__(self) -> None:
        # Ensure ~/.axiom/ exists for the history file
        _AXIOM_HOME.mkdir(parents=True, exist_ok=True)

        self._completer = WordCompleter(
            _SLASH_COMMANDS,
            sentence=True,  # complete full command including the /
        )

        self._session: PromptSession[str] | None = None
        try:
            self._session = PromptSession(
                message=_PROMPT_TEXT,
                style=_PROMPT_STYLE,
                history=FileHistory(str(_HISTORY_FILE)),
                auto_suggest=AutoSuggestFromHistory(),
                completer=self._completer,
                complete_while_typing=True,
                multiline=False,  # Enter submits; Shift+Enter for newlines via KB
                key_bindings=_make_bindings(),
                enable_open_in_editor=False,
            )
        except Exception:
            # Non-interactive terminal (CI, piped stdin, etc.) --
            # fall back to plain input() in get_input / get_approval.
            self._session = None

    # ── Async API (preferred when inside an event loop) ─────────────

    async def get_input_async(self) -> str:
        """Async version of :meth:`get_input`.

        Uses ``prompt_async()`` which cooperates with an already-running
        ``asyncio`` event loop instead of spawning its own.
        """
        try:
            if self._session is not None:
                text: str = await self._session.prompt_async()
            else:
                # Fallback: run blocking input() in a thread so we don't
                # block the event loop.
                loop = asyncio.get_event_loop()
                text = await loop.run_in_executor(None, lambda: input("you > "))
            return text.strip()
        except KeyboardInterrupt:
            return ""
        except EOFError:
            return "/exit"

    async def get_approval_async(
        self,
        tool_name: str,
        args: dict[str, Any],
    ) -> bool:
        """Async version of :meth:`get_approval`."""
        args_summary = _compact_args(args)

        approval_prompt: list[tuple[str, str]] = [
            ("class:prompt", "  Allow "),
            ("bold", tool_name),
        ]
        if args_summary:
            approval_prompt.append(("", f" ({args_summary})"))
        approval_prompt.append(("class:prompt", "? [A]ccept / [D]eny / [E]dit: "))

        try:
            if self._session is not None:
                answer: str = (
                    await self._session.prompt_async(
                        approval_prompt,
                        style=_PROMPT_STYLE,
                    )
                ).strip().lower()
            else:
                loop = asyncio.get_event_loop()
                prompt_str = f"  Allow {tool_name}"
                if args_summary:
                    prompt_str += f" ({args_summary})"
                prompt_str += "? [A]ccept / [D]eny / [E]dit: "
                answer = await loop.run_in_executor(
                    None, lambda: input(prompt_str).strip().lower()
                )
        except (KeyboardInterrupt, EOFError):
            return False

        return answer in ("a", "accept", "y", "yes", "")

    # ── Sync API (for non-async callers) ────────────────────────────

    def get_input(self) -> str:
        """Block until the user submits input.

        Returns
        -------
        str
            The (possibly multi-line) text the user typed, stripped of
            leading/trailing whitespace.  Returns an empty string if the
            user presses Ctrl-C or Ctrl-D.

        .. warning::
            Do **not** call this from within an ``asyncio.run()`` context --
            use :meth:`get_input_async` instead to avoid a
            ``RuntimeError: asyncio.run() cannot be called from a running
            event loop``.
        """
        try:
            if self._session is not None:
                text: str = self._session.prompt()
            else:
                text = input("you > ")
            return text.strip()
        except KeyboardInterrupt:
            return ""
        except EOFError:
            return "/exit"

    def get_approval(
        self,
        tool_name: str,
        args: dict[str, Any],
    ) -> bool:
        """Ask the user to approve a tool call (sync version).

        Displays the tool name and arguments, then prompts for one of:

        * **[A]ccept** -- approve and run the tool.
        * **[D]eny**   -- skip the tool call.
        * **[E]dit**   -- (placeholder) reject for now.

        Returns ``True`` if the user accepted, ``False`` otherwise.

        .. warning::
            Use :meth:`get_approval_async` when inside an event loop.
        """
        args_summary = _compact_args(args)

        approval_prompt: list[tuple[str, str]] = [
            ("class:prompt", "  Allow "),
            ("bold", tool_name),
        ]
        if args_summary:
            approval_prompt.append(("", f" ({args_summary})"))
        approval_prompt.append(("class:prompt", "? [A]ccept / [D]eny / [E]dit: "))

        try:
            if self._session is not None:
                answer: str = self._session.prompt(
                    approval_prompt,
                    style=_PROMPT_STYLE,
                ).strip().lower()
            else:
                prompt_str = f"  Allow {tool_name}"
                if args_summary:
                    prompt_str += f" ({args_summary})"
                prompt_str += "? [A]ccept / [D]eny / [E]dit: "
                answer = input(prompt_str).strip().lower()
        except (KeyboardInterrupt, EOFError):
            return False

        return answer in ("a", "accept", "y", "yes", "")

    # ── Utilities ─────────────────────────────────────────────────────────

    @staticmethod
    def is_command(text: str) -> bool:
        """Return ``True`` if *text* looks like a slash command."""
        return text.startswith("/")


# ── Private Helpers ───────────────────────────────────────────────────────────


def _compact_args(args: dict[str, Any], max_len: int = 80) -> str:
    """Return a compact one-line summary of *args* for the approval prompt."""
    if not args:
        return ""

    parts: list[str] = []
    for key, value in args.items():
        val_str = str(value)
        if len(val_str) > 40:
            val_str = val_str[:37] + "..."
        parts.append(f"{key}={val_str}")

    joined = ", ".join(parts)
    if len(joined) > max_len:
        joined = joined[: max_len - 3] + "..."
    return joined
