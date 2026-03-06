"""Axiom CLI -- Entry point."""

import sys
import asyncio
import logging
import os


def main():
    """Main entry point for the ``axiom`` command.

    Registered via ``pyproject.toml`` as::

        [project.scripts]
        axiom = "axiom.main:main"
    """
    # ── Logging configuration ──────────────────────────────────────
    # Suppress library noise (LiteLLM, httpx, etc.) from the console.
    # Axiom uses Rich for all user-facing output; logger messages should
    # only appear when the user opts in with --trace or AXIOM_LOG_LEVEL.
    log_level_name = os.environ.get("AXIOM_LOG_LEVEL", "ERROR").upper()
    log_level = getattr(logging, log_level_name, logging.ERROR)
    logging.basicConfig(
        level=log_level,
        format="%(levelname)s | %(name)s | %(message)s",
        stream=sys.stderr,
    )
    # Extra-noisy libraries — keep at ERROR regardless of user setting
    for noisy in ("httpx", "httpcore", "litellm", "openai", "chromadb"):
        logging.getLogger(noisy).setLevel(logging.ERROR)

    # Suppress RuntimeWarning about unawaited coroutines from LiteLLM internals
    import warnings
    warnings.filterwarnings("ignore", message="coroutine.*was never awaited")
    warnings.filterwarnings("ignore", message="Enable tracemalloc")

    # ── Fix Windows event loop ────────────────────────────────────
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    # ── Force UTF-8 on Windows ────────────────────────────────────
    # PYTHONUTF8 must be set before interpreter starts, but setting
    # it here covers child processes. For the current process, we
    # reconfigure stdout/stderr to use UTF-8 explicitly.
    os.environ.setdefault("PYTHONUTF8", "1")

    if sys.platform == "win32":
        # Reconfigure stdout/stderr to UTF-8 with error replacement
        # so Unicode characters (arrows, emojis) render correctly or
        # degrade gracefully instead of crashing with UnicodeEncodeError.
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, OSError):
            pass  # Already reconfigured or not a text stream

    from axiom.cli.app import AxiomApp

    app = AxiomApp()

    # ── One-shot mode: ``axiom "do something"`` ───────────────────
    if len(sys.argv) > 1 and not sys.argv[1].startswith("-"):
        task = " ".join(sys.argv[1:])
        asyncio.run(app.run_once(task))
    else:
        # ── Interactive mode with flag parsing ────────────────────
        import argparse

        parser = argparse.ArgumentParser(
            prog="axiom",
            description="Axiom -- The Ultimate AI Agent",
        )
        parser.add_argument(
            "--model", type=str, default=None, help="Start with a specific model"
        )
        parser.add_argument(
            "--yolo", action="store_true", help="Auto-approve all tool invocations"
        )
        parser.add_argument(
            "--visible", action="store_true", help="Show browser window (Playwright)"
        )
        parser.add_argument(
            "--voice", action="store_true", help="Enable voice input"
        )
        parser.add_argument(
            "--offline", action="store_true", help="Local LLMs only (Ollama)"
        )
        parser.add_argument(
            "--trace", action="store_true", help="Show agent trace output"
        )
        parser.add_argument(
            "--telegram", action="store_true", help="Enable Telegram bot bridge"
        )
        args = parser.parse_args()

        asyncio.run(
            app.run_interactive(
                model=args.model,
                yolo=args.yolo,
                visible=args.visible,
                voice=args.voice,
                offline=args.offline,
                trace=args.trace,
                telegram=args.telegram,
            )
        )


if __name__ == "__main__":
    main()
