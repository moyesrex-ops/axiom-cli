"""
Axiom CLI -- Shell presentation layer.

Re-exports the main components so consumers can write::

    from axiom.cli import StreamRenderer, InputHandler, print_banner
"""

from axiom.cli.banner import print_banner
from axiom.cli.input_handler import InputHandler
from axiom.cli.renderer import StreamRenderer
from axiom.cli.theme import (
    AXIOM_CYAN,
    AXIOM_DIM,
    AXIOM_GREEN,
    AXIOM_PURPLE,
    AXIOM_RED,
    AXIOM_THEME,
    AXIOM_YELLOW,
    make_console,
)
from axiom.cli.tool_approval import ToolApproval
from axiom.cli.voice_input import VoiceInput

__all__ = [
    # Classes
    "StreamRenderer",
    "InputHandler",
    "ToolApproval",
    "VoiceInput",
    # Functions
    "print_banner",
    "make_console",
    # Theme constants
    "AXIOM_CYAN",
    "AXIOM_PURPLE",
    "AXIOM_GREEN",
    "AXIOM_YELLOW",
    "AXIOM_RED",
    "AXIOM_DIM",
    "AXIOM_THEME",
]
