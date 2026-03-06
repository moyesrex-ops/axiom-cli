"""Tool system -- base class, error type, singleton registry, and all tools.

Built-in tool implementations live in sibling modules and are registered
into the singleton ``ToolRegistry`` via ``get_registry()``.
"""

from axiom.core.tools.base import AxiomTool, ToolError  # noqa: F401
from axiom.core.tools.registry import (  # noqa: F401
    ToolRegistry,
    get_registry,
    register_default_tools,
)

# ── P0 tools (stateless) ─────────────────────────────────────────
from axiom.core.tools.bash import BashTool  # noqa: F401
from axiom.core.tools.code_exec import CodeExecTool  # noqa: F401
from axiom.core.tools.files import (  # noqa: F401
    EditFileTool,
    GlobTool,
    GrepTool,
    ReadFileTool,
    WriteFileTool,
)
from axiom.core.tools.git import GitTool  # noqa: F401
from axiom.core.tools.web_fetch import WebFetchTool  # noqa: F401
from axiom.core.tools.http import HTTPTool  # noqa: F401

# ── P1 tools (some need constructor args) ────────────────────────
from axiom.core.tools.browser import BrowserTool  # noqa: F401
from axiom.core.tools.research import ResearchTool  # noqa: F401
from axiom.core.tools.desktop import DesktopTool  # noqa: F401
from axiom.core.tools.vision import VisionTool  # noqa: F401
from axiom.core.tools.mcp_client import MCPClientTool  # noqa: F401
from axiom.core.tools.agent_spawn import SpawnAgentTool  # noqa: F401
from axiom.core.tools.tool_create import ToolCreateTool  # noqa: F401

__all__ = [
    # Core
    "AxiomTool",
    "ToolError",
    "ToolRegistry",
    "get_registry",
    "register_default_tools",
    # P0 tools
    "BashTool",
    "CodeExecTool",
    "EditFileTool",
    "GlobTool",
    "GrepTool",
    "GitTool",
    "ReadFileTool",
    "WebFetchTool",
    "HTTPTool",
    "WriteFileTool",
    # P1 tools
    "BrowserTool",
    "ResearchTool",
    "DesktopTool",
    "VisionTool",
    "MCPClientTool",
    "SpawnAgentTool",
    "ToolCreateTool",
]
