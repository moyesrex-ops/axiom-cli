"""Axiom CLI integration test suite.

Tests all critical import chains, module wiring, and basic functionality
to ensure the CLI is ready for installation and use.
"""

import sys
import os
import importlib
import traceback

# Force UTF-8 output on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Ensure axiom package is importable
sys.path.insert(0, os.path.dirname(__file__))

PASS = 0
FAIL = 0
ERRORS = []


def test(name: str, fn):
    global PASS, FAIL
    try:
        result = fn()
        if result is not False:
            PASS += 1
            print(f"  [PASS] {name}")
        else:
            FAIL += 1
            ERRORS.append(f"  [FAIL] {name}: returned False")
            print(f"  [FAIL] {name}: returned False")
    except Exception as exc:
        FAIL += 1
        msg = f"  [FAIL] {name}: {exc}"
        ERRORS.append(msg)
        print(msg)
        traceback.print_exc()


# ── 1. Core Package Imports ──────────────────────────────────────────────────

print("\n--- 1. Core Package Imports ---")


def test_axiom_init():
    import axiom
    return hasattr(axiom, "__version__") or True  # Just ensure importable


def test_config_settings():
    from axiom.config.settings import get_settings
    s = get_settings()
    return s is not None


def test_config_defaults():
    from axiom.config.defaults import DEFAULT_MODEL, MODEL_DEFAULTS, AXIOM_HOME
    return isinstance(MODEL_DEFAULTS, dict) and "anthropic" in MODEL_DEFAULTS


test("axiom.__init__", test_axiom_init)
test("axiom.config.settings", test_config_settings)
test("axiom.config.defaults", test_config_defaults)


# ── 2. CLI Module Imports ────────────────────────────────────────────────────

print("\n--- 2. CLI Module Imports ---")


def test_cli_init():
    from axiom.cli import (
        StreamRenderer, InputHandler, print_banner,
        make_console, ToolApproval, VoiceInput,
        AXIOM_CYAN, AXIOM_GREEN, AXIOM_PURPLE,
        AXIOM_YELLOW, AXIOM_RED, AXIOM_DIM, AXIOM_THEME,
    )
    return True


def test_renderer():
    from axiom.cli.renderer import StreamRenderer
    from axiom.cli.theme import make_console
    c = make_console()
    r = StreamRenderer(c)
    return r is not None


def test_input_handler():
    from axiom.cli.input_handler import InputHandler
    return True  # Just ensure importable


def test_banner():
    from axiom.cli.banner import print_banner
    return callable(print_banner)


def test_theme():
    from axiom.cli.theme import AXIOM_THEME, make_console
    c = make_console()
    return c is not None


def test_tool_approval():
    from axiom.cli.tool_approval import ToolApproval
    from axiom.cli.theme import make_console
    c = make_console()
    ta = ToolApproval(console=c, yolo_mode=True)
    # In yolo mode, should auto-approve everything
    return ta.should_auto_approve("bash", "high") is True


def test_voice_input():
    from axiom.cli.voice_input import VoiceInput
    return True  # Just ensure importable


def test_app_import():
    from axiom.cli.app import AxiomApp
    return True


test("axiom.cli.__init__ (all exports)", test_cli_init)
test("StreamRenderer", test_renderer)
test("InputHandler", test_input_handler)
test("print_banner", test_banner)
test("Theme + make_console", test_theme)
test("ToolApproval (yolo mode)", test_tool_approval)
test("VoiceInput", test_voice_input)
test("AxiomApp", test_app_import)


# ── 3. Core Tool Imports ────────────────────────────────────────────────────

print("\n--- 3. Core Tool Imports ---")


def test_tool_base():
    from axiom.core.tools.base import AxiomTool, ToolError
    # AxiomTool should be abstract
    return hasattr(AxiomTool, "execute")


def test_tool_registry():
    from axiom.core.tools.registry import ToolRegistry, get_registry
    reg = ToolRegistry()
    return reg.count == 0


def test_bash_tool():
    from axiom.core.tools.bash import BashTool
    t = BashTool()
    return t.name == "bash"


def test_file_tools():
    from axiom.core.tools.files import (
        ReadFileTool, WriteFileTool, EditFileTool,
        GlobTool, GrepTool,
    )
    return ReadFileTool().name == "read_file"


def test_git_tool():
    from axiom.core.tools.git import GitTool
    return GitTool().name == "git"


def test_code_exec_tool():
    from axiom.core.tools.code_exec import CodeExecTool
    return CodeExecTool().name == "code_exec"


def test_web_fetch_tool():
    from axiom.core.tools.web_fetch import WebFetchTool
    return WebFetchTool().name == "web_fetch"


def test_research_tool():
    from axiom.core.tools.research import ResearchTool
    return ResearchTool().name == "research"


def test_browser_tool():
    from axiom.core.tools.browser import BrowserTool, _close_browser
    return callable(_close_browser)


def test_desktop_tool():
    from axiom.core.tools.desktop import DesktopTool
    return DesktopTool().name == "desktop"


def test_mcp_client_tool():
    from axiom.core.tools.mcp_client import MCPClientTool
    return MCPClientTool().name == "mcp_client"


def test_vision_tool():
    from axiom.core.tools.vision import VisionTool
    return True  # Needs router arg


def test_spawn_agent_tool():
    from axiom.core.tools.agent_spawn import SpawnAgentTool
    return True  # Needs router+registry args


def test_tool_create_tool():
    from axiom.core.tools.tool_create import ToolCreateTool, load_custom_tools
    return callable(load_custom_tools)


def test_memory_tool():
    from axiom.core.tools.memory_tool import MemorySearchTool, MemorySaveTool
    return True  # Needs memory_manager arg


def test_tools_init_all_exports():
    from axiom.core.tools import (
        AxiomTool, ToolError, ToolRegistry, get_registry,
        BashTool, CodeExecTool, EditFileTool, GlobTool,
        GrepTool, GitTool, ReadFileTool, WebFetchTool,
        WriteFileTool, BrowserTool, ResearchTool, DesktopTool,
        VisionTool, MCPClientTool, SpawnAgentTool, ToolCreateTool,
    )
    return True


test("AxiomTool + ToolError", test_tool_base)
test("ToolRegistry", test_tool_registry)
test("BashTool", test_bash_tool)
test("File tools (5)", test_file_tools)
test("GitTool", test_git_tool)
test("CodeExecTool", test_code_exec_tool)
test("WebFetchTool", test_web_fetch_tool)
test("ResearchTool", test_research_tool)
test("BrowserTool + _close_browser", test_browser_tool)
test("DesktopTool", test_desktop_tool)
test("MCPClientTool", test_mcp_client_tool)
test("VisionTool", test_vision_tool)
test("SpawnAgentTool", test_spawn_agent_tool)
test("ToolCreateTool + load_custom_tools", test_tool_create_tool)
test("MemorySearchTool + MemorySaveTool", test_memory_tool)
test("tools __init__ (all exports)", test_tools_init_all_exports)


# ── 4. LLM Router + Model Switcher ──────────────────────────────────────────

print("\n--- 4. LLM Router + Model Switcher ---")


def test_llm_router():
    from axiom.core.llm.router import UniversalRouter, CircuitBreaker
    return True


def test_model_switcher():
    from axiom.core.llm.model_switcher import ModelSwitcher
    return True


def test_llm_init():
    from axiom.core.llm import UniversalRouter, CircuitBreaker, ModelSwitcher
    return True


test("UniversalRouter + CircuitBreaker", test_llm_router)
test("ModelSwitcher", test_model_switcher)
test("llm __init__ (all exports)", test_llm_init)


# ── 5. Memory System ────────────────────────────────────────────────────────

print("\n--- 5. Memory System ---")


def test_memory_manager():
    from axiom.core.memory.manager import MemoryManager
    return True


def test_vector_store():
    from axiom.core.memory.vector_store import VectorStore
    return True


def test_file_memory():
    from axiom.core.memory.file_memory import FileMemory
    return True


def test_context_compressor():
    from axiom.core.memory.context_compressor import (
        compress_context, should_compress, estimate_tokens,
    )
    return callable(compress_context)


def test_memory_init():
    from axiom.core.memory import (
        MemoryManager, VectorStore, FileMemory,
        compress_context, should_compress, estimate_tokens,
    )
    return True


test("MemoryManager", test_memory_manager)
test("VectorStore", test_vector_store)
test("FileMemory", test_file_memory)
test("Context compressor", test_context_compressor)
test("memory __init__ (all exports)", test_memory_init)


# ── 6. Agent Core ───────────────────────────────────────────────────────────

print("\n--- 6. Agent Core ---")


def test_agent_state():
    from axiom.core.agent.state import AgentState
    return True


def test_agent_graph():
    from axiom.core.agent.graph import run_agent, EventType
    return True


def test_planner():
    from axiom.core.agent.nodes.planner import generate_plan, generate_react_thought
    return callable(generate_plan) and callable(generate_react_thought)


def test_executor():
    from axiom.core.agent.nodes.executor import execute_tool
    return callable(execute_tool)


def test_observer():
    from axiom.core.agent.nodes.observer import observe_progress
    return callable(observe_progress)


def test_synthesizer():
    from axiom.core.agent.nodes.synthesizer import synthesize_answer
    return callable(synthesize_answer)


def test_system_prompt():
    from axiom.core.agent.prompts.system import build_system_prompt
    result = build_system_prompt(
        tool_names=["bash", "read_file"],
        memory_context="test memory",
        model_name="test-model",
    )
    return isinstance(result, str) and "bash" in result


test("AgentState", test_agent_state)
test("AgentGraph + EventType", test_agent_graph)
test("PlannerNode", test_planner)
test("ExecutorNode", test_executor)
test("ObserverNode", test_observer)
test("SynthesizerNode", test_synthesizer)
test("build_system_prompt", test_system_prompt)


# ── 7. Entry Point ──────────────────────────────────────────────────────────

print("\n--- 7. Entry Point ---")


def test_main_module():
    from axiom.main import main
    return callable(main)


test("axiom.main:main entry point", test_main_module)


# ── 8. Tool Registration (E2E) ──────────────────────────────────────────────

print("\n--- 8. Tool Registration (E2E) ---")


def test_tool_registration_e2e():
    from axiom.core.tools.registry import ToolRegistry
    from axiom.core.tools.bash import BashTool
    from axiom.core.tools.files import ReadFileTool, WriteFileTool, GlobTool, GrepTool
    from axiom.core.tools.git import GitTool
    from axiom.core.tools.code_exec import CodeExecTool
    from axiom.core.tools.web_fetch import WebFetchTool
    from axiom.core.tools.research import ResearchTool
    from axiom.core.tools.desktop import DesktopTool
    from axiom.core.tools.mcp_client import MCPClientTool

    reg = ToolRegistry()
    tools = [
        BashTool(), ReadFileTool(), WriteFileTool(),
        GlobTool(), GrepTool(), GitTool(),
        CodeExecTool(), WebFetchTool(), ResearchTool(),
        DesktopTool(), MCPClientTool(),
    ]
    for t in tools:
        reg.register(t)

    # Verify all registered
    assert reg.count == 11, f"Expected 11 tools, got {reg.count}"

    # Verify can list tool names
    names = reg.list_names()
    assert "bash" in names
    assert "read_file" in names
    assert "mcp_client" in names

    # Verify can get schemas for LLM
    schemas = reg.to_llm_schemas()
    assert isinstance(schemas, list)
    assert len(schemas) == 11

    return True


test("Register 11 tools + list + schemas", test_tool_registration_e2e)


# ── 9. AxiomApp Construction ────────────────────────────────────────────────

print("\n--- 9. AxiomApp Construction ---")


def test_axiom_app_creation():
    from axiom.cli.app import AxiomApp
    app = AxiomApp()
    # Check that lazy attributes are None
    assert app.router is None
    assert app.renderer is None
    assert app.registry is None
    assert app.memory is None
    assert app.model_switcher is None
    assert app.tool_approval is None
    assert app.messages == []
    assert app.yolo_mode is False
    return True


def test_axiom_app_renderer():
    from axiom.cli.app import AxiomApp
    app = AxiomApp()
    app._init_renderer()
    assert app.renderer is not None
    return True


def test_axiom_app_tool_approval():
    from axiom.cli.app import AxiomApp
    app = AxiomApp()
    app._init_renderer()
    app._init_tool_approval()
    assert app.tool_approval is not None
    # In non-yolo mode, bash should not auto-approve
    assert app.tool_approval.should_auto_approve("bash", "high") is False
    # read_file should auto-approve (safe tool)
    assert app.tool_approval.should_auto_approve("read_file", "low") is True
    return True


test("AxiomApp() creation", test_axiom_app_creation)
test("AxiomApp._init_renderer()", test_axiom_app_renderer)
test("AxiomApp._init_tool_approval()", test_axiom_app_tool_approval)


# ── 10. Help System ─────────────────────────────────────────────────────────

print("\n--- 10. CLI Help Command ---")


def test_help_system():
    from axiom.cli.app import AxiomApp
    app = AxiomApp()
    app._init_renderer()
    # _show_help should not crash
    app._show_help()
    return True


test("_show_help() renders", test_help_system)


# ── Results Summary ──────────────────────────────────────────────────────────

print(f"\n{'=' * 60}")
print(f"  AXIOM CLI Integration Test Results")
print(f"{'=' * 60}")
print(f"  PASSED: {PASS}")
print(f"  FAILED: {FAIL}")
print(f"  TOTAL:  {PASS + FAIL}")
print(f"{'=' * 60}")

if ERRORS:
    print("\nFailed tests:")
    for err in ERRORS:
        print(err)

sys.exit(0 if FAIL == 0 else 1)
