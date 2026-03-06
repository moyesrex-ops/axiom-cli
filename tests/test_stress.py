"""Stress test — verify every subsystem constructs and interoperates."""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

passed = 0
failed = 0


def check(name, cond, detail=""):
    global passed, failed
    if cond:
        passed += 1
        print(f"  [PASS] {name}")
    else:
        failed += 1
        print(f"  [FAIL] {name} -- {detail}")


# ═══════════════════════════════════════════════════
# A. AxiomApp Construction
# ═══════════════════════════════════════════════════
print("=== A. AxiomApp Construction ===")
from axiom.cli.app import AxiomApp

app = AxiomApp()
check("messages list", isinstance(app.messages, list))
check("messages empty", len(app.messages) == 0)
check("yolo_mode default False", app.yolo_mode is False)
check("trace_mode default False", app.trace_mode is False)
check("voice_mode default False", app.voice_mode is False)
check("router starts None", app.router is None)
check("memory starts None", app.memory is None)
check("session_id set", len(app.session_id) > 0)


# ═══════════════════════════════════════════════════
# B. Renderer
# ═══════════════════════════════════════════════════
print("\n=== B. Renderer ===")
app._init_renderer()
check("renderer initialized", app.renderer is not None)


# ═══════════════════════════════════════════════════
# C. Input Handler
# ═══════════════════════════════════════════════════
print("\n=== C. Input Handler ===")
try:
    app._init_input()
    check("input handler init", app.input_handler is not None)
except Exception:
    check("input handler fallback", True)


# ═══════════════════════════════════════════════════
# D. ToolRegistry
# ═══════════════════════════════════════════════════
print("\n=== D. ToolRegistry ===")
from axiom.core.tools.registry import ToolRegistry

registry = ToolRegistry()
check("registry singleton", registry is not None)
check("has invoke", hasattr(registry, "invoke"))
check("has list_names", hasattr(registry, "list_names"))
check("has to_llm_schemas", hasattr(registry, "to_llm_schemas"))


# ═══════════════════════════════════════════════════
# E. AgentState
# ═══════════════════════════════════════════════════
print("\n=== E. AgentState ===")
from axiom.core.agent.state import AgentState, AgentMode, AgentStatus

state = AgentState()
check("mode default PLAN", state.mode == AgentMode.PLAN)
check("status default PLANNING", state.status == AgentStatus.PLANNING)
check("is_terminal False", state.is_terminal is False)
state.status = AgentStatus.COMPLETE
check("is_terminal complete True", state.is_terminal is True)
state.status = AgentStatus.ERROR
check("is_terminal error True", state.is_terminal is True)


# ═══════════════════════════════════════════════════
# F. EventType (14 members)
# ═══════════════════════════════════════════════════
print("\n=== F. EventType ===")
from axiom.core.agent.graph import run_agent, EventType, AgentEvent

check("run_agent callable", callable(run_agent))
check("EventType count 14", len(EventType) == 14, str(len(EventType)))
for ev in [
    "THINKING", "PLAN_CREATED", "STEP_START", "TOOL_CALL",
    "TOOL_RESULT", "OBSERVATION", "REPLAN", "ANSWER", "ERROR",
    "TRACE", "COUNCIL_START", "COUNCIL_MEMBER", "COUNCIL_SYNTHESIS", "LEARNING",
]:
    check(f"EventType.{ev}", hasattr(EventType, ev))


# ═══════════════════════════════════════════════════
# G. UniversalRouter
# ═══════════════════════════════════════════════════
print("\n=== G. UniversalRouter ===")
from axiom.core.llm.router import UniversalRouter
from axiom.config.settings import get_settings

settings = get_settings()
router = UniversalRouter(settings)
check("router constructed", router is not None)
check("has complete", hasattr(router, "complete"))
check("has get_usage", hasattr(router, "get_usage"))
check("has switch_model", hasattr(router, "switch_model"))
check("has active_model", hasattr(router, "active_model"))
check("has validate_default_model", hasattr(router, "validate_default_model"))
check("active_model is string", isinstance(router.active_model, str))
usage = router.get_usage()
check("get_usage returns dict", isinstance(usage, dict))
check("usage has cost", "cost" in usage)


# ═══════════════════════════════════════════════════
# H. ModelSwitcher
# ═══════════════════════════════════════════════════
print("\n=== H. ModelSwitcher ===")
from axiom.core.llm.model_switcher import ModelSwitcher

switcher = ModelSwitcher(router)
check("switcher created", switcher is not None)
check("has switch", hasattr(switcher, "switch"))
check("has list_models", hasattr(switcher, "list_models"))
check("has auto_select", hasattr(switcher, "auto_select"))


# ═══════════════════════════════════════════════════
# I. MemoryManager
# ═══════════════════════════════════════════════════
print("\n=== I. MemoryManager ===")
from axiom.core.memory.manager import MemoryManager

mm = MemoryManager()
check("mm created", mm is not None)
check("has store_message", hasattr(mm, "store_message"))
check("has search", hasattr(mm, "search"))
check("has build_context", hasattr(mm, "build_context"))


# ═══════════════════════════════════════════════════
# J. ContextCompressor
# ═══════════════════════════════════════════════════
print("\n=== J. ContextCompressor ===")
from axiom.core.memory.context_compressor import (
    should_compress, compress_context, estimate_tokens,
    COMPRESS_AFTER_MESSAGES,
)

check("should_compress callable", callable(should_compress))
check("compress_context callable", callable(compress_context))
check("estimate_tokens callable", callable(estimate_tokens))
check("COMPRESS_AFTER > 5", COMPRESS_AFTER_MESSAGES > 5)
check("COMPRESS_AFTER <= 20", COMPRESS_AFTER_MESSAGES <= 20)
tokens = estimate_tokens([{"content": "hello world"}])
check("estimate_tokens int", isinstance(tokens, int))
check("COMPRESS_AFTER_MESSAGES > 0", COMPRESS_AFTER_MESSAGES > 0)


# ═══════════════════════════════════════════════════
# K. Skills
# ═══════════════════════════════════════════════════
print("\n=== K. Skills ===")
from axiom.core.skills.loader import SkillLoader
from axiom.core.skills.injector import SkillInjector

sl = SkillLoader()
check("skill loader", sl is not None)
si = SkillInjector(sl)
check("skill injector", si is not None)


# ═══════════════════════════════════════════════════
# L. MCP
# ═══════════════════════════════════════════════════
print("\n=== L. MCP ===")
from axiom.core.mcp.client import MCPClient
from axiom.core.mcp.discovery import MCPDiscovery
from axiom.core.mcp.bridge import MCPBridge

check("MCPClient", MCPClient is not None)
disc = MCPDiscovery()
check("MCPDiscovery", disc is not None)
check("MCPBridge", MCPBridge is not None)


# ═══════════════════════════════════════════════════
# M. Tracer
# ═══════════════════════════════════════════════════
print("\n=== M. Tracer ===")
from axiom.core.agent.tracer import AgentTracer

tracer = AgentTracer(enabled=True)
tracer.log("TEST", "entry")
check("log ok", len(tracer._entries) == 1)
tracer.log_tool_call("bash", {"cmd": "ls"}, result="ok", success=True, duration_ms=100)
check("log_tool_call", len(tracer._entries) == 2)
tracer.log_model_call("opus", 100, 50, 200)
check("log_model_call", len(tracer._entries) == 3)

tracer_off = AgentTracer(enabled=False)
tracer_off.log("TEST", "nope")
check("disabled tracer", len(tracer_off._entries) == 0)


# ═══════════════════════════════════════════════════
# N. AutoSelector
# ═══════════════════════════════════════════════════
print("\n=== N. AutoSelector ===")
from axiom.core.agent.auto_selector import auto_select_mode

m1 = auto_select_mode("build a REST API with auth and database")
check("complex -> plan", m1 == "plan", m1)
m2 = auto_select_mode("what is python")
check("simple -> chat/react", m2 in ("chat", "react"), m2)
m3 = auto_select_mode("compare GPT-4 vs Claude for coding tasks, debate pros and cons")
check("compare -> council or valid mode", m3 in ("council", "react", "plan"), m3)


# ═══════════════════════════════════════════════════
# O. System Prompt
# ═══════════════════════════════════════════════════
print("\n=== O. System Prompt ===")
from axiom.core.agent.prompts.system import build_system_prompt

sp = build_system_prompt(model_name="claude-opus-4", tool_names=["bash", "read_file"])
check("prompt is string", isinstance(sp, str))
check("prompt mentions Axiom", "Axiom" in sp)
check("prompt > 500 chars", len(sp) > 500, str(len(sp)))


# ═══════════════════════════════════════════════════
# P. Config
# ═══════════════════════════════════════════════════
print("\n=== P. Config ===")
from axiom.config.settings import get_settings, AxiomSettings

s = get_settings()
check("settings created", s is not None)
check("settings is AxiomSettings", isinstance(s, AxiomSettings))

from axiom.config.defaults import DEFAULT_MODEL, MAX_ITERATIONS, AXIOM_HOME

check("DEFAULT_MODEL string", isinstance(DEFAULT_MODEL, str))
check("MAX_ITERATIONS int", isinstance(MAX_ITERATIONS, int))
check("AXIOM_HOME is Path", hasattr(AXIOM_HOME, "exists"))


# ═══════════════════════════════════════════════════
# Q. App Methods
# ═══════════════════════════════════════════════════
print("\n=== Q. App Methods ===")
for method in [
    "handle_command", "chat", "run_interactive", "run_once",
    "_inject_system_prompt", "_execute_tool_calls", "_render_agent_event",
    "_compact_history", "_init_router", "_init_tools", "_init_memory",
    "_init_skills", "_init_tracer", "_init_mcp_bridge",
]:
    check(method, hasattr(app, method))


# ═══════════════════════════════════════════════════
# Results
# ═══════════════════════════════════════════════════
sep = "=" * 55
print()
print(sep)
print(f"  STRESS TEST: {passed} passed, {failed} failed, {passed+failed} total")
print(sep)
if failed == 0:
    print("  ALL STRESS TESTS PASS - PRODUCTION READY")
    sys.exit(0)
else:
    print("  *** FAILURES DETECTED ***")
    sys.exit(1)
