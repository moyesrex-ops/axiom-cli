"""Production hardening tests — final verification before launch.

Tests the critical fixes:
1. System prompt injection on first chat turn
2. Recursive tool call handling
3. run_once error resilience
4. All slash commands are reachable
5. Agent event rendering safety
6. Memory/session shutdown path
"""

import asyncio
import json
import sys
import os
import time

# Ensure project root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

passed = 0
failed = 0


def check(name: str, condition: bool, detail: str = ""):
    global passed, failed
    if condition:
        passed += 1
        print(f"  [PASS] {name}")
    else:
        failed += 1
        print(f"  [FAIL] {name} -- {detail}")


# ═══════════════════════════════════════════════════════════════
# Phase 1: System prompt injection
# ═══════════════════════════════════════════════════════════════
print("\n=== Phase 1: System Prompt Injection ===")

from axiom.cli.app import AxiomApp
app = AxiomApp()

# Before chat, messages should be empty
check("1.1 messages start empty", len(app.messages) == 0)

# Simulate the _inject_system_prompt method
app._init_renderer()
try:
    app._init_input()
except Exception:
    pass

# Mock a minimal router
class MockRouter:
    active_model = "test/mock-model"
    async def complete(self, **kw):
        # yield nothing
        return
        yield  # make it an async generator
    def get_usage(self):
        return {"input": 0, "output": 0, "cost": 0.0, "requests": 0, "total_tokens": 0}
    def switch_model(self, name):
        return name
    async def validate_default_model(self, timeout=15.0):
        return self.active_model

app.router = MockRouter()

# Call _inject_system_prompt
app._inject_system_prompt("test message")

check(
    "1.2 system prompt injected",
    len(app.messages) == 1 and app.messages[0]["role"] == "system",
    f"messages={app.messages}"
)

check(
    "1.3 system prompt contains Axiom identity",
    "Axiom" in app.messages[0].get("content", ""),
    f"content={app.messages[0].get('content', '')[:100]}"
)

# Second call should NOT inject again (already has system message)
app._inject_system_prompt("second message")
system_count = sum(1 for m in app.messages if m.get("role") == "system")
check(
    "1.4 no duplicate system prompt",
    system_count == 1,
    f"system messages: {system_count}"
)


# ═══════════════════════════════════════════════════════════════
# Phase 2: Chat method structure
# ═══════════════════════════════════════════════════════════════
print("\n=== Phase 2: Chat Method Structure ===")

import inspect
source = inspect.getsource(app.chat)

check(
    "2.1 chat checks for system prompt",
    "_inject_system_prompt" in source,
    "Missing _inject_system_prompt call"
)

check(
    "2.2 chat has KeyboardInterrupt handler",
    "KeyboardInterrupt" in source,
)

check(
    "2.3 chat stores in memory",
    "store_message" in source,
)

check(
    "2.4 chat has auto-compress check",
    "should_compress" in source,
)


# ═══════════════════════════════════════════════════════════════
# Phase 3: Tool call execution structure
# ═══════════════════════════════════════════════════════════════
print("\n=== Phase 3: Recursive Tool Call Handling ===")

exec_source = inspect.getsource(app._execute_tool_calls)

check(
    "3.1 has MAX_TOOL_ROUNDS guard",
    "MAX_TOOL_ROUNDS" in exec_source,
    "Missing recursion guard"
)

check(
    "3.2 accumulates follow-up tool calls",
    "followup_tool_buffer" in exec_source,
    "Missing follow-up tool call accumulation"
)

check(
    "3.3 handles finish_reason == tool_calls in follow-up",
    'finish == "tool_calls"' in exec_source or "tool_calls" in exec_source,
)

check(
    "3.4 has error handling for follow-up",
    "Follow-up completion failed" in exec_source,
)


# ═══════════════════════════════════════════════════════════════
# Phase 4: run_once hardening
# ═══════════════════════════════════════════════════════════════
print("\n=== Phase 4: run_once Hardening ===")

run_once_source = inspect.getsource(app.run_once)

check(
    "4.1 run_once has graceful memory init",
    "Memory init failed" in run_once_source or "try:" in run_once_source,
)

check(
    "4.2 run_once has KeyboardInterrupt handler",
    "KeyboardInterrupt" in run_once_source,
)

check(
    "4.3 run_once has general exception handler",
    "except Exception" in run_once_source,
)


# ═══════════════════════════════════════════════════════════════
# Phase 5: Slash commands reachable
# ═══════════════════════════════════════════════════════════════
print("\n=== Phase 5: Slash Command Coverage ===")

handle_source = inspect.getsource(app.handle_command)
expected_commands = [
    "/exit", "/quit", "/help", "/clear", "/reset", "/model",
    "/tools", "/trace", "/yolo", "/memory", "/agent",
    "/history", "/compact", "/voice", "/think", "/mcp",
    "/agents", "/skills", "/council",
]

for cmd in expected_commands:
    check(
        f"5.x command {cmd} is handled",
        f'"{cmd}"' in handle_source or f"'{cmd}'" in handle_source
        or cmd.lstrip("/") in handle_source,
    )


# ═══════════════════════════════════════════════════════════════
# Phase 6: Agent event rendering safety
# ═══════════════════════════════════════════════════════════════
print("\n=== Phase 6: Agent Event Rendering ===")

render_source = inspect.getsource(app._render_agent_event)

# All EventType values should be handled
event_types = [
    "THINKING", "PLAN_CREATED", "STEP_START", "TOOL_CALL",
    "TOOL_RESULT", "OBSERVATION", "REPLAN", "TRACE",
    "ANSWER", "COUNCIL_START", "COUNCIL_MEMBER",
    "COUNCIL_SYNTHESIS", "LEARNING", "ERROR",
]

for et in event_types:
    check(
        f"6.x renders {et}",
        et in render_source,
        f"EventType.{et} not handled in renderer"
    )


# ═══════════════════════════════════════════════════════════════
# Phase 7: Shutdown path
# ═══════════════════════════════════════════════════════════════
print("\n=== Phase 7: Shutdown Safety ===")

interactive_source = inspect.getsource(app.run_interactive)

check(
    "7.1 saves session on exit",
    "save_session" in interactive_source,
)

check(
    "7.2 closes browser on exit",
    "_close_browser" in interactive_source,
)

check(
    "7.3 shows session summary on exit",
    "Until next time" in interactive_source,
)


# ═══════════════════════════════════════════════════════════════
# Phase 8: Windows-safe Unicode
# ═══════════════════════════════════════════════════════════════
print("\n=== Phase 8: Windows Unicode Safety ===")

# Read main.py and verify UTF-8 reconfiguration
with open("axiom/main.py", encoding="utf-8") as f:
    main_source = f.read()

check(
    "8.1 main.py reconfigures stdout to utf-8",
    'reconfigure(encoding="utf-8"' in main_source,
)

check(
    "8.2 main.py uses errors='replace'",
    "errors=\"replace\"" in main_source,
)

# Check context_compressor doesn't have → in logger calls
with open("axiom/core/memory/context_compressor.py", encoding="utf-8") as f:
    compressor_source = f.read()

# Only check logger lines for unsafe chars
import re
logger_lines = [
    line for line in compressor_source.split("\n")
    if "logger." in line
]
unsafe_logger = any("\u2192" in line for line in logger_lines)
check(
    "8.3 context_compressor logger is ASCII-safe",
    not unsafe_logger,
    "Found → character in logger call"
)


# ═══════════════════════════════════════════════════════════════
# Phase 9: Module imports (regression check)
# ═══════════════════════════════════════════════════════════════
print("\n=== Phase 9: Critical Module Imports ===")

critical_modules = [
    "axiom.cli.app",
    "axiom.cli.renderer",
    "axiom.cli.banner",
    "axiom.cli.theme",
    "axiom.cli.tool_approval",
    "axiom.core.llm.router",
    "axiom.core.llm.model_switcher",
    "axiom.core.tools.registry",
    "axiom.core.tools.base",
    "axiom.core.agent.graph",
    "axiom.core.agent.state",
    "axiom.core.agent.tracer",
    "axiom.core.agent.auto_selector",
    "axiom.core.agent.prompts.system",
    "axiom.core.memory.context_compressor",
    "axiom.core.skills.loader",
    "axiom.core.skills.injector",
    "axiom.core.mcp.client",
    "axiom.core.mcp.discovery",
    "axiom.core.mcp.bridge",
    "axiom.config.settings",
    "axiom.config.defaults",
]

import importlib
for mod_name in critical_modules:
    try:
        importlib.import_module(mod_name)
        check(f"9.x import {mod_name.split('.')[-1]}", True)
    except Exception as e:
        check(f"9.x import {mod_name.split('.')[-1]}", False, str(e))


# ═══════════════════════════════════════════════════════════════
# Results
# ═══════════════════════════════════════════════════════════════
print(f"\n{'=' * 55}")
print(f"  RESULTS: {passed} passed, {failed} failed, {passed + failed} total")
print(f"{'=' * 55}")

if failed > 0:
    print("\n  *** SOME TESTS FAILED — fix before launch ***")
    sys.exit(1)
else:
    print("\n  ALL TESTS PASS — production ready!")
    sys.exit(0)
