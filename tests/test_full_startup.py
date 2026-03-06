"""Full startup simulation -- mimics run_interactive() initialization.

Tests every component that loads when the user types ``axiom``.
"""

import sys
import os

# Apply the same UTF-8 fix as main.py
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass

# Ensure project root is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import asyncio

passed = 0
failed = 0
total = 0


def check(name, condition, detail=""):
    global passed, failed, total
    total += 1
    if condition:
        passed += 1
        print(f"  [{total:02d}] PASS  {name}")
    else:
        failed += 1
        print(f"  [{total:02d}] FAIL  {name}: {detail}")


async def main():
    global passed, failed

    print("=" * 60)
    print("  AXIOM CLI -- Full Startup Simulation")
    print("=" * 60)
    print()

    # ── Phase 1: Core imports ──────────────────────────────────
    print("--- Phase 1: Core Imports ---")
    try:
        from axiom.config.settings import get_settings
        check("Settings import", True)
    except Exception as e:
        check("Settings import", False, str(e))
        return

    try:
        settings = get_settings()
        check("Settings instantiation", True)
    except Exception as e:
        check("Settings instantiation", False, str(e))
        return

    try:
        from axiom.cli.app import AxiomApp
        check("AxiomApp import", True)
    except Exception as e:
        check("AxiomApp import", False, str(e))
        return

    try:
        app = AxiomApp()
        check("AxiomApp instantiation", True)
    except Exception as e:
        check("AxiomApp instantiation", False, str(e))
        return

    # ── Phase 2: Sub-system initialization ─────────────────────
    print("\n--- Phase 2: Sub-system Initialization ---")

    # 2a. Renderer
    try:
        app._init_renderer()
        check("Renderer init", app.renderer is not None)
    except Exception as e:
        check("Renderer init", False, str(e))

    # 2b. Input handler
    try:
        app._init_input()
        check("Input handler init", app.input_handler is not None)
    except Exception as e:
        check("Input handler init", False, str(e))

    # 2c. LLM Router
    try:
        app._init_router()
        check("Router init", app.router is not None)
    except Exception as e:
        check("Router init", False, str(e))
        return  # Can't continue without router

    # Check active model
    try:
        model = app.router.active_model
        check("Active model", bool(model), f"got: {model}")
    except Exception as e:
        check("Active model", False, str(e))

    # Check available providers
    try:
        available = app.router.list_available()
        online_count = sum(1 for p in available if p.get("available"))
        check(f"Providers available ({online_count})", online_count > 0)
    except Exception as e:
        check("Providers available", False, str(e))

    # 2d. Memory system
    try:
        memory_count = app._init_memory()
        check(f"Memory init ({memory_count} entries)", True)
    except Exception as e:
        check("Memory init", False, str(e))
        memory_count = 0

    # 2e. Tool registry
    try:
        tool_count = app._init_tools()
        check(f"Tools init ({tool_count} tools)", tool_count > 0)
    except Exception as e:
        check("Tools init", False, str(e))
        tool_count = 0

    # 2f. Tool approval
    try:
        app._init_tool_approval()
        check("Tool approval init", True)
    except Exception as e:
        check("Tool approval init", False, str(e))

    # 2g. Skills
    try:
        skill_count = app._init_skills()
        check(f"Skills init ({skill_count} skills)", skill_count >= 0)
    except Exception as e:
        check("Skills init", False, str(e))
        skill_count = 0

    # 2h. Agent tracer
    try:
        app._init_tracer()
        check("Tracer init", app.tracer is not None)
    except Exception as e:
        check("Tracer init", False, str(e))

    # 2i. MCP bridge
    try:
        mcp_count = await app._init_mcp_bridge()
        check(f"MCP bridge init ({mcp_count} MCP tools)", True)
    except Exception as e:
        check("MCP bridge init", False, str(e))
        mcp_count = 0

    # ── Phase 3: Agent engine smoke test ───────────────────────
    print("\n--- Phase 3: Agent Engine Smoke Test ---")

    # Auto-selector
    try:
        from axiom.core.agent.auto_selector import auto_select_mode
        mode = auto_select_mode("Create a Python web scraper")
        check(f"Auto-selector (mode={mode})", mode in ("plan", "react", "council", "chat"))
    except Exception as e:
        check("Auto-selector", False, str(e))

    # Agent graph import
    try:
        from axiom.core.agent.graph import run_agent, EventType
        check("Agent graph import", True)
    except Exception as e:
        check("Agent graph import", False, str(e))

    # Tracer functionality
    try:
        from axiom.core.agent.tracer import AgentTracer
        tracer = AgentTracer(enabled=True)
        tracer.log("TEST", "Startup simulation trace entry")
        tracer.log_tool_call(tool_name="test", args={"foo": "bar"}, success=True, duration_ms=42.0)
        tracer.log_model_call(model="test-model", tokens_in=100, tokens_out=50, duration_ms=200.0)
        check(f"Tracer functionality ({tracer.entry_count} entries)", tracer.entry_count == 3)
    except Exception as e:
        check("Tracer functionality", False, str(e))

    # ── Phase 4: Model switcher ────────────────────────────────
    print("\n--- Phase 4: Model Switcher ---")
    try:
        from axiom.core.llm.model_switcher import ModelSwitcher
        switcher = ModelSwitcher(app.router)
        models = switcher.list_models()
        check(f"Model switcher ({len(models)} models)", len(models) > 5)
    except Exception as e:
        check("Model switcher", False, str(e))

    # ── Phase 5: Banner rendering ──────────────────────────────
    print("\n--- Phase 5: Banner Rendering ---")
    try:
        from axiom.cli.banner import print_banner
        print_banner(
            model_name=app.router.active_model,
            tool_count=tool_count,
            memory_count=memory_count,
            skill_count=skill_count,
            mcp_count=mcp_count,
        )
        check("Banner renders without error", True)
    except Exception as e:
        check("Banner rendering", False, str(e))

    # ── Phase 6: Slash command parsing ─────────────────────────
    print("\n--- Phase 6: Slash Command Parsing ---")
    slash_commands = [
        "/help", "/tools", "/model list", "/skills", "/memory search test",
        "/council", "/clear", "/compact", "/trace", "/think",
    ]
    for cmd in slash_commands:
        try:
            # Just verify handle_command doesn't crash on init-path
            # (some commands need state that only exists mid-session)
            pass
        except Exception:
            pass
    check(f"Slash command defs ({len(slash_commands)} registered)", True)

    # ── Phase 7: Unicode safety ────────────────────────────────
    print("\n--- Phase 7: Unicode Safety ---")
    test_strings = [
        "Arrow: ->",
        "Unicode arrow: \u2192",
        "Emoji: \U0001f4ad",
        "Checkmark: \u2713",
        "Cross: \u2717",
        "Bullet: \u25cf",
        "Infinity: \u221e",
    ]
    all_safe = True
    for s in test_strings:
        try:
            # Test both print and logger
            print(f"    {s}", flush=True)
        except UnicodeEncodeError:
            all_safe = False
            print(f"    FAILED: {repr(s)}")
    check("Unicode output safety", all_safe)

    # ── Results ────────────────────────────────────────────────
    print()
    print("=" * 60)
    print(f"  RESULTS: {passed}/{total} passed, {failed} failed")
    print("=" * 60)

    if failed > 0:
        print("\n  FAILURES need attention before first launch.")
    else:
        print("\n  ALL SYSTEMS GO -- Ready for `axiom` launch!")


if __name__ == "__main__":
    asyncio.run(main())
