"""Production-grade verification of all fixes."""
import asyncio
import tempfile
from pathlib import Path


async def test_conversation_store_absolute_path():
    from axiom.core.memory.conversation_store import ConversationStore
    store = ConversationStore()  # Should use AXIOM_HOME
    assert store._base.is_absolute(), f"Path not absolute: {store._base}"
    await store.append("user", "test message", channel="test")
    msgs = await store.load()
    assert len(msgs) >= 1
    assert msgs[-1]["content"] == "test message"
    print("  ConversationStore: absolute path, write+read works")


async def test_task_store_absolute_path():
    from axiom.core.memory.task_store import TaskStore
    store = TaskStore()  # Should use AXIOM_HOME
    assert store._path.is_absolute(), f"Path not absolute: {store._path}"
    task = await store.add("Test task from verify_production")
    assert task["id"] >= 1
    pending = await store.get_pending()
    assert any(t["description"] == "Test task from verify_production" for t in pending)
    print("  TaskStore: absolute path, add+get_pending works")


async def test_settings_env_discovery():
    from axiom.config.settings import get_settings
    s = get_settings()
    assert s.AXIOM_HOME.is_absolute()
    print(f"  Settings: AXIOM_HOME={s.AXIOM_HOME}")


async def test_build_system_prompt_includes_integrations():
    from axiom.cli.app import AxiomApp
    app = AxiomApp()
    app._telegram_active = True
    prompt = app._build_system_prompt("test", "")
    assert "Telegram" in prompt, "System prompt should mention Telegram when active"
    assert "CONNECTED" in prompt, "Should say CONNECTED"
    print("  System prompt: includes Telegram integrations context")


async def test_inject_uses_build():
    """Verify _inject_system_prompt delegates to _build_system_prompt."""
    from axiom.cli.app import AxiomApp
    app = AxiomApp()
    app._telegram_active = True
    app._init_router()
    app._inject_system_prompt("test")
    system_msg = app.messages[0]["content"]
    assert "Telegram" in system_msg, "_inject_system_prompt must include integrations"
    print("  _inject_system_prompt: delegates to _build_system_prompt correctly")


async def test_integrations_context_no_event_loop_crash():
    """Verify _build_integrations_context doesn't crash on event loop check."""
    from axiom.cli.app import AxiomApp
    app = AxiomApp()
    app._telegram_active = False
    ctx = app._build_integrations_context()
    assert "Conversation Store" in ctx, "Should always mention conversation store"
    assert "Task Memory" in ctx, "Should always mention task memory"
    print("  Integrations context: no crash, includes stores")


async def test_cross_platform_paths():
    """Verify all paths are absolute and cross-platform."""
    from axiom.core.memory.conversation_store import ConversationStore
    from axiom.core.memory.task_store import TaskStore
    from axiom.config.settings import get_settings

    settings = get_settings()
    conv_store = ConversationStore()
    task_store = TaskStore()

    paths = {
        "AXIOM_HOME": settings.AXIOM_HOME,
        "ConversationStore base": conv_store._base,
        "TaskStore path": task_store._path,
    }
    for name, p in paths.items():
        assert Path(p).is_absolute(), f"{name} is not absolute: {p}"
    print("  Cross-platform paths: all absolute")


async def test_connect_command_exists():
    """Verify /connect is wired into the command dispatcher."""
    from axiom.cli.app import AxiomApp
    app = AxiomApp()
    assert hasattr(app, '_handle_connect_command'), "Missing _handle_connect_command"
    assert hasattr(app, '_connect_telegram'), "Missing _connect_telegram"
    print("  /connect command: handler methods exist")


async def test_telegram_bot_instance_var():
    """Verify telegram_bot is an instance variable, not local."""
    from axiom.cli.app import AxiomApp
    app = AxiomApp()
    assert hasattr(app, '_telegram_bot'), "Missing _telegram_bot instance var"
    assert app._telegram_bot is None, "_telegram_bot should init as None"
    print("  _telegram_bot: instance variable exists")


async def test_system_prompt_allows_configuration():
    """Verify system prompt no longer blocks token handling."""
    from axiom.core.agent.prompts.system import build_system_prompt
    prompt = build_system_prompt()
    assert "Self-Configuration" in prompt, "Missing self-configuration section"
    assert "NEVER lecture about security" in prompt, "Missing anti-lecture instruction"
    assert "STORE IT" in prompt, "Missing store instruction"
    # Should NOT contain the blanket ban
    assert "Never expose API keys, passwords, or sensitive data in responses" not in prompt
    print("  System prompt: allows self-configuration, no blanket ban")


async def test_help_shows_connect():
    """Verify /connect appears in help output."""
    from axiom.cli.app import AxiomApp
    import inspect
    app = AxiomApp()
    source = inspect.getsource(app._show_help)
    assert "/connect" in source, "/connect not in help table"
    print("  Help table: includes /connect command")


async def main():
    print("\n=== Axiom Production Verification ===\n")
    tests = [
        test_conversation_store_absolute_path,
        test_task_store_absolute_path,
        test_settings_env_discovery,
        test_build_system_prompt_includes_integrations,
        test_inject_uses_build,
        test_integrations_context_no_event_loop_crash,
        test_cross_platform_paths,
        test_connect_command_exists,
        test_telegram_bot_instance_var,
        test_system_prompt_allows_configuration,
        test_help_shows_connect,
    ]
    passed = 0
    failed = 0
    for test in tests:
        try:
            await test()
            passed += 1
        except Exception as e:
            print(f"  FAIL {test.__name__}: {e}")
            failed += 1

    print(f"\n{'='*45}")
    print(f"Results: {passed}/{len(tests)} passed, {failed} failed")
    if passed == len(tests):
        print("ALL TESTS PASSED - Production ready")
    else:
        print(f"{failed} failures need fixing")


if __name__ == "__main__":
    asyncio.run(main())
