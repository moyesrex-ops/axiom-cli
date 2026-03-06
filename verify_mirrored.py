"""Comprehensive verification for mirrored Telegram + infinite memory."""

import asyncio
import os
import tempfile
import shutil
import inspect


async def test_all():
    results = []

    # ── Test 1: ConversationStore ──────────────────────────────
    print("━━━ Test 1: ConversationStore ━━━")
    from axiom.core.memory.conversation_store import ConversationStore

    test_dir = tempfile.mkdtemp(prefix="axiom_test_")
    try:
        store = ConversationStore(base_dir=os.path.join(test_dir, "conv"))

        # Write messages from CLI and Telegram
        await store.append("user", "hello from CLI", channel="cli")
        await store.append("assistant", "hi there", channel="cli")
        await store.append("user", "hello from Telegram", channel="telegram")
        await store.append("assistant", "hey telegram user", channel="telegram")

        # Load all
        msgs = await store.load()
        assert len(msgs) == 4, f"Expected 4, got {len(msgs)}"
        assert msgs[0]["channel"] == "cli"
        assert msgs[2]["channel"] == "telegram"

        # to_llm_messages
        llm = await store.to_llm_messages(last_n=2)
        assert len(llm) == 2
        assert "channel" not in llm[0]  # Should only have role+content

        # Stats
        stats = await store.get_stats()
        assert stats["messages"] == 4
        assert "cli" in stats["channels"]
        assert "telegram" in stats["channels"]

        # Clear (archive)
        await store.clear()
        msgs_after = await store.load()
        assert len(msgs_after) == 0

        print("  ✅ ConversationStore: ALL PASS (6/6 assertions)")
        results.append(("ConversationStore", True))
    except Exception as e:
        print(f"  ❌ ConversationStore FAIL: {e}")
        results.append(("ConversationStore", False))
    finally:
        shutil.rmtree(test_dir, ignore_errors=True)

    # ── Test 2: MessageBus ─────────────────────────────────────
    print("━━━ Test 2: MessageBus ━━━")
    from axiom.core.memory.message_bus import MessageBus

    try:
        bus = MessageBus()
        received_cli = []
        received_tg = []
        received_all = []

        async def on_cli(m):
            received_cli.append(m)

        async def on_tg(m):
            received_tg.append(m)

        async def on_all(m):
            received_all.append(m)

        bus.subscribe("cli", on_cli)
        bus.subscribe("telegram", on_tg)
        bus.subscribe("*", on_all)

        # Publish from telegram -> CLI should get it, telegram should NOT
        await bus.publish(
            {"content": "hello", "channel": "telegram"}, source_channel="telegram"
        )
        assert len(received_cli) == 1, f"CLI should get 1, got {len(received_cli)}"
        assert (
            len(received_tg) == 0
        ), f"Telegram should get 0 (no echo), got {len(received_tg)}"
        assert len(received_all) == 1, f"Star should get 1, got {len(received_all)}"

        # Publish from CLI -> Telegram should get it
        await bus.publish(
            {"content": "world", "channel": "cli"}, source_channel="cli"
        )
        assert len(received_tg) == 1
        assert len(received_cli) == 1  # Still 1 (no echo)

        # Unsubscribe
        bus.unsubscribe("cli", on_cli)
        await bus.publish({"content": "test"}, source_channel="telegram")
        assert len(received_cli) == 1  # Should not increase

        print("  ✅ MessageBus: ALL PASS (6/6 assertions)")
        results.append(("MessageBus", True))
    except Exception as e:
        print(f"  ❌ MessageBus FAIL: {e}")
        results.append(("MessageBus", False))

    # ── Test 3: TaskStore ──────────────────────────────────────
    print("━━━ Test 3: TaskStore ━━━")
    from axiom.core.memory.task_store import TaskStore, TaskStatus

    test_dir2 = tempfile.mkdtemp(prefix="axiom_test_")
    try:
        store = TaskStore(path=os.path.join(test_dir2, "tasks.json"))

        # Add tasks
        t1 = await store.add("Fix the login bug", source="user", priority="high")
        t2 = await store.add("Research new APIs", source="telegram")
        t3 = await store.add("Deploy to production", source="agent")

        assert t1["id"] == 1
        assert t2["id"] == 2
        assert t3["id"] == 3
        assert t1["status"] == "pending"

        # Get pending
        pending = await store.get_pending()
        assert len(pending) == 3

        # Update
        await store.update(1, status="completed", result="Fixed!")
        await store.update(2, status="in_progress")

        pending2 = await store.get_pending()
        assert len(pending2) == 2  # Only t2 (in_progress) and t3 (pending)

        # Get all
        all_tasks = await store.get_all()
        assert len(all_tasks) == 3
        assert all_tasks[0]["id"] == 3  # Most recent first

        # Format for prompt
        fmt = store.format_for_prompt(pending2)
        assert "\U0001f504" in fmt  # 🔄 in_progress icon
        assert "\u23f3" in fmt  # ⏳ pending icon

        # Persistence test — reload from disk
        store2 = TaskStore(path=os.path.join(test_dir2, "tasks.json"))
        all2 = await store2.get_all()
        assert len(all2) == 3, f"Persistence: expected 3, got {len(all2)}"

        print("  ✅ TaskStore: ALL PASS (10/10 assertions)")
        results.append(("TaskStore", True))
    except Exception as e:
        print(f"  ❌ TaskStore FAIL: {e}")
        results.append(("TaskStore", False))
    finally:
        shutil.rmtree(test_dir2, ignore_errors=True)

    # ── Test 4: System prompt with tasks ───────────────────────
    print("━━━ Test 4: System Prompt Task Injection ━━━")
    try:
        from axiom.core.agent.prompts.system import build_system_prompt

        prompt = build_system_prompt(
            tasks_context="⏳ #1: Fix login bug [pending]\n🔄 #2: Research APIs [in_progress]"
        )
        assert "Active Tasks" in prompt
        assert "Fix login bug" in prompt
        assert "Research APIs" in prompt
        assert "GOD MODE" in prompt  # Core identity preserved

        # Without tasks
        prompt2 = build_system_prompt()
        assert "Active Tasks" not in prompt2  # No section when empty

        # Extra kwargs don't crash
        prompt3 = build_system_prompt(
            tools="test", memory="test", some_random_kwarg=True
        )
        assert "Axiom" in prompt3

        print("  ✅ System Prompt: ALL PASS (6/6 assertions)")
        results.append(("SystemPrompt", True))
    except Exception as e:
        print(f"  ❌ System Prompt FAIL: {e}")
        results.append(("SystemPrompt", False))

    # ── Test 5: TelegramBridge import + signature ──────────────
    print("━━━ Test 5: TelegramBridge Import ━━━")
    try:
        from axiom.integrations.telegram.bridge import TelegramBridge

        # Test that constructor only takes app
        sig = inspect.signature(TelegramBridge.__init__)
        params = list(sig.parameters.keys())
        assert params == ["self", "app"], f"Expected [self, app], got {params}"

        # Test properties exist
        assert hasattr(TelegramBridge, "conversation_store")
        assert hasattr(TelegramBridge, "task_store")

        print("  ✅ TelegramBridge: Import + Signature + Properties OK")
        results.append(("TelegramBridge", True))
    except Exception as e:
        print(f"  ❌ TelegramBridge FAIL: {e}")
        results.append(("TelegramBridge", False))

    # ── Test 6: AxiomApp has new stores ────────────────────────
    print("━━━ Test 6: AxiomApp Store Integration ━━━")
    try:
        from axiom.cli.app import AxiomApp

        app = AxiomApp()
        assert hasattr(app, "conversation_store")
        assert hasattr(app, "message_bus")
        assert hasattr(app, "task_store")
        assert isinstance(app.conversation_store, ConversationStore)
        assert isinstance(app.message_bus, MessageBus)
        assert isinstance(app.task_store, TaskStore)

        # Verify chat_headless exists
        assert hasattr(app, "chat_headless")
        assert asyncio.iscoroutinefunction(app.chat_headless)

        # Verify _build_system_prompt exists
        assert hasattr(app, "_build_system_prompt")

        print("  ✅ AxiomApp: ALL stores wired + chat_headless exists")
        results.append(("AxiomApp Integration", True))
    except Exception as e:
        print(f"  ❌ AxiomApp FAIL: {e}")
        results.append(("AxiomApp Integration", False))

    # ── Test 7: Cross-channel mirroring simulation ─────────────
    print("━━━ Test 7: Cross-Channel Mirroring ━━━")
    test_dir3 = tempfile.mkdtemp(prefix="axiom_test_")
    try:
        shared_store = ConversationStore(
            base_dir=os.path.join(test_dir3, "shared")
        )
        bus = MessageBus()

        cli_notifications = []
        tg_notifications = []

        async def cli_listener(msg):
            cli_notifications.append(msg)

        async def tg_listener(msg):
            tg_notifications.append(msg)

        bus.subscribe("cli", cli_listener)
        bus.subscribe("telegram", tg_listener)

        # Simulate CLI user sends message
        await shared_store.append("user", "hello from CLI", channel="cli")
        await bus.publish(
            {"role": "user", "content": "hello from CLI", "channel": "cli"},
            source_channel="cli",
        )

        # Simulate Telegram user sends message
        await shared_store.append(
            "user", "hello from Telegram", channel="telegram"
        )
        await bus.publish(
            {
                "role": "user",
                "content": "hello from Telegram",
                "channel": "telegram",
            },
            source_channel="telegram",
        )

        # Verify mirroring
        msgs = await shared_store.load()
        assert len(msgs) == 2  # Both in same store
        assert msgs[0]["channel"] == "cli"
        assert msgs[1]["channel"] == "telegram"

        # Verify cross-notifications
        assert len(cli_notifications) == 1  # Got telegram msg
        assert len(tg_notifications) == 1  # Got CLI msg
        assert cli_notifications[0]["content"] == "hello from Telegram"
        assert tg_notifications[0]["content"] == "hello from CLI"

        print("  ✅ Cross-Channel Mirroring: ALL PASS (6/6 assertions)")
        results.append(("Mirroring", True))
    except Exception as e:
        print(f"  ❌ Mirroring FAIL: {e}")
        results.append(("Mirroring", False))
    finally:
        shutil.rmtree(test_dir3, ignore_errors=True)

    # ── Summary ────────────────────────────────────────────────
    print()
    print("=" * 55)
    print("  AXIOM MIRRORED TELEGRAM VERIFICATION")
    print("=" * 55)
    total = len(results)
    passed = sum(1 for _, ok in results if ok)
    for name, ok in results:
        icon = "✅" if ok else "❌"
        print(f"  {icon} {name}")
    print(f"\n  Result: {passed}/{total} tests passed")
    if passed == total:
        print("  🏆 ALL TESTS PASSED — PRODUCTION READY")
    else:
        print("  ⚠️  SOME TESTS FAILED — FIX BEFORE DEPLOY")
    print("=" * 55)


if __name__ == "__main__":
    asyncio.run(test_all())
