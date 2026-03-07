"""Axiom CLI -- Main application loop.

This is the heart of the interactive experience.  It orchestrates the
banner, input, LLM routing, tool invocation, and streaming output.

Components used:
    - ``StreamRenderer``  (axiom.cli.renderer)  -- Rich-powered streaming display
    - ``InputHandler``    (axiom.cli.input_handler)  -- prompt_toolkit input
    - ``UniversalRouter`` (axiom.core.llm.router)  -- 15+ provider LLM router
    - ``ToolRegistry``    (axiom.core.tools.registry)  -- singleton tool registry
    - ``ToolApproval``    (axiom.cli.tool_approval)  -- Interactive approval flow
    - ``ModelSwitcher``   (axiom.core.llm.model_switcher)  -- Live model swap
    - ``VoiceInput``      (axiom.cli.voice_input)  -- Whisper STT bridge
"""

from __future__ import annotations

import asyncio
import importlib
import json
import logging
import os
import time
import traceback
from pathlib import Path
from typing import Any, Optional

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from axiom.cli.banner import print_banner
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
from axiom.config.settings import get_settings
from axiom.core.memory.conversation_store import ConversationStore
from axiom.core.memory.message_bus import MessageBus
from axiom.core.memory.task_store import TaskStore

logger = logging.getLogger(__name__)

# ── Console (theme-aware, Windows-safe) ──────────────────────────────────────

console = make_console()


# ── AxiomApp ─────────────────────────────────────────────────────────────────


class AxiomApp:
    """The main Axiom CLI application.

    Manages the interactive REPL, one-shot execution, slash commands,
    LLM routing, tool invocation, and streaming output.
    """

    def __init__(self) -> None:
        self.settings = get_settings()
        self.router: Any = None  # Lazy -- initialised in _init_router
        self.renderer: Any = None  # Lazy -- initialised in _init_renderer
        self.input_handler: Any = None  # Lazy -- initialised in _init_input
        self.registry: Any = None  # Lazy -- initialised in _init_tools
        self.memory: Any = None  # Lazy -- initialised in _init_memory
        self.model_switcher: Any = None  # Lazy -- initialised after router
        self.tool_approval: Any = None  # Lazy -- initialised after renderer
        self.voice: Any = None  # Lazy -- initialised on demand
        self.skill_loader: Any = None  # Lazy -- initialised in _init_skills
        self.skill_injector: Any = None  # Lazy -- initialised in _init_skills
        self.tracer: Any = None  # Lazy -- initialised in _init_tracer
        self.mcp_bridge: Any = None  # Lazy -- initialised in _init_mcp_bridge
        self.messages: list[dict[str, str]] = []
        self.session_start: float = time.time()
        self.session_id: str = time.strftime("%Y-%m-%d_%H-%M-%S")
        self.trace_mode: bool = False
        self.yolo_mode: bool = False
        self.voice_mode: bool = False
        self._telegram_active: bool = False
        self._telegram_bot: Any = None  # TelegramBot instance (set on connect)

        # ── Shared stores (mirrored Telegram + infinite memory) ────
        self.conversation_store = ConversationStore()
        self.message_bus = MessageBus()
        self.task_store = TaskStore()

    # ── Initialisation helpers ────────────────────────────────────

    def _init_renderer(self) -> None:
        """Create the stream renderer."""
        from axiom.cli.renderer import StreamRenderer

        self.renderer = StreamRenderer(console)

    def _init_input(self) -> None:
        """Create the input handler.

        Falls back to a bare ``input()`` wrapper if prompt_toolkit
        cannot initialise (e.g. running inside a non-TTY pipe).
        """
        try:
            from axiom.cli.input_handler import InputHandler

            self.input_handler = InputHandler()
        except Exception as exc:
            logger.debug("InputHandler init failed (%s), using fallback", exc)
            self.input_handler = _FallbackInputHandler()

    def _init_router(
        self, model: Optional[str] = None, offline: bool = False
    ) -> None:
        """Initialise the LLM router with settings."""
        if offline:
            self.settings.DEFAULT_MODEL = "ollama/qwen2.5:7b"
        if model:
            self.settings.DEFAULT_MODEL = model

        from axiom.core.llm.router import UniversalRouter

        self.router = UniversalRouter(self.settings)

        # Initialise model switcher
        from axiom.core.llm.model_switcher import ModelSwitcher

        self.model_switcher = ModelSwitcher(self.router)

    def _init_tool_approval(self) -> None:
        """Initialise the tool approval handler."""
        from axiom.cli.tool_approval import ToolApproval

        self.tool_approval = ToolApproval(
            console=console,
            yolo_mode=self.yolo_mode,
        )

    def _init_voice(self) -> None:
        """Initialise voice input (Whisper STT)."""
        try:
            from axiom.cli.voice_input import VoiceInput

            mode = "cloud" if self.settings.OPENAI_API_KEY else "local"
            self.voice = VoiceInput(
                mode=mode,
                openai_api_key=self.settings.OPENAI_API_KEY or None,
            )
            if self.voice.available:
                console.print(f"  [{AXIOM_GREEN}]Voice input ready ({mode} Whisper).[/]")
            else:
                console.print(
                    f"  [{AXIOM_YELLOW}]Voice deps missing: "
                    f"pip install sounddevice openai-whisper[/]"
                )
                self.voice = None
        except Exception as exc:
            logger.debug("Voice init failed: %s", exc)
            self.voice = None

    def _init_skills(self) -> int:
        """Initialise the skill loader and injector.

        Scans standard skill directories for SKILL.md files and prepares
        the injector for task-based skill selection.

        Returns the number of skills discovered.
        """
        try:
            from axiom.core.skills.loader import SkillLoader
            from axiom.core.skills.injector import SkillInjector

            self.skill_loader = SkillLoader()
            self.skill_loader.load_all()
            self.skill_injector = SkillInjector(
                loader=self.skill_loader,
                token_budget=4000,
            )
            count = self.skill_loader.count
            if count > 0:
                logger.info("Loaded %d skills (%d tokens)", count, self.skill_loader.total_tokens)
            return count
        except Exception as exc:
            logger.debug("Skills init failed: %s", exc)
            self.skill_loader = None
            self.skill_injector = None
            return 0

    def _init_tracer(self) -> None:
        """Initialise the agent tracer for observability."""
        try:
            from axiom.core.agent.tracer import AgentTracer

            self.tracer = AgentTracer(enabled=True)
        except Exception as exc:
            logger.debug("Tracer init failed: %s", exc)
            self.tracer = None

    async def _init_mcp_bridge(self) -> int:
        """Discover MCP servers and bridge their tools into the registry.

        Returns the number of MCP tools bridged.
        """
        if self.registry is None:
            return 0
        try:
            from axiom.core.mcp.discovery import MCPDiscovery
            from axiom.core.mcp.bridge import MCPBridge
            from axiom.core.mcp.client import MCPClient

            discovery = MCPDiscovery()
            discovery.load_configs()
            servers = discovery.list_configs()
            if not servers:
                return 0

            client = MCPClient()
            bridge = MCPBridge(client)
            bridged = 0

            for cfg in servers:
                try:
                    if cfg.command:
                        await client.connect(cfg.name, cfg.command, cfg.args)
                        tools = await bridge.bridge_server(cfg.name)
                        for tool in tools:
                            self.registry.register(tool)
                            bridged += 1
                except Exception as exc:
                    logger.debug("MCP bridge failed for %s: %s", cfg.name, exc)

            if bridged:
                logger.info("Bridged %d MCP tools from %d servers", bridged, len(servers))
            self.mcp_bridge = bridge
            return bridged
        except Exception as exc:
            logger.debug("MCP bridge init failed: %s", exc)
            return 0

    async def _validate_startup_model(self) -> str:
        """Probe the configured default model; auto-fallback if it fails.

        Delegates to ``UniversalRouter.validate_default_model()`` which
        sends a minimal request (5 tokens) and walks the fallback chain on
        failure.  If the model was swapped, we surface a Rich warning so
        the user knows *before* they start chatting.
        """
        original = self.router.active_model
        try:
            validated = await self.router.validate_default_model()
        except Exception as exc:
            logger.warning("Startup validation error: %s", exc)
            validated = original

        if validated != original:
            console.print(
                f"\n  [{AXIOM_YELLOW}]⚠  Default model unavailable:[/] "
                f"[{AXIOM_DIM}]{original}[/]"
            )
            console.print(
                f"  [{AXIOM_GREEN}]✓  Falling back to:[/] {validated}"
            )
            console.print(
                f"  [{AXIOM_DIM}]Tip: /model list  to see all options, "
                f"or /model <name> to switch.[/]\n"
            )
        return validated

    def _init_memory(self) -> int:
        """Initialise the memory system (ChromaDB + file memory).

        Returns the number of vector entries currently stored.
        """
        try:
            from axiom.core.memory import MemoryManager

            memory_dir = str(self.settings.AXIOM_HOME / "memory")
            self.memory = MemoryManager(memory_dir=memory_dir)
            stats = self.memory.get_stats()
            logger.debug(
                "Memory loaded: %d vectors, %d sessions, %d facts",
                stats["vector_entries"],
                stats["sessions"],
                stats["facts"],
            )
            return stats["vector_entries"]
        except Exception as exc:
            logger.debug("Memory init failed: %s", exc)
            self.memory = None
            return 0

    def _init_tools(self) -> int:
        """Register all built-in tools.

        Returns the count of successfully loaded tools.
        """
        from axiom.core.tools.registry import ToolRegistry

        self.registry = ToolRegistry()

        # ── P0: Stateless tools (no constructor args) ────────────
        _stateless_imports: list[str] = [
            "axiom.core.tools.bash:BashTool",
            "axiom.core.tools.files:ReadFileTool",
            "axiom.core.tools.files:WriteFileTool",
            "axiom.core.tools.files:EditFileTool",
            "axiom.core.tools.files:GlobTool",
            "axiom.core.tools.files:GrepTool",
            "axiom.core.tools.web_fetch:WebFetchTool",
            "axiom.core.tools.git:GitTool",
            "axiom.core.tools.code_exec:CodeExecTool",
            "axiom.core.tools.research:ResearchTool",
            "axiom.core.tools.desktop:DesktopTool",
            "axiom.core.tools.mcp_client:MCPClientTool",
        ]
        loaded = 0
        for dotted in _stateless_imports:
            try:
                module_path, class_name = dotted.rsplit(":", 1)
                mod = importlib.import_module(module_path)
                cls = getattr(mod, class_name)
                self.registry.register(cls())
                loaded += 1
            except Exception as exc:
                logger.debug("Skipping tool %s: %s", dotted, exc)

        # ── P1: Tools needing constructor args ───────────────────

        # Browser tool (optional headless flag)
        try:
            from axiom.core.tools.browser import BrowserTool

            self.registry.register(BrowserTool(headless=True))
            loaded += 1
        except Exception as exc:
            logger.debug("Skipping BrowserTool: %s", exc)

        # Vision tool (needs router for multimodal LLM)
        if self.router is not None:
            try:
                from axiom.core.tools.vision import VisionTool

                self.registry.register(VisionTool(router=self.router))
                loaded += 1
            except Exception as exc:
                logger.debug("Skipping VisionTool: %s", exc)

        # Sub-agent spawning (needs router + registry)
        if self.router is not None:
            try:
                from axiom.core.tools.agent_spawn import SpawnAgentTool

                self.registry.register(
                    SpawnAgentTool(router=self.router, registry=self.registry)
                )
                loaded += 1
            except Exception as exc:
                logger.debug("Skipping SpawnAgentTool: %s", exc)

        # Dynamic tool creation (needs registry)
        try:
            from axiom.core.tools.tool_create import ToolCreateTool

            self.registry.register(ToolCreateTool(registry=self.registry))
            loaded += 1
        except Exception as exc:
            logger.debug("Skipping ToolCreateTool: %s", exc)

        # Memory tools (need MemoryManager instance)
        if self.memory is not None:
            try:
                from axiom.core.tools.memory_tool import (
                    MemorySearchTool,
                    MemorySaveTool,
                )

                self.registry.register(MemorySearchTool(memory_manager=self.memory))
                self.registry.register(MemorySaveTool(memory_manager=self.memory))
                loaded += 2
            except Exception as exc:
                logger.debug("Skipping MemoryTools: %s", exc)

        # ── GOD MODE: Self-repair tool ─────────────────────────────
        try:
            from axiom.core.tools.self_repair import SelfRepairTool

            self.registry.register(SelfRepairTool())
            loaded += 1
        except Exception as exc:
            logger.debug("Skipping SelfRepairTool: %s", exc)

        # Load user-created custom tools from ~/.axiom/tools/
        try:
            from axiom.core.tools.tool_create import load_custom_tools

            custom_count = load_custom_tools(self.registry)
            loaded += custom_count
            if custom_count:
                logger.info("Loaded %d custom tools", custom_count)
        except Exception as exc:
            logger.debug("Custom tool loading failed: %s", exc)

        logger.debug("Loaded %d tools total", loaded)
        return self.registry.count

    # ── Slash commands ────────────────────────────────────────────

    async def handle_command(self, cmd: str) -> bool:
        """Handle slash commands.

        Returns ``True`` to signal the REPL should exit.
        """
        parts = cmd.strip().split(maxsplit=1)
        command = parts[0].lower()
        arg = parts[1] if len(parts) > 1 else ""

        if command in ("/exit", "/quit"):
            console.print(f"[{AXIOM_DIM}]Saving session... Goodbye![/]")
            return True

        if command == "/help":
            self._show_help()
            return False

        if command == "/clear":
            console.clear()
            return False

        if command == "/reset":
            self.messages.clear()
            console.print(f"[{AXIOM_GREEN}]Conversation reset.[/]")
            return False

        if command in ("/model", "/models"):
            # /models (no arg) defaults to "list"
            if command == "/models" and not arg:
                arg = "list"
            return self._handle_model_command(arg)

        if command == "/tools":
            self._show_tools()
            return False

        if command == "/trace":
            self.trace_mode = not self.trace_mode
            state = "ON" if self.trace_mode else "OFF"
            console.print(f"[{AXIOM_CYAN}]Trace mode:[/] {state}")
            return False

        if command == "/yolo":
            self.yolo_mode = not self.yolo_mode
            if self.tool_approval:
                self.tool_approval.yolo_mode = self.yolo_mode
            state = "ON" if self.yolo_mode else "OFF"
            console.print(f"[{AXIOM_CYAN}]YOLO mode:[/] {state}")
            return False

        if command == "/memory":
            self._handle_memory_command(arg)
            return False

        if command == "/agent":
            if not arg:
                console.print(
                    f"[{AXIOM_CYAN}]Usage:[/] /agent <task description>"
                )
            else:
                await self._handle_agent_run(arg)
            return False

        if command == "/history":
            self._show_history()
            return False

        if command == "/compact":
            await self._compact_history()
            return False

        if command == "/voice":
            self._toggle_voice()
            return False

        if command == "/think":
            console.print(
                f"[{AXIOM_CYAN}]Extended thinking is always active with "
                f"compatible models (Opus, DeepSeek R1).[/]"
            )
            return False

        if command == "/mcp":
            await self._handle_mcp_command(arg)
            return False

        if command == "/agents":
            await self._handle_agents_command()
            return False

        if command == "/skills":
            self._handle_skills_command(arg)
            return False

        if command == "/fix":
            await self._handle_fix_command(arg)
            return False

        if command == "/selftest":
            await self._handle_selftest_command()
            return False

        if command == "/connect":
            await self._handle_connect_command(arg)
            return False

        if command == "/council":
            if not arg:
                console.print(
                    f"[{AXIOM_CYAN}]Usage:[/] /council <question or decision>"
                )
            else:
                await self._handle_council_run(arg)
            return False

        console.print(
            f"[{AXIOM_YELLOW}]Unknown command:[/] {command}. "
            f"Type [bold]/help[/] for available commands."
        )
        return False

    def _handle_model_command(self, arg: str) -> bool:
        """Process ``/model`` sub-commands."""
        if not arg:
            console.print(
                f"[{AXIOM_CYAN}]Active model:[/] {self.router.active_model}"
            )
            if self.model_switcher and self.model_switcher.auto_mode:
                console.print(f"  [{AXIOM_DIM}]Auto-routing: ON[/]")
            return False

        if arg == "list":
            self._show_models()
            return False

        if arg == "cost":
            if self.model_switcher:
                report = self.model_switcher.format_cost_report()
                console.print(Panel(
                    report,
                    title=f"[bold {AXIOM_CYAN}]Session Usage[/]",
                    border_style=AXIOM_CYAN,
                    expand=False,
                ))
            else:
                usage = self.router.get_usage()
                console.print(
                    f"[{AXIOM_CYAN}]Session usage:[/] "
                    f"{usage['input']:,} in + {usage['output']:,} out = "
                    f"${usage['cost']:.4f} "
                    f"[{AXIOM_DIM}]({usage['requests']} requests)[/]"
                )
            return False

        if self.model_switcher:
            new_model = self.model_switcher.switch(arg)
        else:
            new_model = self.router.switch_model(arg)
        console.print(f"[{AXIOM_GREEN}]Switched to:[/] {new_model}")
        return False

    def _handle_memory_command(self, arg: str) -> None:
        """Process ``/memory`` sub-commands."""
        if self.memory is None:
            console.print(f"[{AXIOM_YELLOW}]Memory system not available.[/]")
            return

        if arg.startswith("search "):
            query = arg[7:].strip()
            if not query:
                console.print(f"[{AXIOM_YELLOW}]Please provide a search query.[/]")
                return
            console.print(f"[{AXIOM_DIM}]Searching memory for: {query}...[/]")
            results = self.memory.search(query, k=5)
            if not results:
                console.print(f"[{AXIOM_DIM}]No matching memories found.[/]")
                return
            for i, r in enumerate(results, 1):
                meta = r.get("metadata", {})
                sim = 1.0 - r.get("distance", 0.0)
                rtype = meta.get("type", "unknown")
                text = r["text"][:200].replace("\n", " ")
                console.print(
                    f"  [{AXIOM_GREEN}]{i}.[/] [{AXIOM_DIM}][{rtype}][/] "
                    f"(sim: {sim:.2f}) {text}"
                )

        elif arg.startswith("save "):
            content = arg[5:].strip()
            if not content:
                console.print(f"[{AXIOM_YELLOW}]Please provide content to save.[/]")
                return
            self.memory.store_fact(content)
            console.print(f"[{AXIOM_GREEN}]Saved to memory:[/] {content[:100]}")

        elif arg == "stats":
            stats = self.memory.get_stats()
            console.print(f"[{AXIOM_CYAN}]Memory Statistics:[/]")
            console.print(f"  Vectors: {stats['vector_entries']}")
            console.print(f"  Sessions: {stats['sessions']}")
            console.print(f"  Facts: {stats['facts']}")
            console.print(f"  Core memory: {'yes' if stats['core_exists'] else 'no'}")

        else:
            console.print(
                f"[{AXIOM_CYAN}]Usage:[/] /memory search <query> | "
                f"/memory save <text> | /memory stats"
            )

    async def _handle_mcp_command(self, arg: str) -> None:
        """Process ``/mcp`` sub-commands."""
        if not arg or arg == "list":
            # Use the MCP client tool to list servers
            if self.registry and self.registry.get("mcp_client"):
                result = await self.registry.invoke("mcp_client", action="list_servers")
                console.print(result.result if result.success else f"[{AXIOM_RED}]{result.result}[/]")
            else:
                console.print(f"[{AXIOM_YELLOW}]MCP client tool not available.[/]")
        elif arg.startswith("connect "):
            server_name = arg[8:].strip()
            if self.registry and self.registry.get("mcp_client"):
                result = await self.registry.invoke(
                    "mcp_client", action="connect", server_name=server_name
                )
                console.print(result.result if result.success else f"[{AXIOM_RED}]{result.result}[/]")
            else:
                console.print(f"[{AXIOM_YELLOW}]MCP client tool not available.[/]")
        else:
            console.print(
                f"[{AXIOM_CYAN}]Usage:[/] /mcp list | /mcp connect <server>"
            )

    async def _handle_agents_command(self) -> None:
        """List running sub-agents."""
        if self.registry and self.registry.get("spawn_agent"):
            result = await self.registry.invoke("spawn_agent", action="list")
            console.print(result.result if result.success else f"[{AXIOM_RED}]{result.result}[/]")
        else:
            console.print(f"[{AXIOM_DIM}]No sub-agent tool available.[/]")

    def _handle_skills_command(self, arg: str) -> None:
        """Process ``/skills`` sub-commands."""
        if self.skill_loader is None:
            console.print(f"[{AXIOM_YELLOW}]Skills system not available.[/]")
            return

        if not arg or arg == "list":
            skills = self.skill_loader.list_skills()
            if not skills:
                console.print(f"[{AXIOM_DIM}]No skills loaded.[/]")
                return
            console.print(
                f"[{AXIOM_CYAN}]Available skills ({len(skills)}):[/]"
            )
            for s in skills[:30]:  # Cap display at 30
                desc = (s.description or "")[:60]
                console.print(f"  [{AXIOM_GREEN}]@{s.name}[/] -- {desc}")
            if len(skills) > 30:
                console.print(
                    f"  [{AXIOM_DIM}]... and {len(skills) - 30} more. "
                    f"Use /skills search <query> to filter.[/]"
                )

        elif arg.startswith("search "):
            query = arg[7:].strip()
            if not query:
                console.print(f"[{AXIOM_YELLOW}]Please provide a search query.[/]")
                return
            results = self.skill_loader.search(query)
            if not results:
                console.print(f"[{AXIOM_DIM}]No matching skills found.[/]")
                return
            console.print(f"[{AXIOM_CYAN}]Skills matching '{query}':[/]")
            for s in results[:15]:
                desc = (s.description or "")[:60]
                console.print(f"  [{AXIOM_GREEN}]@{s.name}[/] -- {desc}")

        elif arg.startswith("show "):
            name = arg[5:].strip()
            skill = self.skill_loader.get(name)
            if skill is None:
                console.print(f"[{AXIOM_YELLOW}]Skill '{name}' not found.[/]")
                return
            console.print(Panel(
                Markdown(skill.content or "No content"),
                title=f"[bold {AXIOM_CYAN}]@{name}[/]",
                border_style=AXIOM_CYAN,
                expand=True,
                padding=(1, 2),
            ))
        else:
            console.print(
                f"[{AXIOM_CYAN}]Usage:[/] /skills [list] | "
                f"/skills search <query> | /skills show <name>"
            )

    # ── GOD MODE: Self-Repair Commands ──────────────────────────

    async def _handle_fix_command(self, context: str = "") -> None:
        """GOD MODE: Self-diagnose and optionally auto-repair errors.

        This is the user's panic button.  When Axiom hits an error, running
        ``/fix`` surfaces the last traceback, identifies the faulty module,
        and offers to feed the diagnosis into the agent for auto-repair.
        """
        try:
            from axiom.core.tools.self_repair import SelfRepairTool

            tool = SelfRepairTool()

            # Step 1: Diagnose
            diagnosis = tool._diagnose(context)
            console.print(Panel(
                Text(diagnosis, overflow="fold"),
                title=Text.from_markup(
                    f"[bold {AXIOM_CYAN}]🔧 GOD MODE — Self-Diagnosis[/]"
                ),
                title_align="left",
                border_style=Style(color=AXIOM_CYAN),
                padding=(0, 1),
                expand=False,
            ))

            # Check if there's actually an error to fix
            if not SelfRepairTool._last_error:
                console.print(
                    f"[{AXIOM_GREEN}]✓ No errors detected. System is healthy.[/]"
                )
                return

            # Step 2: Offer to auto-repair via the agent
            console.print(
                f"\n[{AXIOM_YELLOW}]Would you like Axiom to attempt auto-repair? "
                f"[bold](y/n)[/bold][/]"
            )
            try:
                answer = await self.input_handler.get_input_async()
                if answer and answer.strip().lower() in ("y", "yes"):
                    # Feed the error into the agent with self-repair instructions
                    repair_prompt = (
                        f"I detected an error in myself. Here is the diagnosis:\n\n"
                        f"```\n{diagnosis}\n```\n\n"
                        f"Use the self_repair tool to:\n"
                        f"1. introspect the faulty module\n"
                        f"2. identify the bug\n"
                        f"3. apply a self_edit fix\n"
                        f"4. hot_reload the patched module\n"
                        f"5. verify the fix\n\n"
                        f"Fix this error now."
                    )
                    await self.chat(repair_prompt)
                else:
                    console.print(
                        f"[{AXIOM_DIM}]Auto-repair skipped. You can manually fix "
                        f"the issue or try again with /fix[/]"
                    )
            except (EOFError, KeyboardInterrupt):
                console.print(f"\n[{AXIOM_DIM}]Cancelled.[/]")

        except ImportError:
            console.print(
                f"[{AXIOM_RED}]Self-repair tool not available. "
                f"Check axiom/core/tools/self_repair.py[/]"
            )
        except Exception as exc:
            console.print(f"[{AXIOM_RED}]Self-diagnosis failed: {exc}[/]")

    async def _handle_selftest_command(self) -> None:
        """GOD MODE: Run a quick self-test of all major subsystems.

        Checks: model router, tool registry, memory, self-repair, and
        reports what's working vs broken.
        """
        console.print(
            f"\n[bold {AXIOM_CYAN}]🧪 GOD MODE — Self-Test[/]\n"
        )
        results: list[tuple[str, bool, str]] = []

        # 1. Model Router
        try:
            model = self.router.active_model if self.router else None
            if model:
                results.append(("LLM Router", True, f"Active: {model}"))
            else:
                results.append(("LLM Router", False, "No active model"))
        except Exception as exc:
            results.append(("LLM Router", False, str(exc)))

        # 2. Tool Registry
        try:
            if self.registry:
                tool_count = len(self.registry.list_tools())
                tool_names = ", ".join(
                    t.name for t in self.registry.list_tools()[:8]
                )
                if tool_count > 8:
                    tool_names += f" (+{tool_count - 8} more)"
                results.append(("Tool Registry", True, f"{tool_count} tools: {tool_names}"))
            else:
                results.append(("Tool Registry", False, "Not initialized"))
        except Exception as exc:
            results.append(("Tool Registry", False, str(exc)))

        # 3. Self-Repair Tool
        try:
            from axiom.core.tools.self_repair import SelfRepairTool
            sr = SelfRepairTool()
            cmap = sr._codebase_map()
            line_count = cmap.count("\n")
            results.append(("Self-Repair", True, f"Codebase visible ({line_count} entries)"))
        except Exception as exc:
            results.append(("Self-Repair", False, str(exc)))

        # 4. Memory System
        try:
            if self.memory:
                results.append(("Memory", True, "ChromaDB + File memory active"))
            else:
                results.append(("Memory", False, "Not initialized"))
        except Exception as exc:
            results.append(("Memory", False, str(exc)))

        # 5. Individual tool health check
        if self.registry:
            for tool in self.registry.list_tools():
                try:
                    # Verify tool has required attributes
                    assert tool.name, "Missing name"
                    assert tool.description, "Missing description"
                    assert callable(getattr(tool, "execute", None)), "Not callable"
                    results.append((f"  └ {tool.name}", True, tool.risk_level))
                except Exception as exc:
                    results.append((f"  └ {tool.name}", False, str(exc)))

        # 6. Last error status
        try:
            from axiom.core.tools.self_repair import SelfRepairTool
            if SelfRepairTool._last_error:
                results.append(("Last Error", False, SelfRepairTool._last_error[:80]))
            else:
                results.append(("Last Error", True, "None — system healthy"))
        except Exception:
            results.append(("Last Error", True, "Cannot check (self_repair not loaded)"))

        # Render results
        table = Table(
            title=Text.from_markup(f"[bold {AXIOM_CYAN}]System Health Report[/]"),
            border_style=Style(color=AXIOM_DIM),
            padding=(0, 1),
            expand=False,
        )
        table.add_column("Subsystem", style=f"bold {AXIOM_CYAN}", min_width=18)
        table.add_column("Status", justify="center", width=8)
        table.add_column("Details", min_width=40)

        passed = 0
        for name, ok, detail in results:
            icon = f"[{AXIOM_GREEN}]✓[/]" if ok else f"[{AXIOM_RED}]✗[/]"
            detail_style = AXIOM_DIM if ok else AXIOM_RED
            table.add_row(name, Text.from_markup(icon), Text(detail, style=detail_style))
            if ok:
                passed += 1

        console.print(table)
        total = len(results)
        pct = int(100 * passed / total) if total else 0
        color = AXIOM_GREEN if pct >= 80 else AXIOM_YELLOW if pct >= 50 else AXIOM_RED
        console.print(
            f"\n[{color}]{passed}/{total} checks passed ({pct}%)[/]\n"
        )

    # ── /connect command ──────────────────────────────────────────

    async def _handle_connect_command(self, arg: str) -> None:
        """Handle /connect <service> [config] — connect integrations live."""
        parts = arg.strip().split(maxsplit=1)
        if not parts:
            console.print(
                f"[{AXIOM_CYAN}]Usage:[/]\n"
                f"  /connect telegram <bot_token>\n"
                f"  /connect telegram disconnect"
            )
            return

        service = parts[0].lower()
        config = parts[1].strip() if len(parts) > 1 else ""

        if service == "telegram":
            await self._connect_telegram(config)
        else:
            console.print(
                f"[{AXIOM_YELLOW}]Unknown service:[/] {service}. "
                f"Available: telegram"
            )

    async def _connect_telegram(self, token_or_cmd: str) -> None:
        """Connect or disconnect Telegram bot live."""
        from dotenv import set_key

        env_path = self.settings.AXIOM_HOME / ".env"
        env_path.parent.mkdir(parents=True, exist_ok=True)

        # ── Disconnect ──────────────────────────────────────────
        if token_or_cmd.lower() == "disconnect":
            if self._telegram_bot is not None:
                try:
                    await self._telegram_bot.stop()
                except Exception:
                    pass
                self._telegram_bot = None
                self._telegram_active = False
            set_key(str(env_path), "TELEGRAM_ENABLED", "false")
            console.print(f"[{AXIOM_GREEN}]✓ Telegram disconnected.[/]")
            return

        # ── Connect — need a token ──────────────────────────────
        token = token_or_cmd.strip()
        if not token:
            console.print(
                f"[{AXIOM_YELLOW}]Please provide your bot token:[/]\n"
                f"  /connect telegram <your_bot_token>"
            )
            return

        # Validate token format (basic check: contains colon)
        if ":" not in token:
            console.print(
                f"[{AXIOM_RED}]Invalid token format.[/] "
                f"Telegram bot tokens look like: 123456:ABC-DEF..."
            )
            return

        # Stop existing bot if running
        if self._telegram_bot is not None:
            try:
                await self._telegram_bot.stop()
            except Exception:
                pass
            self._telegram_bot = None
            self._telegram_active = False

        # Write token to ~/.axiom/.env
        console.print(f"[{AXIOM_DIM}]Writing config to {env_path}...[/]")
        set_key(str(env_path), "TELEGRAM_BOT_TOKEN", token)
        set_key(str(env_path), "TELEGRAM_ENABLED", "true")

        # Reload settings
        get_settings.cache_clear()
        self.settings = get_settings()

        # Hot-start Telegram bot
        try:
            from axiom.integrations.telegram.bridge import TelegramBridge
            from axiom.integrations.telegram.handler import TelegramBot

            allowed_users = None
            if self.settings.TELEGRAM_ALLOWED_USERS:
                try:
                    allowed_users = {
                        int(uid.strip())
                        for uid in self.settings.TELEGRAM_ALLOWED_USERS.split(",")
                        if uid.strip()
                    }
                except ValueError:
                    pass

            tg_bridge = TelegramBridge(app=self)
            self._telegram_bot = TelegramBot(
                token=token,
                bridge=tg_bridge,
                allowed_users=allowed_users,
            )
            await self._telegram_bot.start()
            self._telegram_active = True

            # Re-inject system prompt so LLM knows Telegram is now active
            for i, msg in enumerate(self.messages):
                if msg.get("role") == "system":
                    self.messages.pop(i)
                    break
            tasks_text = getattr(self, "_pending_tasks_text", "")
            system_prompt = self._build_system_prompt("", tasks_text)
            self.messages.insert(0, {"role": "system", "content": system_prompt})

            console.print(
                f"[{AXIOM_GREEN}]✓ Telegram bot CONNECTED and mirrored![/]\n"
                f"[{AXIOM_DIM}]Token saved to {env_path}[/]\n"
                f"[{AXIOM_DIM}]Bot will auto-start on next launch.[/]"
            )
        except Exception as exc:
            console.print(
                f"[{AXIOM_RED}]✗ Telegram connect failed: {str(exc)[:120]}[/]"
            )
            logger.warning("Telegram hot-connect failed: %s", exc)

    async def _handle_council_run(self, question: str) -> None:
        """Run a question through COUNCIL mode (multi-LLM consensus)."""
        try:
            from axiom.core.agent.graph import run_agent, EventType
            from axiom.core.agent.state import AgentMode
        except ImportError as exc:
            console.print(f"[{AXIOM_RED}]Agent graph not available: {exc}[/]")
            return

        # Build system prompt
        memory_context = ""
        if self.memory is not None:
            try:
                memory_context = self.memory.build_context(question, k=3)
            except Exception:
                pass

        try:
            from axiom.core.agent.prompts.system import build_system_prompt

            tool_names = self.registry.list_names() if self.registry else []
            system_prompt = build_system_prompt(
                tool_names=tool_names,
                memory_context=memory_context,
                model_name=self.router.active_model if self.router else "unknown",
                mode="council",
            )
        except Exception:
            system_prompt = "You are Axiom. Provide a thorough, well-reasoned answer."

        agent_messages = [
            {"role": "system", "content": system_prompt},
        ]
        if self.messages:
            agent_messages.extend(self.messages[-4:])
        agent_messages.append({"role": "user", "content": question})

        console.print(
            Panel(
                f"[bold {AXIOM_PURPLE}]Council Mode[/] — multi-LLM deliberation",
                border_style=AXIOM_PURPLE,
                expand=False,
            )
        )

        final_answer = ""
        try:
            async for event in run_agent(
                router=self.router,
                registry=self.registry,
                messages=agent_messages,
                mode=AgentMode.COUNCIL,
                tracer=self.tracer,
            ):
                final_answer = self._render_agent_event(event, final_answer)

        except KeyboardInterrupt:
            console.print(f"\n[{AXIOM_YELLOW}]Council interrupted.[/]")
        except Exception as exc:
            error_msg = str(exc)
            if self.trace_mode:
                error_msg += f"\n{traceback.format_exc()}"
            console.print(f"[{AXIOM_RED}]Council error:[/] {error_msg}")

        if final_answer:
            self.messages.append({"role": "user", "content": f"[council] {question}"})
            self.messages.append({"role": "assistant", "content": final_answer})

    def _toggle_voice(self) -> None:
        """Toggle voice input mode."""
        if self.voice is None:
            self._init_voice()
            if self.voice is None:
                return
        self.voice_mode = not self.voice_mode
        state = "ON" if self.voice_mode else "OFF"
        console.print(f"[{AXIOM_CYAN}]Voice input:[/] {state}")

    # ── Display helpers ───────────────────────────────────────────

    def _show_help(self) -> None:
        """Display grouped command reference."""
        from rich.tree import Tree

        groups = [
            ("Core", AXIOM_CYAN, [
                ("/help", "Show this help"),
                ("/clear", "Clear screen"),
                ("/reset", "Clear conversation history"),
                ("/history", "Show conversation summary"),
                ("/compact", "Compress history via LLM summary"),
                ("/exit", "Save session and quit"),
            ]),
            ("LLM", AXIOM_PURPLE, [
                ("/model <name>", "Switch LLM (opus, sonnet, gemini, groq, etc.)"),
                ("/models", "Show all available models"),
                ("/model cost", "Show token usage and cost"),
                ("/model auto", "Enable smart auto-routing"),
                ("/think", "Extended thinking info"),
                ("/council <q>", "Multi-LLM consensus deliberation"),
            ]),
            ("Tools & Agent", AXIOM_GREEN, [
                ("/tools", "List available tools"),
                ("/agent <task>", "Run task through agent (PLAN/REACT)"),
                ("/yolo", "Toggle auto-approve tool calls"),
                ("/trace", "Toggle agent trace visibility"),
            ]),
            ("Memory & Skills", AXIOM_YELLOW, [
                ("/memory search <q>", "Search agent memory"),
                ("/memory save <text>", "Save a fact to memory"),
                ("/memory stats", "Show memory statistics"),
                ("/skills", "Browse domain knowledge skills"),
            ]),
            ("Integrations", AXIOM_CYAN, [
                ("/connect telegram <token>", "Connect Telegram bot"),
                ("/mcp list", "List MCP server connections"),
                ("/mcp connect <name>", "Connect to an MCP server"),
                ("/voice", "Toggle voice input (Whisper STT)"),
            ]),
            ("GOD MODE", AXIOM_RED, [
                ("/fix [context]", "Self-diagnose and auto-repair"),
                ("/selftest", "Verify all systems operational"),
                ("/agents", "List running sub-agents"),
            ]),
        ]

        tree = Tree(Text.from_markup(f"[bold {AXIOM_CYAN}]Axiom Commands[/]"))

        for group_name, color, commands in groups:
            branch = tree.add(Text.from_markup(f"[bold {color}]{group_name}[/]"))
            for cmd, desc in commands:
                branch.add(Text.from_markup(
                    f"[bold {color}]{cmd:<30}[/]  [{AXIOM_DIM}]{desc}[/]"
                ))

        console.print()
        console.print(Panel(tree, border_style=AXIOM_CYAN, padding=(0, 1), expand=False))
        console.print()

    def _show_models(self) -> None:
        """Show all available models grouped by provider with aliases."""
        if self.model_switcher:
            # Rich formatted catalog with groups, aliases, and availability
            formatted = self.model_switcher.format_model_list()
            console.print(Panel(
                formatted,
                title=f"[bold {AXIOM_CYAN}]Available Models[/]",
                border_style=AXIOM_CYAN,
                expand=False,
                padding=(0, 1),
            ))
        else:
            # Fallback to simple provider table
            available = self.router.list_available()
            table = Table(title="Available Models", border_style=AXIOM_CYAN)
            table.add_column("Provider", style=f"bold {AXIOM_CYAN}", min_width=14)
            table.add_column("Model", min_width=30)
            table.add_column("Status", justify="center", min_width=10)
            table.add_column("Circuit", justify="center", min_width=8)
            for m in available:
                if m["is_active"]:
                    status = f"[bold {AXIOM_CYAN}]ACTIVE[/]"
                elif m["available"]:
                    status = f"[{AXIOM_GREEN}]Online[/]"
                else:
                    status = f"[{AXIOM_RED}]No Key[/]"
                cb = m.get("circuit_state", "closed")
                if cb == "closed":
                    circuit = f"[{AXIOM_GREEN}]OK[/]"
                elif cb == "half_open":
                    circuit = f"[{AXIOM_YELLOW}]Test[/]"
                else:
                    circuit = f"[{AXIOM_RED}]Open[/]"
                table.add_row(m["display_name"], m["model"], status, circuit)
            console.print(table)

    def _show_tools(self) -> None:
        if self.registry is None or self.registry.count == 0:
            console.print(f"[{AXIOM_DIM}]No tools loaded.[/]")
            return
        console.print(
            f"[{AXIOM_CYAN}]Available tools ({self.registry.count}):[/]"
        )
        for tool in self.registry.list_tools():
            color = {
                "low": AXIOM_GREEN,
                "medium": AXIOM_YELLOW,
                "high": AXIOM_RED,
            }.get(tool.risk_level, AXIOM_DIM)
            console.print(
                f"  [{color}]*[/] [bold]{tool.name}[/] -- {tool.description}"
            )

    def _show_history(self) -> None:
        if not self.messages:
            console.print(f"[{AXIOM_DIM}]No messages in conversation.[/]")
            return
        console.print(
            f"[{AXIOM_CYAN}]Conversation ({len(self.messages)} messages):[/]"
        )
        for i, msg in enumerate(self.messages):
            role = msg["role"]
            content = msg.get("content", "")
            preview = content[:80].replace("\n", " ") if content else ""
            if content and len(content) > 80:
                preview += "..."
            color = AXIOM_CYAN if role == "user" else AXIOM_GREEN
            console.print(f"  [{color}]{i + 1}. {role}:[/] {preview}")

    async def _compact_history(self) -> None:
        """Compress older messages via LLM summarization."""
        from axiom.core.memory.context_compressor import (
            compress_context,
            should_compress,
        )

        if not await should_compress(self.messages):
            console.print(
                f"[{AXIOM_DIM}]History is compact enough "
                f"({len(self.messages)} messages).[/]"
            )
            return

        before = len(self.messages)
        console.print(f"[{AXIOM_DIM}]Compressing {before} messages...[/]")

        if self.router is None:
            console.print(f"[{AXIOM_YELLOW}]No LLM router -- cannot compress.[/]")
            return

        try:
            self.messages = await compress_context(
                router=self.router,
                messages=self.messages,
            )
            after = len(self.messages)
            console.print(
                f"[{AXIOM_GREEN}]Compressed:[/] {before} -> {after} messages."
            )
        except Exception as exc:
            logger.debug("Compression failed: %s", exc)
            # Fallback: simple truncation
            if before > 20:
                self.messages = self.messages[-20:]
                console.print(
                    f"[{AXIOM_YELLOW}]Fallback truncation:[/] "
                    f"{before} -> {len(self.messages)} messages."
                )

    def _inject_system_prompt(self, user_input: str = "") -> None:
        """Inject the system prompt as the first message (idempotent).

        Delegates to _build_system_prompt() which includes all context:
        tools, memory, skills, tasks, AND integrations.
        """
        if any(m.get("role") == "system" for m in self.messages):
            return  # Already injected

        tasks_text = getattr(self, "_pending_tasks_text", "")
        system_prompt = self._build_system_prompt(user_input, tasks_text)
        self.messages.insert(0, {"role": "system", "content": system_prompt})

    # ── Agent mode ────────────────────────────────────────────────

    async def _handle_agent_run(self, task: str) -> None:
        """Run a task through the tri-mode agent graph.

        Yields events and renders them in real-time using Rich panels.
        """
        try:
            from axiom.core.agent.graph import run_agent, EventType
        except ImportError as exc:
            console.print(f"[{AXIOM_RED}]Agent graph not available: {exc}[/]")
            return

        # Build memory context for the agent
        memory_context = ""
        if self.memory is not None:
            try:
                memory_context = self.memory.build_context(task, k=5)
            except Exception as exc:
                logger.debug("Agent memory context build failed: %s", exc)

        # Inject relevant skills for this task
        skills_context = ""
        if self.skill_injector is not None:
            try:
                skills_context = self.skill_injector.inject(task)
            except Exception as exc:
                logger.debug("Agent skills injection failed: %s", exc)

        # Build system prompt with tool names, memory, skills, and mode hint
        try:
            from axiom.core.agent.prompts.system import build_system_prompt

            tool_names = self.registry.list_names() if self.registry else []
            system_prompt = build_system_prompt(
                tool_names=tool_names,
                memory_context=memory_context,
                model_name=self.router.active_model if self.router else "unknown",
                skills_context=skills_context,
            )
        except Exception:
            system_prompt = (
                "You are Axiom, an advanced AI agent. Complete the task using "
                "available tools efficiently."
            )

        # Prepare messages with system prompt
        agent_messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
        ]
        # Include recent conversation for context
        if self.messages:
            agent_messages.extend(self.messages[-6:])
        agent_messages.append({"role": "user", "content": task})

        console.print(
            Panel(
                f"[bold {AXIOM_CYAN}]Agent Mode[/] — executing task",
                border_style=AXIOM_CYAN,
                expand=False,
            )
        )

        final_answer = ""
        try:
            async for event in run_agent(
                router=self.router,
                registry=self.registry,
                messages=agent_messages,
                tracer=self.tracer,
                skills_dir=Path("memory/skills") if skills_context else None,
            ):
                final_answer = self._render_agent_event(event, final_answer)

        except KeyboardInterrupt:
            console.print(f"\n[{AXIOM_YELLOW}]Agent interrupted.[/]")
        except Exception as exc:
            error_msg = str(exc)
            if self.trace_mode:
                error_msg += f"\n{traceback.format_exc()}"
            console.print(f"[{AXIOM_RED}]Agent error:[/] {error_msg}")

        # Store the interaction in history
        if final_answer:
            self.messages.append({"role": "user", "content": f"[agent task] {task}"})
            self.messages.append({"role": "assistant", "content": final_answer})

            # Store in memory
            if self.memory is not None:
                try:
                    self.memory.store_message("user", task)
                    self.memory.store_message("assistant", final_answer)
                except Exception:
                    pass

    def _render_agent_event(self, event: Any, current_answer: str) -> str:
        """Render a single AgentEvent to the console.

        Returns the updated answer string.
        """
        from axiom.core.agent.graph import EventType

        data = event.data
        etype = event.type

        if etype == EventType.THINKING:
            phase = data.get("phase", data.get("mode", ""))
            iteration = data.get("iteration", "")
            label = phase or f"iteration {iteration}" if iteration else "thinking"
            console.print(f"  [{AXIOM_DIM}]⟳ {label}...[/]")

        elif etype == EventType.PLAN_CREATED:
            plan = data.get("plan", [])
            console.print(f"  [{AXIOM_CYAN}]Plan ({len(plan)} steps):[/]")
            for i, step in enumerate(plan, 1):
                desc = step.get("description", step.get("action", ""))
                console.print(f"    [{AXIOM_DIM}]{i}.[/] {desc}")

        elif etype == EventType.STEP_START:
            step = data.get("step", "?")
            total = data.get("total", "?")
            desc = data.get("description", "")
            console.print(
                f"  [{AXIOM_CYAN}]Step {step}/{total}:[/] {desc}"
            )

        elif etype == EventType.TOOL_CALL:
            tool = data.get("tool", "?")
            args = data.get("args", {})
            args_str = json.dumps(args, indent=None)[:120]
            console.print(
                f"  [{AXIOM_PURPLE}]⚡ {tool}[/] [{AXIOM_DIM}]{args_str}[/]"
            )

        elif etype == EventType.TOOL_RESULT:
            tool = data.get("tool", "")
            success = data.get("success", False)
            duration = data.get("duration_ms", 0)
            result_preview = data.get("result", "")[:200]
            icon = f"[{AXIOM_GREEN}]✓[/]" if success else f"[{AXIOM_RED}]✗[/]"
            console.print(
                f"  {icon} [{AXIOM_DIM}]{tool} ({duration}ms)[/]"
            )
            if self.trace_mode and result_preview:
                console.print(f"    [{AXIOM_DIM}]{result_preview}[/]")

        elif etype == EventType.OBSERVATION:
            status = data.get("status", "")
            confidence = data.get("confidence", 0)
            reason = data.get("reason", "")
            color = AXIOM_GREEN if status == "complete" else AXIOM_YELLOW
            console.print(
                f"  [{color}]Observer:[/] {status} "
                f"[{AXIOM_DIM}](confidence: {confidence:.0%})[/]"
            )
            if reason and self.trace_mode:
                console.print(f"    [{AXIOM_DIM}]{reason}[/]")

        elif etype == EventType.REPLAN:
            attempt = data.get("attempt", "?")
            reason = data.get("reason", "")
            console.print(
                f"  [{AXIOM_YELLOW}]↻ Replan #{attempt}:[/] {reason[:120]}"
            )

        elif etype == EventType.TRACE:
            thought = data.get("thought", "")
            action = data.get("action", "")
            console.print(f"  [{AXIOM_DIM}]💭 {thought[:200]}[/]")
            if action != "finish":
                console.print(f"  [{AXIOM_PURPLE}]\u2192 {action}[/]")

        elif etype == EventType.ANSWER:
            answer = data.get("answer", "")
            current_answer = answer
            console.print()
            console.print(
                Panel(
                    Markdown(answer),
                    title=f"[bold {AXIOM_GREEN}]Agent Answer[/]",
                    border_style=AXIOM_GREEN,
                    expand=True,
                    padding=(1, 2),
                )
            )

        elif etype == EventType.COUNCIL_START:
            phase = data.get("phase", "deliberation")
            console.print(
                f"\n  [{AXIOM_PURPLE}]🏛  Council {phase}[/] "
                f"[{AXIOM_DIM}]— querying multiple models...[/]"
            )

        elif etype == EventType.COUNCIL_MEMBER:
            model = data.get("model", "?")
            score = data.get("score", 0)
            latency = data.get("latency_ms", 0)
            error = data.get("error")
            if error:
                console.print(
                    f"  [{AXIOM_RED}]✗[/] [{AXIOM_DIM}]{model} — error: {error}[/]"
                )
            else:
                score_color = AXIOM_GREEN if score >= 7 else AXIOM_YELLOW if score >= 4 else AXIOM_RED
                console.print(
                    f"  [{AXIOM_GREEN}]✓[/] [{AXIOM_DIM}]{model}[/] "
                    f"[{score_color}]score:{score:.1f}[/] "
                    f"[{AXIOM_DIM}]({latency}ms)[/]"
                )
                if self.trace_mode:
                    preview = data.get("response_preview", "")
                    if preview:
                        console.print(f"    [{AXIOM_DIM}]{preview[:150]}...[/]")

        elif etype == EventType.COUNCIL_SYNTHESIS:
            consensus = data.get("consensus_score", 0)
            members = data.get("member_count", 0)
            chairman = data.get("chairman", "?")
            total_ms = data.get("total_time_ms", 0)
            consensus_color = AXIOM_GREEN if consensus >= 0.7 else AXIOM_YELLOW
            console.print(
                f"\n  [{AXIOM_PURPLE}]🏛  Synthesis[/] "
                f"[{consensus_color}]consensus:{consensus:.0%}[/] "
                f"[{AXIOM_DIM}]({members} models, chair:{chairman}, "
                f"{total_ms}ms)[/]"
            )

        elif etype == EventType.LEARNING:
            skill = data.get("skill", "")
            saved = data.get("saved", False)
            icon = f"[{AXIOM_GREEN}]📚[/]" if saved else f"[{AXIOM_DIM}]📝[/]"
            console.print(
                f"\n  {icon} [{AXIOM_DIM}]Learned: {skill[:120]}[/]"
            )

        elif etype == EventType.ERROR:
            message = data.get("message", "Unknown error")
            console.print(f"  [{AXIOM_RED}]✗ Error:[/] {message}")

        return current_answer

    # ── Chat turn ─────────────────────────────────────────────────

    async def chat(self, user_input: str) -> None:
        """Process a single chat turn.

        Streams the LLM response, handles tool calls, and updates
        conversation history.
        """
        # ── Preload pending tasks for system prompt injection ────
        try:
            pending = await self.task_store.get_pending()
            self._pending_tasks_text = self.task_store.format_for_prompt(pending)
        except Exception:
            self._pending_tasks_text = ""

        # ── Inject system prompt on first message ────────────────
        if not any(m.get("role") == "system" for m in self.messages):
            self._inject_system_prompt(user_input)

        self.messages.append({"role": "user", "content": user_input})

        # Persist to shared conversation store (mirrored with Telegram)
        try:
            await self.conversation_store.append("user", user_input, channel="cli")
        except Exception as exc:
            logger.warning("Conversation store write failed: %s", exc)

        # Store user message in memory
        if self.memory is not None:
            try:
                self.memory.store_message("user", user_input)
            except Exception as exc:
                logger.debug("Memory store failed: %s", exc)

        # Auto-select model if in auto mode
        if self.model_switcher and self.model_switcher.auto_mode:
            selected = self.model_switcher.auto_select(user_input)
            if selected and self.trace_mode:
                console.print(
                    f"  [{AXIOM_DIM}]Auto-routed to: {selected}[/]"
                )

        # Auto-compress if context is getting large
        try:
            from axiom.core.memory.context_compressor import should_compress

            if await should_compress(self.messages):
                await self._compact_history()
        except Exception as exc:
            logger.debug("Context compression check failed: %s", exc)

        # Build tool schemas for function calling
        tools: Optional[list[dict[str, Any]]] = None
        if self.registry is not None and self.registry.count > 0:
            try:
                tools = self.registry.to_llm_schemas()
            except Exception as exc:
                logger.debug("Failed to build tool schemas: %s", exc)

        self.renderer.start_thinking()

        full_response = ""
        tool_call_buffer: dict[int, dict[str, Any]] = {}  # index -> {id, name, args_str}

        try:
            async for chunk in self.router.complete(
                messages=self.messages,
                tools=tools,
                stream=True,
            ):
                self.renderer.stop_thinking()

                if not hasattr(chunk, "choices") or not chunk.choices:
                    continue

                delta = chunk.choices[0].delta

                # ── Text content ──────────────────────────────────
                if hasattr(delta, "content") and delta.content:
                    self.renderer.stream_token(delta.content)
                    full_response += delta.content

                # ── Tool call accumulation ────────────────────────
                if hasattr(delta, "tool_calls") and delta.tool_calls:
                    for tc in delta.tool_calls:
                        idx = getattr(tc, "index", 0) or 0
                        if idx not in tool_call_buffer:
                            tool_call_buffer[idx] = {
                                "id": "",
                                "name": "",
                                "arguments": "",
                            }
                        buf = tool_call_buffer[idx]
                        if hasattr(tc, "id") and tc.id:
                            buf["id"] = tc.id
                        if hasattr(tc, "function") and tc.function:
                            if tc.function.name:
                                buf["name"] = tc.function.name
                            if tc.function.arguments:
                                buf["arguments"] += tc.function.arguments

                # ── Check for finish reason ───────────────────────
                finish = getattr(chunk.choices[0], "finish_reason", None)
                if finish == "tool_calls" and tool_call_buffer:
                    self.renderer.finish_stream()
                    full_response = await self._execute_tool_calls(
                        tool_call_buffer, full_response
                    )
                    tool_call_buffer.clear()

            self.renderer.finish_stream()

            if full_response:
                self.messages.append(
                    {"role": "assistant", "content": full_response}
                )
                # Store assistant response in memory
                if self.memory is not None:
                    try:
                        self.memory.store_message("assistant", full_response)
                    except Exception as exc:
                        logger.debug("Memory store (assistant) failed: %s", exc)

                # Persist to shared conversation store (mirrored)
                try:
                    await self.conversation_store.append(
                        "assistant", full_response, channel="cli"
                    )
                except Exception as exc:
                    logger.warning("Conversation store write failed: %s", exc)

        except KeyboardInterrupt:
            self.renderer.stop_thinking()
            self.renderer.finish_stream()
            console.print(f"\n[{AXIOM_DIM}]Interrupted.[/]")
        except Exception as exc:
            self.renderer.stop_thinking()
            self.renderer.finish_stream()
            error_msg = str(exc)
            tb_str = traceback.format_exc()
            if self.trace_mode:
                error_msg += f"\n{tb_str}"
            self.renderer.show_error(error_msg)

            # ── GOD MODE: Capture error for self-repair ───────
            try:
                from axiom.core.tools.self_repair import SelfRepairTool
                SelfRepairTool.capture_error(exc, tb_str)
            except Exception:
                pass

    # ── Headless chat (for Telegram / API callers) ─────────────

    async def chat_headless(
        self,
        user_input: str,
        *,
        channel: str = "telegram",
        user_id: Optional[int] = None,
    ) -> str:
        """Run the LLM pipeline without CLI rendering — returns plain text.

        This is the method Telegram (and future API callers) use.
        It shares the same ConversationStore as the CLI, creating
        a mirrored conversation experience.

        Args:
            user_input: The user's message text.
            channel: Source channel identifier (default ``"telegram"``).
            user_id: Optional user ID for multi-user support.

        Returns:
            The agent's text response.
        """
        # 1. Persist user message to shared conversation store
        await self.conversation_store.append(
            "user", user_input, channel=channel,
        )
        await self.message_bus.publish(
            {"role": "user", "content": user_input, "channel": channel},
            source_channel=channel,
        )

        # 2. Load shared conversation history for LLM context
        messages = await self.conversation_store.to_llm_messages(last_n=40)

        # 3. Build task context + inject system prompt if not present
        tasks_text = ""
        try:
            pending = await self.task_store.get_pending()
            tasks_text = self.task_store.format_for_prompt(pending)
        except Exception:
            pass

        if not any(m.get("role") == "system" for m in messages):
            system_text = self._build_system_prompt(user_input, tasks_text)
            messages.insert(0, {"role": "system", "content": system_text})

        # 4. Build tool schemas
        tools: Optional[list[dict[str, Any]]] = None
        if self.registry is not None and self.registry.count > 0:
            try:
                tools = self.registry.to_llm_schemas()
            except Exception:
                pass

        # 5. Run LLM completion with tool call support
        full_response = ""
        tool_call_buffer: dict[int, dict[str, Any]] = {}
        try:
            async for chunk in self.router.complete(
                messages=messages,
                tools=tools,
                stream=True,
            ):
                if not hasattr(chunk, "choices") or not chunk.choices:
                    continue
                delta = chunk.choices[0].delta

                # ── Text content ──
                if hasattr(delta, "content") and delta.content:
                    full_response += delta.content

                # ── Tool call accumulation ──
                if hasattr(delta, "tool_calls") and delta.tool_calls:
                    for tc in delta.tool_calls:
                        idx = getattr(tc, "index", 0) or 0
                        if idx not in tool_call_buffer:
                            tool_call_buffer[idx] = {
                                "id": "", "name": "", "arguments": "",
                            }
                        buf = tool_call_buffer[idx]
                        if hasattr(tc, "id") and tc.id:
                            buf["id"] = tc.id
                        if hasattr(tc, "function") and tc.function:
                            if tc.function.name:
                                buf["name"] = tc.function.name
                            if tc.function.arguments:
                                buf["arguments"] += tc.function.arguments

                # ── Check finish reason for tool execution ──
                finish = getattr(chunk.choices[0], "finish_reason", None)
                if finish == "tool_calls" and tool_call_buffer:
                    full_response = await self._execute_tool_calls_headless(
                        tool_call_buffer, messages, full_response,
                    )
                    tool_call_buffer.clear()

        except Exception as exc:
            logger.error("chat_headless error: %s", exc)
            full_response = f"Error processing your message: {str(exc)[:200]}"

        # 6. Persist assistant response
        await self.conversation_store.append(
            "assistant", full_response, channel=channel,
        )
        await self.message_bus.publish(
            {"role": "assistant", "content": full_response, "channel": channel},
            source_channel=channel,
        )

        # 7. Store in vector memory for semantic search
        if self.memory is not None:
            try:
                self.memory.store_message("user", user_input)
                self.memory.store_message("assistant", full_response)
            except Exception:
                pass

        return full_response

    def _build_system_prompt(
        self, user_input: str = "", tasks_text: str = "",
    ) -> str:
        """Build the system prompt with task context (shared by headless + CLI)."""
        from axiom.core.agent.prompts.system import build_system_prompt

        try:
            tool_names = self.registry.list_names() if self.registry else []

            memory_context = ""
            if self.memory is not None:
                try:
                    memory_context = self.memory.build_context(user_input, k=3)
                except Exception as exc:
                    logger.debug("Memory context build failed: %s", exc)

            skills_context = ""
            if self.skill_injector is not None:
                try:
                    skills_context = self.skill_injector.inject(user_input)
                except Exception as exc:
                    logger.debug("Skills injection failed: %s", exc)

            # Build active integrations context
            integrations_context = self._build_integrations_context()

            return build_system_prompt(
                tool_names=tool_names,
                memory_context=memory_context,
                model_name=(
                    self.router.active_model if self.router else "unknown"
                ),
                skills_context=skills_context,
                tasks_context=tasks_text,
                integrations_context=integrations_context,
            )
        except Exception:
            return (
                "You are Axiom, an advanced AI assistant. "
                "Be helpful, precise, and action-oriented."
            )

    def _build_integrations_context(self) -> str:
        """Detect and describe active integrations for the system prompt."""
        lines: list[str] = []

        # Telegram bridge
        if getattr(self, "_telegram_active", False):
            lines.append(
                "🔗 **Telegram Bot**: CONNECTED and running. "
                "Messages sent on Telegram are mirrored here in the CLI. "
                "Both CLI and Telegram share the same conversation history "
                "via ConversationStore. The user can chat on either channel."
            )

        # Conversation store is always active
        lines.append(
            "💾 **Shared Conversation Store**: Active (JSONL persistence). "
            "Messages from all channels are stored and shared across sessions."
        )

        # Task store
        lines.append(
            "📋 **Task Memory**: Active. Tasks persist across restarts. "
            "The user can ask you to remember tasks, and you can track them."
        )

        return "\n".join(lines)

    async def _execute_tool_calls(
        self,
        tool_call_buffer: dict[int, dict[str, Any]],
        current_response: str,
        _depth: int = 0,
    ) -> str:
        """Execute accumulated tool calls and append results to messages.

        The ``_depth`` parameter guards against runaway recursive tool
        invocation.  Each follow-up completion that itself returns tool
        calls increments the depth; once ``MAX_TOOL_DEPTH`` is reached
        the agent stops calling tools and returns whatever text it has.
        """
        MAX_TOOL_DEPTH = 10
        if _depth >= MAX_TOOL_DEPTH:
            console.print(
                f"[{AXIOM_YELLOW}]Reached max tool depth ({MAX_TOOL_DEPTH}) "
                f"-- stopping tool chain.[/]"
            )
            return current_response

        if self.registry is None:
            console.print(
                f"[{AXIOM_YELLOW}]No tool registry -- cannot execute tools.[/]"
            )
            return current_response

        # Build the assistant message with tool_calls metadata
        assistant_tool_calls = []
        for idx in sorted(tool_call_buffer.keys()):
            buf = tool_call_buffer[idx]
            assistant_tool_calls.append({
                "id": buf["id"],
                "type": "function",
                "function": {
                    "name": buf["name"],
                    "arguments": buf["arguments"],
                },
            })

        # Add the assistant message that triggered the tool calls
        self.messages.append({
            "role": "assistant",
            "content": current_response or None,
            "tool_calls": assistant_tool_calls,
        })

        # Execute each tool call
        for tc in assistant_tool_calls:
            tool_name = tc["function"]["name"]
            raw_args = tc["function"]["arguments"]
            tool_call_id = tc["id"]

            try:
                tool_args = json.loads(raw_args) if raw_args else {}
            except (json.JSONDecodeError, TypeError):
                tool_args = {}

            # Look up the tool for risk-level check
            tool = self.registry.get(tool_name)
            risk = tool.risk_level if tool else "medium"
            desc = tool.description if tool else ""

            # ── Approval gate (using ToolApproval) ────────────────
            if self.tool_approval:
                approved, tool_args = self.tool_approval.request_approval(
                    tool_name=tool_name,
                    tool_args=tool_args,
                    risk_level=risk,
                    description=desc,
                )
                if not approved:
                    console.print(f"  [{AXIOM_DIM}]Skipped by user.[/]")
                    self.messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "content": "Tool call denied by user.",
                    })
                    continue
            elif not self.yolo_mode and risk != "low":
                # Fallback to basic approval
                approved = await self.input_handler.get_approval_async(tool_name, tool_args)
                if not approved:
                    console.print(f"  [{AXIOM_DIM}]Skipped by user.[/]")
                    self.messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "content": "Tool call denied by user.",
                    })
                    continue

            # Show tool call — compact by default, full panels only in /trace
            if self.trace_mode:
                self.renderer.show_tool_call(tool_name, tool_args, risk)
            else:
                self.renderer.show_tool_call_compact(tool_name, tool_args)

            # Execute the tool
            try:
                record = await self.registry.invoke(tool_name, **tool_args)

                if self.trace_mode:
                    self.renderer.show_tool_result(
                        tool_name,
                        record.result,
                        record.success,
                        int(record.duration_ms),
                    )
                else:
                    self.renderer.show_tool_result_compact(
                        tool_name,
                        record.result,
                        record.success,
                        int(record.duration_ms),
                    )

                # Append tool result to conversation
                self.messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": record.result,
                })

            except Exception as tool_exc:
                import traceback as _tb
                tb_str = _tb.format_exc()
                error_result = f"Tool '{tool_name}' crashed: {tool_exc}"
                # Errors always show full detail
                self.renderer.show_tool_result_compact(
                    tool_name, error_result, False, 0,
                )
                self.messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": error_result,
                })

        # After tool results are added, do a follow-up completion.
        # The follow-up itself may trigger MORE tool calls (recursive).
        # We loop up to MAX_TOOL_ROUNDS to prevent infinite recursion.
        MAX_TOOL_ROUNDS = 10
        for _round in range(MAX_TOOL_ROUNDS):
            self.renderer.start_thinking()
            followup_response = ""
            followup_tool_buffer: dict[int, dict[str, Any]] = {}

            try:
                tools_schemas = self.registry.to_llm_schemas()
                async for chunk in self.router.complete(
                    messages=self.messages,
                    tools=tools_schemas,
                    stream=True,
                ):
                    self.renderer.stop_thinking()
                    if not hasattr(chunk, "choices") or not chunk.choices:
                        continue
                    delta = chunk.choices[0].delta

                    if hasattr(delta, "content") and delta.content:
                        self.renderer.stream_token(delta.content)
                        followup_response += delta.content

                    # Accumulate any new tool calls from this follow-up
                    if hasattr(delta, "tool_calls") and delta.tool_calls:
                        for tc in delta.tool_calls:
                            idx = getattr(tc, "index", 0) or 0
                            if idx not in followup_tool_buffer:
                                followup_tool_buffer[idx] = {
                                    "id": "", "name": "", "arguments": "",
                                }
                            buf = followup_tool_buffer[idx]
                            if hasattr(tc, "id") and tc.id:
                                buf["id"] = tc.id
                            if hasattr(tc, "function") and tc.function:
                                if tc.function.name:
                                    buf["name"] = tc.function.name
                                if tc.function.arguments:
                                    buf["arguments"] += tc.function.arguments

                    # Check for finish reason
                    finish = getattr(chunk.choices[0], "finish_reason", None)
                    if finish == "tool_calls" and followup_tool_buffer:
                        self.renderer.finish_stream()
                        # Recursively execute these new tool calls (depth+1)
                        followup_response = await self._execute_tool_calls(
                            followup_tool_buffer, followup_response,
                            _depth=_depth + 1,
                        )
                        # _execute_tool_calls already handles further recursion
                        return followup_response or current_response

                self.renderer.finish_stream()
                return followup_response or current_response

            except Exception as exc:
                self.renderer.stop_thinking()
                self.renderer.finish_stream()
                self.renderer.show_error(f"Follow-up completion failed: {exc}")
                # ── GOD MODE: Capture follow-up failures ─────
                try:
                    from axiom.core.tools.self_repair import SelfRepairTool
                    SelfRepairTool.capture_error(exc, traceback.format_exc())
                except Exception:
                    pass
                return current_response

        # Safety: if we somehow exhaust all rounds
        return followup_response or current_response

    # ── Headless tool execution (for Telegram / API) ──────────────

    async def _execute_tool_calls_headless(
        self,
        tool_call_buffer: dict[int, dict[str, Any]],
        messages: list[dict[str, Any]],
        current_response: str,
        _depth: int = 0,
    ) -> str:
        """Execute tool calls without CLI rendering — for chat_headless().

        Same logic as ``_execute_tool_calls`` but:
        - Takes ``messages`` list as parameter (not self.messages)
        - Skips Rich rendering (no renderer calls)
        - Auto-approves all tool calls (no interactive approval)
        - Still handles recursive follow-up completions
        """
        MAX_TOOL_DEPTH = 10
        if _depth >= MAX_TOOL_DEPTH:
            logger.warning("Headless: max tool depth (%d) reached", MAX_TOOL_DEPTH)
            return current_response

        if self.registry is None:
            return current_response

        # Build the assistant message with tool_calls metadata
        assistant_tool_calls = []
        for idx in sorted(tool_call_buffer.keys()):
            buf = tool_call_buffer[idx]
            assistant_tool_calls.append({
                "id": buf["id"],
                "type": "function",
                "function": {
                    "name": buf["name"],
                    "arguments": buf["arguments"],
                },
            })

        # Add assistant message that triggered the tool calls
        messages.append({
            "role": "assistant",
            "content": current_response or None,
            "tool_calls": assistant_tool_calls,
        })

        # Execute each tool call (auto-approved for headless)
        for tc in assistant_tool_calls:
            tool_name = tc["function"]["name"]
            raw_args = tc["function"]["arguments"]
            tool_call_id = tc["id"]

            try:
                tool_args = json.loads(raw_args) if raw_args else {}
            except (json.JSONDecodeError, TypeError):
                tool_args = {}

            try:
                record = await self.registry.invoke(tool_name, **tool_args)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": record.result,
                })
            except Exception as tool_exc:
                error_result = f"Tool '{tool_name}' crashed: {tool_exc}"
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": error_result,
                })

        # Follow-up completion after tool results
        followup_response = ""
        followup_tool_buffer: dict[int, dict[str, Any]] = {}

        try:
            tools_schemas = self.registry.to_llm_schemas()
            async for chunk in self.router.complete(
                messages=messages,
                tools=tools_schemas,
                stream=True,
            ):
                if not hasattr(chunk, "choices") or not chunk.choices:
                    continue
                delta = chunk.choices[0].delta

                if hasattr(delta, "content") and delta.content:
                    followup_response += delta.content

                if hasattr(delta, "tool_calls") and delta.tool_calls:
                    for tc in delta.tool_calls:
                        idx = getattr(tc, "index", 0) or 0
                        if idx not in followup_tool_buffer:
                            followup_tool_buffer[idx] = {
                                "id": "", "name": "", "arguments": "",
                            }
                        buf = followup_tool_buffer[idx]
                        if hasattr(tc, "id") and tc.id:
                            buf["id"] = tc.id
                        if hasattr(tc, "function") and tc.function:
                            if tc.function.name:
                                buf["name"] = tc.function.name
                            if tc.function.arguments:
                                buf["arguments"] += tc.function.arguments

                finish = getattr(chunk.choices[0], "finish_reason", None)
                if finish == "tool_calls" and followup_tool_buffer:
                    followup_response = await self._execute_tool_calls_headless(
                        followup_tool_buffer, messages,
                        followup_response, _depth=_depth + 1,
                    )
                    return followup_response or current_response

            return followup_response or current_response

        except Exception as exc:
            logger.error("Headless follow-up failed: %s", exc)
            return current_response

    # ── Run modes ─────────────────────────────────────────────────

    async def run_once(self, task: str) -> None:
        """One-shot mode: execute a single task and exit.

        ``axiom "do something"`` — runs the task then exits.
        """
        self._init_renderer()
        self._init_input()
        self._init_router()

        # Validate the default model (fallback if it's broken)
        validated = await self._validate_startup_model()

        # Memory (graceful)
        try:
            self._init_memory()
        except Exception as exc:
            logger.debug("Memory init failed: %s", exc)

        # Tools (graceful)
        try:
            self._init_tools()
        except Exception as exc:
            logger.debug("Tool init failed: %s", exc)

        self._init_tool_approval()

        # Skills (graceful)
        try:
            self._init_skills()
        except Exception as exc:
            logger.debug("Skills init failed: %s", exc)

        self._init_tracer()

        try:
            await self._init_mcp_bridge()
        except Exception:
            pass

        console.print(f"[{AXIOM_CYAN}]Axiom[/] [{AXIOM_DIM}]({validated})[/] executing: {task}\n")

        try:
            await self.chat(task)
        except KeyboardInterrupt:
            console.print(f"\n[{AXIOM_DIM}]Interrupted.[/]")
        except Exception as exc:
            self.renderer.show_error(str(exc))

    async def run_interactive(
        self,
        model: Optional[str] = None,
        yolo: bool = False,
        visible: bool = False,
        voice: bool = False,
        offline: bool = False,
        trace: bool = False,
        telegram: bool = False,
    ) -> None:
        """Interactive REPL mode."""
        self._init_renderer()
        self._init_input()
        self._init_router(model=model, offline=offline)
        self.yolo_mode = yolo
        self.trace_mode = trace

        # If --trace, lower the log level so debug messages appear
        if trace:
            logging.getLogger("axiom").setLevel(logging.DEBUG)

        # Validate default model (probes with a tiny request; auto-fallback)
        validated = await self._validate_startup_model()

        # Memory must come before tools (memory tools depend on it)
        try:
            memory_count = self._init_memory()
        except Exception as exc:
            logger.debug("Memory init failed: %s", exc)
            memory_count = 0

        # Attempt to load tools (graceful if some fail)
        try:
            tool_count = self._init_tools()
        except Exception as exc:
            logger.debug("Tool init failed: %s", exc)
            tool_count = 0

        # Tool approval system
        self._init_tool_approval()

        # Skills system (load SKILL.md files for domain knowledge injection)
        try:
            skill_count = self._init_skills()
        except Exception as exc:
            logger.debug("Skills init failed: %s", exc)
            skill_count = 0

        # Agent tracer for observability
        self._init_tracer()

        # MCP bridge (discover + bridge external tool servers)
        mcp_tool_count = 0
        try:
            mcp_tool_count = await self._init_mcp_bridge()
            if mcp_tool_count > 0:
                tool_count += mcp_tool_count
        except Exception as exc:
            logger.debug("MCP bridge init failed: %s", exc)

        # Voice input (if requested)
        if voice:
            self._init_voice()
            self.voice_mode = True

        # ── Telegram bot bridge (optional) ─────────────────────────
        # telegram_bot is self._telegram_bot (instance var for hot-connect)
        telegram_enabled = telegram or self.settings.TELEGRAM_ENABLED
        if telegram_enabled and self.settings.TELEGRAM_BOT_TOKEN:
            try:
                from axiom.integrations.telegram.bridge import TelegramBridge
                from axiom.integrations.telegram.handler import TelegramBot

                # Parse allowed user IDs from comma-separated string
                allowed_users = None
                if self.settings.TELEGRAM_ALLOWED_USERS:
                    try:
                        allowed_users = {
                            int(uid.strip())
                            for uid in self.settings.TELEGRAM_ALLOWED_USERS.split(",")
                            if uid.strip()
                        }
                    except ValueError:
                        logger.warning("Invalid TELEGRAM_ALLOWED_USERS format")

                tg_bridge = TelegramBridge(app=self)
                self._telegram_bot = TelegramBot(
                    token=self.settings.TELEGRAM_BOT_TOKEN,
                    bridge=tg_bridge,
                    allowed_users=allowed_users,
                )
                await self._telegram_bot.start()
                self._telegram_active = True
                console.print(f"  [{AXIOM_GREEN}]✓ Telegram bot online (mirrored)[/]")
            except Exception as exc:
                logger.warning("Telegram init failed: %s", exc)
                console.print(
                    f"  [{AXIOM_YELLOW}]⚠ Telegram failed: {str(exc)[:80]}[/]"
                )
                self._telegram_bot = None

        # ── Heartbeat daemon (optional) ────────────────────────────
        heartbeat_daemon = None
        if self.settings.HEARTBEAT_ENABLED:
            try:
                from axiom.integrations.heartbeat.daemon import HeartbeatDaemon

                heartbeat_file = Path("workspace/HEARTBEAT.md")
                if heartbeat_file.exists():
                    heartbeat_daemon = HeartbeatDaemon(
                        heartbeat_file=str(heartbeat_file),
                    )
                    await heartbeat_daemon.start()
                    checks = heartbeat_daemon.status
                    console.print(
                        f"  [{AXIOM_GREEN}]✓ Heartbeat daemon "
                        f"({checks['total_checks']} checks)[/]"
                    )
                else:
                    console.print(
                        f"  [{AXIOM_DIM}]Heartbeat: no workspace/HEARTBEAT.md found[/]"
                    )
            except Exception as exc:
                logger.warning("Heartbeat init failed: %s", exc)
                heartbeat_daemon = None

        # ── Session resume + task display ────────────────────────
        try:
            conv_stats = await self.conversation_store.get_stats()
            if conv_stats["messages"] > 0:
                console.print(
                    f"  [{AXIOM_DIM}]Resuming conversation "
                    f"({conv_stats['messages']} messages)...[/]"
                )
        except Exception:
            pass

        try:
            pending = await self.task_store.get_pending()
            if pending:
                console.print(
                    f"  [{AXIOM_YELLOW}]📋 {len(pending)} pending tasks "
                    f"from previous session[/]"
                )
                for t in pending[:5]:
                    console.print(
                        f"    [{AXIOM_DIM}]#{t['id']}: {t['description']}[/]"
                    )
        except Exception:
            pass

        # ── MessageBus subscriber: show Telegram messages in CLI ──
        async def _on_remote_message(msg: dict) -> None:
            ch = msg.get("channel", "")
            role = msg.get("role", "")
            content = msg.get("content", "")

            if ch == "cli":
                return  # Don't echo our own messages

            if role == "user":
                display = content[:80]
                ellipsis = "..." if len(content) > 80 else ""
                console.print(
                    f"\n  [{AXIOM_CYAN}]📱 {ch}: {display}{ellipsis}[/]"
                )
            elif role == "assistant":
                display = content[:120]
                ellipsis = "..." if len(content) > 120 else ""
                console.print(
                    Panel(
                        Text.from_markup(
                            f"[{AXIOM_PURPLE}]{display}{ellipsis}[/]"
                        ),
                        title=Text.from_markup(
                            f"[bold {AXIOM_PURPLE}]🤖 axiom → {ch}[/]"
                        ),
                        title_align="left",
                        border_style=AXIOM_PURPLE,
                        padding=(0, 1),
                        expand=False,
                    )
                )

        self.message_bus.subscribe("cli", _on_remote_message)

        # Print startup banner
        print_banner(
            model_name=self.router.active_model,
            tool_count=tool_count,
            memory_count=memory_count,
            skill_count=skill_count,
            mcp_count=mcp_tool_count,
            telegram_active=self._telegram_active,
        )

        # Hint for first-time users
        console.print(
            f"  [{AXIOM_DIM}]Type a message to chat, or /help for commands.[/]\n"
        )

        # ── Main REPL loop ────────────────────────────────────────
        while True:
            try:
                # Voice input mode
                if self.voice_mode and self.voice:
                    console.print(
                        f"  [{AXIOM_CYAN}]🎤 Listening... "
                        f"(speak now, or type text)[/]"
                    )
                    user_input = self.voice.listen(duration=10.0)
                    if user_input:
                        console.print(
                            f"  [{AXIOM_DIM}]Heard: {user_input}[/]"
                        )
                    else:
                        # Fallback to text input
                        user_input = await self.input_handler.get_input_async()
                else:
                    user_input = await self.input_handler.get_input_async()

                if not user_input or not user_input.strip():
                    continue

                stripped = user_input.strip()

                # InputHandler returns "/exit" on EOF
                if stripped in ("/exit", "/quit"):
                    console.print(f"[{AXIOM_DIM}]Saving session... Goodbye![/]")
                    break

                # Slash commands
                if stripped.startswith("/"):
                    should_exit = await self.handle_command(stripped)
                    if should_exit:
                        break
                    continue

                # Regular chat turn
                await self.chat(stripped)

            except KeyboardInterrupt:
                console.print(f"\n[{AXIOM_DIM}]Use /exit to quit.[/]")
            except EOFError:
                break

        # ── Shutdown ──────────────────────────────────────────────
        elapsed = time.time() - self.session_start
        minutes = int(elapsed // 60)
        seconds = int(elapsed % 60)
        usage = self.router.get_usage()

        # Save session to memory
        if self.memory is not None and self.messages:
            try:
                summary = (
                    f"Session {self.session_id}: {len(self.messages)} messages, "
                    f"{minutes}m {seconds}s, model={self.router.active_model}, "
                    f"cost=${usage['cost']:.4f}"
                )
                self.memory.save_session(
                    summary=summary,
                    session_id=self.session_id,
                )
                console.print(f"[{AXIOM_GREEN}]Session saved to memory.[/]")
            except Exception as exc:
                logger.debug("Session save failed: %s", exc)

        # Stop Telegram bot if running
        if self._telegram_bot is not None:
            try:
                await self._telegram_bot.stop()
                console.print(f"[{AXIOM_DIM}]Telegram bot stopped.[/]")
            except Exception as exc:
                logger.debug("Telegram stop error: %s", exc)

        # Stop Heartbeat daemon if running
        if heartbeat_daemon is not None:
            try:
                await heartbeat_daemon.stop()
                console.print(f"[{AXIOM_DIM}]Heartbeat daemon stopped.[/]")
            except Exception as exc:
                logger.debug("Heartbeat stop error: %s", exc)

        # Clean up browser if it was used
        try:
            from axiom.core.tools.browser import _close_browser

            await _close_browser()
        except Exception:
            pass

        console.print(
            f"[{AXIOM_DIM}]Session: {len(self.messages)} messages, "
            f"{minutes}m {seconds}s, "
            f"${usage['cost']:.4f}. Until next time.[/]"
        )


# ── Fallback InputHandler (bare input() for non-TTY environments) ────────────


class _FallbackInputHandler:
    """Minimal prompt handler used when prompt_toolkit cannot initialise."""

    def get_input(self) -> str:
        """Block until the user submits input (sync)."""
        try:
            return input("you > ").strip()
        except KeyboardInterrupt:
            return ""
        except EOFError:
            return "/exit"

    async def get_input_async(self) -> str:
        """Async version — runs blocking input() in a thread."""
        loop = asyncio.get_event_loop()
        try:
            text = await loop.run_in_executor(None, lambda: input("you > "))
            return text.strip()
        except KeyboardInterrupt:
            return ""
        except EOFError:
            return "/exit"

    def get_approval(self, tool_name: str, tool_args: dict[str, Any]) -> bool:
        """Ask for tool-call approval (sync)."""
        try:
            response = input(
                f"  Allow {tool_name}? [Y/n] "
            ).strip().lower()
            return response in ("", "y", "yes")
        except (EOFError, KeyboardInterrupt):
            return False

    async def get_approval_async(
        self, tool_name: str, tool_args: dict[str, Any]
    ) -> bool:
        """Async version — runs blocking input() in a thread."""
        loop = asyncio.get_event_loop()
        try:
            prompt_str = f"  Allow {tool_name}? [Y/n] "
            response = await loop.run_in_executor(
                None, lambda: input(prompt_str).strip().lower()
            )
            return response in ("", "y", "yes")
        except (EOFError, KeyboardInterrupt):
            return False
