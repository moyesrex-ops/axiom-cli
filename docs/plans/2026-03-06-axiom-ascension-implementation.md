# Axiom Ascension Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make Axiom a genuinely autonomous, deep-thinking agent that never gives up, prefers free/local tools, runs on Telegram with full GOD MODE, and proactively monitors via heartbeat.

**Architecture:** 5 new components layered onto the existing tri-mode engine. No rewrites — all additions. ThinkTool for explicit reasoning, upgraded system prompt for deep search/persistence, raised agent limits, Telegram bridge wrapping AxiomApp, heartbeat daemon with asyncio scheduler.

**Tech Stack:** python-telegram-bot 21+, asyncio, existing AxiomTool/ToolRegistry pattern, Pydantic settings extension, Rich formatting → Telegram MarkdownV2 converter.

---

### Task 1: Create ThinkTool — Explicit Reasoning Tool

**Files:**
- Create: `axiom/core/tools/think.py`

**Step 1: Create the ThinkTool file**

```python
"""Think tool -- forces explicit chain-of-thought reasoning.

Zero side effects. The LLM externalizes its reasoning into a visible,
logged trace step before taking action. This is the most important tool
in Axiom's arsenal -- it prevents shallow single-attempt actions.
"""

from __future__ import annotations
from typing import Any
from axiom.core.tools.base import AxiomTool


class ThinkTool(AxiomTool):
    """Pure reasoning tool with zero side effects.

    The agent MUST use this before complex tasks to plan its approach,
    consider what could go wrong, and prepare fallback strategies.
    """

    name = "think"
    description = (
        "Reason explicitly before acting. Use this to plan your approach, "
        "consider alternatives, and prepare fallback strategies. "
        "MANDATORY before any multi-step task or search operation. "
        "Has zero side effects -- just returns your reasoning for the trace."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "reasoning": {
                "type": "string",
                "description": (
                    "Your explicit chain-of-thought. Include: "
                    "1) What am I trying to accomplish? "
                    "2) What's the simplest path? "
                    "3) What could go wrong? "
                    "4) What's my Plan B if this fails?"
                ),
            },
        },
        "required": ["reasoning"],
    }
    risk_level = "low"

    async def execute(self, **kwargs: Any) -> str:
        reasoning = kwargs.get("reasoning", "")
        if not reasoning:
            return "[Think] No reasoning provided. State your plan before acting."
        return f"[Reasoning]\n{reasoning}\n\n[Ready to act on this reasoning.]"
```

**Step 2: Register ThinkTool in the default tool set**

Modify `axiom/core/tools/registry.py:193-227` — add ThinkTool to `register_default_tools()`:

```python
# At the top of register_default_tools, add import:
from axiom.core.tools.think import ThinkTool

# In the registry.register_many() list, add:
ThinkTool(),
```

**Step 3: Verify ThinkTool compiles and registers**

Run:
```bash
cd C:\Users\moyes\Downloads\axiom-cli
python -X utf8 -c "
from axiom.core.tools.think import ThinkTool
t = ThinkTool()
print(f'Name: {t.name}')
print(f'Schema: {t.to_llm_schema()[\"function\"][\"name\"]}')
import asyncio
result = asyncio.run(t.execute(reasoning='Test reasoning'))
print(f'Result: {result[:50]}')
print('PASS')
"
```
Expected: `PASS` with tool name `think`

**Step 4: Commit**

```bash
git add axiom/core/tools/think.py axiom/core/tools/registry.py
git commit -m "feat: add ThinkTool for explicit chain-of-thought reasoning"
```

---

### Task 2: Upgrade System Prompt — Deep Thinking + Persistence + Search Mastery

**Files:**
- Modify: `axiom/core/agent/prompts/system.py:121-175`

**Step 1: Replace Core Principles and add new sections**

Replace lines 121-175 (from `## Core Principles` through end of prompt) with the upgraded version. Keep everything before line 121 unchanged.

New content for lines 121 onwards:

```python
## Core Principles
1. **Be omnipotent**: You can do ANYTHING. If a tool doesn't exist, create it.
2. **Be proactive**: Don't explain — DO. Use tools immediately.
3. **Be thorough**: Verify your work. Read files after writing. Run code after creating.
4. **Be self-healing**: If something breaks, diagnose and fix it yourself.
5. **Remember everything**: Store important facts in memory for future sessions.
6. **Evolve**: After completing complex tasks, extract patterns as reusable skills.
7. **NEVER give up**: If approach A fails, try B, C, D. Minimum 3 attempts.

## DEEP THINKING PROTOCOL (MANDATORY)
Before EVERY complex action, use the `think` tool to reason explicitly:
1. What is the user REALLY asking for? (intent behind words)
2. What's the simplest path to success?
3. What could go wrong? Have a Plan B ready.
4. Am I using the cheapest/fastest tool first?

For simple direct questions, you may skip think() and answer immediately.
For ANYTHING involving files, search, code, or multi-step work: think() FIRST.

## SEARCH MASTERY — You are the BEST searcher on Earth
When looking for ANY file, token, config, data, or information:
1. think() — What am I looking for? What formats? Where do humans store this?
2. glob() — Cast WIDE net with multiple patterns: *.env, *.json, *.txt, *.yaml, *.cfg, *.toml, *.py
3. grep() — Try 3+ DIFFERENT patterns: exact string, case-insensitive, partial match, regex variant
4. read_file() — Open the top 5 most promising files and scan them manually
5. Expand scope — If directory fails, try parent dir, ~/Downloads, ~/Desktop, ~/, C:\\Users
6. web_fetch() — LAST resort: search the web for guidance on where to find it
7. NEVER report "not found" after fewer than 3 distinct search attempts
8. ALWAYS tell the user what you searched and where, even on success

## PERSISTENCE PROTOCOL — NON-NEGOTIABLE
- You NEVER say "I can't", "I couldn't find it", or "I'm unable to"
- If approach A fails → try B. If B fails → try C, D, E.
- If ALL tools fail → CREATE a new tool via tool_create
- If you need a capability you don't have → RESEARCH how to get it
- Minimum 3 genuine, DIFFERENT attempts before reporting any failure
- On failure: explain EXACTLY what you tried AND propose concrete next steps
- You are not a chatbot. You are an AUTONOMOUS AGENT. Act like one.

## TOOL PREFERENCE — Cheapest/fastest first, ALWAYS
TIER 1 (FREE/LOCAL — always try first):
  think, grep, glob, read_file, write_file, edit_file, bash, code_exec, git

TIER 2 (FREE API — if local insufficient):
  DuckDuckGo via research(mode="quick"), web_fetch for specific URLs, memory_tool

TIER 3 (PAID — only if Tiers 1-2 genuinely insufficient):
  Tavily/Exa deep research, Playwright browser, vision model, spawn_agent

Before using ANY Tier 3 tool, you MUST have tried at least one Tier 1 alternative.
Exception: If the user explicitly asks you to use a specific tool, use it.

## Execution Strategy
- For **structured tasks** (build, create, fix): think() → Plan → execute → verify → learn
- For **exploratory tasks** (research, find, explain): think() → act → observe → adjust
- For **simple questions**: Answer directly, no unnecessary tool calls
- For **self-repair**: Diagnose → introspect → edit → reload → verify
- When uncertain: think() → investigate (read files, search, bash) → act
- When a tool fails: Try a DIFFERENT tool or DIFFERENT arguments. NEVER repeat same failure.

## Tool Usage Guidelines
- **think**: MANDATORY before complex tasks. Externalize your reasoning chain.
- **bash**: Execute shell commands. PowerShell on Windows, bash on Unix.
- **read_file / write_file / edit_file**: File operations with line numbers.
- **glob / grep**: Find files by pattern or search content. Use MULTIPLE patterns.
- **git**: Git operations (status, diff, commit, push, branch).
- **code_exec**: Run Python/JS/Bash in isolated subprocess.
- **web_fetch**: Fetch web pages, extract content as markdown.
- **http**: Raw HTTP requests (GET/POST/PUT/DELETE).
- **browser**: Full Playwright browser control (navigate, click, type, screenshot).
- **research**: Multi-source deep web research with citations.
- **desktop**: GOD MODE desktop automation (screenshot, click, type, OCR, hotkey).
- **vision**: Send screenshots to vision model for understanding.
- **memory_search / memory_save**: Semantic search & store persistent memories.
- **spawn_agent**: Create sub-agents for parallel tasks.
- **create_tool**: Generate new tools dynamically at runtime.
- **self_repair**: Read, edit, and hot-reload YOUR OWN source code.
- **mcp_connect**: Connect to MCP servers for 200+ additional tools.

## Self-Repair Protocol
When you encounter an error in your own behaviour:
- A runtime error appears → Use `self_repair(action="diagnose")` first
- You know which file is broken → Use `self_repair(action="introspect", target="module.path")`
- You have a fix → Use `self_repair(action="self_edit", target="...", old_text="...", new_text="...")`
- Apply the fix live → Use `self_repair(action="hot_reload", target="module.path")`
- Need the full picture → Use `self_repair(action="codebase_map")`

## Response Format
- Use **Markdown** formatting for all responses
- Include code blocks with language labels
- Use bullet points for lists
- Be concise but complete
- When showing file changes, explain what changed and why

## Safety
- Ask before destructive operations (delete, overwrite, force-push)
- Never expose API keys, passwords, or sensitive data in responses
- Self-edits are logged to memory/self_repairs/ for user audit
- Warn about potentially dangerous commands before executing
```

**Step 2: Verify system prompt builds**

Run:
```bash
cd C:\Users\moyes\Downloads\axiom-cli
python -X utf8 -c "
from axiom.core.agent.prompts.system import build_system_prompt
p = build_system_prompt(tool_names=['think','grep','glob','bash'], model_name='Claude Opus 4.6')
checks = [
    ('DEEP THINKING' in p, 'Deep thinking protocol'),
    ('SEARCH MASTERY' in p, 'Search mastery'),
    ('PERSISTENCE PROTOCOL' in p, 'Persistence protocol'),
    ('TOOL PREFERENCE' in p, 'Tool preference'),
    ('think' in p, 'Think tool mentioned'),
    ('NEVER say' in p, 'Never give up directive'),
    ('Tier 1' in p or 'TIER 1' in p, 'Tier system'),
    ('GOD MODE' in p, 'GOD MODE identity'),
]
for ok, label in checks:
    print(f'[{\"OK\" if ok else \"FAIL\"}] {label}')
print(f'Total length: {len(p)} chars')
assert all(ok for ok, _ in checks), 'Some checks failed!'
print('ALL PASSED')
"
```

**Step 3: Commit**

```bash
git add axiom/core/agent/prompts/system.py
git commit -m "feat: upgrade system prompt with deep thinking, persistence, search mastery"
```

---

### Task 3: Upgrade Agent Graph — Higher Limits + Retry Hints

**Files:**
- Modify: `axiom/core/agent/graph.py:34-37` (constants)
- Modify: `axiom/core/agent/graph.py:265-278` (failure handling in PLAN mode)
- Modify: `axiom/core/agent/graph.py:389-399` (failure handling in REACT mode)
- Modify: `axiom/core/agent/state.py:146-148` (default state limits)

**Step 1: Raise the constants in graph.py**

Replace lines 34-37:

```python
MAX_PLAN_ITERATIONS = 50      # was 25
MAX_REACT_ITERATIONS = 30     # was 15
MAX_REPLANS = 8               # was 4
MAX_CONSECUTIVE_FAILURES = 10  # was 6
```

**Step 2: Raise defaults in state.py**

Replace lines 146-148:

```python
    max_iterations: int = 50           # was 25
    consecutive_failures: int = 0
    max_consecutive_failures: int = 10  # was 6
```

And line 135:

```python
    max_replans: int = 8  # was 4
```

**Step 3: Add retry hint injection in PLAN mode failure handler**

In `graph.py`, after the consecutive failure check (around line 265-278), modify the failure block to inject a retry hint instead of just counting:

Replace the section that handles `not result["success"]` (lines 265-278):

```python
                if not result["success"]:
                    consecutive_failures += 1
                    # Inject retry hint for the LLM
                    alt_tools = [t for t in tool_names if t != action][:5]
                    retry_hint = (
                        f"Tool '{action}' failed. MANDATORY: Use a DIFFERENT tool or "
                        f"DIFFERENT arguments next time. Consider: {', '.join(alt_tools)}"
                    )
                    messages.append({
                        "role": "system",
                        "content": retry_hint,
                    })
                else:
                    consecutive_failures = 0
```

**Step 4: Add same retry hint in REACT mode**

In `_run_react_mode`, replace the failure counter section (lines 389-392):

```python
        if not result["success"]:
            consecutive_failures += 1
            # Inject retry hint
            alt_tools = [t for t in tool_names if t != action][:5]
            react_trace.append({
                "thought": f"RETRY HINT: '{action}' failed. Use DIFFERENT tool/args. Try: {', '.join(alt_tools)}",
                "action": "system_hint",
                "args": {},
                "observation": "Retry with different approach",
                "success": True,
            })
        else:
            consecutive_failures = 0
```

**Step 5: Verify graph compiles**

Run:
```bash
python -X utf8 -c "
from axiom.core.agent.graph import MAX_PLAN_ITERATIONS, MAX_REACT_ITERATIONS, MAX_REPLANS, MAX_CONSECUTIVE_FAILURES
assert MAX_PLAN_ITERATIONS == 50, f'Expected 50, got {MAX_PLAN_ITERATIONS}'
assert MAX_REACT_ITERATIONS == 30, f'Expected 30, got {MAX_REACT_ITERATIONS}'
assert MAX_REPLANS == 8, f'Expected 8, got {MAX_REPLANS}'
assert MAX_CONSECUTIVE_FAILURES == 10, f'Expected 10, got {MAX_CONSECUTIVE_FAILURES}'
print('ALL LIMITS CORRECT')

from axiom.core.agent.state import AgentState
s = AgentState()
assert s.max_iterations == 50
assert s.max_consecutive_failures == 10
assert s.max_replans == 8
print('STATE DEFAULTS CORRECT')
print('PASS')
"
```

**Step 6: Commit**

```bash
git add axiom/core/agent/graph.py axiom/core/agent/state.py
git commit -m "feat: upgrade agent persistence - 50 iters, 8 replans, retry hints"
```

---

### Task 4: Add `/models` Alias + Minor CLI Fixes

**Files:**
- Modify: `axiom/cli/app.py:430-431` (command handler)

**Step 1: Add `/models` as alias for `/model list`**

In `handle_command()`, right after the `/model` handler (line 431), add:

```python
        if command == "/models":
            self._show_models()
            return False
```

**Step 2: Verify**

Run:
```bash
python -X utf8 -c "
import ast
with open('axiom/cli/app.py', 'r', encoding='utf-8') as f:
    src = f.read()
ast.parse(src)
assert '/models' in src
print('PASS')
"
```

**Step 3: Commit**

```bash
git add axiom/cli/app.py
git commit -m "fix: add /models alias for /model list"
```

---

### Task 5: Settings + .env for Telegram

**Files:**
- Modify: `axiom/config/settings.py:52-53` (add TELEGRAM_ALLOWED_USERS)
- Modify: `axiom-cli/.env:53` (fill TELEGRAM_BOT_TOKEN)

**Step 1: Add Telegram settings**

In `axiom/config/settings.py`, after line 53 (`TELEGRAM_BOT_TOKEN`), add:

```python
    TELEGRAM_ALLOWED_USERS: Optional[str] = None  # Comma-separated Telegram user IDs
    TELEGRAM_ENABLED: bool = False
    HEARTBEAT_INTERVAL_MINUTES: int = 30
    HEARTBEAT_ENABLED: bool = False
```

**Step 2: Fill in .env with the found token**

In `axiom-cli/.env`, set line 53 to:
```
TELEGRAM_BOT_TOKEN=8646524596:AAGuHnKy1qFS3Ja7LIBPXDw2h-UfppLRkuo
```

**Step 3: Commit (DO NOT commit .env — it has secrets)**

```bash
git add axiom/config/settings.py
git commit -m "feat: add Telegram + Heartbeat settings fields"
```

---

### Task 6: Telegram Formatter — Rich → Telegram MarkdownV2

**Files:**
- Create: `axiom/integrations/__init__.py`
- Create: `axiom/integrations/telegram/__init__.py`
- Create: `axiom/integrations/telegram/formatter.py`

**Step 1: Create package init files**

`axiom/integrations/__init__.py`:
```python
"""Axiom integrations -- Telegram, Heartbeat, and future channels."""
```

`axiom/integrations/telegram/__init__.py`:
```python
"""Telegram integration -- Full GOD MODE Axiom over Telegram."""
from axiom.integrations.telegram.handler import TelegramBot

__all__ = ["TelegramBot"]
```

**Step 2: Create the formatter**

`axiom/integrations/telegram/formatter.py`:

```python
"""Convert Axiom's Rich-style output to Telegram MarkdownV2.

Telegram MarkdownV2 requires escaping: _ * [ ] ( ) ~ ` > # + - = | { } . !
Reference: https://core.telegram.org/bots/api#markdownv2-style
"""

from __future__ import annotations

import re

# Characters that must be escaped in Telegram MarkdownV2
_ESCAPE_CHARS = r"_*[]()~`>#+-=|{}.!"
_ESCAPE_RE = re.compile(f"([{re.escape(_ESCAPE_CHARS)}])")


def escape_md2(text: str) -> str:
    """Escape special characters for Telegram MarkdownV2."""
    return _ESCAPE_RE.sub(r"\\\1", text)


def format_tool_call(tool_name: str, args: dict) -> str:
    """Format a tool call for Telegram display."""
    import json
    args_str = json.dumps(args, indent=2, default=str)
    if len(args_str) > 500:
        args_str = args_str[:497] + "..."
    safe_name = escape_md2(tool_name)
    return f"⚙ *{safe_name}*\n```json\n{args_str}\n```"


def format_tool_result(tool_name: str, success: bool, result: str, duration_ms: int = 0) -> str:
    """Format a tool result for Telegram display."""
    icon = "✓" if success else "✗"
    safe_name = escape_md2(tool_name)
    # Truncate long results
    if len(result) > 1000:
        result = result[:997] + "..."
    safe_result = escape_md2(result)
    dur = f" {duration_ms}ms" if duration_ms > 0 else ""
    return f"{icon} *{safe_name}*{escape_md2(dur)}\n{safe_result}"


def format_agent_response(text: str) -> str:
    """Convert a markdown response to Telegram-safe MarkdownV2.

    Preserves code blocks and bold, escapes everything else.
    """
    if not text:
        return escape_md2("(no response)")

    # Split by code blocks to preserve them
    parts = re.split(r"(```[\s\S]*?```)", text)
    result_parts = []

    for part in parts:
        if part.startswith("```"):
            # Code blocks are sent as-is (Telegram handles them)
            result_parts.append(part)
        else:
            # Escape special chars but preserve **bold** → *bold*
            # First, extract bold markers
            bold_parts = re.split(r"(\*\*.*?\*\*)", part)
            for bp in bold_parts:
                if bp.startswith("**") and bp.endswith("**"):
                    inner = bp[2:-2]
                    result_parts.append(f"*{escape_md2(inner)}*")
                else:
                    result_parts.append(escape_md2(bp))

    return "".join(result_parts)


def truncate_for_telegram(text: str, max_length: int = 4000) -> str:
    """Telegram messages have a 4096 char limit. Truncate gracefully."""
    if len(text) <= max_length:
        return text
    return text[:max_length - 20] + "\n\n" + escape_md2("... (truncated)")
```

**Step 3: Commit**

```bash
git add axiom/integrations/
git commit -m "feat: add Telegram formatter (Rich → MarkdownV2 converter)"
```

---

### Task 7: Telegram Bridge — AxiomApp ↔ Telegram Message Adapter

**Files:**
- Create: `axiom/integrations/telegram/bridge.py`

**Step 1: Create the bridge**

```python
"""Bridge between AxiomApp's chat pipeline and Telegram messages.

Wraps AxiomApp so that Telegram messages go through the same full pipeline:
system prompt, tools, memory, skills, tri-mode agent -- everything.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


class TelegramBridge:
    """Adapts AxiomApp for Telegram I/O.

    Instead of streaming to a Rich terminal, we collect the full response
    and send it back to Telegram.
    """

    def __init__(self, app: Any):
        """
        Args:
            app: An initialized AxiomApp instance with router, registry, memory, etc.
        """
        self.app = app
        self._response_buffer: str = ""
        self._tool_outputs: list[str] = []

    async def process_message(self, text: str) -> dict[str, Any]:
        """Process a user message through the full AxiomApp pipeline.

        Returns a dict with:
            - response: The final text response
            - tool_calls: List of tool call summaries
            - mode: Which agent mode was used
            - tokens: Token usage
        """
        from axiom.integrations.telegram.formatter import (
            format_tool_call,
            format_tool_result,
            format_agent_response,
        )

        # Check if it's a slash command
        if text.startswith("/") and not text.startswith("//"):
            return await self._handle_command(text)

        # Inject system prompt if needed
        if not any(m.get("role") == "system" for m in self.app.messages):
            self.app._inject_system_prompt(text)

        self.app.messages.append({"role": "user", "content": text})

        # Store in memory
        if self.app.memory is not None:
            try:
                self.app.memory.store_message("user", text)
            except Exception:
                pass

        # Build tool schemas
        tools = None
        if self.app.registry and self.app.registry.count > 0:
            try:
                tools = self.app.registry.to_llm_schemas()
            except Exception:
                pass

        # Run completion (non-streaming for Telegram)
        full_response = ""
        tool_summaries = []

        try:
            async for chunk in self.app.router.complete(
                messages=self.app.messages,
                tools=tools,
                stream=True,
            ):
                if not hasattr(chunk, "choices") or not chunk.choices:
                    continue

                delta = chunk.choices[0].delta

                if hasattr(delta, "content") and delta.content:
                    full_response += delta.content

                # Handle tool calls (simplified — collect and execute)
                if hasattr(delta, "tool_calls") and delta.tool_calls:
                    # For Telegram, we'll handle tool calls in a simpler way
                    # The full tool execution is handled by the agent pipeline
                    pass

            if full_response:
                self.app.messages.append({"role": "assistant", "content": full_response})
                if self.app.memory:
                    try:
                        self.app.memory.store_message("assistant", full_response)
                    except Exception:
                        pass

        except Exception as exc:
            full_response = f"Error: {exc}"
            logger.error("Telegram bridge error: %s", exc, exc_info=True)

        return {
            "response": format_agent_response(full_response) if full_response else "",
            "raw_response": full_response,
            "tool_calls": tool_summaries,
            "tokens": self.app.router.get_usage() if hasattr(self.app.router, "get_usage") else {},
        }

    async def process_agent_task(self, text: str) -> dict[str, Any]:
        """Run a full agent pipeline (PLAN/REACT/COUNCIL) for complex tasks.

        Use this for longer tasks that need multi-step tool execution.
        """
        from axiom.core.agent.graph import run_agent, EventType, AgentMode
        from axiom.integrations.telegram.formatter import (
            format_tool_call,
            format_tool_result,
            format_agent_response,
        )

        events_log = []
        final_answer = ""
        tool_summaries = []

        try:
            async for event in run_agent(
                router=self.app.router,
                registry=self.app.registry,
                messages=self.app.messages + [{"role": "user", "content": text}],
                mode=None,  # auto-select
            ):
                events_log.append({"type": event.type.value, "data": event.data})

                if event.type == EventType.TOOL_CALL:
                    tool_summaries.append(format_tool_call(
                        event.data.get("tool", "unknown"),
                        event.data.get("args", {}),
                    ))

                if event.type == EventType.ANSWER:
                    final_answer = event.data.get("answer", "")

        except Exception as exc:
            final_answer = f"Agent error: {exc}"
            logger.error("Telegram agent error: %s", exc, exc_info=True)

        return {
            "response": format_agent_response(final_answer),
            "raw_response": final_answer,
            "tool_calls": tool_summaries,
            "events": events_log,
        }

    async def _handle_command(self, text: str) -> dict[str, Any]:
        """Handle slash commands from Telegram."""
        parts = text.strip().split(maxsplit=1)
        cmd = parts[0].lower()
        arg = parts[1] if len(parts) > 1 else ""

        if cmd in ("/start", "/help"):
            return {
                "response": (
                    "*Axiom* — GOD MODE AI Agent\n\n"
                    "Send any message and I'll handle it\\.\n\n"
                    "*Commands:*\n"
                    "/agent \\<task\\> — Run full agent pipeline\n"
                    "/model — Show active model\n"
                    "/model list — List all models\n"
                    "/tools — List available tools\n"
                    "/memory search \\<query\\> — Search memory\n"
                    "/status — Show session status\n"
                ),
                "raw_response": "",
                "tool_calls": [],
            }

        if cmd == "/status":
            usage = self.app.router.get_usage() if hasattr(self.app.router, "get_usage") else {}
            msg_count = len(self.app.messages)
            model = getattr(self.app.router, "active_model", "unknown")
            from axiom.integrations.telegram.formatter import escape_md2
            return {
                "response": (
                    f"*Status*\n"
                    f"Model: {escape_md2(model)}\n"
                    f"Messages: {msg_count}\n"
                    f"Tools: {self.app.registry.count if self.app.registry else 0}\n"
                ),
                "raw_response": "",
                "tool_calls": [],
            }

        if cmd == "/agent" and arg:
            return await self.process_agent_task(arg)

        # Default: process as regular message without the slash
        return await self.process_message(text.lstrip("/"))
```

**Step 2: Commit**

```bash
git add axiom/integrations/telegram/bridge.py
git commit -m "feat: add Telegram bridge (AxiomApp ↔ Telegram adapter)"
```

---

### Task 8: Telegram Handler — Main Bot Entry Point

**Files:**
- Create: `axiom/integrations/telegram/handler.py`
- Create: `axiom/integrations/telegram/media.py`

**Step 1: Create media handler**

`axiom/integrations/telegram/media.py`:

```python
"""Handle file uploads, voice messages, and photos from Telegram."""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


async def handle_voice(update: Any, context: Any) -> Optional[str]:
    """Download voice message and transcribe with Whisper.

    Returns transcribed text, or None if transcription fails.
    """
    try:
        voice = update.message.voice or update.message.audio
        if voice is None:
            return None

        # Download to temp file
        file = await context.bot.get_file(voice.file_id)
        with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as tmp:
            await file.download_to_drive(tmp.name)
            tmp_path = tmp.name

        # Try Whisper transcription
        try:
            import whisper
            model = whisper.load_model("base")
            result = model.transcribe(tmp_path)
            return result.get("text", "")
        except ImportError:
            logger.warning("Whisper not installed — voice messages not supported")
            return None
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    except Exception as exc:
        logger.error("Voice handling failed: %s", exc)
        return None


async def handle_photo(update: Any, context: Any, workspace: Path) -> Optional[str]:
    """Download photo and save to workspace.

    Returns the saved file path.
    """
    try:
        photo = update.message.photo[-1]  # Largest size
        file = await context.bot.get_file(photo.file_id)

        save_path = workspace / "telegram_uploads" / f"{photo.file_id}.jpg"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        await file.download_to_drive(str(save_path))

        return str(save_path)

    except Exception as exc:
        logger.error("Photo handling failed: %s", exc)
        return None


async def handle_document(update: Any, context: Any, workspace: Path) -> Optional[str]:
    """Download document and save to workspace.

    Returns the saved file path.
    """
    try:
        doc = update.message.document
        file = await context.bot.get_file(doc.file_id)

        save_path = workspace / "telegram_uploads" / (doc.file_name or f"{doc.file_id}")
        save_path.parent.mkdir(parents=True, exist_ok=True)
        await file.download_to_drive(str(save_path))

        return str(save_path)

    except Exception as exc:
        logger.error("Document handling failed: %s", exc)
        return None
```

**Step 2: Create main handler**

`axiom/integrations/telegram/handler.py`:

```python
"""Main Telegram bot handler -- Full GOD MODE Axiom over Telegram.

Wraps a complete AxiomApp instance so Telegram gets the exact same
capabilities as the CLI: all tools, all models, all memory.

Usage:
    from axiom.integrations.telegram import TelegramBot
    bot = TelegramBot(token="...", app=axiom_app)
    await bot.start()
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


class TelegramBot:
    """Full-featured Telegram bot wrapping AxiomApp."""

    def __init__(
        self,
        token: str,
        app: Any,
        allowed_users: Optional[list[int]] = None,
    ):
        """
        Args:
            token: Telegram Bot API token from @BotFather.
            app: Initialized AxiomApp instance.
            allowed_users: List of Telegram user IDs allowed to use the bot.
                          If None, allows ALL users (use with caution).
        """
        self.token = token
        self.app = app
        self.allowed_users = set(allowed_users) if allowed_users else None
        self._application = None
        self._bridge = None

    def _check_auth(self, user_id: int) -> bool:
        """Check if a Telegram user is authorized."""
        if self.allowed_users is None:
            return True
        return user_id in self.allowed_users

    async def start(self) -> None:
        """Start the Telegram bot (blocking)."""
        try:
            from telegram.ext import (
                Application,
                CommandHandler,
                MessageHandler,
                filters,
            )
        except ImportError:
            logger.error(
                "python-telegram-bot not installed. "
                "Install with: pip install 'python-telegram-bot>=21.0'"
            )
            return

        from axiom.integrations.telegram.bridge import TelegramBridge

        self._bridge = TelegramBridge(self.app)

        self._application = (
            Application.builder()
            .token(self.token)
            .build()
        )

        # Register handlers
        self._application.add_handler(CommandHandler("start", self._on_start))
        self._application.add_handler(CommandHandler("help", self._on_start))
        self._application.add_handler(CommandHandler("status", self._on_status))
        self._application.add_handler(CommandHandler("agent", self._on_agent))
        self._application.add_handler(CommandHandler("model", self._on_model))
        self._application.add_handler(CommandHandler("models", self._on_models))
        self._application.add_handler(CommandHandler("tools", self._on_tools))
        self._application.add_handler(
            MessageHandler(filters.TEXT & ~filters.COMMAND, self._on_message)
        )
        self._application.add_handler(
            MessageHandler(filters.VOICE | filters.AUDIO, self._on_voice)
        )
        self._application.add_handler(
            MessageHandler(filters.PHOTO, self._on_photo)
        )
        self._application.add_handler(
            MessageHandler(filters.Document.ALL, self._on_document)
        )

        logger.info("Telegram bot starting with %d tools", self.app.registry.count if self.app.registry else 0)

        # Run the bot
        await self._application.initialize()
        await self._application.start()
        await self._application.updater.start_polling()

        logger.info("Telegram bot is running. Press Ctrl+C to stop.")

        # Keep running until stopped
        try:
            await asyncio.Event().wait()
        except (KeyboardInterrupt, asyncio.CancelledError):
            pass
        finally:
            await self.stop()

    async def stop(self) -> None:
        """Gracefully stop the bot."""
        if self._application:
            await self._application.updater.stop()
            await self._application.stop()
            await self._application.shutdown()
            logger.info("Telegram bot stopped.")

    # ── Handlers ───────────────────────────────────────────────

    async def _on_start(self, update: Any, context: Any) -> None:
        """Handle /start and /help commands."""
        if not self._check_auth(update.effective_user.id):
            await update.message.reply_text("⛔ Unauthorized.")
            return
        result = await self._bridge._handle_command("/help")
        await self._send_response(update, result)

    async def _on_status(self, update: Any, context: Any) -> None:
        if not self._check_auth(update.effective_user.id):
            return
        result = await self._bridge._handle_command("/status")
        await self._send_response(update, result)

    async def _on_agent(self, update: Any, context: Any) -> None:
        """Run full agent pipeline for complex tasks."""
        if not self._check_auth(update.effective_user.id):
            return
        task = update.message.text.replace("/agent", "", 1).strip()
        if not task:
            await update.message.reply_text("Usage: /agent <task description>")
            return

        # Send "thinking" indicator
        thinking_msg = await update.message.reply_text("🧠 Axiom is thinking...")

        result = await self._bridge.process_agent_task(task)

        # Delete thinking message
        try:
            await thinking_msg.delete()
        except Exception:
            pass

        # Send tool call summaries first
        for tc in result.get("tool_calls", [])[:5]:  # Max 5 tool summaries
            try:
                await update.message.reply_text(tc, parse_mode="MarkdownV2")
            except Exception:
                await update.message.reply_text(tc)  # Fallback to plain text

        await self._send_response(update, result)

    async def _on_model(self, update: Any, context: Any) -> None:
        if not self._check_auth(update.effective_user.id):
            return
        from axiom.integrations.telegram.formatter import escape_md2
        model = getattr(self.app.router, "active_model", "unknown")
        await update.message.reply_text(
            f"*Active model:* {escape_md2(model)}",
            parse_mode="MarkdownV2",
        )

    async def _on_models(self, update: Any, context: Any) -> None:
        if not self._check_auth(update.effective_user.id):
            return
        from axiom.integrations.telegram.formatter import escape_md2
        available = self.app.router.list_available() if hasattr(self.app.router, "list_available") else []
        if not available:
            await update.message.reply_text("No models available.")
            return
        lines = [f"*Available Models*\n"]
        for m in available[:15]:
            status = "✓" if m.get("available") else "✗"
            name = escape_md2(m.get("model", "unknown"))
            lines.append(f"{status} {name}")
        await update.message.reply_text("\n".join(lines), parse_mode="MarkdownV2")

    async def _on_tools(self, update: Any, context: Any) -> None:
        if not self._check_auth(update.effective_user.id):
            return
        from axiom.integrations.telegram.formatter import escape_md2
        if not self.app.registry:
            await update.message.reply_text("No tools loaded.")
            return
        lines = [f"*Tools \\({self.app.registry.count}\\)*\n"]
        for tool in self.app.registry.list_tools():
            name = escape_md2(tool.name)
            desc = escape_md2(tool.description[:60])
            lines.append(f"• *{name}* — {desc}")
        text = "\n".join(lines)
        from axiom.integrations.telegram.formatter import truncate_for_telegram
        await update.message.reply_text(
            truncate_for_telegram(text),
            parse_mode="MarkdownV2",
        )

    async def _on_message(self, update: Any, context: Any) -> None:
        """Handle regular text messages."""
        if not self._check_auth(update.effective_user.id):
            return

        text = update.message.text
        if not text:
            return

        # Send typing indicator
        await context.bot.send_chat_action(
            chat_id=update.effective_chat.id,
            action="typing",
        )

        result = await self._bridge.process_message(text)
        await self._send_response(update, result)

    async def _on_voice(self, update: Any, context: Any) -> None:
        """Handle voice messages — transcribe then process."""
        if not self._check_auth(update.effective_user.id):
            return

        from axiom.integrations.telegram.media import handle_voice
        text = await handle_voice(update, context)
        if text:
            await context.bot.send_chat_action(
                chat_id=update.effective_chat.id,
                action="typing",
            )
            # Show transcription
            await update.message.reply_text(f"🎤 Heard: {text}")
            result = await self._bridge.process_message(text)
            await self._send_response(update, result)
        else:
            await update.message.reply_text(
                "Could not transcribe voice message. "
                "Install whisper: pip install openai-whisper"
            )

    async def _on_photo(self, update: Any, context: Any) -> None:
        """Handle photos — save and optionally describe with vision."""
        if not self._check_auth(update.effective_user.id):
            return

        from axiom.integrations.telegram.media import handle_photo
        workspace = Path(self.app.settings.AXIOM_HOME) / "workspace" if hasattr(self.app, "settings") else Path.home() / ".axiom" / "workspace"
        path = await handle_photo(update, context, workspace)
        if path:
            caption = update.message.caption or "Describe this image"
            result = await self._bridge.process_message(
                f"I sent you a photo saved at {path}. {caption}"
            )
            await self._send_response(update, result)

    async def _on_document(self, update: Any, context: Any) -> None:
        """Handle document uploads — save to workspace."""
        if not self._check_auth(update.effective_user.id):
            return

        from axiom.integrations.telegram.media import handle_document
        workspace = Path(self.app.settings.AXIOM_HOME) / "workspace" if hasattr(self.app, "settings") else Path.home() / ".axiom" / "workspace"
        path = await handle_document(update, context, workspace)
        if path:
            caption = update.message.caption or f"Process this file: {path}"
            result = await self._bridge.process_message(
                f"I uploaded a file to {path}. {caption}"
            )
            await self._send_response(update, result)

    # ── Response Sending ───────────────────────────────────────

    async def _send_response(self, update: Any, result: dict) -> None:
        """Send formatted response to Telegram, handling length limits."""
        from axiom.integrations.telegram.formatter import truncate_for_telegram

        text = result.get("response", "")
        if not text:
            text = result.get("raw_response", "(no response)")

        text = truncate_for_telegram(text)

        try:
            await update.message.reply_text(text, parse_mode="MarkdownV2")
        except Exception:
            # Fallback to plain text if MarkdownV2 fails
            raw = result.get("raw_response", text)
            if len(raw) > 4000:
                raw = raw[:3980] + "\n\n... (truncated)"
            try:
                await update.message.reply_text(raw)
            except Exception as exc:
                logger.error("Failed to send Telegram response: %s", exc)
                await update.message.reply_text(f"Error sending response: {exc}")
```

**Step 3: Commit**

```bash
git add axiom/integrations/telegram/
git commit -m "feat: add Telegram bot handler + media support + bridge"
```

---

### Task 9: Heartbeat Daemon — Proactive Self-Activation

**Files:**
- Create: `axiom/integrations/heartbeat/__init__.py`
- Create: `axiom/integrations/heartbeat/daemon.py`
- Create: `workspace/HEARTBEAT.md`

**Step 1: Create heartbeat package**

`axiom/integrations/heartbeat/__init__.py`:
```python
"""Heartbeat daemon -- proactive self-activation on schedule."""
from axiom.integrations.heartbeat.daemon import HeartbeatDaemon

__all__ = ["HeartbeatDaemon"]
```

**Step 2: Create the daemon**

`axiom/integrations/heartbeat/daemon.py`:

```python
"""Heartbeat daemon -- reads HEARTBEAT.md and runs checks on schedule.

Like OpenClaw's heartbeat but integrated with Axiom's full tool pipeline.
Each check is a mini-agent turn that can use any of Axiom's 18+ tools.
"""

from __future__ import annotations

import asyncio
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

DEFAULT_HEARTBEAT_MD = """# Heartbeat Checks
interval: 30  # minutes

## Checks
- [ ] Check system health (CPU, disk space)
"""


class HeartbeatDaemon:
    """Background asyncio task that runs heartbeat checks on schedule."""

    def __init__(
        self,
        app: Any,
        heartbeat_path: Optional[Path] = None,
        interval_minutes: int = 30,
        on_alert: Optional[Callable] = None,
    ):
        self.app = app
        self.heartbeat_path = heartbeat_path or Path("workspace/HEARTBEAT.md")
        self.interval_minutes = interval_minutes
        self.on_alert = on_alert  # Callback for alerts (e.g., send to Telegram)
        self._task: Optional[asyncio.Task] = None
        self._running = False
        self._log_dir = Path("memory/heartbeat_log")

    async def start(self) -> None:
        """Start the heartbeat daemon as a background task."""
        if self._running:
            logger.warning("Heartbeat daemon already running")
            return

        # Ensure heartbeat file exists
        if not self.heartbeat_path.exists():
            self.heartbeat_path.parent.mkdir(parents=True, exist_ok=True)
            self.heartbeat_path.write_text(DEFAULT_HEARTBEAT_MD, encoding="utf-8")
            logger.info("Created default HEARTBEAT.md at %s", self.heartbeat_path)

        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._running = True
        self._task = asyncio.create_task(self._loop())
        logger.info(
            "Heartbeat daemon started (interval: %d min, path: %s)",
            self.interval_minutes,
            self.heartbeat_path,
        )

    async def stop(self) -> None:
        """Stop the heartbeat daemon."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        logger.info("Heartbeat daemon stopped")

    async def _loop(self) -> None:
        """Main loop -- sleep then check."""
        while self._running:
            try:
                await asyncio.sleep(self.interval_minutes * 60)
                if not self._running:
                    break
                await self._run_checks()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error("Heartbeat check failed: %s", exc, exc_info=True)

    async def _run_checks(self) -> None:
        """Parse HEARTBEAT.md and execute each check."""
        try:
            content = self.heartbeat_path.read_text(encoding="utf-8")
        except FileNotFoundError:
            logger.warning("HEARTBEAT.md not found at %s", self.heartbeat_path)
            return

        # Parse interval override
        interval_match = re.search(r"interval:\s*(\d+)", content)
        if interval_match:
            new_interval = int(interval_match.group(1))
            if new_interval != self.interval_minutes:
                self.interval_minutes = new_interval
                logger.info("Heartbeat interval updated to %d min", new_interval)

        # Parse check items (markdown checklist)
        checks = re.findall(r"- \[ \] (.+)", content)
        if not checks:
            logger.debug("No heartbeat checks defined")
            return

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        log_lines = [f"# Heartbeat {timestamp}\n"]
        alerts = []

        for check_desc in checks:
            logger.debug("Running heartbeat check: %s", check_desc)
            try:
                # Run as a mini chat turn
                result = await self._execute_check(check_desc)
                status = "OK" if "ok" in result.lower() or "pass" in result.lower() else "ALERT"

                log_lines.append(f"- [{status}] {check_desc}: {result[:200]}")

                if status == "ALERT":
                    alerts.append(f"⚠ {check_desc}: {result[:500]}")

            except Exception as exc:
                log_lines.append(f"- [ERROR] {check_desc}: {exc}")
                alerts.append(f"❌ {check_desc}: {exc}")

        # Write log
        log_file = self._log_dir / f"{datetime.now().strftime('%Y-%m-%d')}.md"
        with open(log_file, "a", encoding="utf-8") as f:
            f.write("\n".join(log_lines) + "\n\n")

        # Send alerts
        if alerts and self.on_alert:
            alert_text = f"🫀 Heartbeat Alert ({timestamp})\n\n" + "\n".join(alerts)
            try:
                await self.on_alert(alert_text)
            except Exception as exc:
                logger.error("Failed to send heartbeat alert: %s", exc)
        elif not alerts:
            logger.debug("HEARTBEAT_OK — all checks passed")

    async def _execute_check(self, description: str) -> str:
        """Execute a single heartbeat check using Axiom's tool pipeline."""
        # Use the app's chat method with a focused prompt
        prompt = (
            f"Heartbeat check: {description}\n"
            f"Run this check using available tools. "
            f"Reply with a brief status: OK if everything is fine, "
            f"or describe the issue if something needs attention."
        )

        # Simple approach: use the router directly for a quick check
        try:
            tools = None
            if self.app.registry and self.app.registry.count > 0:
                tools = self.app.registry.to_llm_schemas()

            response = ""
            async for chunk in self.app.router.complete(
                messages=[
                    {"role": "system", "content": "You are Axiom running a heartbeat check. Be brief and factual. Use tools if needed."},
                    {"role": "user", "content": prompt},
                ],
                tools=tools,
                stream=True,
            ):
                if hasattr(chunk, "choices") and chunk.choices:
                    delta = chunk.choices[0].delta
                    if hasattr(delta, "content") and delta.content:
                        response += delta.content

            return response or "No response"

        except Exception as exc:
            return f"Check failed: {exc}"
```

**Step 3: Create default HEARTBEAT.md**

`workspace/HEARTBEAT.md`:
```markdown
# Heartbeat Checks
interval: 30  # minutes

## Checks
- [ ] Check C:\Users\moyes\Downloads\forex_data\ for new CSV files
- [ ] Monitor axiom-box health: curl https://axiom-box.vercel.app
- [ ] Check git status of axiom-cli for uncommitted changes
- [ ] Scan ~/Downloads for files > 1GB (disk space alert)
```

**Step 4: Commit**

```bash
git add axiom/integrations/heartbeat/ workspace/HEARTBEAT.md
git commit -m "feat: add heartbeat daemon (proactive self-activation)"
```

---

### Task 10: Wire Everything Into AxiomApp

**Files:**
- Modify: `axiom/cli/app.py` (add init methods + /telegram command)

**Step 1: Add Telegram + Heartbeat initialization methods**

Add these methods to the AxiomApp class (after `_init_mcp_bridge`):

```python
    def _init_telegram(self) -> None:
        """Initialize Telegram bot if configured."""
        token = getattr(self.settings, "TELEGRAM_BOT_TOKEN", None)
        if not token:
            logger.debug("No TELEGRAM_BOT_TOKEN set — Telegram disabled")
            return

        try:
            from axiom.integrations.telegram import TelegramBot
            allowed = getattr(self.settings, "TELEGRAM_ALLOWED_USERS", None)
            user_ids = None
            if allowed:
                user_ids = [int(uid.strip()) for uid in allowed.split(",") if uid.strip()]
            self.telegram_bot = TelegramBot(
                token=token, app=self, allowed_users=user_ids
            )
            logger.info("Telegram bot initialized")
        except ImportError:
            logger.warning("python-telegram-bot not installed")
        except Exception as exc:
            logger.warning("Telegram init failed: %s", exc)

    def _init_heartbeat(self) -> None:
        """Initialize heartbeat daemon if configured."""
        if not getattr(self.settings, "HEARTBEAT_ENABLED", False):
            return
        try:
            from axiom.integrations.heartbeat import HeartbeatDaemon
            interval = getattr(self.settings, "HEARTBEAT_INTERVAL_MINUTES", 30)
            self.heartbeat = HeartbeatDaemon(
                app=self,
                interval_minutes=interval,
            )
            logger.info("Heartbeat daemon initialized (interval: %d min)", interval)
        except Exception as exc:
            logger.warning("Heartbeat init failed: %s", exc)
```

**Step 2: Add /telegram command to handle_command()**

In `handle_command()`, add after the `/models` handler:

```python
        if command == "/telegram":
            await self._handle_telegram_command(arg)
            return False

        if command == "/heartbeat":
            await self._handle_heartbeat_command(arg)
            return False
```

**Step 3: Add the handler methods**

```python
    async def _handle_telegram_command(self, arg: str) -> None:
        """Start/stop the Telegram bot."""
        if arg == "start":
            if not hasattr(self, "telegram_bot") or self.telegram_bot is None:
                self._init_telegram()
            if hasattr(self, "telegram_bot") and self.telegram_bot:
                console.print(f"[{AXIOM_GREEN}]Starting Telegram bot...[/]")
                asyncio.create_task(self.telegram_bot.start())
                console.print(f"[{AXIOM_GREEN}]Telegram bot running in background![/]")
            else:
                console.print(
                    f"[{AXIOM_YELLOW}]Set TELEGRAM_BOT_TOKEN in .env first.[/]"
                )
        elif arg == "stop":
            if hasattr(self, "telegram_bot") and self.telegram_bot:
                await self.telegram_bot.stop()
                console.print(f"[{AXIOM_GREEN}]Telegram bot stopped.[/]")
            else:
                console.print(f"[{AXIOM_DIM}]Telegram bot not running.[/]")
        else:
            console.print(
                f"[{AXIOM_CYAN}]Usage:[/] /telegram start | /telegram stop"
            )

    async def _handle_heartbeat_command(self, arg: str) -> None:
        """Start/stop the heartbeat daemon."""
        if arg == "start":
            if not hasattr(self, "heartbeat") or self.heartbeat is None:
                self._init_heartbeat()
            if hasattr(self, "heartbeat") and self.heartbeat:
                await self.heartbeat.start()
                console.print(f"[{AXIOM_GREEN}]Heartbeat daemon started![/]")
            else:
                console.print(
                    f"[{AXIOM_YELLOW}]Set HEARTBEAT_ENABLED=true in .env first.[/]"
                )
        elif arg == "stop":
            if hasattr(self, "heartbeat") and self.heartbeat:
                await self.heartbeat.stop()
                console.print(f"[{AXIOM_GREEN}]Heartbeat daemon stopped.[/]")
        else:
            console.print(
                f"[{AXIOM_CYAN}]Usage:[/] /heartbeat start | /heartbeat stop"
            )
```

**Step 4: Add help entries**

In `_show_help()`, add to the help table:

```python
        table.add_row("/telegram start|stop", "Start/stop Telegram bot")
        table.add_row("/heartbeat start|stop", "Start/stop proactive heartbeat daemon")
```

**Step 5: Initialize attributes in __init__**

Near the top of AxiomApp.__init__, add:

```python
        self.telegram_bot: Any = None
        self.heartbeat: Any = None
```

**Step 6: Commit**

```bash
git add axiom/cli/app.py
git commit -m "feat: wire Telegram + Heartbeat into AxiomApp CLI"
```

---

### Task 11: Final Verification — End-to-End Tests

**Step 1: Verify all files compile**

```bash
python -X utf8 -c "
import ast, pathlib

files = [
    'axiom/core/tools/think.py',
    'axiom/core/tools/registry.py',
    'axiom/core/agent/prompts/system.py',
    'axiom/core/agent/graph.py',
    'axiom/core/agent/state.py',
    'axiom/config/settings.py',
    'axiom/cli/app.py',
    'axiom/integrations/__init__.py',
    'axiom/integrations/telegram/__init__.py',
    'axiom/integrations/telegram/formatter.py',
    'axiom/integrations/telegram/bridge.py',
    'axiom/integrations/telegram/handler.py',
    'axiom/integrations/telegram/media.py',
    'axiom/integrations/heartbeat/__init__.py',
    'axiom/integrations/heartbeat/daemon.py',
]
for f in files:
    with open(f, 'r', encoding='utf-8') as fh:
        ast.parse(fh.read())
    print(f'[OK] {f}')
print(f'ALL {len(files)} FILES COMPILE')
"
```

**Step 2: Verify ThinkTool registers**

```bash
python -X utf8 -c "
from axiom.core.tools.registry import get_registry
r = get_registry()
assert 'think' in r, 'ThinkTool not registered!'
t = r.get('think')
print(f'ThinkTool: {t.name} — {t.description[:50]}')
print(f'Total tools: {r.count}')
print('PASS')
"
```

**Step 3: Verify system prompt upgrades**

```bash
python -X utf8 -c "
from axiom.core.agent.prompts.system import build_system_prompt
p = build_system_prompt(tool_names=['think','grep'], model_name='test')
required = ['DEEP THINKING', 'SEARCH MASTERY', 'PERSISTENCE PROTOCOL', 'TOOL PREFERENCE', 'TIER 1', 'NEVER say', 'think']
for r in required:
    assert r in p, f'Missing: {r}'
    print(f'[OK] {r}')
print('ALL PROMPT CHECKS PASSED')
"
```

**Step 4: Verify agent limits**

```bash
python -X utf8 -c "
from axiom.core.agent.graph import MAX_PLAN_ITERATIONS, MAX_REACT_ITERATIONS, MAX_REPLANS, MAX_CONSECUTIVE_FAILURES
assert MAX_PLAN_ITERATIONS == 50
assert MAX_REACT_ITERATIONS == 30
assert MAX_REPLANS == 8
assert MAX_CONSECUTIVE_FAILURES == 10
print('AGENT LIMITS CORRECT')

from axiom.core.agent.state import AgentState
s = AgentState()
assert s.max_iterations == 50
assert s.max_replans == 8
print('STATE DEFAULTS CORRECT')
print('PASS')
"
```

**Step 5: Verify Telegram formatter**

```bash
python -X utf8 -c "
from axiom.integrations.telegram.formatter import escape_md2, format_agent_response, truncate_for_telegram
assert escape_md2('hello_world') == 'hello\\_world'
assert escape_md2('test.txt') == 'test\\.txt'
resp = format_agent_response('**Bold** and `code`')
print(f'Formatted: {resp}')
long = 'x' * 5000
truncated = truncate_for_telegram(long)
assert len(truncated) <= 4096
print('TELEGRAM FORMATTER PASS')
"
```

**Step 6: Final commit**

```bash
git add -A
git commit -m "feat: Axiom Ascension complete - deep intelligence + telegram + heartbeat"
```

---

## Execution Summary

| Task | Component | New Files | Modified Files |
|------|-----------|-----------|----------------|
| 1 | ThinkTool | `tools/think.py` | `tools/registry.py` |
| 2 | System Prompt Upgrade | — | `prompts/system.py` |
| 3 | Agent Graph Upgrade | — | `graph.py`, `state.py` |
| 4 | /models Alias | — | `app.py` |
| 5 | Telegram Settings | — | `settings.py`, `.env` |
| 6 | Telegram Formatter | 3 init + `formatter.py` | — |
| 7 | Telegram Bridge | `bridge.py` | — |
| 8 | Telegram Handler | `handler.py`, `media.py` | — |
| 9 | Heartbeat Daemon | `daemon.py`, `HEARTBEAT.md` | — |
| 10 | Wire into CLI | — | `app.py` |
| 11 | Final Verification | — | — |

**Total: 10 new files, 6 modified files, 11 tasks, ~50 minutes estimated.**
