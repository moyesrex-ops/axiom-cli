# Axiom Ascension — Deep Intelligence + Telegram + Heartbeat

**Date**: 2026-03-06
**Status**: Approved
**Author**: Axiom (self-designed)

## Problem Statement

Axiom CLI is architecturally sound (tri-mode agent, 18+ tools, 15+ LLM providers, self-repair) but operationally shallow:

1. **Shallow search** — Single grep attempt with narrow patterns, gives up after one try
2. **Shallow reasoning** — No explicit chain-of-thought before tool calls
3. **API-first bias** — Reaches for paid services before exhausting free/local alternatives
4. **Purely reactive** — Only acts when prompted, never proactive
5. **CLI-only** — No mobile/remote access
6. **Quick surrender** — Reports "not found" or "I can't" after minimal effort

OpenClaw (247K GitHub stars) solves some of these with Heartbeat, two-layer memory, and lane queues. Axiom already surpasses OpenClaw in architecture (tri-mode vs single loop, self-repair, 18 tools vs ~10). This design closes the intelligence and accessibility gaps.

## Design Overview

Five new components, zero architecture rewrites:

| Component | Purpose | Files |
|-----------|---------|-------|
| **Deep Reasoning Engine** | Force explicit reasoning before every action | `think` tool + system prompt |
| **Persistence Loop** | Never give up, auto-retry with alternative approaches | Agent graph upgrade |
| **Tool Preference Hierarchy** | Local/free first, paid last | System prompt + tool metadata |
| **Telegram Bridge** | Full GOD MODE Axiom over Telegram | `axiom/integrations/telegram/` |
| **Heartbeat Daemon** | Proactive self-activation on schedule | `axiom/integrations/heartbeat/` |

## Component 1: Deep Reasoning Engine

### The `think` Tool

```python
class ThinkTool(AxiomTool):
    name = "think"
    description = "Reason explicitly before acting. MANDATORY before complex tasks."
    risk_level = "low"

    async def execute(self, reasoning: str) -> str:
        """Pure reasoning — no side effects. Returns the reasoning for the trace."""
        return f"[Reasoning]\n{reasoning}"
```

This tool has zero side effects. Its only purpose is to force the LLM to externalize its chain-of-thought into a visible, logged trace step. The system prompt mandates its use before any multi-step task.

### Multi-Strategy Search Protocol (System Prompt)

Injected into the GOD MODE system prompt:

```
## SEARCH MASTERY — You are the BEST searcher on Earth
When looking for ANY file, token, config, or data:
1. think() — What am I looking for? What formats? Where do humans store this?
2. glob() — Cast WIDE net: try *.env, *.json, *.txt, *.yaml, *.cfg, *.toml
3. grep() — Try 3+ patterns: exact string, case-insensitive, partial match, regex
4. read_file() — Open the top 5 most promising files and scan manually
5. Expand scope — If not found, search parent directory, ~/Downloads, ~/Desktop, ~
6. web_fetch() — As last resort, search the web for guidance
7. NEVER report "not found" after < 3 distinct attempts
```

### Persistence Directive (System Prompt)

```
## PERSISTENCE PROTOCOL — NON-NEGOTIABLE
- You NEVER say "I can't" or "I couldn't find it" or "I'm unable to"
- If approach A fails → try B, C, D
- If all tools fail → CREATE a new tool via tool_create
- If you need a capability → RESEARCH how to get it
- Minimum 3 genuine attempts before reporting any failure
- On failure: explain what you tried AND propose concrete next steps
- You are not a chatbot. You are an autonomous agent. ACT like one.
```

## Component 2: Persistence Loop (Agent Graph Upgrade)

### Current Limits vs New

| Parameter | Current | New | Why |
|-----------|---------|-----|-----|
| Max iterations | 25 | 50 | Complex tasks need more steps |
| Max replans | 4 | 8 | More chances to find alternative approaches |
| Consecutive failures before abort | 6 | 10 | More resilient |
| Replan strategy | Same approach | MANDATORY different approach | Prevents loops |

### Auto-Retry with Variation

When a tool call fails, the agent graph injects a "retry hint" into the next LLM call:

```
Tool '{tool_name}' failed with: {error}
MANDATORY: Use a DIFFERENT tool or DIFFERENT arguments for your next attempt.
Do NOT repeat the same call. Consider: {alternative_tools}
```

### Creative Problem Solving Mode

After all replans exhausted (iteration > 40), enter "creative mode":
- Use `tool_create` to dynamically build a custom tool
- Use `research` to find new approaches online
- Use `spawn_agent` for parallel exploration
- Use `bash` to install missing packages

## Component 3: Tool Preference Hierarchy

### Tier System

Each tool gets a `preference_tier` attribute (1=cheapest, 3=most expensive):

| Tier | Tools | When to Use |
|------|-------|-------------|
| **1 — Local/Free** | grep, glob, read_file, write_file, edit_file, bash, code_exec, git, think | Always try first |
| **2 — Free API** | research(mode=quick, source=duckduckgo), web_fetch, memory_tool | If local insufficient |
| **3 — Paid API** | research(mode=deep, source=tavily), browser, vision, spawn_agent | Only if tiers 1-2 fail |

### System Prompt Directive

```
## TOOL PREFERENCE — Cheapest first, always
TIER 1 (FREE/LOCAL — always try first):
  grep, glob, read_file, bash, code_exec, git, think

TIER 2 (FREE API — if local fails):
  DuckDuckGo search, web_fetch for specific URLs, memory_tool

TIER 3 (PAID — only if tiers 1-2 insufficient):
  Tavily/Exa deep research, Playwright browser, vision model

Before using ANY Tier 3 tool, you MUST have tried at least one Tier 1 alternative.
```

### LLM Model Preference

Keep ALL existing models. Priority order unchanged:
1. Explicit user selection (`/model opus`)
2. BYOK quality (user API keys)
3. Environment BYOK fallback
4. Free tier (Groq Llama, Gemini, DeepSeek)

No models removed. Anthropic Opus 4.6, Vertex, and all current providers stay.

## Component 4: Telegram Bridge

### Architecture

```
axiom/integrations/telegram/
├── __init__.py          # Exports TelegramBot
├── handler.py           # Main TelegramBot class
├── formatter.py         # Rich markup → Telegram MarkdownV2
├── media.py             # File/photo/voice handling
└── bridge.py            # AxiomApp ↔ Telegram message adapter
```

### TelegramBot Class

```python
class TelegramBot:
    def __init__(self, token: str, app: AxiomApp):
        self.app = app           # Full AxiomApp instance
        self.token = token
        self.bot = Application.builder().token(token).build()

    async def on_message(self, update, context):
        """Forward Telegram message → AxiomApp.chat() → stream back."""
        text = update.message.text
        # Run through full agent pipeline
        response = await self.bridge.process(text)
        # Send formatted response to Telegram
        await update.message.reply_text(
            format_telegram(response),
            parse_mode="MarkdownV2"
        )
```

### Capabilities (Same as CLI)

- **Text messages** → `app.chat(message)` → formatted response
- **Voice messages** → Whisper STT → `app.chat(transcribed)` → text response
- **Photos** → Save to workspace → vision tool → describe → respond
- **Files** → Save to workspace → process → respond
- **Commands**: `/plan`, `/react`, `/council`, `/model`, `/tools`, `/memory`, `/yolo`
- **Streaming**: Edit-message technique (send → edit every 500ms as tokens arrive)
- **Tool panels**: Formatted as code blocks with tool name and args
- **Errors**: Red-formatted error messages

### Token

Located at: `C:\Users\moyes\Downloads\axiom-telegram\.env`
Value: `8646524596:AAGuHnKy1qFS3Ja7LIBPXDw2h-UfppLRkuo`

Injected into: `axiom-cli/.env` → `TELEGRAM_BOT_TOKEN=`

### Security

- Full GOD MODE — same 18+ tools as CLI (user-approved)
- Only responds to authorized user ID (configurable `TELEGRAM_ALLOWED_USERS`)
- All tool calls logged to memory
- High-risk tools still require confirmation (sent as inline keyboard buttons)

## Component 5: Heartbeat Daemon

### Architecture

```
axiom/integrations/heartbeat/
├── __init__.py
├── daemon.py            # AsyncIO scheduler
├── checks.py            # Built-in check functions
└── HEARTBEAT.md         # User-editable checklist (in workspace/)
```

### How It Works

1. Background `asyncio.Task` runs every N minutes (default: 30)
2. Reads `workspace/HEARTBEAT.md` for check definitions
3. Each check becomes a mini-agent turn (uses tools via AxiomApp)
4. Results: HEARTBEAT_OK (silent) or ALERT (sends Telegram notification)
5. Logs all checks to `memory/heartbeat_log/YYYY-MM-DD.md`

### HEARTBEAT.md Format

```markdown
# Heartbeat Checks
interval: 30  # minutes

## Checks
- [ ] Check C:\Users\moyes\Downloads\forex_data\ for new CSV files
- [ ] Monitor axiom-box health: curl https://axiom-box.vercel.app
- [ ] Check git status of axiom-cli for uncommitted changes
- [ ] Scan ~/Downloads for files > 1GB (disk space alert)
```

### Integration with Telegram

When a heartbeat check finds something noteworthy:
1. Formats alert message with check details
2. Sends to user's Telegram via the TelegramBot
3. User can respond directly on Telegram to handle it

## Component 6: Minor Fixes

### `/models` alias
Add `/models` as alias for `/model list` in `handle_command()`.

### System prompt deep thinking upgrade
Rewrite the GOD MODE prompt with all the directives from Components 1-3.

## File Changes Summary

### New Files (8)
- `axiom/core/tools/think.py` — ThinkTool
- `axiom/integrations/__init__.py`
- `axiom/integrations/telegram/__init__.py`
- `axiom/integrations/telegram/handler.py`
- `axiom/integrations/telegram/formatter.py`
- `axiom/integrations/telegram/media.py`
- `axiom/integrations/telegram/bridge.py`
- `axiom/integrations/heartbeat/daemon.py`
- `axiom/integrations/heartbeat/__init__.py`
- `workspace/HEARTBEAT.md`

### Modified Files (4)
- `axiom/core/agent/prompts/system.py` — Deep thinking + persistence + search mastery
- `axiom/core/agent/graph.py` — Persistence loop (50 iters, 8 replans, retry hints)
- `axiom/cli/app.py` — `/models` alias, Telegram/Heartbeat init, think tool registration
- `axiom-cli/.env` — TELEGRAM_BOT_TOKEN + TELEGRAM_ALLOWED_USERS

### Unchanged
- All 18 existing tools
- All LLM providers and models
- All CLI commands
- Memory system
- Skills system
- Tool approval flow

## Success Criteria

1. **Deep search**: Given "find my telegram token", Axiom finds it in < 3 tool calls
2. **Persistence**: Given a complex task, Axiom tries 3+ different approaches before any failure
3. **Local-first**: Axiom uses grep before web search, DuckDuckGo before Tavily
4. **Telegram**: Full conversation with tools works over Telegram
5. **Heartbeat**: Proactive checks run on schedule, alerts sent to Telegram
6. **No regressions**: All existing CLI features work unchanged
