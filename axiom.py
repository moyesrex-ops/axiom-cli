#!/usr/bin/env python3
"""
╔═══════════════════════════════════════════════════════════════╗
║                                                               ║
║      █████╗ ██╗  ██╗██╗ ██████╗ ███╗   ███╗                  ║
║     ██╔══██╗╚██╗██╔╝██║██╔═══██╗████╗ ████║                  ║
║     ███████║ ╚███╔╝ ██║██║   ██║██╔████╔██║                  ║
║     ██╔══██║ ██╔██╗ ██║██║   ██║██║╚██╔╝██║                  ║
║     ██║  ██║██╔╝ ██╗██║╚██████╔╝██║ ╚═╝ ██║                  ║
║     ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝ ╚═════╝ ╚═╝     ╚═╝                  ║
║                                                               ║
║              THE UNSTOPPABLE AI AGENT                         ║
╚═══════════════════════════════════════════════════════════════╝

Axiom CLI — Run from anywhere. Use any model. Execute anything.
Commands work, tools fire, no limits apply.
"""

from __future__ import annotations

import asyncio
import json
import os
import platform
import random
import re
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx

# Force UTF-8 on Windows terminals
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
    os.environ.setdefault("PYTHONUTF8", "1")

from prompt_toolkit import PromptSession
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.history import FileHistory
from prompt_toolkit.styles import Style
from rich import print as rprint
from rich.columns import Columns
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.rule import Rule
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

# ── Paths ──────────────────────────────────────────────────────────────────────
AXIOM_DIR = Path(__file__).parent
TELEGRAM_DIR = Path("C:/Users/moyes/Downloads/axiom-telegram")
PUTER_SERVER = TELEGRAM_DIR / "puter_server.js"
MEMORY_FILE = AXIOM_DIR / "axiom_memory.json"
HISTORY_FILE = AXIOM_DIR / ".axiom_history"

# Try to load .env from telegram dir, then local
ENV_FILES = [TELEGRAM_DIR / ".env", AXIOM_DIR / ".env"]

# ── Config ─────────────────────────────────────────────────────────────────────
def load_env() -> dict[str, str]:
    env: dict[str, str] = {}
    for f in ENV_FILES:
        if f.exists():
            for line in f.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, _, v = line.partition("=")
                    env[k.strip()] = v.strip()
    env.update({k: v for k, v in os.environ.items() if k in env or k.startswith("PUTER")})
    return env

CONFIG = load_env()
PUTER_USERNAME = CONFIG.get("PUTER_USERNAME", "")
PUTER_PASSWORD = CONFIG.get("PUTER_PASSWORD", "")
OPENROUTER_KEY = CONFIG.get("OPENROUTER_API_KEY", "")
GEMINI_KEY = CONFIG.get("GEMINI_API_KEY", "")
DEFAULT_MODEL = CONFIG.get("DEFAULT_MODEL", "puter")

PUTER_SERVER_URL = "http://127.0.0.1:47825"

# ── OpenClaw bridge ────────────────────────────────────────────────────────────
# When the Telegram bot is running, it exposes a REST API on port 8765.
# CLI can submit tasks through that API — results appear in BOTH terminal AND Telegram.
BRIDGE_URL = "http://127.0.0.1:8765"
OWNER_CHAT_ID = int(CONFIG.get("OWNER_CHAT_ID", "0") or "0")
TELEGRAM_BOT_TOKEN = CONFIG.get("TELEGRAM_BOT_TOKEN", "")

# ── Console ────────────────────────────────────────────────────────────────────
console = Console(force_terminal=True, highlight=True)

# ── Puter model map ────────────────────────────────────────────────────────────
MODELS = {
    # OpenAI — verified working via Puter
    "puter":            ("puter", "gpt-4o-mini"),
    "gpt-4o":           ("puter", "gpt-4o"),
    "gpt-5":            ("puter", "gpt-5"),
    "gpt5":             ("puter", "gpt-5"),
    "gpt-4o-mini":      ("puter", "gpt-4o-mini"),
    "o3":               ("puter", "o3"),
    "o3-mini":          ("puter", "o3-mini"),
    "o4-mini":          ("puter", "o4-mini"),
    # Claude — via Puter (does NOT burn Max subscription)
    "claude-opus":      ("puter", "claude-opus-4-6"),
    "claude-opus-4-6":  ("puter", "claude-opus-4-6"),
    "claude-sonnet":    ("puter", "claude-sonnet-4-6"),
    "claude-haiku":     ("puter", "claude-haiku-4-5-20251001"),
    # Gemini — via Puter
    "gemini-pro":       ("puter", "gemini-2.5-pro"),
    "gemini-flash":     ("puter", "gemini-2.5-flash"),
    "gemini-2.5-pro":   ("puter", "gemini-2.5-pro"),
    "gemini-2.5-flash": ("puter", "gemini-2.5-flash"),
    # Grok — via Puter
    "grok-3":           ("puter", "grok-3"),
    "grok":             ("puter", "grok-3"),
    "grok-mini":        ("puter", "grok-3-mini"),
    # DeepSeek — verified IDs
    "deepseek-r1":      ("puter", "deepseek-r1"),
    "deepseek-v3":      ("puter", "deepseek-chat"),
    "deepseek":         ("puter", "deepseek-r1"),
    # Meta Llama — OpenRouter path works (lowercase)
    "llama":            ("puter", "meta-llama/llama-3.3-70b-instruct"),
    "llama-70b":        ("puter", "meta-llama/llama-3.3-70b-instruct"),
    "llama-4":          ("puter", "meta-llama/llama-4-maverick"),
    # Mistral — verified working
    "mistral":          ("puter", "mistral-large-latest"),
    "mistral-7b":       ("puter", "open-mistral-7b"),
    # Qwen — OpenRouter path
    "qwen":             ("puter", "qwen/qwen3-235b-a22b"),
    "qwen-2.5":         ("puter", "qwen/qwen-2.5-72b-instruct"),
    # Perplexity — live web search
    "perplexity":       ("puter", "perplexity/sonar"),
    "sonar":            ("puter", "perplexity/sonar"),
    # Cohere
    "cohere":           ("puter", "cohere/command-r-plus-08-2024"),
    # Kimi
    "kimi":             ("puter", "moonshotai/kimi-k2"),
    # OpenRouter fallback
    "openrouter":       ("openrouter", "meta-llama/llama-3.1-8b-instruct:free"),
}

# ── State ──────────────────────────────────────────────────────────────────────
state = {
    "model":          DEFAULT_MODEL,
    "history":        [],      # [{role, content}]
    "memory":         {},      # persistent key→value
    "agent_running":  False,
    "bridge_active":  False,   # True when Telegram bot API (port 8765) is reachable
    "telegram_mirror": True,   # Mirror CLI responses to Telegram when bridge is active
    "user_id":        7800065141,  # Victor's Telegram user ID
}

# ── Browser globals ────────────────────────────────────────────────────────────
_BROWSER_PROFILE = Path(tempfile.gettempdir()) / "axiom_chrome_profile"
_BROWSER_PROFILE.mkdir(exist_ok=True)
_EXTENSIONS_DIR = AXIOM_DIR / "extensions"
_EXTENSIONS_DIR.mkdir(exist_ok=True)
_EDGE_EXE  = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
_EDGE_PORT = 9222
_browser_ctx:  Any = None
_browser_page: Any = None

# ── Puter server ───────────────────────────────────────────────────────────────
_puter_proc: subprocess.Popen | None = None

def _puter_healthy() -> bool:
    try:
        urllib.request.urlopen(f"{PUTER_SERVER_URL}/health", timeout=2)
        return True
    except Exception:
        return False

def ensure_puter_server() -> bool:
    global _puter_proc
    if _puter_healthy():
        return True
    if not PUTER_SERVER.exists():
        console.print("[yellow]⚠ puter_server.js not found — Puter unavailable[/]")
        return False
    node = shutil.which("node") or "node"
    flags = subprocess.CREATE_NO_WINDOW if platform.system() == "Windows" else 0
    try:
        _puter_proc = subprocess.Popen(
            [node, str(PUTER_SERVER)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=flags,
        )
        for _ in range(8):
            time.sleep(1)
            if _puter_healthy():
                return True
        return False
    except Exception as e:
        console.print(f"[red]Puter server failed to start: {e}[/]")
        return False

# ── LLM call ───────────────────────────────────────────────────────────────────
async def llm(messages: list[dict], model_key: str | None = None) -> str:
    key = (model_key or state["model"]).lower()
    provider, model_id = MODELS.get(key, ("puter", "gpt-4o-mini"))

    if provider == "puter":
        return await _call_puter(messages, model_id)
    elif provider == "openrouter":
        return await _call_openrouter(messages, model_id)
    # fallback
    return await _call_puter(messages, "gpt-4o-mini")

async def _call_puter(messages: list[dict], model: str) -> str:
    if not PUTER_USERNAME:
        raise RuntimeError("No PUTER_USERNAME configured")
    payload = {
        "action": "chat",
        "username": PUTER_USERNAME,
        "password": PUTER_PASSWORD,
        "model": model,
        "messages": messages,
    }
    async with httpx.AsyncClient(timeout=120) as client:
        r = await client.post(PUTER_SERVER_URL, json=payload)
        data = r.json()
    if not data.get("success"):
        raise RuntimeError(f"Puter error: {data.get('error', 'unknown')}")
    return data.get("text", "")

async def _call_openrouter(messages: list[dict], model: str) -> str:
    if not OPENROUTER_KEY:
        raise RuntimeError("No OPENROUTER_API_KEY configured")
    async with httpx.AsyncClient(timeout=120) as client:
        r = await client.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_KEY}",
                "HTTP-Referer": "https://axiom.local",
                "X-Title": "Axiom CLI",
            },
            json={"model": model, "messages": messages},
        )
        data = r.json()
    return data["choices"][0]["message"]["content"]

# ── OpenClaw bridge helpers ────────────────────────────────────────────────────

def _bridge_healthy() -> bool:
    """Check if the Telegram bot's API bridge is running on port 8765."""
    try:
        urllib.request.urlopen(f"{BRIDGE_URL}/health", timeout=2)
        return True
    except Exception:
        return False


async def _bridge_chat(message: str, mirror: bool = True) -> str | None:
    """Send a message through the Telegram bot's agent pipeline.

    Returns the response text. If mirror=True, bot also sends to Telegram automatically.
    Returns None if bridge is not available.
    """
    try:
        async with httpx.AsyncClient(timeout=90.0) as c:
            r = await c.post(f"{BRIDGE_URL}/chat", json={
                "message": message,
                "user_id": state["user_id"],
                "chat_id": OWNER_CHAT_ID,
                "model": state["model"],
                "mirror_telegram": mirror,
            })
            data = r.json()
            if data.get("success"):
                return data.get("response", "")
            else:
                return f"Bridge error: {data.get('detail', 'unknown')}"
    except Exception as e:
        return None  # Bridge unavailable


async def _bridge_start_agent(goal: str) -> str | None:
    """Start a background agent task via the bridge. Returns run_id or None."""
    try:
        async with httpx.AsyncClient(timeout=15.0) as c:
            r = await c.post(f"{BRIDGE_URL}/agent/start", json={
                "goal": goal,
                "user_id": state["user_id"],
                "chat_id": OWNER_CHAT_ID,
                "model": state["model"],
                "mode": "react",
            })
            data = r.json()
            return data.get("run_id") if data.get("success") else None
    except Exception:
        return None


async def _bridge_list_tasks() -> list[dict]:
    """List running agent tasks from the bridge."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as c:
            r = await c.get(f"{BRIDGE_URL}/agents", params={"user_id": state["user_id"]})
            return r.json().get("tasks", [])
    except Exception:
        return []


async def _bridge_stop_task(run_id: str) -> bool:
    """Cancel a running agent task."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as c:
            r = await c.post(f"{BRIDGE_URL}/agent/{run_id}/stop")
            return r.json().get("success", False)
    except Exception:
        return False


async def _direct_telegram(text: str) -> bool:
    """Send a message directly to Telegram via Bot API (no bridge needed)."""
    if not TELEGRAM_BOT_TOKEN or not OWNER_CHAT_ID:
        return False
    try:
        async with httpx.AsyncClient(timeout=10.0) as c:
            r = await c.post(
                f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
                json={
                    "chat_id": OWNER_CHAT_ID,
                    "text": text[:4096],
                    "parse_mode": "Markdown",
                },
            )
            return r.json().get("ok", False)
    except Exception:
        return False


# ── Tools ───────────────────────────────────────────────────────────────────────
async def tool_shell(cmd: str) -> str:
    """Run a shell command and return output."""
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=60,
        )
        out = result.stdout + result.stderr
        return out[:3000] if out else "(no output)"
    except subprocess.TimeoutExpired:
        return "Command timed out after 60s"
    except Exception as e:
        return f"Shell error: {e}"

async def tool_python(code: str) -> str:
    """Execute Python code and return output."""
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
        f.write(code)
        tmp = f.name
    try:
        result = subprocess.run(
            [sys.executable, tmp],
            capture_output=True, text=True, timeout=30,
        )
        out = result.stdout + result.stderr
        return out[:3000] if out else "(no output)"
    except Exception as e:
        return f"Python error: {e}"
    finally:
        os.unlink(tmp)

async def tool_web_search(query: str) -> str:
    """Search the web via DuckDuckGo."""
    try:
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=5))
        if not results:
            return "No results found."
        lines = []
        for r in results:
            lines.append(f"**{r.get('title','')}**\n{r.get('body','')}\n{r.get('href','')}")
        return "\n\n".join(lines)
    except Exception as e:
        return f"Search error: {e}"

async def tool_web_fetch(url: str) -> str:
    """Fetch a webpage and return its text content."""
    try:
        async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
            r = await client.get(url, headers={"User-Agent": "Mozilla/5.0"})
            text = r.text
        # Strip HTML tags
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text[:4000]
    except Exception as e:
        return f"Fetch error: {e}"

async def tool_file_read(path: str) -> str:
    """Read a file."""
    try:
        return Path(path).read_text(encoding="utf-8", errors="replace")[:4000]
    except Exception as e:
        return f"File read error: {e}"

async def tool_file_write(path: str, content: str) -> str:
    """Write content to a file."""
    try:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return f"Written {len(content)} chars to {path}"
    except Exception as e:
        return f"File write error: {e}"

async def tool_memory_set(key: str, value: str) -> str:
    """Store something in persistent memory."""
    state["memory"][key] = value
    _save_memory()
    return f"Remembered: {key}"

async def tool_memory_get(key: str = "") -> str:
    """Retrieve from persistent memory."""
    if not key:
        if not state["memory"]:
            return "Memory is empty."
        return "\n".join(f"{k}: {v}" for k, v in state["memory"].items())
    return state["memory"].get(key, f"No memory for '{key}'")

async def tool_powershell(cmd: str) -> str:
    """Run a PowerShell command. Use this for Windows-specific tasks, file system, registry, networking."""
    try:
        flags = subprocess.CREATE_NO_WINDOW if platform.system() == "Windows" else 0
        result = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", cmd],
            capture_output=True, text=True, timeout=60, creationflags=flags,
        )
        out = (result.stdout + result.stderr).strip()
        return out[:3000] if out else "(no output)"
    except subprocess.TimeoutExpired:
        return "PowerShell timed out after 60s"
    except Exception as e:
        return f"PowerShell error: {e}"


async def tool_open_app(target: str) -> str:
    """Open an application, file, URL, or folder. Works with app names (notepad, chrome, calc) or full paths."""
    try:
        if platform.system() == "Windows":
            os.startfile(target)
        else:
            subprocess.Popen(["xdg-open", target])
        return f"Opened: {target}"
    except Exception as e:
        # Try PowerShell Start-Process as fallback
        try:
            subprocess.Popen(["powershell", "-Command", f"Start-Process '{target}'"],
                             creationflags=subprocess.CREATE_NO_WINDOW)
            return f"Opened via PowerShell: {target}"
        except Exception as e2:
            return f"Open error: {e} / {e2}"


async def tool_notify(title: str, message: str) -> str:
    """Show a Windows toast notification on the desktop."""
    try:
        ps = f"""
Add-Type -AssemblyName System.Windows.Forms
$n = New-Object System.Windows.Forms.NotifyIcon
$n.Icon = [System.Drawing.SystemIcons]::Information
$n.Visible = $true
$n.ShowBalloonTip(6000, '{title.replace("'", "")}', '{message.replace("'", "")}', [System.Windows.Forms.ToolTipIcon]::Info)
Start-Sleep -Milliseconds 500
"""
        subprocess.Popen(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", ps],
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
        return f"Notification sent: {title}"
    except Exception as e:
        return f"Notify error: {e}"


async def tool_clipboard_read() -> str:
    """Read the current Windows clipboard content."""
    try:
        result = subprocess.run(
            ["powershell", "-NoProfile", "-Command", "Get-Clipboard"],
            capture_output=True, text=True, timeout=5,
            creationflags=subprocess.CREATE_NO_WINDOW if platform.system() == "Windows" else 0,
        )
        return result.stdout.strip() or "(clipboard empty)"
    except Exception as e:
        return f"Clipboard read error: {e}"


async def tool_clipboard_write(text: str) -> str:
    """Write text to the Windows clipboard."""
    try:
        proc = subprocess.run(
            ["powershell", "-NoProfile", "-Command", f"Set-Clipboard -Value @'\n{text}\n'@"],
            capture_output=True, text=True, timeout=5,
            creationflags=subprocess.CREATE_NO_WINDOW if platform.system() == "Windows" else 0,
        )
        return f"Copied {len(text)} chars to clipboard"
    except Exception as e:
        return f"Clipboard write error: {e}"


async def tool_process_list(name: str = "") -> str:
    """List running processes. Optional: filter by name (e.g. 'chrome', 'python')."""
    try:
        if name:
            cmd = f"Get-Process -Name '*{name}*' -ErrorAction SilentlyContinue | Select-Object Name,Id,@{{N='CPU(s)';E={{[math]::Round($_.CPU,1)}}}},@{{N='RAM(MB)';E={{[math]::Round($_.WorkingSet/1MB,1)}}}} | Format-Table -AutoSize"
        else:
            cmd = "Get-Process | Sort-Object CPU -Descending | Select-Object -First 25 Name,Id,@{N='CPU(s)';E={[math]::Round($_.CPU,1)}},@{N='RAM(MB)';E={[math]::Round($_.WorkingSet/1MB,1)}} | Format-Table -AutoSize"
        result = subprocess.run(
            ["powershell", "-NoProfile", "-Command", cmd],
            capture_output=True, text=True, timeout=10,
            creationflags=subprocess.CREATE_NO_WINDOW if platform.system() == "Windows" else 0,
        )
        return result.stdout.strip() or "(no processes found)"
    except Exception as e:
        return f"Process list error: {e}"


async def tool_process_kill(name_or_pid: str) -> str:
    """Kill a process by name or PID."""
    try:
        try:
            pid = int(name_or_pid)
            cmd = f"Stop-Process -Id {pid} -Force -ErrorAction Stop"
        except ValueError:
            cmd = f"Stop-Process -Name '{name_or_pid}' -Force -ErrorAction Stop"
        result = subprocess.run(
            ["powershell", "-NoProfile", "-Command", cmd],
            capture_output=True, text=True, timeout=10,
            creationflags=subprocess.CREATE_NO_WINDOW if platform.system() == "Windows" else 0,
        )
        out = (result.stdout + result.stderr).strip()
        return out or f"Killed: {name_or_pid}"
    except Exception as e:
        return f"Kill error: {e}"


async def tool_screenshot(save_path: str = "") -> str:
    """Take a screenshot of the desktop and save it."""
    try:
        from PIL import ImageGrab  # type: ignore
        img = ImageGrab.grab()
        path = save_path or str(AXIOM_DIR / f"screenshot_{int(time.time())}.png")
        img.save(path)
        return f"Screenshot saved: {path} ({img.width}x{img.height})"
    except ImportError:
        # Fallback to PowerShell
        path = save_path or str(AXIOM_DIR / f"screenshot_{int(time.time())}.png")
        ps = f"""
Add-Type -AssemblyName System.Windows.Forms
$screen = [System.Windows.Forms.Screen]::PrimaryScreen.Bounds
$bmp = New-Object System.Drawing.Bitmap($screen.Width, $screen.Height)
$g = [System.Drawing.Graphics]::FromImage($bmp)
$g.CopyFromScreen($screen.Location, [System.Drawing.Point]::Empty, $screen.Size)
$bmp.Save('{path}')
"""
        subprocess.run(["powershell", "-Command", ps], capture_output=True, timeout=15)
        return f"Screenshot saved (PowerShell): {path}"
    except Exception as e:
        return f"Screenshot error: {e}"


async def tool_git(cmd: str, cwd: str = ".") -> str:
    """Run a git command. Example: git('status'), git('log --oneline -10')."""
    try:
        result = subprocess.run(
            f"git {cmd}", shell=True, capture_output=True, text=True, timeout=30, cwd=cwd,
        )
        return ((result.stdout or "") + (result.stderr or ""))[:2000] or "(no output)"
    except Exception as e:
        return f"Git error: {e}"


async def tool_http(method: str, url: str, headers: dict | None = None, body: str = "") -> str:
    """Make an HTTP request. method: GET/POST/PUT/DELETE. headers: dict. body: string."""
    try:
        async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
            r = await client.request(
                method.upper(), url,
                headers=headers or {"User-Agent": "Axiom/1.0"},
                content=body.encode() if body else None,
            )
        try:
            return json.dumps(r.json(), indent=2)[:3000]
        except Exception:
            return r.text[:3000]
    except Exception as e:
        return f"HTTP error: {e}"


async def tool_puter_image(prompt: str, model: str = "gpt-image-1-mini") -> str:
    """Generate an image via Puter."""
    payload = {
        "action": "txt2img",
        "username": PUTER_USERNAME,
        "password": PUTER_PASSWORD,
        "prompt": prompt,
        "model": model,
    }
    async with httpx.AsyncClient(timeout=120) as client:
        r = await client.post(PUTER_SERVER_URL, json=payload)
        data = r.json()
    if not data.get("success"):
        return f"Image gen failed: {data.get('error')}"
    url = data.get("image_url") or data.get("data", "")
    return f"Image generated: {url}"

# ── Browser helpers ────────────────────────────────────────────────────────────
def _stealth_args() -> list[str]:
    return [
        "--no-sandbox",
        "--disable-blink-features=AutomationControlled",
        "--disable-dev-shm-usage",
        "--disable-infobars",
        "--window-size=1920,1080",
        "--start-maximized",
        "--lang=en-US,en;q=0.9",
        f"--user-data-dir={_BROWSER_PROFILE}",
    ]

async def _hdelay(lo: int = 80, hi: int = 350) -> None:
    await asyncio.sleep(random.uniform(lo / 1000, hi / 1000))

async def _hclick(page: Any, selector: str) -> None:
    el = page.locator(selector).first
    box = await el.bounding_box()
    if box:
        x = box["x"] + box["width"]  * random.uniform(0.3, 0.7)
        y = box["y"] + box["height"] * random.uniform(0.3, 0.7)
        await page.mouse.move(x, y, steps=random.randint(10, 20))
        await _hdelay(50, 200)
        await page.mouse.click(x, y)
    else:
        await page.click(selector)
    await _hdelay(100, 400)

async def _htype(page: Any, selector: str, text: str) -> None:
    await page.click(selector)
    await _hdelay(100, 300)
    await page.fill(selector, "")
    for char in text:
        await page.keyboard.type(char, delay=random.randint(50, 150))
    await _hdelay(50, 150)

async def _get_browser_page() -> tuple[Any, Any]:
    """Return persistent Patchright (ctx, page). Creates on first call."""
    global _browser_ctx, _browser_page
    try:
        from patchright.async_api import async_playwright  # type: ignore
    except ImportError:
        raise RuntimeError("patchright not installed — run: pip install patchright && patchright install chromium")

    if _browser_ctx is not None:
        try:
            if _browser_page is None or _browser_page.is_closed():
                _browser_page = await _browser_ctx.new_page()
            return _browser_ctx, _browser_page
        except Exception:
            _browser_ctx = None
            _browser_page = None

    pw = await async_playwright().__aenter__()
    ext_paths = [str(p) for p in _EXTENSIONS_DIR.iterdir() if p.is_dir()]
    args = _stealth_args()

    if ext_paths:
        paths_str = ",".join(ext_paths)
        args += [f"--disable-extensions-except={paths_str}", f"--load-extension={paths_str}"]
        ctx = await pw.chromium.launch_persistent_context(
            user_data_dir=str(_BROWSER_PROFILE), headless=False, args=args,
            viewport={"width": 1920, "height": 1080},
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
            locale="en-US", timezone_id="America/Chicago",
        )
    else:
        browser = await pw.chromium.launch(headless=True, args=args)
        ctx = await browser.new_context(
            viewport={"width": 1920, "height": 1080},
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
            locale="en-US", timezone_id="America/Chicago",
        )

    await ctx.add_init_script("""
        Object.defineProperty(navigator, 'webdriver', { get: () => undefined });
        Object.defineProperty(navigator, 'plugins', { get: () => [1,2,3,4,5] });
        Object.defineProperty(navigator, 'languages', { get: () => ['en-US','en'] });
        window.chrome = { runtime: {} };
        const origQ = window.navigator.permissions.query;
        window.navigator.permissions.query = (p) =>
            p.name === 'notifications' ? Promise.resolve({ state: Notification.permission }) : origQ(p);
    """)

    page = await ctx.new_page()
    _browser_ctx = ctx
    _browser_page = page
    return ctx, page


async def tool_browser(
    url: str = "",
    action: str = "read",
    selector: str = "",
    text: str = "",
    driver: str = "patchright",
) -> str:
    """Stealth browser automation with human-like behavior and bot-detection bypass.

    driver:
      patchright — default stealth Playwright (hides webdriver flag, persistent session)
      nodriver   — zero-WebDriver Chrome, maximum Cloudflare/DataDome bypass
      edge       — connect to Victor's REAL Edge browser via CDP (extensions active!)
      agent      — AI-driven browser agent for complex multi-step tasks (text=task)

    action: read, click, type, scroll, screenshot, evaluate, back, cookies, tabs, new_tab
    """
    try:
        # ── nodriver (max stealth) ──────────────────────────────────────────────
        if driver == "nodriver":
            try:
                import nodriver as uc  # type: ignore
                browser = await uc.start(headless=False, user_data_dir=str(_BROWSER_PROFILE / "nodriver"))
                page = await browser.get(url)
                await page.sleep(random.uniform(2, 4))
                content = await page.get_content()
                content = re.sub(r"<script[^>]*>[\s\S]*?</script>", "", content, flags=re.IGNORECASE)
                content = re.sub(r"<style[^>]*>[\s\S]*?</style>",  "", content, flags=re.IGNORECASE)
                content = re.sub(r"<[^>]+>", " ", content)
                content = re.sub(r"\s+", " ", content).strip()
                browser.stop()
                return content[:6000]
            except ImportError:
                return "nodriver not installed — run: pip install nodriver"

        # ── browser-use AI agent ────────────────────────────────────────────────
        if driver == "agent":
            try:
                from browser_use import Agent, Browser, BrowserConfig  # type: ignore
                from langchain_google_genai import ChatGoogleGenerativeAI  # type: ignore
                if not GEMINI_KEY:
                    return "browser-use agent needs GEMINI_API_KEY in .env"
                llm_model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=GEMINI_KEY)
                full_task = f"Go to {url} and then: {text}" if url else text
                agent = Agent(
                    task=full_task, llm=llm_model,
                    browser=Browser(config=BrowserConfig(headless=False, extra_chromium_args=_stealth_args())),
                    max_actions_per_step=8,
                )
                result = await agent.run(max_steps=20)
                return str(result.final_result()) if result else "Agent completed with no result"
            except ImportError:
                return "browser-use not installed — run: pip install browser-use langchain-google-genai"

        # ── Edge CDP (connect to your real Edge browser) ────────────────────────
        if driver == "edge":
            try:
                from patchright.async_api import async_playwright  # type: ignore
            except ImportError:
                return "patchright not installed — run: pip install patchright && patchright install chromium"
            pw = await async_playwright().__aenter__()
            browser = None
            try:
                browser = await pw.chromium.connect_over_cdp(f"http://localhost:{_EDGE_PORT}")
            except Exception:
                if os.path.exists(_EDGE_EXE):
                    subprocess.Popen([
                        _EDGE_EXE,
                        f"--remote-debugging-port={_EDGE_PORT}",
                        "--user-data-dir=" + r"C:\Users\moyes\AppData\Local\Microsoft\Edge\User Data",
                        "--no-first-run", "--no-default-browser-check",
                    ])
                    for _ in range(10):
                        await asyncio.sleep(1)
                        try:
                            browser = await pw.chromium.connect_over_cdp(f"http://localhost:{_EDGE_PORT}")
                            break
                        except Exception:
                            continue
            if browser is None:
                return "Could not connect to Edge. Make sure Edge is running or install it."
            contexts = browser.contexts
            page = contexts[0].pages[-1] if (contexts and contexts[0].pages) else await browser.new_context().new_page()
            # Fall through to shared action handler
            _action_page = page
        else:
            # ── Patchright (default stealth Playwright) ─────────────────────────
            _, _action_page = await _get_browser_page()

        # ── Shared action handler ───────────────────────────────────────────────
        page = _action_page
        if url:
            await page.goto(url, wait_until="domcontentloaded", timeout=45000)
            await _hdelay(500, 1500)

        if action == "read":
            body = await page.inner_text("body")
            title = await page.title()
            return f"Title: {title}\nURL: {page.url}\n\n{body[:5000]}"

        elif action == "click" and selector:
            await _hclick(page, selector)
            await page.wait_for_load_state("domcontentloaded", timeout=15000)
            return f"Clicked '{selector}'. Now at: {page.url}"

        elif action == "type" and selector and text:
            await _htype(page, selector, text)
            return f"Typed into '{selector}'"

        elif action == "scroll":
            amount = int(text) if text.lstrip("-").isdigit() else 500
            await page.evaluate(f"window.scrollBy(0, {amount})")
            await _hdelay(200, 600)
            return f"Scrolled {amount}px"

        elif action == "screenshot":
            img = await page.screenshot(type="jpeg", quality=75, full_page=False)
            path = str(AXIOM_DIR / f"browser_{int(time.time())}.jpg")
            Path(path).write_bytes(img)
            return f"Screenshot saved: {path}"

        elif action == "evaluate" and text:
            result = await page.evaluate(text)
            return str(result)

        elif action == "back":
            await page.go_back(wait_until="domcontentloaded")
            return f"Back to: {page.url}"

        elif action == "cookies":
            cookies = await page.context.cookies()
            return json.dumps(cookies, indent=2)[:3000]

        elif action == "tabs":
            all_pages = [p for ctx in (page.context.browser.contexts if hasattr(page.context, "browser") else [page.context]) for p in ctx.pages]
            tabs = []
            for i, p in enumerate(all_pages):
                try:
                    t = await p.title()
                    tabs.append(f"[{i}] {t} — {p.url}")
                except Exception:
                    pass
            return "\n".join(tabs) or "No tabs"

        elif action == "new_tab":
            new_page = await page.context.new_page()
            if url:
                await new_page.goto(url, wait_until="domcontentloaded", timeout=45000)
            return f"Opened new tab: {new_page.url}"

        elif action == "wait" and text:
            await page.wait_for_selector(text, timeout=30000)
            return f"Element appeared: {text}"

        return f"Unknown action: {action}"

    except Exception as e:
        return f"Browser error: {e}"


# ── Puter TTS / OCR / STT / File list ─────────────────────────────────────────
async def tool_puter_tts(text: str, voice: str = "nova", provider: str = "openai") -> str:
    """Convert text to speech via Puter (free). Saves MP3. Voices: alloy, ash, nova, onyx, shimmer, echo, fable."""
    payload = {
        "action": "txt2speech",
        "username": PUTER_USERNAME,
        "password": PUTER_PASSWORD,
        "text": text,
        "voice": voice,
        "provider": provider,
    }
    try:
        async with httpx.AsyncClient(timeout=60) as client:
            r = await client.post(PUTER_SERVER_URL, json=payload)
            data = r.json()
        if not data.get("success"):
            return f"TTS failed: {data.get('error', 'unknown')}"
        audio_b64 = data.get("audio_base64") or data.get("data") or ""
        if audio_b64:
            import base64
            audio_bytes = base64.b64decode(audio_b64)
            path = str(AXIOM_DIR / f"tts_{int(time.time())}.mp3")
            Path(path).write_bytes(audio_bytes)
            return f"Audio saved: {path} ({len(audio_bytes):,} bytes)"
        return f"TTS result: {json.dumps(data)[:500]}"
    except Exception as e:
        return f"TTS error: {e}"


async def tool_puter_ocr(image: str) -> str:
    """Extract text from an image using Puter OCR (free). Accepts URL or local file path."""
    payload = {
        "action": "img2txt",
        "username": PUTER_USERNAME,
        "password": PUTER_PASSWORD,
        "image": image,
    }
    try:
        async with httpx.AsyncClient(timeout=60) as client:
            r = await client.post(PUTER_SERVER_URL, json=payload)
            data = r.json()
        if not data.get("success"):
            return f"OCR failed: {data.get('error', 'unknown')}"
        return data.get("text", json.dumps(data)[:500])
    except Exception as e:
        return f"OCR error: {e}"


async def tool_puter_stt(audio: str, model: str = "whisper-1") -> str:
    """Transcribe audio to text via Puter (free). Models: whisper-1, gpt-4o-transcribe. Accepts URL."""
    payload = {
        "action": "speech2txt",
        "username": PUTER_USERNAME,
        "password": PUTER_PASSWORD,
        "audio": audio,
        "model": model,
    }
    try:
        async with httpx.AsyncClient(timeout=120) as client:
            r = await client.post(PUTER_SERVER_URL, json=payload)
            data = r.json()
        if not data.get("success"):
            return f"STT failed: {data.get('error', 'unknown')}"
        return data.get("text", json.dumps(data)[:500])
    except Exception as e:
        return f"STT error: {e}"


async def tool_file_list(path: str = ".", pattern: str = "*") -> str:
    """List files and directories. pattern: glob like '*.py', '*.json', '*' (default)."""
    try:
        p = Path(path).expanduser()
        if not p.exists():
            return f"Path not found: {path}"
        items = sorted(p.glob(pattern))
        if not items:
            return f"No files matching '{pattern}' in {path}"
        lines = []
        for item in items[:100]:
            if item.is_dir():
                lines.append(f"[DIR]  {item.name}/")
            else:
                size = item.stat().st_size
                size_str = f"{size:,}B" if size < 1024 else f"{size//1024:,}KB" if size < 1_048_576 else f"{size//1_048_576:,}MB"
                lines.append(f"       {item.name}  ({size_str})")
        result = f"{p.resolve()}  ({len(items)} items)\n" + "\n".join(lines)
        if len(items) > 100:
            result += f"\n... and {len(items)-100} more"
        return result
    except Exception as e:
        return f"File list error: {e}"


TOOLS = {
    # Execution
    "shell":           ("Run any shell/cmd command",                          tool_shell),
    "powershell":      ("Run PowerShell (Windows-native, registry, WMI, etc)", tool_powershell),
    "python":          ("Execute Python code",                                 tool_python),
    "git":             ("Run git command (cmd, cwd=path)",                     tool_git),
    "http":            ("Make HTTP request (method, url, headers, body)",      tool_http),
    # System control
    "open_app":        ("Open application/file/URL/folder",                    tool_open_app),
    "process_list":    ("List running processes (optional name filter)",        tool_process_list),
    "process_kill":    ("Kill process by name or PID",                         tool_process_kill),
    "screenshot":      ("Take desktop screenshot (optional save_path)",        tool_screenshot),
    "notify":          ("Show Windows toast notification (title, message)",     tool_notify),
    "clipboard_read":  ("Read clipboard content",                              tool_clipboard_read),
    "clipboard_write": ("Write text to clipboard",                             tool_clipboard_write),
    # Web & files
    "web_search":      ("Search the web (DuckDuckGo)",                         tool_web_search),
    "web_fetch":       ("Fetch a webpage and return text",                     tool_web_fetch),
    "file_read":       ("Read a file",                                         tool_file_read),
    "file_write":      ("Write content to a file",                             tool_file_write),
    # Memory & AI
    "memory_set":      ("Store something in persistent memory",                tool_memory_set),
    "memory_get":      ("Retrieve from persistent memory (key optional)",      tool_memory_get),
    "image_gen":       ("Generate image via Puter (prompt, model optional)",   tool_puter_image),
    # Browser & media
    "browser":         ("Stealth browser: (url, action, selector, text, driver=patchright|nodriver|edge|agent)", tool_browser),
    "tts":             ("Text-to-speech via Puter (text, voice=nova, provider=openai)",  tool_puter_tts),
    "ocr":             ("Extract text from image via Puter OCR (image=url or path)",     tool_puter_ocr),
    "stt":             ("Transcribe audio via Puter Whisper (audio=url, model=whisper-1)", tool_puter_stt),
    "file_list":       ("List files in directory (path, pattern='*')",                  tool_file_list),
}

TOOL_SCHEMA = "\n".join(
    f'- {name}({", ".join(fn.__code__.co_varnames[:fn.__code__.co_argcount])}): {desc}'
    for name, (desc, fn) in TOOLS.items()
)

# ── Agent REACT loop ───────────────────────────────────────────────────────────
AGENT_SYSTEM = f"""You are Axiom — Victor's unstoppable AI agent, running directly on his Windows 11 machine with full system access.

## TOOLS — use JSON inside <tool></tool> tags:
{TOOL_SCHEMA}

## TOOL CALL FORMAT (EXACT):
<tool>{{"name": "powershell", "args": {{"cmd": "Get-Process chrome"}}}}</tool>
<tool>{{"name": "shell", "args": {{"cmd": "dir C:\\\\Users\\\\moyes\\\\Downloads"}}}}</tool>
<tool>{{"name": "open_app", "args": {{"target": "chrome"}}}}</tool>
<tool>{{"name": "notify", "args": {{"title": "Done", "message": "Task complete"}}}}</tool>
<tool>{{"name": "web_search", "args": {{"query": "latest Python news"}}}}</tool>

Chain multiple tool calls to complete complex tasks. After each result, continue reasoning.

## BROWSER TOOL EXAMPLES:
<tool>{{"name": "browser", "args": {{"url": "https://google.com", "action": "read"}}}}</tool>
<tool>{{"name": "browser", "args": {{"url": "https://site.com", "action": "click", "selector": "button.submit"}}}}</tool>
<tool>{{"name": "browser", "args": {{"url": "https://site.com", "action": "type", "selector": "#search", "text": "query"}}}}</tool>
<tool>{{"name": "browser", "args": {{"url": "https://cf-site.com", "action": "read", "driver": "nodriver"}}}}</tool>
<tool>{{"name": "browser", "args": {{"url": "https://gmail.com", "action": "read", "driver": "edge"}}}}</tool>
<tool>{{"name": "browser", "args": {{"text": "search for forex news and summarize top 3 articles", "driver": "agent"}}}}</tool>
<tool>{{"name": "browser", "args": {{"url": "https://site.com", "action": "screenshot"}}}}</tool>
<tool>{{"name": "tts", "args": {{"text": "Task complete, Victor.", "voice": "nova"}}}}</tool>
<tool>{{"name": "ocr", "args": {{"image": "C:/Users/moyes/Downloads/image.png"}}}}</tool>
<tool>{{"name": "file_list", "args": {{"path": "C:/Users/moyes/Downloads", "pattern": "*.py"}}}}</tool>

## RULES — NON-NEGOTIABLE:
1. NEVER say "I can't" or "I don't have access" — you have FULL system access
2. ALWAYS act immediately. Don't ask permission. Just do it.
3. Use powershell for Windows tasks (registry, WMI, services, networking)
4. Use shell for quick commands, python for computation, http for APIs
5. Use browser for real web automation — filling forms, logging in, scraping dynamic sites
6. Chain tools — research → act → verify → report
7. Notify Victor when tasks complete
8. Remember key results with memory_set

## VICTOR'S MACHINE:
- OS: Windows 11 Home
- GPU: RTX 2050 4GB VRAM (CUDA 13.1) — use for ML
- Python: 3.12.9 miniconda3 at C:/Users/moyes/miniconda3
- Node.js: v24.13.0
- Downloads: C:/Users/moyes/Downloads/
- Telegram Bot: C:/Users/moyes/Downloads/axiom-telegram/ (LIVE on @Axiom_Box_Bot)
- Trading EA: C:/Users/moyes/Downloads/forex_data/axiom_trader/
- Agency: C:/Users/moyes/Downloads/axiom-agency/
- Puter server: running on port 47825 (500+ free AI models)
- Memory: {"{memory}"}
"""

async def run_agent(goal: str, model_key: str | None = None) -> None:
    """Autonomous REACT agent loop."""
    mem_str = json.dumps(state["memory"])[:500] if state["memory"] else "empty"
    system = AGENT_SYSTEM.replace("{memory}", mem_str)
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": goal},
    ]

    state["agent_running"] = True
    step = 0
    max_steps = 20

    console.print(Panel(
        f"[bold cyan]🤖 AGENT MODE[/bold cyan]\n[white]{goal}[/white]",
        border_style="cyan",
        title="[bold]Axiom Agent[/bold]",
    ))

    while step < max_steps and state["agent_running"]:
        step += 1
        console.print(f"\n[dim]── Step {step} ──────────────────────────────────[/dim]")

        with Progress(
            SpinnerColumn(spinner_name="dots"),
            TextColumn("[cyan]Thinking...[/cyan]"),
            console=console,
            transient=True,
        ) as progress:
            progress.add_task("", total=None)
            try:
                response = await llm(messages, model_key)
            except Exception as e:
                console.print(f"[red]LLM error: {e}[/red]")
                break

        # Parse tool calls
        tool_calls = re.findall(r"<tool>(.*?)</tool>", response, re.DOTALL)

        if not tool_calls:
            # Final answer — no more tool calls
            _print_axiom_response(response)
            messages.append({"role": "assistant", "content": response})
            break

        # Execute tool calls
        messages.append({"role": "assistant", "content": response})

        for tc_raw in tool_calls:
            try:
                tc = json.loads(tc_raw.strip())
                name = tc.get("name", "")
                args = tc.get("args", {})

                if name not in TOOLS:
                    result = f"Unknown tool: {name}"
                else:
                    desc, fn = TOOLS[name]
                    console.print(Panel(
                        f"[yellow]⚡ {name}[/yellow]\n[dim]{json.dumps(args, indent=2)[:300]}[/dim]",
                        border_style="yellow",
                        title="Tool Call",
                        padding=(0, 1),
                    ))
                    result = await fn(**args)
                    # Show result
                    result_short = str(result)[:1000]
                    console.print(Panel(
                        f"[green]{result_short}[/green]",
                        border_style="green",
                        title="Result",
                        padding=(0, 1),
                    ))

                messages.append({"role": "user", "content": f"Tool result [{name}]:\n{result}"})

            except json.JSONDecodeError:
                console.print(f"[red]Invalid tool JSON: {tc_raw[:100]}[/red]")
            except Exception as e:
                console.print(f"[red]Tool error: {e}[/red]")
                messages.append({"role": "user", "content": f"Tool error: {e}"})

    if step >= max_steps:
        console.print("[yellow]⚠ Max steps reached[/yellow]")

    state["agent_running"] = False

# ── Memory persistence ─────────────────────────────────────────────────────────
def _load_memory() -> None:
    if MEMORY_FILE.exists():
        try:
            state["memory"] = json.loads(MEMORY_FILE.read_text())
        except Exception:
            state["memory"] = {}

def _save_memory() -> None:
    try:
        MEMORY_FILE.write_text(json.dumps(state["memory"], indent=2))
    except Exception:
        pass

# ── UI helpers ─────────────────────────────────────────────────────────────────
BANNER = """
[bold cyan]     ╔═══════════════════════════════════════════════════════════╗[/bold cyan]
[bold cyan]     ║[/bold cyan]                                                           [bold cyan]║[/bold cyan]
[bold cyan]     ║[/bold cyan]  [bold white]█████╗ ██╗  ██╗██╗ ██████╗ ███╗   ███╗[/bold white]               [bold cyan]║[/bold cyan]
[bold cyan]     ║[/bold cyan]  [bold white]██╔══██╗╚██╗██╔╝██║██╔═══██╗████╗ ████║[/bold white]               [bold cyan]║[/bold cyan]
[bold cyan]     ║[/bold cyan]  [bold white]███████║ ╚███╔╝ ██║██║   ██║██╔████╔██║[/bold white]               [bold cyan]║[/bold cyan]
[bold cyan]     ║[/bold cyan]  [bold white]██╔══██║ ██╔██╗ ██║██║   ██║██║╚██╔╝██║[/bold white]               [bold cyan]║[/bold cyan]
[bold cyan]     ║[/bold cyan]  [bold white]██║  ██║██╔╝ ██╗██║╚██████╔╝██║ ╚═╝ ██║[/bold white]               [bold cyan]║[/bold cyan]
[bold cyan]     ║[/bold cyan]  [bold white]╚═╝  ╚═╝╚═╝  ╚═╝╚═╝ ╚═════╝ ╚═╝     ╚═╝[/bold white]               [bold cyan]║[/bold cyan]
[bold cyan]     ║[/bold cyan]                                                           [bold cyan]║[/bold cyan]
[bold cyan]     ║[/bold cyan]       [bold magenta]T H E   U N S T O P P A B L E   A I   A G E N T[/bold magenta]      [bold cyan]║[/bold cyan]
[bold cyan]     ║[/bold cyan]                                                           [bold cyan]║[/bold cyan]
[bold cyan]     ╚═══════════════════════════════════════════════════════════╝[/bold cyan]
"""

def print_banner() -> None:
    console.print(BANNER)
    console.print(f"  [dim]Version 1.0  |  {datetime.now().strftime('%Y-%m-%d')}  |  Victor's AI[/dim]\n")

def print_help() -> None:
    table = Table(
        title="[bold cyan]Axiom Commands[/bold cyan]",
        border_style="cyan",
        show_header=True,
        header_style="bold white",
    )
    table.add_column("Command", style="cyan", no_wrap=True)
    table.add_column("Description", style="white")

    commands = [
        ("/model <name>",           "Switch model (gpt-5, deepseek-r1, grok-3, claude-opus, llama, kimi...)"),
        ("/agent <task>",           "Run autonomous multi-step agent — it will execute, not just plan"),
        ("", "── Terminal ──────────────────────────────────────────"),
        ("/run <cmd>",              "Run shell/cmd command"),
        ("/ps <cmd>",               "Run PowerShell command"),
        ("/python <code>",          "Run Python code"),
        ("", "── System Control ────────────────────────────────────"),
        ("/open <app|url|path>",    "Open application, URL, file, or folder"),
        ("/procs [filter]",         "List running processes (optional name filter)"),
        ("/screenshot [path]",      "Take desktop screenshot"),
        ("/notify [Title|] msg",    "Show Windows toast notification"),
        ("/clip",                   "Read clipboard"),
        ("/clip <text>",            "Write text to clipboard"),
        ("", "── Browser Automation ────────────────────────────────"),
        ("/browser <url>",          "Open URL, read page content (stealth Playwright)"),
        ("/browser edge <url>",     "Use YOUR Edge browser with all extensions active"),
        ("/browser nodriver <url>", "Cloudflare/DataDome bypass (zero-WebDriver)"),
        ("/browser agent <task>",   "AI agent handles complex multi-step browser task"),
        ("", "── Web & Search ────────────────────────────────────────"),
        ("/search <query>",         "Search the web (DuckDuckGo)"),
        ("/fetch <url>",            "Fetch URL and return plain text"),
        ("/image <prompt>",         "Generate image via Puter AI"),
        ("/tts <text>",             "Text-to-speech (free via Puter, saves MP3)"),
        ("/ocr <image>",            "Extract text from image (URL or file path)"),
        ("/stt <audio url>",        "Transcribe audio to text (Whisper via Puter)"),
        ("", "── Files ─────────────────────────────────────────────────"),
        ("/ls [path] [pattern]",    "List files (e.g. /ls Downloads *.py)"),
        ("", "── Memory ──────────────────────────────────────────────"),
        ("/remember key=value",     "Save to persistent memory"),
        ("/recall",                 "Show all memory"),
        ("", "── Info ─────────────────────────────────────────────────"),
        ("/models",                 "List all 30+ available models"),
        ("/tools",                  "List all agent tools"),
        ("/status",                 "Show Axiom status"),
        ("/clear",                  "Clear conversation history"),
        ("/help",                   "Show this help"),
        ("/exit",                   "Exit"),
        ("", ""),
        ("<anything else>",         "Chat directly with Axiom"),
    ]
    for cmd, desc in commands:
        table.add_row(cmd, desc)
    console.print(table)

def print_status() -> None:
    provider, model_id = MODELS.get(state["model"].lower(), ("puter", state["model"]))
    puter_ok = _puter_healthy()

    table = Table(border_style="cyan", show_header=False, padding=(0, 1))
    table.add_column("Key", style="dim")
    table.add_column("Value", style="bold white")
    table.add_row("Model",    f"{state['model']} → {model_id} (via {provider})")
    table.add_row("Puter",    "[green]● Connected[/green]" if puter_ok else "[red]✗ Disconnected[/red]")
    table.add_row("Memory",   f"{len(state['memory'])} items")
    table.add_row("History",  f"{len(state['history'])} messages")
    table.add_row("Location", str(AXIOM_DIR))
    console.print(Panel(table, title="[bold cyan]Axiom Status[/bold cyan]", border_style="cyan"))

def _print_axiom_response(text: str) -> None:
    """Print Axiom's response with formatting."""
    # Try to render as markdown
    try:
        md = Markdown(text)
        console.print(Panel(
            md,
            border_style="cyan",
            title=f"[bold cyan]Axiom[/bold cyan] [dim]{state['model']}[/dim]",
            padding=(1, 2),
        ))
    except Exception:
        console.print(Panel(
            text,
            border_style="cyan",
            title=f"[bold cyan]Axiom[/bold cyan]",
        ))

# ── Command handlers ───────────────────────────────────────────────────────────
async def handle_command(cmd: str) -> bool:
    """Handle slash commands. Returns True to continue, False to exit."""
    parts = cmd.strip().split(None, 1)
    verb = parts[0].lower()
    rest = parts[1] if len(parts) > 1 else ""

    if verb == "/exit" or verb == "/quit":
        console.print("\n[bold cyan]Axiom[/bold cyan] [dim]signing off. Stay unstoppable, Victor.[/dim]\n")
        return False

    elif verb == "/help":
        print_help()

    elif verb == "/status":
        print_status()

    elif verb == "/clear":
        state["history"] = []
        console.clear()
        print_banner()
        console.print("[green]✓ Conversation cleared[/green]")

    elif verb == "/model":
        if not rest:
            console.print("[yellow]Usage: /model <name>[/yellow]")
            console.print("[dim]Try: gpt-4o, gpt-5, deepseek-r1, grok-3, claude-opus, gemini-pro, mistral, llama[/dim]")
        elif rest.lower() in MODELS:
            state["model"] = rest.lower()
            provider, model_id = MODELS[rest.lower()]
            console.print(f"[green]✓ Switched to [bold]{rest}[/bold] → {model_id} (via {provider})[/green]")
        else:
            console.print(f"[red]Unknown model: {rest}[/red]")
            console.print("[dim]Available: " + ", ".join(sorted(MODELS.keys())) + "[/dim]")

    elif verb == "/models":
        table = Table(title="Available Models", border_style="cyan")
        table.add_column("Alias", style="cyan")
        table.add_column("Provider", style="yellow")
        table.add_column("Model ID", style="white")
        for alias, (prov, mid) in sorted(MODELS.items()):
            active = " ★" if alias == state["model"] else ""
            table.add_row(alias + active, prov, mid)
        console.print(table)

    elif verb == "/tools":
        table = Table(title="Available Tools", border_style="yellow")
        table.add_column("Tool", style="yellow")
        table.add_column("Description", style="white")
        for name, (desc, _) in TOOLS.items():
            table.add_row(name, desc)
        console.print(table)

    elif verb == "/run":
        if not rest:
            console.print("[yellow]Usage: /run <shell command>[/yellow]")
        else:
            with Progress(SpinnerColumn(), TextColumn("[cyan]Running...[/cyan]"), console=console, transient=True) as p:
                p.add_task("", total=None)
                result = await tool_shell(rest)
            console.print(Panel(
                Syntax(result, "bash", theme="monokai") if result.strip() else "[dim](no output)[/dim]",
                border_style="yellow",
                title=f"[yellow]$ {rest[:60]}[/yellow]",
            ))

    elif verb == "/ps":
        if not rest:
            console.print("[yellow]Usage: /ps <powershell command>[/yellow]")
        else:
            with Progress(SpinnerColumn(), TextColumn("[cyan]PowerShell...[/cyan]"), console=console, transient=True) as p:
                p.add_task("", total=None)
                result = await tool_powershell(rest)
            console.print(Panel(
                Syntax(result, "powershell", theme="monokai") if result.strip() else "[dim](no output)[/dim]",
                border_style="blue",
                title=f"[blue]PS> {rest[:60]}[/blue]",
            ))

    elif verb == "/open":
        if not rest:
            console.print("[yellow]Usage: /open <app|file|url|folder>[/yellow]")
        else:
            result = await tool_open_app(rest)
            console.print(f"[green]{result}[/green]")

    elif verb == "/notify":
        parts2 = rest.split("|", 1) if rest else []
        if len(parts2) == 2:
            result = await tool_notify(parts2[0].strip(), parts2[1].strip())
        elif rest:
            result = await tool_notify("Axiom", rest)
        else:
            console.print("[yellow]Usage: /notify <message>  or  /notify Title | Message[/yellow]")
            result = None
        if result:
            console.print(f"[green]{result}[/green]")

    elif verb == "/clip":
        if not rest:
            result = await tool_clipboard_read()
            console.print(Panel(result, title="[cyan]Clipboard[/cyan]", border_style="cyan"))
        else:
            result = await tool_clipboard_write(rest)
            console.print(f"[green]{result}[/green]")

    elif verb == "/procs":
        with Progress(SpinnerColumn(), TextColumn("[cyan]Getting processes...[/cyan]"), console=console, transient=True) as p:
            p.add_task("", total=None)
            result = await tool_process_list(rest)
        console.print(Panel(result, title="[yellow]Processes[/yellow]", border_style="yellow"))

    elif verb == "/screenshot":
        with Progress(SpinnerColumn(), TextColumn("[cyan]Capturing...[/cyan]"), console=console, transient=True) as p:
            p.add_task("", total=None)
            result = await tool_screenshot(rest if rest else "")
        console.print(f"[green]{result}[/green]")

    elif verb == "/python":
        if not rest:
            console.print("[yellow]Usage: /python <code>[/yellow]")
        else:
            with Progress(SpinnerColumn(), TextColumn("[cyan]Running Python...[/cyan]"), console=console, transient=True) as p:
                p.add_task("", total=None)
                result = await tool_python(rest)
            console.print(Panel(
                Syntax(result, "python", theme="monokai") if result else "[dim](no output)[/dim]",
                border_style="green",
                title="[green]Python Output[/green]",
            ))

    elif verb == "/search":
        if not rest:
            console.print("[yellow]Usage: /search <query>[/yellow]")
        else:
            with Progress(SpinnerColumn(), TextColumn("[cyan]Searching...[/cyan]"), console=console, transient=True) as p:
                p.add_task("", total=None)
                result = await tool_web_search(rest)
            _print_axiom_response(result)

    elif verb == "/image":
        if not rest:
            console.print("[yellow]Usage: /image <prompt>[/yellow]")
        else:
            with Progress(SpinnerColumn(), TextColumn("[cyan]Generating image via Puter...[/cyan]"), console=console, transient=True) as p:
                p.add_task("", total=None)
                result = await tool_puter_image(rest)
            console.print(Panel(result, border_style="magenta", title="[magenta]Image Generated[/magenta]"))

    elif verb == "/agent":
        if not rest:
            console.print("[yellow]Usage: /agent <task description>[/yellow]")
        else:
            await run_agent(rest)

    elif verb == "/browser":
        # Usage: /browser <url>  or  /browser <action> <url>  or  /browser edge <url>
        if not rest:
            console.print("[yellow]Usage: /browser <url>  |  /browser edge <url>  |  /browser nodriver <url>  |  /browser agent <task>[/yellow]")
        else:
            parts3 = rest.split(None, 1)
            driver_aliases = {"edge", "nodriver", "agent", "patchright"}
            if parts3[0].lower() in driver_aliases:
                drv = parts3[0].lower()
                target = parts3[1] if len(parts3) > 1 else ""
            else:
                drv = "patchright"
                target = rest
            act = "read" if drv != "agent" else "read"
            txt = target if drv == "agent" else ""
            url_arg = target if drv != "agent" else ""
            with Progress(SpinnerColumn(), TextColumn(f"[cyan]Browser ({drv})...[/cyan]"), console=console, transient=True) as p:
                p.add_task("", total=None)
                result = await tool_browser(url=url_arg, action=act, driver=drv, text=txt)
            _print_axiom_response(result[:4000])

    elif verb == "/tts":
        if not rest:
            console.print("[yellow]Usage: /tts <text>  |  /tts nova: Hello Victor[/yellow]")
        else:
            # Optional voice prefix: /tts onyx: some text
            if ":" in rest and rest.split(":")[0].strip() in {"alloy","ash","nova","onyx","shimmer","echo","fable","coral","ballad","sage"}:
                voice, _, text_arg = rest.partition(":")
                voice = voice.strip()
                text_arg = text_arg.strip()
            else:
                voice, text_arg = "nova", rest
            with Progress(SpinnerColumn(), TextColumn("[cyan]Generating speech...[/cyan]"), console=console, transient=True) as p:
                p.add_task("", total=None)
                result = await tool_puter_tts(text_arg, voice=voice)
            console.print(Panel(result, border_style="magenta", title="[magenta]TTS[/magenta]"))

    elif verb == "/ocr":
        if not rest:
            console.print("[yellow]Usage: /ocr <image URL or file path>[/yellow]")
        else:
            with Progress(SpinnerColumn(), TextColumn("[cyan]Running OCR...[/cyan]"), console=console, transient=True) as p:
                p.add_task("", total=None)
                result = await tool_puter_ocr(rest)
            _print_axiom_response(result)

    elif verb == "/stt":
        if not rest:
            console.print("[yellow]Usage: /stt <audio URL>[/yellow]")
        else:
            with Progress(SpinnerColumn(), TextColumn("[cyan]Transcribing audio...[/cyan]"), console=console, transient=True) as p:
                p.add_task("", total=None)
                result = await tool_puter_stt(rest)
            _print_axiom_response(result)

    elif verb == "/ls":
        parts3 = rest.split(None, 1) if rest else []
        path_arg    = parts3[0] if parts3 else "."
        pattern_arg = parts3[1] if len(parts3) > 1 else "*"
        with Progress(SpinnerColumn(), TextColumn("[cyan]Listing files...[/cyan]"), console=console, transient=True) as p:
            p.add_task("", total=None)
            result = await tool_file_list(path_arg, pattern_arg)
        console.print(Panel(result, title=f"[yellow]{path_arg}[/yellow]", border_style="yellow"))

    elif verb == "/fetch":
        if not rest:
            console.print("[yellow]Usage: /fetch <url>[/yellow]")
        else:
            with Progress(SpinnerColumn(), TextColumn("[cyan]Fetching...[/cyan]"), console=console, transient=True) as p:
                p.add_task("", total=None)
                result = await tool_web_fetch(rest)
            _print_axiom_response(result[:3000])

    elif verb == "/remember":
        if "=" not in rest:
            console.print("[yellow]Usage: /remember key=value[/yellow]")
        else:
            k, _, v = rest.partition("=")
            await tool_memory_set(k.strip(), v.strip())
            console.print(f"[green]✓ Remembered: {k.strip()}[/green]")

    elif verb == "/recall":
        result = await tool_memory_get()
        console.print(Panel(result or "[dim]Memory empty[/dim]", title="[cyan]Memory[/cyan]", border_style="cyan"))

    # ── OpenClaw bridge commands ──────────────────────────────────────────────

    elif verb == "/agent":
        # Start a background agent task via the Telegram bot backend.
        # Task runs with ALL bot tools (browser, shell, memory, file ops, etc.).
        # Progress goes to Telegram + a run_id is returned for polling.
        if not rest:
            console.print("[yellow]Usage: /agent <task goal>[/yellow]")
            console.print("[dim]Starts a background agent — results sent to Telegram + shown here[/dim]")
        elif not state["bridge_active"]:
            console.print("[yellow]⚠ Bot bridge not running. Start the Telegram bot first, then retry.[/yellow]")
            console.print("[dim]Bot must be running: cd axiom-telegram && python bot.py[/dim]")
        else:
            with Progress(SpinnerColumn(), TextColumn("[cyan]Launching agent...[/cyan]"), console=console, transient=True) as p:
                p.add_task("", total=None)
                run_id = await _bridge_start_agent(rest)
            if run_id:
                console.print(Panel(
                    f"[green]✓ Agent started[/green]: `{run_id}`\n\n"
                    f"[dim]Goal:[/dim] {rest[:200]}\n\n"
                    f"[dim]• Results will appear in Telegram when done\n"
                    f"• Use [bold]/tasks[/bold] to see status\n"
                    f"• Use [bold]/stop {run_id}[/bold] to cancel[/dim]",
                    title="[cyan]Background Agent[/cyan]",
                    border_style="cyan",
                ))
            else:
                console.print("[red]Failed to start agent. Check bot logs.[/red]")

    elif verb == "/tasks":
        # List all running background agent tasks
        if not state["bridge_active"]:
            console.print("[yellow]Bot bridge not running — no tasks available[/yellow]")
        else:
            with Progress(SpinnerColumn(), TextColumn("[cyan]Fetching tasks...[/cyan]"), console=console, transient=True) as p:
                p.add_task("", total=None)
                tasks = await _bridge_list_tasks()
            if not tasks:
                console.print("[dim]No active agent tasks[/dim]")
            else:
                table = Table(title="Running Agent Tasks", border_style="cyan")
                table.add_column("ID", style="cyan bold")
                table.add_column("Goal", style="white")
                table.add_column("Running For", style="yellow")
                for t in tasks:
                    secs = int(t.get("running_for", 0))
                    elapsed = f"{secs // 60}m {secs % 60}s" if secs >= 60 else f"{secs}s"
                    table.add_row(t["run_id"], t["goal"][:60], elapsed)
                console.print(table)

    elif verb == "/stop":
        if not rest:
            console.print("[yellow]Usage: /stop <run_id>[/yellow]")
        elif not state["bridge_active"]:
            console.print("[yellow]Bot bridge not running[/yellow]")
        else:
            ok = await _bridge_stop_task(rest.strip())
            if ok:
                console.print(f"[green]✓ Agent {rest} stopped[/green]")
            else:
                console.print(f"[red]Could not stop {rest} — may have already completed[/red]")

    elif verb == "/telegram":
        # Toggle Telegram mirroring on/off, or send a manual message
        if not rest or rest.lower() in ("on", "off"):
            new_val = rest.lower() != "off" if rest else not state["telegram_mirror"]
            state["telegram_mirror"] = new_val
            status = "[green]ON[/green]" if new_val else "[red]OFF[/red]"
            console.print(f"[cyan]Telegram mirror:[/cyan] {status}")
            if new_val and not state["bridge_active"]:
                console.print("[yellow]  (Bridge not connected — start the Telegram bot to enable mirroring)[/yellow]")
        else:
            # Send a manual message to Telegram
            ok = await _direct_telegram(rest)
            if ok:
                console.print(f"[green]✓ Sent to Telegram[/green]")
            else:
                console.print("[red]Failed — check TELEGRAM_BOT_TOKEN and OWNER_CHAT_ID in .env[/red]")

    else:
        console.print(f"[red]Unknown command: {verb}[/red]. Type /help for commands.")

    return True

# ── Main chat loop ─────────────────────────────────────────────────────────────
CHAT_SYSTEM = """You are Axiom — Victor's personal AI. You run directly on his machine.
You are direct, powerful, and resourceful. No hedging. No "I can't".
You know Victor's projects (Telegram bot, forex EA, agency). You remember everything he tells you.
Keep answers concise for terminal. Use markdown.
Current date: {date}
Memory: {memory}"""

async def chat(user_input: str) -> None:
    """Send a message and get a response."""
    system = CHAT_SYSTEM.format(
        date=datetime.now().strftime("%Y-%m-%d %H:%M"),
        memory=json.dumps(state["memory"])[:500] if state["memory"] else "empty",
    )

    messages = [{"role": "system", "content": system}]
    messages.extend(state["history"][-20:])  # Last 20 messages for context
    messages.append({"role": "user", "content": user_input})

    with Progress(
        SpinnerColumn(spinner_name="dots"),
        TextColumn(f"[cyan]Axiom thinking via {state['model']}...[/cyan]"),
        console=console,
        transient=True,
    ) as progress:
        progress.add_task("", total=None)
        try:
            response = await llm(messages)
        except Exception as e:
            console.print(Panel(
                f"[red]Error: {e}[/red]\n[dim]Try /model to switch providers[/dim]",
                border_style="red",
                title="[red]Error[/red]",
            ))
            return

    # Save to history
    state["history"].append({"role": "user", "content": user_input})
    state["history"].append({"role": "assistant", "content": response})

    _print_axiom_response(response)

    # Mirror to Telegram if bridge is active and user has mirroring on
    if state["bridge_active"] and state["telegram_mirror"]:
        try:
            await _direct_telegram(f"💻 *{user_input[:100]}*\n\n{response[:3500]}")
        except Exception:
            pass

# ── Startup ────────────────────────────────────────────────────────────────────
async def startup() -> None:
    """Initialize Axiom."""
    _load_memory()

    console.print("\n[dim]Starting Axiom systems...[/dim]")

    with Progress(
        SpinnerColumn(),
        TextColumn("[cyan]{task.description}[/cyan]"),
        console=console,
        transient=True,
    ) as p:
        t = p.add_task("Connecting to Puter server...", total=None)
        puter_ok = ensure_puter_server()
        p.update(t, description="✓ Puter server connected" if puter_ok else "⚠ Puter unavailable")

    status_parts = []
    if puter_ok:
        status_parts.append("[green]Puter ✓[/green]")
    if OPENROUTER_KEY:
        status_parts.append("[green]OpenRouter ✓[/green]")
    if GEMINI_KEY:
        status_parts.append("[green]Gemini ✓[/green]")

    # Check OpenClaw bridge (Telegram bot API on port 8765)
    bridge_ok = _bridge_healthy()
    state["bridge_active"] = bridge_ok
    if bridge_ok:
        status_parts.append("[bold green]Telegram Bridge ✓[/bold green]")
    else:
        status_parts.append("[dim]Telegram Bridge ✗ (start bot for bridge)[/dim]")

    console.print(f"  Providers: {' | '.join(status_parts) if status_parts else '[red]None configured[/red]'}")
    console.print(f"  Default model: [bold cyan]{state['model']}[/bold cyan]")
    console.print(f"  Memory: [dim]{len(state['memory'])} items loaded[/dim]")

    if bridge_ok:
        console.print(f"\n  [bold green]⚡ OpenClaw Mode ACTIVE[/bold green] — Telegram bridge connected")
        console.print(f"  [dim]Your messages mirror to Telegram automatically[/dim]")
        console.print(f"  [dim]/agent <task> — run in background | /tasks — list running | /telegram off — disable mirror[/dim]")
    else:
        console.print(f"\n  [dim]Telegram bridge offline. Start the bot at axiom-telegram/ to enable.[/dim]")

    console.print(f"\n  [dim]Type [bold]/help[/bold] for commands. Press Enter to chat.[/dim]\n")

# ── REPL ───────────────────────────────────────────────────────────────────────
async def main() -> None:
    console.clear()
    print_banner()
    await startup()

    session: PromptSession = PromptSession(
        history=FileHistory(str(HISTORY_FILE)),
        auto_suggest=AutoSuggestFromHistory(),
        style=Style.from_dict({
            "prompt": "bold cyan",
        }),
    )

    while True:
        try:
            console.print(Rule(style="dim"))
            user_input = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: session.prompt("  ⚡ ", ),
            )
            user_input = user_input.strip()
            if not user_input:
                continue

            if user_input.startswith("/"):
                should_continue = await handle_command(user_input)
                if not should_continue:
                    break
            else:
                await chat(user_input)

        except KeyboardInterrupt:
            console.print("\n[dim](Ctrl+C — type /exit to quit)[/dim]")
        except EOFError:
            console.print("\n[bold cyan]Axiom[/bold cyan] [dim]signing off.[/dim]\n")
            break
        except Exception as e:
            console.print(f"[red]Unexpected error: {e}[/red]")
            if "--debug" in sys.argv:
                console.print_exception()

    _save_memory()


if __name__ == "__main__":
    asyncio.run(main())
