"""Default constants for Axiom CLI."""

from pathlib import Path

# ── Core Defaults ──────────────────────────────────────────────
DEFAULT_MODEL = "anthropic/claude-opus-4-6"
MAX_ITERATIONS = 25
CONTEXT_WINDOW = 200_000

# ── Directory Paths ────────────────────────────────────────────
# Resolve project root: walk up from this file until we find pyproject.toml
# This allows paths to work both in dev (pip install -e .) and installed mode.
def _find_project_root() -> Path:
    """Locate the project root by searching for pyproject.toml."""
    current = Path(__file__).resolve().parent
    for _ in range(5):
        if (current / "pyproject.toml").exists():
            return current
        current = current.parent
    # Fallback: use CWD if pyproject.toml not found (installed mode)
    cwd = Path.cwd()
    if (cwd / "memory").exists():
        return cwd
    return Path.home() / ".axiom"

PROJECT_ROOT = _find_project_root()

# When running from the project directory, use local paths.
# When installed globally, fall back to ~/.axiom.
AXIOM_HOME = PROJECT_ROOT if (PROJECT_ROOT / "memory").exists() else Path.home() / ".axiom"
MEMORY_DIR = AXIOM_HOME / "memory"
SESSIONS_DIR = MEMORY_DIR / "sessions"
FACTS_DIR = MEMORY_DIR / "facts"
SKILLS_DIR = MEMORY_DIR / "skills"
TRACES_DIR = MEMORY_DIR / "traces"
CHROMA_DIR = MEMORY_DIR / "chroma"
TOOLS_DIR = AXIOM_HOME / "tools"
AGENTS_DIR = AXIOM_HOME / "agents"
WORKSPACE_DIR = AXIOM_HOME / "workspace"
MCP_SERVERS_FILE = AXIOM_HOME / "mcp_servers" / "servers.json"

# ── Provider → Best Model Mapping (litellm format) ────────────
MODEL_DEFAULTS: dict[str, str] = {
    "anthropic": "anthropic/claude-opus-4-6",
    "openai": "openai/gpt-4o",
    "groq": "groq/llama-3.3-70b-versatile",
    "gemini": "gemini/gemini-2.5-flash",
    "nvidia_nim": "nvidia_nim/meta/llama-3.3-70b-instruct",
    "deepseek": "deepseek/deepseek-chat",
    "together_ai": "together_ai/meta-llama/Llama-4-Scout-17B-16E-Instruct",
    "mistral": "mistral/mistral-small-latest",
    "cohere_chat": "cohere_chat/command-r-plus",
    "openrouter": "openrouter/auto",
    "huggingface": "huggingface/meta-llama/Llama-3.3-70B-Instruct",
    "ollama": "ollama/qwen2.5:7b",
    "vertex_ai": "vertex_ai/gemini-2.5-pro",
}

# ── Display Names ──────────────────────────────────────────────
PROVIDER_DISPLAY_NAMES: dict[str, str] = {
    "anthropic": "Anthropic",
    "openai": "OpenAI",
    "groq": "Groq",
    "gemini": "Google Gemini",
    "nvidia_nim": "NVIDIA NIM",
    "deepseek": "DeepSeek",
    "together_ai": "Together AI",
    "mistral": "Mistral AI",
    "cohere_chat": "Cohere",
    "openrouter": "OpenRouter",
    "huggingface": "Hugging Face",
    "ollama": "Ollama (Local)",
    "vertex_ai": "Google Vertex AI",
}
