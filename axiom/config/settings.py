"""Pydantic Settings for Axiom CLI -- loads from .env and environment."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict

from axiom.config.defaults import MODEL_DEFAULTS


class AxiomSettings(BaseSettings):
    """Central configuration loaded from environment / .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── LLM Provider API Keys ──────────────────────────────────
    ANTHROPIC_API_KEY: Optional[str] = None
    OPENAI_API_KEY: Optional[str] = None
    VERTEX_AI_PROJECT: Optional[str] = None
    VERTEX_AI_LOCATION: Optional[str] = None
    GOOGLE_APPLICATION_CREDENTIALS: Optional[str] = None
    TOGETHER_API_KEY: Optional[str] = None
    NVIDIA_API_KEY: Optional[str] = None
    GROQ_API_KEY: Optional[str] = None
    DEEPSEEK_API_KEY: Optional[str] = None
    GEMINI_API_KEY: Optional[str] = None
    MISTRAL_API_KEY: Optional[str] = None
    COHERE_API_KEY: Optional[str] = None
    OPENROUTER_API_KEY: Optional[str] = None
    HUGGINGFACE_API_KEY: Optional[str] = None

    # ── Puter Cloud Credentials ────────────────────────────────
    PUTER_USERNAME: Optional[str] = None
    PUTER_PASSWORD: Optional[str] = None

    # ── Local Model Backends ───────────────────────────────────
    OLLAMA_BASE_URL: str = "http://localhost:11434"

    # ── Tool / Service Keys ────────────────────────────────────
    TAVILY_API_KEY: Optional[str] = None
    EXA_API_KEY: Optional[str] = None
    FIRECRAWL_API_KEY: Optional[str] = None
    GITHUB_TOKEN: Optional[str] = None

    # ── Integrations ───────────────────────────────────────────
    TELEGRAM_BOT_TOKEN: Optional[str] = None
    TELEGRAM_ALLOWED_USERS: Optional[str] = None  # Comma-separated Telegram user IDs
    TELEGRAM_ENABLED: bool = False

    # ── Heartbeat Daemon ─────────────────────────────────────
    HEARTBEAT_ENABLED: bool = False
    HEARTBEAT_INTERVAL_MINUTES: int = 30

    # ── Axiom Behaviour ────────────────────────────────────────
    DEFAULT_MODEL: str = "anthropic/claude-opus-4-6"
    AXIOM_YOLO: bool = False
    AXIOM_VISIBLE_BROWSER: bool = False
    AXIOM_HOME: Path = Path.home() / ".axiom"

    # ── Provider key mapping (litellm provider name -> settings attr) ──
    _PROVIDER_KEY_MAP: dict[str, str] = {
        "anthropic": "ANTHROPIC_API_KEY",
        "openai": "OPENAI_API_KEY",
        "groq": "GROQ_API_KEY",
        "gemini": "GEMINI_API_KEY",
        "nvidia_nim": "NVIDIA_API_KEY",
        "deepseek": "DEEPSEEK_API_KEY",
        "together_ai": "TOGETHER_API_KEY",
        "mistral": "MISTRAL_API_KEY",
        "cohere_chat": "COHERE_API_KEY",
        "openrouter": "OPENROUTER_API_KEY",
        "huggingface": "HUGGINGFACE_API_KEY",
        "vertex_ai": "GOOGLE_APPLICATION_CREDENTIALS",
    }

    def available_providers(self) -> list[str]:
        """Return provider names that have their API key configured.

        Ollama is always included (local, no key needed).
        Vertex AI is included if VERTEX_AI_PROJECT is set (uses ADC or explicit creds).
        """
        providers: list[str] = []
        for provider, key_attr in self._PROVIDER_KEY_MAP.items():
            if getattr(self, key_attr, None):
                providers.append(provider)
        # Vertex AI: also available via project ID + Application Default Credentials
        if self.VERTEX_AI_PROJECT and "vertex_ai" not in providers:
            providers.append("vertex_ai")
        # Ollama is local -- always available
        providers.append("ollama")
        return sorted(set(providers))

    def get_model_for_provider(self, provider: str) -> str:
        """Return the best default model string for a given provider.

        Falls back to DEFAULT_MODEL if the provider is unknown.
        """
        return MODEL_DEFAULTS.get(provider, self.DEFAULT_MODEL)


@lru_cache(maxsize=1)
def get_settings() -> AxiomSettings:
    """Singleton accessor -- import and call once, reuse everywhere."""
    return AxiomSettings()
