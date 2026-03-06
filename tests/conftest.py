"""Shared fixtures for Axiom CLI test suite."""

from __future__ import annotations

from typing import Any, Optional
from unittest.mock import MagicMock

import pytest


class FakeSettings:
    """Minimal AxiomSettings stand-in for unit tests.

    No .env is loaded — only the explicitly set keys are "available".
    """

    def __init__(
        self,
        default_model: str = "anthropic/claude-opus-4-6",
        available: Optional[set[str]] = None,
        **api_keys: str,
    ):
        self.DEFAULT_MODEL = default_model
        self._available = available or {"vertex_ai", "ollama"}

        # API keys default to None
        self.ANTHROPIC_API_KEY: Optional[str] = api_keys.get("ANTHROPIC_API_KEY")
        self.OPENAI_API_KEY: Optional[str] = api_keys.get("OPENAI_API_KEY")
        self.GROQ_API_KEY: Optional[str] = api_keys.get("GROQ_API_KEY")
        self.GEMINI_API_KEY: Optional[str] = api_keys.get("GEMINI_API_KEY")
        self.NVIDIA_API_KEY: Optional[str] = api_keys.get("NVIDIA_API_KEY")
        self.DEEPSEEK_API_KEY: Optional[str] = api_keys.get("DEEPSEEK_API_KEY")
        self.TOGETHER_API_KEY: Optional[str] = api_keys.get("TOGETHER_API_KEY")
        self.MISTRAL_API_KEY: Optional[str] = api_keys.get("MISTRAL_API_KEY")
        self.COHERE_API_KEY: Optional[str] = api_keys.get("COHERE_API_KEY")
        self.OPENROUTER_API_KEY: Optional[str] = api_keys.get("OPENROUTER_API_KEY")
        self.HUGGINGFACE_API_KEY: Optional[str] = api_keys.get("HUGGINGFACE_API_KEY")
        self.TAVILY_API_KEY: Optional[str] = api_keys.get("TAVILY_API_KEY")

        self.VERTEX_AI_PROJECT: Optional[str] = api_keys.get(
            "VERTEX_AI_PROJECT", "test-project"
        )
        self.VERTEX_AI_LOCATION: Optional[str] = api_keys.get(
            "VERTEX_AI_LOCATION", "us-east5"
        )
        self.GOOGLE_APPLICATION_CREDENTIALS: Optional[str] = api_keys.get(
            "GOOGLE_APPLICATION_CREDENTIALS"
        )
        self.OLLAMA_BASE_URL: str = api_keys.get(
            "OLLAMA_BASE_URL", "http://localhost:11434"
        )

    def available_providers(self) -> list[str]:
        return sorted(self._available)


@pytest.fixture
def fake_settings() -> FakeSettings:
    """Default fake settings with vertex_ai + ollama available."""
    return FakeSettings()


@pytest.fixture
def fake_settings_all_providers() -> FakeSettings:
    """Fake settings with every provider available."""
    return FakeSettings(
        available={
            "anthropic",
            "openai",
            "groq",
            "gemini",
            "nvidia_nim",
            "deepseek",
            "together_ai",
            "mistral",
            "cohere_chat",
            "openrouter",
            "huggingface",
            "vertex_ai",
            "ollama",
        },
    )
