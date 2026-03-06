"""Tests for axiom/config/settings.py — AxiomSettings + provider discovery."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from axiom.config.settings import AxiomSettings
from axiom.config.defaults import MODEL_DEFAULTS


class TestAxiomSettings:
    """Verify settings defaults and provider discovery logic."""

    @pytest.fixture
    def settings(self):
        """Create settings without loading .env."""
        # Override env_file to prevent loading real .env
        with patch.dict("os.environ", {}, clear=False):
            s = AxiomSettings(
                _env_file=None,
                ANTHROPIC_API_KEY=None,
                OPENAI_API_KEY=None,
                VERTEX_AI_PROJECT=None,
            )
        return s

    def test_default_model_is_opus_4_6(self, settings):
        assert settings.DEFAULT_MODEL == "anthropic/claude-opus-4-6"

    def test_yolo_defaults_false(self, settings):
        assert settings.AXIOM_YOLO is False

    def test_visible_browser_defaults_false(self, settings):
        assert settings.AXIOM_VISIBLE_BROWSER is False

    def test_ollama_default_url(self, settings):
        assert settings.OLLAMA_BASE_URL == "http://localhost:11434"


class TestAvailableProviders:
    """Test the available_providers() method."""

    def test_ollama_always_available(self):
        s = AxiomSettings(
            _env_file=None,
            VERTEX_AI_PROJECT=None,
        )
        providers = s.available_providers()
        assert "ollama" in providers

    def test_vertex_ai_available_with_project(self):
        s = AxiomSettings(
            _env_file=None,
            VERTEX_AI_PROJECT="my-project",
        )
        providers = s.available_providers()
        assert "vertex_ai" in providers

    def test_anthropic_available_with_key(self):
        s = AxiomSettings(
            _env_file=None,
            ANTHROPIC_API_KEY="sk-test",
            VERTEX_AI_PROJECT=None,
        )
        providers = s.available_providers()
        assert "anthropic" in providers

    def test_no_keys_means_only_ollama(self):
        # Must explicitly null every key — Pydantic BaseSettings reads
        # from the real environment even with _env_file=None.
        s = AxiomSettings(
            _env_file=None,
            VERTEX_AI_PROJECT=None,
            GOOGLE_APPLICATION_CREDENTIALS=None,
            ANTHROPIC_API_KEY=None,
            OPENAI_API_KEY=None,
            GROQ_API_KEY=None,
            GEMINI_API_KEY=None,
            NVIDIA_API_KEY=None,
            DEEPSEEK_API_KEY=None,
            TOGETHER_API_KEY=None,
            MISTRAL_API_KEY=None,
            COHERE_API_KEY=None,
            OPENROUTER_API_KEY=None,
            HUGGINGFACE_API_KEY=None,
        )
        providers = s.available_providers()
        # Only ollama (always included)
        assert providers == ["ollama"]

    def test_multiple_providers(self):
        s = AxiomSettings(
            _env_file=None,
            ANTHROPIC_API_KEY="sk-test",
            GROQ_API_KEY="gsk-test",
            VERTEX_AI_PROJECT="proj",
        )
        providers = s.available_providers()
        assert "anthropic" in providers
        assert "groq" in providers
        assert "vertex_ai" in providers
        assert "ollama" in providers

    def test_providers_sorted(self):
        s = AxiomSettings(
            _env_file=None,
            ANTHROPIC_API_KEY="sk-test",
            OPENAI_API_KEY="sk-test2",
            GROQ_API_KEY="gsk-test",
            VERTEX_AI_PROJECT="proj",
        )
        providers = s.available_providers()
        assert providers == sorted(providers)

    def test_no_duplicates(self):
        s = AxiomSettings(
            _env_file=None,
            VERTEX_AI_PROJECT="proj",
            GOOGLE_APPLICATION_CREDENTIALS="/path/to/creds.json",
        )
        providers = s.available_providers()
        assert len(providers) == len(set(providers))

    def test_get_model_for_known_provider(self):
        s = AxiomSettings(_env_file=None, VERTEX_AI_PROJECT=None)
        model = s.get_model_for_provider("groq")
        assert model == MODEL_DEFAULTS["groq"]

    def test_get_model_for_unknown_provider(self):
        s = AxiomSettings(_env_file=None, VERTEX_AI_PROJECT=None)
        model = s.get_model_for_provider("nonexistent")
        assert model == s.DEFAULT_MODEL
