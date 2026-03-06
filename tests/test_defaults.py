"""Tests for axiom/config/defaults.py — default constants and model mappings."""

from axiom.config.defaults import (
    DEFAULT_MODEL,
    MODEL_DEFAULTS,
    PROVIDER_DISPLAY_NAMES,
)


class TestDefaultModel:
    """Verify DEFAULT_MODEL is Claude Opus 4.6."""

    def test_default_is_opus_4_6(self):
        assert DEFAULT_MODEL == "anthropic/claude-opus-4-6"

    def test_default_has_provider_prefix(self):
        assert "/" in DEFAULT_MODEL
        provider = DEFAULT_MODEL.split("/")[0]
        assert provider == "anthropic"


class TestModelDefaults:
    """Verify every provider has a sane default model mapping."""

    EXPECTED_PROVIDERS = {
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
        "ollama",
        "vertex_ai",
    }

    def test_all_providers_present(self):
        assert set(MODEL_DEFAULTS.keys()) == self.EXPECTED_PROVIDERS

    def test_each_model_string_has_provider_prefix(self):
        for provider, model in MODEL_DEFAULTS.items():
            assert "/" in model, f"{provider} model missing '/': {model}"

    def test_anthropic_default_matches_global(self):
        assert MODEL_DEFAULTS["anthropic"] == DEFAULT_MODEL

    def test_vertex_ai_default_is_gemini(self):
        # Vertex AI default should be Gemini (auto-available, good fallback)
        assert "gemini" in MODEL_DEFAULTS["vertex_ai"]


class TestProviderDisplayNames:
    """Verify display names exist for every provider."""

    def test_all_providers_have_display_names(self):
        for provider in MODEL_DEFAULTS:
            assert provider in PROVIDER_DISPLAY_NAMES, (
                f"Missing display name for {provider}"
            )

    def test_display_names_are_nonempty_strings(self):
        for provider, name in PROVIDER_DISPLAY_NAMES.items():
            assert isinstance(name, str) and len(name) > 0
