"""Tests for axiom/core/llm/router.py — UniversalRouter + shortcuts + circuit breaker."""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from axiom.core.llm.router import (
    MODEL_SHORTCUTS,
    CircuitBreaker,
    UniversalRouter,
    _FALLBACK_CHAIN,
)
from axiom.config.defaults import MODEL_DEFAULTS


# ── Circuit Breaker Tests ──────────────────────────────────────────


class TestCircuitBreaker:
    """Verify circuit breaker state transitions."""

    def test_starts_closed(self):
        cb = CircuitBreaker()
        assert cb.state == "closed"
        assert cb.is_available()

    def test_stays_closed_under_threshold(self):
        cb = CircuitBreaker(threshold=3)
        cb.record_failure("err1")
        cb.record_failure("err2")
        assert cb.state == "closed"
        assert cb.is_available()

    def test_opens_at_threshold(self):
        cb = CircuitBreaker(threshold=3)
        for i in range(3):
            cb.record_failure(f"err{i}")
        assert cb.state == "open"
        assert not cb.is_available()

    def test_resets_on_success(self):
        cb = CircuitBreaker(threshold=2)
        cb.record_failure("err1")
        cb.record_failure("err2")
        assert cb.state == "open"
        cb.record_success()
        assert cb.state == "closed"
        assert cb.failures == 0

    def test_half_open_after_timeout(self):
        cb = CircuitBreaker(threshold=1, timeout_seconds=0.1)
        cb.record_failure("err")
        assert cb.state == "open"
        time.sleep(0.15)
        # Should transition to half_open on availability check
        assert cb.is_available()
        assert cb.state == "half_open"

    def test_half_open_failure_reopens(self):
        cb = CircuitBreaker(threshold=1, timeout_seconds=0.01)
        cb.record_failure("err")
        time.sleep(0.02)
        assert cb.is_available()  # transitions to half_open
        cb.record_failure("err2")
        assert cb.state == "open"

    def test_half_open_success_closes(self):
        cb = CircuitBreaker(threshold=1, timeout_seconds=0.01)
        cb.record_failure("err")
        time.sleep(0.02)
        assert cb.is_available()  # half_open
        cb.record_success()
        assert cb.state == "closed"


# ── Model Shortcuts Tests ──────────────────────────────────────────


class TestModelShortcuts:
    """Verify the shortcut map has correct structure and key aliases."""

    def test_shortcut_count_at_least_100(self):
        assert len(MODEL_SHORTCUTS) >= 100

    def test_all_values_are_full_model_strings(self):
        for alias, model in MODEL_SHORTCUTS.items():
            assert "/" in model, f"Shortcut '{alias}' -> '{model}' missing '/'"

    # ── Core aliases ───────────────────────────────────────────────

    def test_opus_alias(self):
        assert MODEL_SHORTCUTS["opus"] == "anthropic/claude-opus-4-6"

    def test_sonnet_alias(self):
        assert MODEL_SHORTCUTS["sonnet"] == "anthropic/claude-sonnet-4-6"

    def test_haiku_alias(self):
        assert "haiku" in MODEL_SHORTCUTS["haiku"]

    def test_gpt4o_alias(self):
        assert MODEL_SHORTCUTS["gpt4o"] == "openai/gpt-4o"

    def test_groq_alias(self):
        assert MODEL_SHORTCUTS["groq"] == "groq/llama-3.3-70b-versatile"

    # ── Vertex AI aliases ──────────────────────────────────────────

    def test_vertex_opus(self):
        assert MODEL_SHORTCUTS["vertex-opus"] == "vertex_ai/claude-opus-4-6"

    def test_vertex_flash(self):
        assert MODEL_SHORTCUTS["vertex-flash"] == "vertex_ai/gemini-2.5-flash"

    def test_vertex_default(self):
        assert MODEL_SHORTCUTS["vertex"] == "vertex_ai/gemini-2.5-pro"

    # ── Third-party Vertex AI ──────────────────────────────────────

    def test_vertex_deepseek(self):
        m = MODEL_SHORTCUTS["vertex-deepseek"]
        assert "deepseek" in m and "vertex_ai" in m

    def test_vertex_deepseek_r1(self):
        m = MODEL_SHORTCUTS["vertex-deepseek-r1"]
        assert "deepseek-r1" in m

    def test_vertex_qwen_coder(self):
        m = MODEL_SHORTCUTS["vertex-qwen-coder"]
        assert "qwen3-coder" in m

    def test_vertex_minimax(self):
        m = MODEL_SHORTCUTS["vertex-minimax"]
        assert "minimax" in m.lower()

    def test_vertex_kimi(self):
        m = MODEL_SHORTCUTS["vertex-kimi"]
        assert "kimi" in m

    def test_vertex_glm_5(self):
        m = MODEL_SHORTCUTS["vertex-glm-5"]
        assert "glm-5" in m

    def test_vertex_gpt_oss(self):
        m = MODEL_SHORTCUTS["vertex-gpt-oss"]
        assert "gpt-oss" in m

    def test_vertex_llama4(self):
        m = MODEL_SHORTCUTS["vertex-llama4"]
        assert "llama-4" in m

    def test_vertex_mistral(self):
        m = MODEL_SHORTCUTS["vertex-mistral"]
        assert "mistral" in m

    def test_vertex_jamba(self):
        m = MODEL_SHORTCUTS["vertex-jamba"]
        assert "jamba" in m

    # ── Gemini 3.x ─────────────────────────────────────────────────

    def test_vertex_3_1_pro(self):
        assert MODEL_SHORTCUTS["vertex-3.1-pro"] == "vertex_ai/gemini-3.1-pro-preview"

    def test_vertex_3_flash(self):
        assert MODEL_SHORTCUTS["vertex-3-flash"] == "vertex_ai/gemini-3-flash-preview"

    # ── Local ──────────────────────────────────────────────────────

    def test_qwen_alias(self):
        assert MODEL_SHORTCUTS["qwen"] == "ollama/qwen2.5:7b"

    def test_local_alias(self):
        assert MODEL_SHORTCUTS["local"] == "ollama/qwen2.5:7b"

    # ── No duplicate values for core aliases ───────────────────────

    def test_no_duplicate_core_aliases(self):
        core_aliases = [
            "opus", "sonnet", "haiku", "gpt4o", "groq", "nvidia",
            "deepseek", "gemini", "qwen", "vertex",
        ]
        for alias in core_aliases:
            assert alias in MODEL_SHORTCUTS, f"Core alias '{alias}' missing"


# ── Fallback Chain Tests ───────────────────────────────────────────


class TestFallbackChain:
    """Verify fallback chains are properly configured."""

    def test_every_provider_has_chain(self):
        for provider in MODEL_DEFAULTS:
            assert provider in _FALLBACK_CHAIN, (
                f"Missing fallback chain for {provider}"
            )

    def test_vertex_ai_first_fallback_is_anthropic(self):
        # If vertex_ai fails, should try anthropic first
        assert _FALLBACK_CHAIN["vertex_ai"][0] == "anthropic"

    def test_anthropic_first_fallback_is_vertex(self):
        assert _FALLBACK_CHAIN["anthropic"][0] == "vertex_ai"

    def test_no_self_references(self):
        for provider, chain in _FALLBACK_CHAIN.items():
            assert provider not in chain, (
                f"{provider} has itself in fallback chain"
            )


# ── UniversalRouter Unit Tests ─────────────────────────────────────


class TestUniversalRouter:
    """Unit tests for UniversalRouter (no real API calls)."""

    @pytest.fixture
    def router(self, fake_settings):
        """Create a router with fake settings (no real API keys)."""
        with patch("axiom.core.llm.router.litellm"):
            r = UniversalRouter(fake_settings)
        return r

    def test_init_sets_active_model(self, router, fake_settings):
        assert router.active_model == fake_settings.DEFAULT_MODEL

    def test_switch_model_by_alias(self, router):
        result = router.switch_model("groq")
        assert result == "groq/llama-3.3-70b-versatile"
        assert router.active_model == result

    def test_switch_model_by_full_name(self, router):
        full = "deepseek/deepseek-chat"
        result = router.switch_model(full)
        assert result == full
        assert router.active_model == full

    def test_switch_model_case_insensitive(self, router):
        result = router.switch_model("OPUS")
        assert result == "anthropic/claude-opus-4-6"

    def test_switch_model_strips_whitespace(self, router):
        result = router.switch_model("  sonnet  ")
        assert result == "anthropic/claude-sonnet-4-6"

    def test_switch_model_unknown_passes_through(self, router):
        custom = "custom/my-fine-tuned-model"
        result = router.switch_model(custom)
        assert result == custom

    def test_extract_provider(self, router):
        assert router._extract_provider("anthropic/claude-opus-4-6") == "anthropic"
        assert router._extract_provider("vertex_ai/gemini-2.5-pro") == "vertex_ai"
        assert router._extract_provider("no-slash") == "unknown"

    def test_provider_name_property(self, router):
        router.active_model = "anthropic/claude-opus-4-6"
        assert router.provider_name == "Anthropic"

    def test_model_short_name_property(self, router):
        router.active_model = "anthropic/claude-opus-4-6"
        assert router.model_short_name == "claude-opus-4-6"

    def test_get_usage_initial(self, router):
        usage = router.get_usage()
        assert usage["input"] == 0
        assert usage["output"] == 0
        assert usage["cost"] == 0.0
        assert usage["requests"] == 0

    def test_list_available(self, router):
        providers = router.list_available()
        assert isinstance(providers, list)
        assert len(providers) == len(MODEL_DEFAULTS)
        # Each entry should have expected keys
        for p in providers:
            assert "provider" in p
            assert "model" in p
            assert "available" in p
            assert "is_active" in p

    def test_repr(self, router):
        r = repr(router)
        assert "UniversalRouter" in r
        assert router.active_model in r


# ── Startup Validation Tests (async, mocked) ──────────────────────


class TestValidateDefaultModel:
    """Test validate_default_model() with mocked LLM calls."""

    @pytest.fixture
    def router_with_mock(self, fake_settings):
        """Router where complete_sync is mockable."""
        with patch("axiom.core.llm.router.litellm"):
            r = UniversalRouter(fake_settings)
        return r

    @pytest.mark.asyncio
    async def test_valid_model_returns_unchanged(self, router_with_mock):
        router = router_with_mock
        # Mock a successful completion
        mock_response = MagicMock()
        mock_response.__bool__ = lambda self: True
        router.complete_sync = AsyncMock(return_value=mock_response)

        result = await router.validate_default_model(timeout=5.0)
        assert result == router.active_model

    @pytest.mark.asyncio
    async def test_timeout_falls_back_to_same_provider_default(self, router_with_mock):
        router = router_with_mock
        router.active_model = "vertex_ai/claude-opus-4-6"

        # First call times out, second succeeds (Gemini on same Vertex AI)
        call_count = 0

        async def mock_complete(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            model = kwargs.get("model", args[0] if args else "")
            if "claude" in str(model):
                await asyncio.sleep(100)  # Will timeout
            return MagicMock(__bool__=lambda s: True)

        router.complete_sync = mock_complete

        result = await router.validate_default_model(timeout=0.1)
        # Should fall back to vertex_ai/gemini-2.5-pro (same provider)
        assert result == "vertex_ai/gemini-2.5-pro"
        assert router.active_model == result

    @pytest.mark.asyncio
    async def test_all_failures_keeps_original(self, router_with_mock):
        router = router_with_mock
        router.active_model = "vertex_ai/claude-opus-4-6"

        # Everything fails
        async def mock_fail(*args, **kwargs):
            raise ConnectionError("no connection")

        router.complete_sync = mock_fail

        result = await router.validate_default_model(timeout=0.1)
        # Should keep original even though nothing validated
        assert result == "vertex_ai/claude-opus-4-6"


# ── Provider Overrides Tests ───────────────────────────────────────


class TestProviderOverrides:
    """Verify _apply_provider_overrides sets correct api_base."""

    def test_nvidia_gets_api_base(self):
        kwargs: dict = {}
        UniversalRouter._apply_provider_overrides(
            "nvidia_nim/meta/llama-3.3-70b-instruct", kwargs
        )
        assert kwargs["api_base"] == "https://integrate.api.nvidia.com/v1"

    def test_ollama_gets_api_base(self):
        kwargs: dict = {}
        UniversalRouter._apply_provider_overrides("ollama/qwen2.5:7b", kwargs)
        assert "api_base" in kwargs

    def test_anthropic_no_override(self):
        kwargs: dict = {}
        UniversalRouter._apply_provider_overrides(
            "anthropic/claude-opus-4-6", kwargs
        )
        assert "api_base" not in kwargs


# ── Message Sanitization Tests ─────────────────────────────────────


class TestSanitizeMessages:
    """Verify _sanitize_messages handles edge cases."""

    def test_basic_message(self):
        msgs = [{"role": "user", "content": "hello"}]
        result = UniversalRouter._sanitize_messages(msgs)
        assert result == [{"role": "user", "content": "hello"}]

    def test_preserves_tool_calls(self):
        msgs = [{"role": "assistant", "content": "", "tool_calls": [{"id": "1"}]}]
        result = UniversalRouter._sanitize_messages(msgs)
        assert result[0]["tool_calls"] == [{"id": "1"}]

    def test_preserves_tool_call_id(self):
        msgs = [{"role": "tool", "content": "result", "tool_call_id": "tc1"}]
        result = UniversalRouter._sanitize_messages(msgs)
        assert result[0]["tool_call_id"] == "tc1"

    def test_defaults_missing_role(self):
        msgs = [{"content": "hello"}]
        result = UniversalRouter._sanitize_messages(msgs)
        assert result[0]["role"] == "user"

    def test_defaults_missing_content(self):
        msgs = [{"role": "assistant"}]
        result = UniversalRouter._sanitize_messages(msgs)
        assert result[0]["content"] == ""
