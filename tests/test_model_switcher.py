"""Tests for axiom/core/llm/model_switcher.py — live model switching + auto-routing."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from axiom.core.llm.model_switcher import (
    MODEL_CATALOG,
    ModelSwitcher,
    _AUTO_ROUTES,
    _CODING_KEYWORDS,
    _REASONING_KEYWORDS,
    _SIMPLE_KEYWORDS,
)
from axiom.core.llm.router import MODEL_SHORTCUTS


# ── Model Catalog Tests ────────────────────────────────────────────


class TestModelCatalog:
    """Verify MODEL_CATALOG structure and content."""

    def test_catalog_has_at_least_10_groups(self):
        assert len(MODEL_CATALOG) >= 10

    def test_total_entries_at_least_50(self):
        total = sum(len(v) for v in MODEL_CATALOG.values())
        assert total >= 50

    def test_every_entry_has_required_keys(self):
        for group, models in MODEL_CATALOG.items():
            for entry in models:
                assert "alias" in entry, f"Missing 'alias' in {group}"
                assert "model" in entry, f"Missing 'model' in {group}"
                assert "note" in entry, f"Missing 'note' in {group}"

    def test_every_model_string_has_provider(self):
        for group, models in MODEL_CATALOG.items():
            for entry in models:
                assert "/" in entry["model"], (
                    f"Model '{entry['model']}' in '{group}' missing provider prefix"
                )

    def test_every_alias_is_in_shortcuts(self):
        """Every catalog alias should resolve in MODEL_SHORTCUTS."""
        for group, models in MODEL_CATALOG.items():
            for entry in models:
                alias = entry["alias"]
                assert alias in MODEL_SHORTCUTS, (
                    f"Catalog alias '{alias}' from '{group}' not in MODEL_SHORTCUTS"
                )

    def test_catalog_alias_matches_shortcut_target(self):
        """Catalog alias should map to the same model in shortcuts."""
        for group, models in MODEL_CATALOG.items():
            for entry in models:
                alias = entry["alias"]
                catalog_model = entry["model"]
                shortcut_model = MODEL_SHORTCUTS.get(alias)
                assert shortcut_model == catalog_model, (
                    f"Alias '{alias}': catalog says '{catalog_model}' "
                    f"but shortcut says '{shortcut_model}'"
                )

    # ── Group existence ────────────────────────────────────────────

    def test_has_anthropic_group(self):
        assert "Claude (Anthropic API)" in MODEL_CATALOG

    def test_has_vertex_claude_group(self):
        assert "Claude via Vertex AI" in MODEL_CATALOG

    def test_has_vertex_gemini_group(self):
        assert "Gemini via Vertex AI" in MODEL_CATALOG

    def test_has_deepseek_group(self):
        assert "DeepSeek on Vertex AI" in MODEL_CATALOG

    def test_has_qwen_group(self):
        assert "Qwen on Vertex AI" in MODEL_CATALOG

    def test_has_chinese_ai_group(self):
        assert any("Chinese AI" in g for g in MODEL_CATALOG)

    def test_has_free_apis_group(self):
        assert "Free Cloud APIs" in MODEL_CATALOG

    def test_has_local_group(self):
        assert "Local (Ollama)" in MODEL_CATALOG


# ── Auto Routes Tests ──────────────────────────────────────────────


class TestAutoRoutes:
    """Verify auto-routing recommendations."""

    def test_all_task_types_present(self):
        expected = {"coding", "reasoning", "simple", "research", "bulk"}
        assert set(_AUTO_ROUTES.keys()) == expected

    def test_coding_includes_opus(self):
        coding = _AUTO_ROUTES["coding"]
        assert any("opus" in m for m in coding)

    def test_coding_includes_qwen_coder(self):
        coding = _AUTO_ROUTES["coding"]
        assert any("qwen3-coder" in m for m in coding)

    def test_reasoning_includes_deepseek_r1(self):
        reasoning = _AUTO_ROUTES["reasoning"]
        assert any("deepseek-r1" in m for m in reasoning)

    def test_reasoning_includes_kimi(self):
        reasoning = _AUTO_ROUTES["reasoning"]
        assert any("kimi" in m for m in reasoning)

    def test_simple_starts_with_fast_model(self):
        simple = _AUTO_ROUTES["simple"]
        # Groq is fastest for simple tasks
        assert "groq" in simple[0]

    def test_research_starts_with_gemini_3_1(self):
        research = _AUTO_ROUTES["research"]
        assert "gemini-3.1" in research[0]

    def test_bulk_starts_with_local(self):
        bulk = _AUTO_ROUTES["bulk"]
        assert "ollama" in bulk[0]

    def test_all_route_models_are_valid_litellm_strings(self):
        for task, models in _AUTO_ROUTES.items():
            for model in models:
                assert "/" in model, (
                    f"Invalid model string in {task}: {model}"
                )


# ── Task Classification Tests ─────────────────────────────────────


class TestTaskClassification:
    """Verify _classify_task heuristics."""

    @pytest.fixture
    def switcher(self):
        mock_router = MagicMock()
        mock_router.active_model = "anthropic/claude-opus-4-6"
        mock_router.list_available.return_value = []
        return ModelSwitcher(mock_router)

    def test_coding_task(self, switcher):
        assert switcher._classify_task(
            "implement a function to parse JSON and fix the error"
        ) == "coding"

    def test_reasoning_task(self, switcher):
        assert switcher._classify_task(
            "analyze and compare the tradeoff between these design architectures"
        ) == "reasoning"

    def test_simple_task(self, switcher):
        assert switcher._classify_task("what is Python?") == "simple"

    def test_research_task(self, switcher):
        assert switcher._classify_task(
            "research the latest trends in machine learning"
        ) == "research"

    def test_long_message_is_reasoning(self, switcher):
        long_msg = "explain " + "something " * 100
        assert switcher._classify_task(long_msg) == "reasoning"

    def test_keyword_sets_not_empty(self):
        assert len(_CODING_KEYWORDS) > 5
        assert len(_REASONING_KEYWORDS) > 5
        assert len(_SIMPLE_KEYWORDS) > 5


# ── ModelSwitcher Tests ────────────────────────────────────────────


class TestModelSwitcher:
    """Test the ModelSwitcher class."""

    @pytest.fixture
    def mock_router(self):
        router = MagicMock()
        router.active_model = "anthropic/claude-opus-4-6"
        router.switch_model.side_effect = lambda name: MODEL_SHORTCUTS.get(
            name.lower().strip(), name
        )
        router.get_usage.return_value = {
            "input": 100,
            "output": 50,
            "cost": 0.005,
            "total_tokens": 150,
            "requests": 3,
        }
        router.list_available.return_value = [
            {"provider": "vertex_ai", "model": "vertex_ai/gemini-2.5-pro", "available": True},
            {"provider": "ollama", "model": "ollama/qwen2.5:7b", "available": True},
        ]
        return router

    def test_switch_updates_router(self, mock_router):
        switcher = ModelSwitcher(mock_router)
        result = switcher.switch("groq")
        mock_router.switch_model.assert_called_once_with("groq")
        assert switcher.auto_mode is False

    def test_switch_auto_mode(self, mock_router):
        switcher = ModelSwitcher(mock_router)
        result = switcher.switch("auto")
        assert switcher.auto_mode is True
        assert "auto" in result

    def test_switch_saves_previous(self, mock_router):
        switcher = ModelSwitcher(mock_router)
        switcher.switch("groq")
        assert switcher._previous_model == "anthropic/claude-opus-4-6"

    def test_switch_back(self, mock_router):
        switcher = ModelSwitcher(mock_router)
        switcher.switch("groq")
        switcher.switch_back()
        mock_router.switch_model.assert_called_with("anthropic/claude-opus-4-6")

    def test_switch_back_none_when_no_previous(self, mock_router):
        switcher = ModelSwitcher(mock_router)
        result = switcher.switch_back()
        assert result is None

    def test_current_model_property(self, mock_router):
        switcher = ModelSwitcher(mock_router)
        assert switcher.current_model == "anthropic/claude-opus-4-6"

    def test_auto_select_off_returns_none(self, mock_router):
        switcher = ModelSwitcher(mock_router)
        result = switcher.auto_select("hello")
        assert result is None

    def test_get_model_info(self, mock_router):
        switcher = ModelSwitcher(mock_router)
        info = switcher.get_model_info()
        assert info["active_model"] == "anthropic/claude-opus-4-6"
        assert info["auto_mode"] is False
        assert "usage" in info

    def test_format_cost_report(self, mock_router):
        switcher = ModelSwitcher(mock_router)
        report = switcher.format_cost_report()
        assert "claude-opus-4-6" in report
        assert "100" in report  # input tokens
        assert "50" in report   # output tokens

    def test_list_models(self, mock_router):
        switcher = ModelSwitcher(mock_router)
        models = switcher.list_models()
        assert isinstance(models, list)
        assert len(models) > 0
        # Each entry should have standard keys
        for m in models:
            assert "group" in m
            assert "alias" in m
            assert "model" in m
            assert "available" in m
            assert "is_active" in m

    def test_format_model_list(self, mock_router):
        switcher = ModelSwitcher(mock_router)
        output = switcher.format_model_list()
        assert "Claude" in output
        assert "opus" in output
        assert "/model" in output  # help text
