"""Model switcher — live model swap mid-conversation.

Provides the logic for the ``/model`` command, including listing
available models, switching providers, and auto-routing based on
task complexity.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Task type heuristics for auto-routing
_CODING_KEYWORDS = {
    "code", "function", "class", "implement", "debug", "fix", "error",
    "refactor", "test", "unittest", "api", "endpoint", "deploy",
    "compile", "build", "script", "module", "package", "library",
}

_REASONING_KEYWORDS = {
    "analyze", "design", "architecture", "plan", "strategy", "compare",
    "evaluate", "tradeoff", "decision", "explain", "why", "how",
    "understand", "complex", "think", "reason", "proof", "math",
}

_SIMPLE_KEYWORDS = {
    "what", "when", "where", "who", "list", "show", "tell",
    "name", "define", "meaning", "translate",
}

# Model recommendations per task type
_AUTO_ROUTES = {
    "coding": [
        "anthropic/claude-opus-4-6",
        "vertex_ai/claude-opus-4-6",
        "anthropic/claude-sonnet-4-6",
        "vertex_ai/qwen/qwen3-coder-480b-a35b-instruct-maas",
        "vertex_ai/codestral-2@001",
        "mistral/codestral-latest",
    ],
    "reasoning": [
        "anthropic/claude-opus-4-6",
        "vertex_ai/claude-opus-4-6",
        "vertex_ai/deepseek-ai/deepseek-r1-0528-maas",
        "vertex_ai/moonshotai/kimi-k2-thinking-maas",
        "deepseek/deepseek-reasoner",
        "openai/o1",
    ],
    "simple": [
        "groq/llama-3.3-70b-versatile",
        "vertex_ai/gemini-2.0-flash-lite",
        "nvidia_nim/meta/llama-3.3-70b-instruct",
        "deepseek/deepseek-chat",
    ],
    "research": [
        "vertex_ai/gemini-3.1-pro-preview",
        "vertex_ai/gemini-2.5-pro",
        "gemini/gemini-2.5-flash",
        "anthropic/claude-opus-4-6",
    ],
    "bulk": [
        "ollama/qwen2.5:7b",
        "ollama/llama3.3:8b",
        "groq/llama-3.3-70b-versatile",
        "vertex_ai/gemini-2.0-flash-lite",
    ],
}

# ── Model catalog (for /model list display) ──────────────────────────

MODEL_CATALOG: dict[str, list[dict[str, str]]] = {
    "Claude (Anthropic API)": [
        {"alias": "opus", "model": "anthropic/claude-opus-4-6", "note": "LATEST — best coding + agents"},
        {"alias": "sonnet", "model": "anthropic/claude-sonnet-4-6", "note": "LATEST — fast + intelligent"},
        {"alias": "haiku", "model": "anthropic/claude-haiku-4-5-20251001", "note": "Fastest, $1/MTok"},
        {"alias": "opus-4.5", "model": "anthropic/claude-opus-4-5-20251101", "note": "Legacy Opus 4.5"},
        {"alias": "sonnet-4.5", "model": "anthropic/claude-sonnet-4-5-20250929", "note": "Legacy Sonnet 4.5"},
    ],
    "Claude via Vertex AI": [
        {"alias": "vertex-opus", "model": "vertex_ai/claude-opus-4-6", "note": "LATEST Opus on GCP"},
        {"alias": "vertex-sonnet", "model": "vertex_ai/claude-sonnet-4-6", "note": "LATEST Sonnet on GCP"},
        {"alias": "vertex-haiku", "model": "vertex_ai/claude-haiku-4-5@20251001", "note": "Haiku on GCP billing"},
        {"alias": "vertex-opus-4.5", "model": "vertex_ai/claude-opus-4-5@20251101", "note": "Legacy Opus 4.5"},
        {"alias": "vertex-sonnet-4.5", "model": "vertex_ai/claude-sonnet-4-5@20250929", "note": "Legacy Sonnet 4.5"},
        {"alias": "vertex-opus-4.0", "model": "vertex_ai/claude-opus-4@20250514", "note": "Legacy Opus 4.0"},
    ],
    "Gemini via Vertex AI": [
        {"alias": "vertex-3.1-pro", "model": "vertex_ai/gemini-3.1-pro-preview", "note": "LATEST Gemini 3.1 Pro"},
        {"alias": "vertex-3.1-flash", "model": "vertex_ai/gemini-3.1-flash-lite-preview", "note": "Gemini 3.1 Flash Lite"},
        {"alias": "vertex-3-pro", "model": "vertex_ai/gemini-3-pro-preview", "note": "Gemini 3 Pro"},
        {"alias": "vertex-3-flash", "model": "vertex_ai/gemini-3-flash-preview", "note": "Gemini 3 Flash"},
        {"alias": "vertex-gemini-pro", "model": "vertex_ai/gemini-2.5-pro", "note": "Gemini 2.5 Pro, 1M ctx"},
        {"alias": "vertex-flash", "model": "vertex_ai/gemini-2.5-flash", "note": "Gemini 2.5 Flash, 1M ctx"},
        {"alias": "vertex-2.0-flash", "model": "vertex_ai/gemini-2.0-flash", "note": "Balanced speed/quality"},
        {"alias": "vertex-lite", "model": "vertex_ai/gemini-2.0-flash-lite", "note": "Cheapest Gemini"},
        {"alias": "vertex-1.5-pro", "model": "vertex_ai/gemini-1.5-pro", "note": "Stable, 2M ctx"},
    ],
    "DeepSeek on Vertex AI": [
        {"alias": "vertex-deepseek", "model": "vertex_ai/deepseek-ai/deepseek-v3.2-maas", "note": "DeepSeek V3.2 (latest)"},
        {"alias": "vertex-deepseek-v3.1", "model": "vertex_ai/deepseek-ai/deepseek-v3.1-maas", "note": "DeepSeek V3.1"},
        {"alias": "vertex-deepseek-r1", "model": "vertex_ai/deepseek-ai/deepseek-r1-0528-maas", "note": "DeepSeek R1 reasoning"},
        {"alias": "vertex-deepseek-ocr", "model": "vertex_ai/deepseek-ai/deepseek-ocr-maas", "note": "DeepSeek OCR"},
    ],
    "Qwen on Vertex AI": [
        {"alias": "vertex-qwen-coder", "model": "vertex_ai/qwen/qwen3-coder-480b-a35b-instruct-maas", "note": "Qwen3 Coder 480B MoE"},
        {"alias": "vertex-qwen-235b", "model": "vertex_ai/qwen/qwen3-235b-a22b-instruct-2507-maas", "note": "Qwen3 235B MoE"},
        {"alias": "vertex-qwen-next", "model": "vertex_ai/qwen/qwen3-next-80b-a3b-instruct-maas", "note": "Qwen3 Next 80B"},
        {"alias": "vertex-qwen-thinking", "model": "vertex_ai/qwen/qwen3-next-80b-a3b-thinking-maas", "note": "Qwen3 Next thinking"},
    ],
    "Chinese AI on Vertex AI (MiniMax, Kimi, GLM)": [
        {"alias": "vertex-minimax", "model": "vertex_ai/minimaxai/minimax-m2-maas", "note": "MiniMax M2"},
        {"alias": "vertex-kimi", "model": "vertex_ai/moonshotai/kimi-k2-thinking-maas", "note": "Kimi K2 Thinking"},
        {"alias": "vertex-glm-5", "model": "vertex_ai/zai-org/glm-5-maas", "note": "GLM-5 (ZAI.org)"},
        {"alias": "vertex-glm-4.7", "model": "vertex_ai/zai-org/glm-4.7-maas", "note": "GLM-4.7 (ZAI.org)"},
    ],
    "Llama + Mistral on Vertex AI": [
        {"alias": "vertex-llama4", "model": "vertex_ai/meta/llama-4-scout-17b-16e-instruct-maas", "note": "Llama 4 Scout"},
        {"alias": "vertex-llama4-maverick", "model": "vertex_ai/meta/llama-4-maverick-17b-128e-instruct-maas", "note": "Llama 4 Maverick"},
        {"alias": "vertex-llama3", "model": "vertex_ai/meta/llama-3.3-70b-instruct-maas", "note": "Llama 3.3 70B"},
        {"alias": "vertex-mistral", "model": "vertex_ai/mistral-large@latest", "note": "Mistral Large"},
        {"alias": "vertex-codestral", "model": "vertex_ai/codestral-2@001", "note": "Codestral 2 (code)"},
        {"alias": "vertex-mistral-medium", "model": "vertex_ai/mistral-medium-3@001", "note": "Mistral Medium 3"},
    ],
    "Other on Vertex AI (GPT-OSS, Jamba)": [
        {"alias": "vertex-gpt-oss", "model": "vertex_ai/openai/gpt-oss-120b-maas", "note": "GPT-OSS 120B"},
        {"alias": "vertex-gpt-oss-20b", "model": "vertex_ai/openai/gpt-oss-20b-maas", "note": "GPT-OSS 20B"},
        {"alias": "vertex-jamba", "model": "vertex_ai/jamba-1.5-large@001", "note": "AI21 Jamba 1.5 Large"},
        {"alias": "vertex-jamba-mini", "model": "vertex_ai/jamba-1.5-mini@001", "note": "AI21 Jamba 1.5 Mini"},
    ],
    "Gemini (Google AI Studio)": [
        {"alias": "gemini-pro", "model": "gemini/gemini-2.5-pro", "note": "Free tier available"},
        {"alias": "gemini", "model": "gemini/gemini-2.5-flash", "note": "Free 15 RPM"},
        {"alias": "gemini-2.0", "model": "gemini/gemini-2.0-flash", "note": "Free, stable"},
    ],
    "OpenAI": [
        {"alias": "gpt4o", "model": "openai/gpt-4o", "note": "GPT-4o latest"},
        {"alias": "o1", "model": "openai/o1", "note": "Reasoning model"},
        {"alias": "o3", "model": "openai/o3-mini", "note": "Efficient reasoning"},
    ],
    "Free Cloud APIs": [
        {"alias": "groq", "model": "groq/llama-3.3-70b-versatile", "note": "Fastest, 14K req/day"},
        {"alias": "nvidia", "model": "nvidia_nim/meta/llama-3.3-70b-instruct", "note": "40 RPM free"},
        {"alias": "deepseek", "model": "deepseek/deepseek-chat", "note": "No rate limit"},
        {"alias": "deepseek-r1", "model": "deepseek/deepseek-reasoner", "note": "Reasoning, free tier"},
        {"alias": "together", "model": "together_ai/meta-llama/Llama-4-Scout-17B-16E-Instruct", "note": "$25 free"},
        {"alias": "mistral", "model": "mistral/mistral-small-latest", "note": "Free eval tier"},
        {"alias": "codestral", "model": "mistral/codestral-latest", "note": "Coding specialist"},
        {"alias": "cohere", "model": "cohere_chat/command-r-plus", "note": "1K calls/mo free"},
    ],
    "Local (Ollama)": [
        {"alias": "qwen", "model": "ollama/qwen2.5:7b", "note": "Best local coder"},
        {"alias": "llama", "model": "ollama/llama3.3:8b", "note": "General purpose"},
        {"alias": "deepseek-local", "model": "ollama/deepseek-r1:7b", "note": "Local reasoning"},
        {"alias": "phi", "model": "ollama/phi3.5:latest", "note": "Lightweight, fast"},
    ],
}


class ModelSwitcher:
    """Manages live model switching and auto-routing.

    Works with UniversalRouter to provide intelligent model selection
    and mid-conversation swaps.
    """

    def __init__(self, router: Any):
        self.router = router
        self.auto_mode = False
        self._previous_model: Optional[str] = None

    @property
    def current_model(self) -> str:
        """The currently active model string."""
        return self.router.active_model

    def switch(self, name: str) -> str:
        """Switch to a named model.

        Args:
            name: Model shortcut or full name (e.g., "opus", "vertex-sonnet").

        Returns:
            The resolved model name that was activated.
        """
        if name == "auto":
            self.auto_mode = True
            return "auto (smart routing enabled)"

        self.auto_mode = False
        self._previous_model = self.router.active_model
        new_model = self.router.switch_model(name)
        return new_model

    def switch_back(self) -> Optional[str]:
        """Switch back to the previous model."""
        if self._previous_model:
            result = self.router.switch_model(self._previous_model)
            self._previous_model = None
            return result
        return None

    def auto_select(self, user_message: str) -> Optional[str]:
        """Auto-select the best model based on task type.

        Only active when auto_mode is True.

        Returns:
            The selected model name, or None if auto-mode is off.
        """
        if not self.auto_mode:
            return None

        task_type = self._classify_task(user_message)
        candidates = _AUTO_ROUTES.get(task_type, _AUTO_ROUTES["simple"])

        # Try each candidate until we find an available one
        available = {
            m["model"] for m in self.router.list_available() if m["available"]
        }

        for model in candidates:
            if model in available:
                if model != self.router.active_model:
                    self.router.switch_model(model)
                    logger.info(
                        "Auto-routed to %s for %s task", model, task_type
                    )
                return model

        # Fallback: keep current model
        return self.router.active_model

    def list_models(self) -> list[dict[str, Any]]:
        """Return the full model catalog with availability status.

        Each entry includes: group, alias, model, note, available, is_active.
        """
        available_providers = set()
        try:
            available_providers = {
                m["provider"]
                for m in self.router.list_available()
                if m["available"]
            }
        except Exception:
            pass

        result = []
        for group, models in MODEL_CATALOG.items():
            for entry in models:
                model_str = entry["model"]
                provider = model_str.split("/")[0] if "/" in model_str else model_str
                result.append({
                    "group": group,
                    "alias": entry["alias"],
                    "model": model_str,
                    "note": entry["note"],
                    "available": provider in available_providers,
                    "is_active": model_str == self.router.active_model,
                })
        return result

    def format_model_list(self) -> str:
        """Format the model catalog as a rich-printable string for /model list."""
        models = self.list_models()
        lines: list[str] = []
        current_group = ""

        for m in models:
            if m["group"] != current_group:
                current_group = m["group"]
                lines.append(f"\n  [bold cyan]{current_group}[/]")

            # Status indicator
            if m["is_active"]:
                icon = "[bold green]>>>[/]"
            elif m["available"]:
                icon = " [green]*[/] "
            else:
                icon = " [dim]-[/] "

            alias = f"[bold]{m['alias']}[/]"
            note = f"[dim]{m['note']}[/]"
            lines.append(f"  {icon} {alias:<22} {note}")

        lines.append("")
        lines.append("  [dim]>>> = active  * = available  - = no API key[/]")
        lines.append("  [dim]Switch: /model <alias>  |  Auto: /model auto[/]")
        return "\n".join(lines)

    def _classify_task(self, message: str) -> str:
        """Classify the task type based on message content."""
        words = set(message.lower().split())

        coding_score = len(words & _CODING_KEYWORDS)
        reasoning_score = len(words & _REASONING_KEYWORDS)
        simple_score = len(words & _SIMPLE_KEYWORDS)

        # Short messages are usually simple
        if len(message) < 50 and simple_score > 0:
            return "simple"

        # Check for research indicators
        if "research" in message.lower() or "search" in message.lower():
            return "research"

        # Score-based classification
        if coding_score >= 2:
            return "coding"
        if reasoning_score >= 2:
            return "reasoning"
        if len(message) > 500:
            return "reasoning"

        return "simple"

    def get_model_info(self) -> dict[str, Any]:
        """Get current model information including auto-route status."""
        usage = self.router.get_usage()
        return {
            "active_model": self.router.active_model,
            "auto_mode": self.auto_mode,
            "previous_model": self._previous_model,
            "usage": usage,
        }

    def format_cost_report(self) -> str:
        """Generate a formatted cost/usage report."""
        usage = self.router.get_usage()
        lines = [
            f"Model: {self.router.active_model}",
            f"Auto-routing: {'ON' if self.auto_mode else 'OFF'}",
            f"Requests: {usage['requests']}",
            f"Input tokens: {usage['input']:,}",
            f"Output tokens: {usage['output']:,}",
            f"Total tokens: {usage['total_tokens']:,}",
            f"Estimated cost: ${usage['cost']:.4f}",
        ]
        return "\n".join(lines)
