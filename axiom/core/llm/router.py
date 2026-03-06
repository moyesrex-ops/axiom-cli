"""Universal LLM Router -- routes to 15+ providers via LiteLLM.

The router handles:
    - API key injection from AxiomSettings into environment variables
    - Circuit breaking per provider (auto-fallback on repeated failures)
    - Live model switching via short aliases (``/model sonnet``)
    - Token and cost tracking across the session
    - Streaming and non-streaming completions with automatic fallback

Usage:
    from axiom.config import get_settings
    from axiom.core.llm.router import UniversalRouter

    router = UniversalRouter(get_settings())
    async for chunk in router.complete(messages, stream=True):
        print(chunk)
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Optional

import litellm
from litellm import acompletion

from axiom.config.defaults import MODEL_DEFAULTS, PROVIDER_DISPLAY_NAMES

logger = logging.getLogger(__name__)


# ── Circuit Breaker ───────────────────────────────────────────────────


@dataclass
class CircuitBreaker:
    """Per-provider circuit breaker to avoid hammering failing endpoints.

    States:
        closed    -- healthy, requests pass through
        open      -- failing, requests are blocked until timeout expires
        half_open -- timeout expired, next request is a probe
    """

    threshold: int = 3
    timeout_seconds: float = 60.0
    failures: int = 0
    last_failure_time: float = 0.0
    state: str = "closed"  # closed | open | half_open
    last_error: str = ""

    def is_available(self) -> bool:
        """Check if the provider should receive requests."""
        if self.state == "closed":
            return True
        if self.state == "open":
            elapsed = time.time() - self.last_failure_time
            if elapsed >= self.timeout_seconds:
                self.state = "half_open"
                logger.info("Circuit breaker half-open (testing after %.0fs)", elapsed)
                return True
            return False
        # half_open -- allow one probe
        return True

    def record_success(self) -> None:
        """Reset on successful request."""
        if self.state != "closed":
            logger.info("Circuit breaker closed (provider recovered)")
        self.failures = 0
        self.state = "closed"
        self.last_error = ""

    def record_failure(self, error: str) -> None:
        """Increment failure count; trip open if threshold exceeded."""
        self.failures += 1
        self.last_failure_time = time.time()
        self.last_error = error
        if self.failures >= self.threshold:
            self.state = "open"
            logger.warning(
                "Circuit breaker OPEN after %d failures: %s",
                self.failures,
                error[:120],
            )
        elif self.state == "half_open":
            # Probe failed -- back to open
            self.state = "open"
            logger.warning("Circuit breaker re-opened (probe failed): %s", error[:120])


# ── Model Shortcut Map ────────────────────────────────────────────────

MODEL_SHORTCUTS: dict[str, str] = {
    # ── Claude (Anthropic API) ─────────────────────────────────
    "opus": "anthropic/claude-opus-4-6",
    "claude": "anthropic/claude-opus-4-6",
    "claude-opus": "anthropic/claude-opus-4-6",
    "opus-4.6": "anthropic/claude-opus-4-6",
    "sonnet": "anthropic/claude-sonnet-4-6",
    "claude-sonnet": "anthropic/claude-sonnet-4-6",
    "sonnet-4.6": "anthropic/claude-sonnet-4-6",
    "haiku": "anthropic/claude-haiku-4-5-20251001",
    "claude-haiku": "anthropic/claude-haiku-4-5-20251001",
    "haiku-4.5": "anthropic/claude-haiku-4-5-20251001",
    # Legacy Claude
    "opus-4.5": "anthropic/claude-opus-4-5-20251101",
    "sonnet-4.5": "anthropic/claude-sonnet-4-5-20250929",
    "opus-4.1": "anthropic/claude-opus-4-1-20250805",
    "opus-4.0": "anthropic/claude-opus-4-20250514",
    "sonnet-4.0": "anthropic/claude-sonnet-4-20250514",

    # ── Claude via Vertex AI (Google Cloud billing) ────────────
    "vertex-opus": "vertex_ai/claude-opus-4-6",
    "vertex-opus-4.6": "vertex_ai/claude-opus-4-6",
    "vertex-sonnet": "vertex_ai/claude-sonnet-4-6",
    "vertex-sonnet-4.6": "vertex_ai/claude-sonnet-4-6",
    "vertex-haiku": "vertex_ai/claude-haiku-4-5@20251001",
    "vertex-haiku-4.5": "vertex_ai/claude-haiku-4-5@20251001",
    "vertex-claude": "vertex_ai/claude-opus-4-6",
    # Legacy Claude on Vertex
    "vertex-opus-4.5": "vertex_ai/claude-opus-4-5@20251101",
    "vertex-sonnet-4.5": "vertex_ai/claude-sonnet-4-5@20250929",
    "vertex-opus-4.1": "vertex_ai/claude-opus-4-1@20250805",
    "vertex-opus-4.0": "vertex_ai/claude-opus-4@20250514",
    "vertex-sonnet-4.0": "vertex_ai/claude-sonnet-4@20250514",

    # ── Gemini via Vertex AI (Google Cloud billing) ────────────
    "vertex-gemini": "vertex_ai/gemini-2.5-pro",
    "vertex-gemini-pro": "vertex_ai/gemini-2.5-pro",
    "vertex-pro": "vertex_ai/gemini-2.5-pro",
    "vertex-flash": "vertex_ai/gemini-2.5-flash",
    "vertex-gemini-flash": "vertex_ai/gemini-2.5-flash",
    "vertex-2.0-flash": "vertex_ai/gemini-2.0-flash",
    "vertex-2.0": "vertex_ai/gemini-2.0-flash",
    "vertex-2.0-lite": "vertex_ai/gemini-2.0-flash-lite",
    "vertex-lite": "vertex_ai/gemini-2.0-flash-lite",
    "vertex-1.5-pro": "vertex_ai/gemini-1.5-pro",
    "vertex-1.5-flash": "vertex_ai/gemini-1.5-flash",
    "vertex": "vertex_ai/gemini-2.5-pro",
    # Gemini 3.x (preview, latest generation)
    "vertex-3.1-pro": "vertex_ai/gemini-3.1-pro-preview",
    "vertex-3.1-flash": "vertex_ai/gemini-3.1-flash-lite-preview",
    "vertex-3-pro": "vertex_ai/gemini-3-pro-preview",
    "vertex-3-flash": "vertex_ai/gemini-3-flash-preview",

    # ── DeepSeek on Vertex AI (Model Garden MaaS) ────────────
    "vertex-deepseek": "vertex_ai/deepseek-ai/deepseek-v3.2-maas",
    "vertex-deepseek-v3": "vertex_ai/deepseek-ai/deepseek-v3.2-maas",
    "vertex-deepseek-v3.2": "vertex_ai/deepseek-ai/deepseek-v3.2-maas",
    "vertex-deepseek-v3.1": "vertex_ai/deepseek-ai/deepseek-v3.1-maas",
    "vertex-deepseek-r1": "vertex_ai/deepseek-ai/deepseek-r1-0528-maas",
    "vertex-deepseek-ocr": "vertex_ai/deepseek-ai/deepseek-ocr-maas",

    # ── Qwen on Vertex AI (Model Garden MaaS) ────────────────
    "vertex-qwen": "vertex_ai/qwen/qwen3-coder-480b-a35b-instruct-maas",
    "vertex-qwen-coder": "vertex_ai/qwen/qwen3-coder-480b-a35b-instruct-maas",
    "vertex-qwen-235b": "vertex_ai/qwen/qwen3-235b-a22b-instruct-2507-maas",
    "vertex-qwen-next": "vertex_ai/qwen/qwen3-next-80b-a3b-instruct-maas",
    "vertex-qwen-thinking": "vertex_ai/qwen/qwen3-next-80b-a3b-thinking-maas",

    # ── MiniMax on Vertex AI (Model Garden MaaS) ─────────────
    "vertex-minimax": "vertex_ai/minimaxai/minimax-m2-maas",
    "vertex-minimax-m2": "vertex_ai/minimaxai/minimax-m2-maas",

    # ── Kimi (Moonshot) on Vertex AI (Model Garden MaaS) ─────
    "vertex-kimi": "vertex_ai/moonshotai/kimi-k2-thinking-maas",
    "vertex-kimi-k2": "vertex_ai/moonshotai/kimi-k2-thinking-maas",

    # ── GLM (ZAI.org) on Vertex AI (Model Garden MaaS) ───────
    "vertex-glm": "vertex_ai/zai-org/glm-5-maas",
    "vertex-glm-5": "vertex_ai/zai-org/glm-5-maas",
    "vertex-glm-4.7": "vertex_ai/zai-org/glm-4.7-maas",

    # ── OpenAI GPT-OSS on Vertex AI (Model Garden MaaS) ──────
    "vertex-gpt-oss": "vertex_ai/openai/gpt-oss-120b-maas",
    "vertex-gpt-oss-120b": "vertex_ai/openai/gpt-oss-120b-maas",
    "vertex-gpt-oss-20b": "vertex_ai/openai/gpt-oss-20b-maas",

    # ── Llama on Vertex AI (Model Garden MaaS) ───────────────
    "vertex-llama4-scout": "vertex_ai/meta/llama-4-scout-17b-16e-instruct-maas",
    "vertex-llama4-maverick": "vertex_ai/meta/llama-4-maverick-17b-128e-instruct-maas",
    "vertex-llama4": "vertex_ai/meta/llama-4-scout-17b-16e-instruct-maas",
    "vertex-llama3": "vertex_ai/meta/llama-3.3-70b-instruct-maas",
    "vertex-llama3-405b": "vertex_ai/meta/llama-3.1-405b-instruct-maas",

    # ── Mistral on Vertex AI (Model Garden) ──────────────────
    "vertex-mistral": "vertex_ai/mistral-large@latest",
    "vertex-mistral-large": "vertex_ai/mistral-large@latest",
    "vertex-mistral-medium": "vertex_ai/mistral-medium-3@001",
    "vertex-mistral-small": "vertex_ai/mistral-small-2503@001",
    "vertex-codestral": "vertex_ai/codestral-2@001",
    "vertex-codestral-2": "vertex_ai/codestral-2@001",

    # ── AI21 Jamba on Vertex AI ───────────────────────────────
    "vertex-jamba": "vertex_ai/jamba-1.5-large@001",
    "vertex-jamba-large": "vertex_ai/jamba-1.5-large@001",
    "vertex-jamba-mini": "vertex_ai/jamba-1.5-mini@001",

    # ── Gemini (Google AI Studio - free tier) ──────────────────
    "gemini": "gemini/gemini-2.5-flash",
    "gemini-flash": "gemini/gemini-2.5-flash",
    "gemini-pro": "gemini/gemini-2.5-pro",
    "gemini-2.0": "gemini/gemini-2.0-flash",

    # ── OpenAI ─────────────────────────────────────────────────
    "gpt4o": "openai/gpt-4o",
    "gpt4": "openai/gpt-4o",
    "gpt": "openai/gpt-4o",
    "o1": "openai/o1",
    "o3": "openai/o3-mini",
    "o3-mini": "openai/o3-mini",

    # ── Free-tier / generous cloud APIs ────────────────────────
    "groq": "groq/llama-3.3-70b-versatile",
    "nvidia": "nvidia_nim/meta/llama-3.3-70b-instruct",
    "deepseek": "deepseek/deepseek-chat",
    "deepseek-r1": "deepseek/deepseek-reasoner",
    "together": "together_ai/meta-llama/Llama-4-Scout-17B-16E-Instruct",
    "mistral": "mistral/mistral-small-latest",
    "codestral": "mistral/codestral-latest",
    "cohere": "cohere_chat/command-r-plus",
    "openrouter": "openrouter/auto",

    # ── Local LLMs (Ollama) ────────────────────────────────────
    "qwen": "ollama/qwen2.5:7b",
    "qwen-14b": "ollama/qwen2.5:14b",
    "llama": "ollama/llama3.3:8b",
    "local": "ollama/qwen2.5:7b",
    "deepseek-local": "ollama/deepseek-r1:7b",
    "mistral-local": "ollama/mistral:7b",
    "phi": "ollama/phi3.5:latest",
}

# Provider-level fallback chain: if provider X fails, try Y
_FALLBACK_CHAIN: dict[str, list[str]] = {
    "anthropic": ["vertex_ai", "openai", "groq", "deepseek", "gemini"],
    "openai": ["anthropic", "vertex_ai", "groq", "deepseek", "gemini"],
    "groq": ["deepseek", "gemini", "nvidia_nim", "together_ai"],
    "gemini": ["vertex_ai", "groq", "deepseek", "nvidia_nim"],
    "nvidia_nim": ["groq", "deepseek", "together_ai"],
    "deepseek": ["groq", "gemini", "nvidia_nim"],
    "together_ai": ["groq", "deepseek", "nvidia_nim"],
    "mistral": ["groq", "deepseek", "gemini"],
    "cohere_chat": ["groq", "deepseek", "gemini"],
    "openrouter": ["groq", "deepseek", "gemini"],
    "huggingface": ["groq", "deepseek", "gemini"],
    "vertex_ai": ["anthropic", "gemini", "openai", "groq"],
    "ollama": ["groq", "deepseek", "gemini"],
}


# ── Universal Router ──────────────────────────────────────────────────


class UniversalRouter:
    """Routes LLM requests to 15+ providers via LiteLLM with circuit breaking.

    The router:
        1. Sets environment variables for each configured API key
        2. Delegates to ``litellm.acompletion()`` with the model string
        3. Tracks per-provider health via circuit breakers
        4. Falls back to alternative providers on failure
        5. Accumulates token usage and estimated cost
    """

    def __init__(self, settings: Any) -> None:
        self.settings = settings
        self.active_model: str = settings.DEFAULT_MODEL or "anthropic/claude-opus-4-6"
        self._circuit_breakers: dict[str, CircuitBreaker] = {}
        self._token_usage: dict[str, Any] = {
            "input": 0,
            "output": 0,
            "cost": 0.0,
        }
        self._request_count: int = 0
        self._setup_litellm()

    # ── Setup ─────────────────────────────────────────────────────

    def _setup_litellm(self) -> None:
        """Push all available API keys into env vars for LiteLLM."""
        s = self.settings

        # Map settings fields to the env vars LiteLLM expects
        _key_map: list[tuple[Optional[str], str]] = [
            (s.ANTHROPIC_API_KEY, "ANTHROPIC_API_KEY"),
            (s.OPENAI_API_KEY, "OPENAI_API_KEY"),
            (s.GROQ_API_KEY, "GROQ_API_KEY"),
            (s.GEMINI_API_KEY, "GEMINI_API_KEY"),
            (s.NVIDIA_API_KEY, "NVIDIA_API_KEY"),
            (s.DEEPSEEK_API_KEY, "DEEPSEEK_API_KEY"),
            (s.TOGETHER_API_KEY, "TOGETHER_API_KEY"),
            (s.MISTRAL_API_KEY, "MISTRAL_API_KEY"),
            (s.COHERE_API_KEY, "COHERE_API_KEY"),
            (s.OPENROUTER_API_KEY, "OPENROUTER_API_KEY"),
            (s.HUGGINGFACE_API_KEY, "HUGGINGFACE_API_KEY"),
            (s.TAVILY_API_KEY, "TAVILY_API_KEY"),
        ]

        for value, env_name in _key_map:
            if value:
                os.environ[env_name] = value

        # Vertex AI needs project + location + credentials
        if s.VERTEX_AI_PROJECT:
            os.environ["VERTEXAI_PROJECT"] = s.VERTEX_AI_PROJECT
        if s.VERTEX_AI_LOCATION:
            os.environ["VERTEXAI_LOCATION"] = s.VERTEX_AI_LOCATION
        if s.GOOGLE_APPLICATION_CREDENTIALS:
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = s.GOOGLE_APPLICATION_CREDENTIALS

        # Ollama base URL
        if s.OLLAMA_BASE_URL:
            os.environ["OLLAMA_API_BASE"] = s.OLLAMA_BASE_URL

        # LiteLLM global config
        litellm.drop_params = True  # Don't fail on unsupported params
        litellm.set_verbose = False  # Suppress LiteLLM debug logs

        logger.debug(
            "LiteLLM configured. Available providers: %s",
            ", ".join(s.available_providers()),
        )

    # ── Core Completion ───────────────────────────────────────────

    async def complete(
        self,
        messages: list[dict[str, Any]],
        model: Optional[str] = None,
        tools: Optional[list[dict[str, Any]]] = None,
        stream: bool = True,
        _fallback_depth: int = 0,
        **kwargs: Any,
    ) -> AsyncIterator[Any]:
        """Send a completion request, yielding chunks (stream) or a single response.

        Args:
            messages: Chat messages in OpenAI format.
            model: Override the active model (full LiteLLM model string).
            tools: OpenAI-compatible tool schemas.
            stream: Whether to stream the response.
            _fallback_depth: Internal recursion guard for fallback chains.
            **kwargs: Extra params forwarded to ``litellm.acompletion()``.

        Yields:
            Streaming chunks or a single ModelResponse.
        """
        model = model or self.active_model
        provider = self._extract_provider(model)

        # Circuit breaker check
        cb = self._get_breaker(provider)
        if not cb.is_available():
            fallback_model = self._get_fallback_model(provider)
            if fallback_model and fallback_model != model and _fallback_depth < 3:
                logger.info(
                    "Provider %s circuit open, falling back to %s",
                    provider,
                    fallback_model,
                )
                async for chunk in self.complete(
                    messages,
                    model=fallback_model,
                    tools=tools,
                    stream=stream,
                    _fallback_depth=_fallback_depth + 1,
                    **kwargs,
                ):
                    yield chunk
                return
            else:
                logger.warning("No fallback available for %s, attempting anyway", provider)

        # Build the sanitized message list
        sanitized = self._sanitize_messages(messages)

        # Build call kwargs
        call_kwargs: dict[str, Any] = {
            "model": model,
            "messages": sanitized,
            "stream": stream,
            **kwargs,
        }
        if tools:
            call_kwargs["tools"] = tools

        # Add provider-specific overrides
        self._apply_provider_overrides(model, call_kwargs)

        self._request_count += 1

        try:
            if stream:
                response = await acompletion(**call_kwargs)
                async for chunk in response:
                    yield chunk
                # Track usage from the final chunk if available
                # (LiteLLM accumulates usage in stream_options for some providers)
                cb.record_success()
            else:
                response = await acompletion(**call_kwargs)
                self._track_usage(response)
                yield response
                cb.record_success()

        except Exception as e:
            error_str = f"{type(e).__name__}: {e}"
            logger.warning("LLM call failed [%s]: %s", model, error_str[:200])
            cb.record_failure(error_str)

            # Attempt fallback
            fallback_model = self._get_fallback_model(provider)
            if (
                fallback_model
                and fallback_model != model
                and _fallback_depth < 3
            ):
                logger.info("Retrying with fallback model: %s", fallback_model)
                async for chunk in self.complete(
                    messages,
                    model=fallback_model,
                    tools=tools,
                    stream=stream,
                    _fallback_depth=_fallback_depth + 1,
                    **kwargs,
                ):
                    yield chunk
            else:
                raise

    async def complete_sync(
        self,
        messages: list[dict[str, Any]],
        model: Optional[str] = None,
        tools: Optional[list[dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> Any:
        """Non-streaming convenience -- returns the full ModelResponse directly."""
        result = None
        async for chunk in self.complete(
            messages, model=model, tools=tools, stream=False, **kwargs
        ):
            result = chunk
        return result

    # ── Model Switching ───────────────────────────────────────────

    def switch_model(self, model_name: str) -> str:
        """Switch the active model.  Accepts short aliases or full model strings.

        Args:
            model_name: Alias (e.g. ``"sonnet"``) or full string
                        (e.g. ``"anthropic/claude-sonnet-4-20250514"``).

        Returns:
            The resolved full model string now active.
        """
        full = MODEL_SHORTCUTS.get(model_name.lower().strip(), model_name)
        old = self.active_model
        self.active_model = full
        logger.info("Model switched: %s -> %s", old, full)
        return full

    # ── Introspection ─────────────────────────────────────────────

    def list_available(self) -> list[dict[str, Any]]:
        """Return all known providers with availability and model info.

        Returns a list of dicts, each with:
            provider, display_name, model, available, circuit_state
        """
        available_providers = set(self.settings.available_providers())
        result: list[dict[str, Any]] = []

        for provider, default_model in MODEL_DEFAULTS.items():
            is_available = provider in available_providers
            cb = self._circuit_breakers.get(provider)
            circuit_state = cb.state if cb else "closed"

            result.append({
                "provider": provider,
                "display_name": PROVIDER_DISPLAY_NAMES.get(provider, provider),
                "model": default_model,
                "available": is_available,
                "circuit_state": circuit_state,
                "is_active": default_model == self.active_model
                             or self._extract_provider(self.active_model) == provider,
            })

        return result

    def get_usage(self) -> dict[str, Any]:
        """Return cumulative token usage and estimated cost for this session."""
        return {
            **self._token_usage,
            "total_tokens": self._token_usage["input"] + self._token_usage["output"],
            "requests": self._request_count,
        }

    @property
    def provider_name(self) -> str:
        """Human-readable name of the current provider."""
        provider = self._extract_provider(self.active_model)
        return PROVIDER_DISPLAY_NAMES.get(provider, provider)

    @property
    def model_short_name(self) -> str:
        """Short display name for the active model (after the slash)."""
        if "/" in self.active_model:
            return self.active_model.split("/", 1)[1]
        return self.active_model

    # ── Startup Validation ──────────────────────────────────────────

    async def validate_default_model(self, timeout: float = 15.0) -> str:
        """Test the default model with a minimal request.

        If the default model fails (error *or* timeout), tries the fallback
        chain until a working model is found.  Returns the validated
        (possibly changed) model name.

        Args:
            timeout: Max seconds to wait per model probe.  Keeps startup
                     snappy even if a provider hangs (e.g. Claude on Vertex
                     when Model Garden isn't enabled returns 404 fast, but
                     some regions can time out).
        """
        import asyncio as _aio

        model = self.active_model
        provider = self._extract_provider(model)

        # Quick test with minimal tokens
        try:
            response = await _aio.wait_for(
                self.complete_sync(
                    messages=[{"role": "user", "content": "Hi"}],
                    model=model,
                    max_tokens=5,
                ),
                timeout=timeout,
            )
            if response:
                logger.info("Default model validated: %s", model)
                return model
        except _aio.TimeoutError:
            logger.warning("Default model %s timed out (%.0fs)", model, timeout)
        except Exception as e:
            logger.warning(
                "Default model %s failed validation: %s", model, str(e)[:120]
            )

        # Helper to probe a single model
        async def _probe(m: str) -> bool:
            try:
                resp = await _aio.wait_for(
                    self.complete_sync(
                        messages=[{"role": "user", "content": "Hi"}],
                        model=m,
                        max_tokens=5,
                    ),
                    timeout=timeout,
                )
                return bool(resp)
            except Exception:
                return False

        # If a third-party model on Vertex AI failed, first try the
        # provider's own default (Gemini) before jumping to other providers.
        provider_default = MODEL_DEFAULTS.get(provider)
        if provider_default and provider_default != model:
            if await _probe(provider_default):
                logger.info(
                    "Startup fallback (same provider): %s -> %s",
                    model, provider_default,
                )
                self.active_model = provider_default
                return provider_default

        # Try cross-provider fallback chain
        chain = _FALLBACK_CHAIN.get(provider, [])
        available = set(self.settings.available_providers())

        for alt_provider in chain:
            if alt_provider not in available:
                continue
            alt_model = MODEL_DEFAULTS.get(alt_provider)
            if not alt_model:
                continue
            if await _probe(alt_model):
                logger.info(
                    "Startup fallback: %s -> %s", model, alt_model
                )
                self.active_model = alt_model
                return alt_model

        # Keep original even if validation failed — user might fix keys later
        logger.warning("No model validated at startup; keeping %s", model)
        return model

    # ── Internals ─────────────────────────────────────────────────

    @staticmethod
    def _extract_provider(model: str) -> str:
        """Extract the provider prefix from a LiteLLM model string."""
        if "/" in model:
            return model.split("/")[0]
        return "unknown"

    def _get_breaker(self, provider: str) -> CircuitBreaker:
        """Get or create a circuit breaker for the given provider."""
        if provider not in self._circuit_breakers:
            self._circuit_breakers[provider] = CircuitBreaker()
        return self._circuit_breakers[provider]

    def _get_fallback_model(self, failed_provider: str) -> Optional[str]:
        """Find the first healthy fallback model for a failed provider."""
        chain = _FALLBACK_CHAIN.get(failed_provider, [])
        available = set(self.settings.available_providers())

        for alt_provider in chain:
            if alt_provider not in available:
                continue
            cb = self._circuit_breakers.get(alt_provider)
            if cb and not cb.is_available():
                continue
            model = MODEL_DEFAULTS.get(alt_provider)
            if model:
                return model

        return None

    @staticmethod
    def _sanitize_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Ensure messages have at minimum role + content for LiteLLM."""
        sanitized = []
        for m in messages:
            entry: dict[str, Any] = {
                "role": m.get("role", "user"),
                "content": m.get("content", ""),
            }
            # Preserve tool_calls and tool_call_id if present
            if "tool_calls" in m:
                entry["tool_calls"] = m["tool_calls"]
            if "tool_call_id" in m:
                entry["tool_call_id"] = m["tool_call_id"]
            if "name" in m:
                entry["name"] = m["name"]
            sanitized.append(entry)
        return sanitized

    @staticmethod
    def _apply_provider_overrides(model: str, kwargs: dict[str, Any]) -> None:
        """Apply provider-specific parameter overrides."""
        provider = model.split("/")[0] if "/" in model else ""

        # NVIDIA NIM needs explicit api_base
        if provider == "nvidia_nim":
            kwargs.setdefault("api_base", "https://integrate.api.nvidia.com/v1")

        # Ollama may need custom base
        if provider == "ollama":
            base = os.environ.get("OLLAMA_API_BASE", "http://localhost:11434")
            kwargs.setdefault("api_base", base)

    def _track_usage(self, response: Any) -> None:
        """Extract token counts and cost from a LiteLLM ModelResponse."""
        usage = getattr(response, "usage", None)
        if usage is None:
            return

        prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
        completion_tokens = getattr(usage, "completion_tokens", 0) or 0

        self._token_usage["input"] += prompt_tokens
        self._token_usage["output"] += completion_tokens

        # LiteLLM sometimes includes cost via response._hidden_params
        hidden = getattr(response, "_hidden_params", {})
        if isinstance(hidden, dict):
            cost = hidden.get("response_cost", 0.0)
            if cost:
                self._token_usage["cost"] += cost

    # ── Display ───────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"<UniversalRouter model={self.active_model!r} "
            f"requests={self._request_count}>"
        )
