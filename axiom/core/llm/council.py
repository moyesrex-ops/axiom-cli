"""LLM Council -- multi-model consensus for high-stakes decisions.

Queries multiple LLM providers in parallel, collects their responses,
runs a peer-review phase, then synthesizes the best answer using a
chairman model. Implements the multi-agent debate pattern.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from axiom.config.defaults import MODEL_DEFAULTS

logger = logging.getLogger(__name__)


@dataclass
class CouncilMember:
    """A member of the LLM council."""

    model: str
    response: str = ""
    review: str = ""
    score: float = 0.0
    latency_ms: float = 0.0
    error: str = ""


@dataclass
class CouncilResult:
    """Result of a council deliberation."""

    synthesis: str
    members: list[CouncilMember] = field(default_factory=list)
    chairman_model: str = ""
    total_time_ms: float = 0.0
    consensus_score: float = 0.0


COUNCIL_SYSTEM = """You are participating in a multi-model council.
Provide your best, most thorough answer to the user's question.
Be specific, cite reasoning, and show your work."""

REVIEW_SYSTEM = """You are reviewing another AI's response. Rate it 1-10 and provide brief feedback.
Focus on: accuracy, completeness, reasoning quality, and practical usefulness.
Format: SCORE: X/10\nFEEDBACK: your feedback"""

SYNTHESIS_SYSTEM = """You are the chairman of an AI council. Multiple models have answered the same question.
Synthesize the BEST answer by combining the strongest points from all responses.
Resolve any disagreements using the highest-scored responses as primary sources.
Do NOT mention the council process -- just provide the best unified answer."""


class LLMCouncil:
    """Multi-LLM consensus engine.

    Flow:
    1. Query N models in parallel with the same prompt
    2. Each model peer-reviews another's response
    3. Chairman model synthesizes the best answer
    """

    def __init__(self, router: Any) -> None:
        self._router = router

    async def deliberate(
        self,
        messages: list[dict[str, str]],
        models: list[str] | None = None,
        chairman: str | None = None,
    ) -> CouncilResult:
        """Run a full council deliberation.

        Args:
            messages: The conversation/question to deliberate on.
            models: List of model identifiers to query. If None, uses defaults.
            chairman: Model to use for final synthesis. Defaults to router's active model.

        Returns:
            CouncilResult with synthesized answer and member details.
        """
        start = time.time()

        # Default council members (mix of providers for diversity)
        if not models:
            models = self._select_default_models()

        if len(models) < 2:
            # Need at least 2 for a council; fall back to single model
            logger.warning("Council needs 2+ models, only %d available", len(models))
            models = models * 2 if models else [self._router.active_model]

        chairman = chairman or self._router.active_model

        # Phase 1: Parallel query
        logger.info("Council Phase 1: Querying %d models", len(models))
        members = await self._phase_query(messages, models)

        # Filter out failed members
        active_members = [m for m in members if m.response and not m.error]
        if not active_members:
            return CouncilResult(
                synthesis="Council deliberation failed -- no models responded successfully.",
                members=members,
                chairman_model=chairman,
                total_time_ms=(time.time() - start) * 1000,
            )

        # Phase 2: Peer review (if 3+ members)
        if len(active_members) >= 3:
            logger.info("Council Phase 2: Peer review")
            await self._phase_review(messages, active_members)

        # Phase 3: Synthesis
        logger.info("Council Phase 3: Chairman synthesis")
        synthesis = await self._phase_synthesize(
            messages, active_members, chairman
        )

        total_ms = (time.time() - start) * 1000

        # Calculate consensus score (agreement between members)
        consensus = self._calculate_consensus(active_members)

        return CouncilResult(
            synthesis=synthesis,
            members=members,
            chairman_model=chairman,
            total_time_ms=total_ms,
            consensus_score=consensus,
        )

    def _select_default_models(self) -> list[str]:
        """Select default council members from available models."""
        # Try to pick diverse providers -- order by preference
        preferred_providers = [
            "anthropic",
            "groq",
            "deepseek",
            "gemini",
            "openai",
            "vertex_ai",
            "together_ai",
        ]

        # Get providers that have API keys configured
        try:
            available_set = set(self._router.settings.available_providers())
        except Exception:
            available_set = set()

        available: list[str] = []
        for provider in preferred_providers:
            if provider in available_set:
                model = MODEL_DEFAULTS.get(provider)
                if model:
                    available.append(model)

        # Need at least 2; fall back to active model duplicated
        if len(available) < 2:
            active = self._router.active_model
            available = [active, active]

        return available[:5]  # Max 5 members

    async def _phase_query(
        self,
        messages: list[dict[str, str]],
        models: list[str],
    ) -> list[CouncilMember]:
        """Phase 1: Query all models in parallel."""

        async def _query_one(model: str) -> CouncilMember:
            member = CouncilMember(model=model)
            start = time.time()
            try:
                council_messages = [
                    {"role": "system", "content": COUNCIL_SYSTEM},
                    *messages,
                ]

                response_text = ""
                async for chunk in self._router.complete(
                    messages=council_messages,
                    model=model,
                    stream=True,
                ):
                    if hasattr(chunk, "choices") and chunk.choices:
                        delta = chunk.choices[0].delta
                        if hasattr(delta, "content") and delta.content:
                            response_text += delta.content

                member.response = response_text
                member.latency_ms = (time.time() - start) * 1000

            except Exception as exc:
                member.error = str(exc)
                member.latency_ms = (time.time() - start) * 1000
                logger.debug("Council member %s failed: %s", model, exc)

            return member

        members = await asyncio.gather(
            *[_query_one(model) for model in models],
            return_exceptions=False,
        )
        return list(members)

    async def _phase_review(
        self,
        messages: list[dict[str, str]],
        members: list[CouncilMember],
    ) -> None:
        """Phase 2: Each member reviews another's response."""

        async def _review_one(
            reviewer: CouncilMember, reviewee: CouncilMember
        ) -> None:
            try:
                review_messages = [
                    {"role": "system", "content": REVIEW_SYSTEM},
                    {
                        "role": "user",
                        "content": messages[-1].get("content", "")
                        if messages
                        else "",
                    },
                    {
                        "role": "assistant",
                        "content": f"Response to review:\n\n{reviewee.response[:3000]}",
                    },
                    {"role": "user", "content": "Please rate and review this response."},
                ]

                review_text = ""
                async for chunk in self._router.complete(
                    messages=review_messages,
                    model=reviewer.model,
                    stream=True,
                ):
                    if hasattr(chunk, "choices") and chunk.choices:
                        delta = chunk.choices[0].delta
                        if hasattr(delta, "content") and delta.content:
                            review_text += delta.content

                reviewee.review = review_text

                # Extract score
                score_match = re.search(r"SCORE:\s*(\d+)", review_text)
                if score_match:
                    reviewee.score = float(score_match.group(1))

            except Exception as exc:
                logger.debug("Review by %s failed: %s", reviewer.model, exc)

        # Circular review: each member reviews the next
        tasks = []
        for i, member in enumerate(members):
            reviewer = members[(i + 1) % len(members)]
            tasks.append(_review_one(reviewer, member))

        await asyncio.gather(*tasks, return_exceptions=True)

    async def _phase_synthesize(
        self,
        messages: list[dict[str, str]],
        members: list[CouncilMember],
        chairman_model: str,
    ) -> str:
        """Phase 3: Chairman synthesizes the best answer."""
        # Build context with all responses
        responses_context = []
        for i, member in enumerate(members, 1):
            entry = f"### Response {i} (score: {member.score:.0f}/10)\n{member.response[:3000]}"
            if member.review:
                entry += f"\n\n**Peer Review:** {member.review[:500]}"
            responses_context.append(entry)

        user_question = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                user_question = msg.get("content", "")
                break

        synth_messages = [
            {"role": "system", "content": SYNTHESIS_SYSTEM},
            {
                "role": "user",
                "content": (
                    f"**Original Question:** {user_question}\n\n"
                    f"**Council Responses:**\n\n"
                    + "\n\n---\n\n".join(responses_context)
                    + "\n\n**Synthesize the best unified answer.**"
                ),
            },
        ]

        synthesis = ""
        try:
            async for chunk in self._router.complete(
                messages=synth_messages,
                model=chairman_model,
                stream=True,
            ):
                if hasattr(chunk, "choices") and chunk.choices:
                    delta = chunk.choices[0].delta
                    if hasattr(delta, "content") and delta.content:
                        synthesis += delta.content
        except Exception as exc:
            logger.error("Chairman synthesis failed: %s", exc)
            # Fall back to highest-scored response
            best = max(members, key=lambda m: m.score)
            synthesis = best.response

        return synthesis or "Council deliberation produced no result."

    def _calculate_consensus(self, members: list[CouncilMember]) -> float:
        """Calculate a consensus score (0-1) based on review scores."""
        scores = [m.score for m in members if m.score > 0]
        if not scores:
            return 0.5

        avg = sum(scores) / len(scores)
        # Normalize: all 10s = 1.0, all 1s = 0.1
        return min(avg / 10.0, 1.0)
