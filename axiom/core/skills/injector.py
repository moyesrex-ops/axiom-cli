"""Skill injector — inject relevant skills into the agent's system prompt.

Uses semantic similarity (when available) or keyword matching to select
the most relevant skills for the current task, respecting a token budget.
"""

from __future__ import annotations

import logging
import re
from typing import Optional

from axiom.core.skills.loader import Skill, SkillLoader

logger = logging.getLogger(__name__)

# Default token budget for skill injection (don't overwhelm the context)
DEFAULT_SKILL_BUDGET = 4000  # ~4K tokens for skills


class SkillInjector:
    """Select and format skills for injection into the system prompt.

    Strategies:
    1. Explicit: User specifies skills by name (@skill-name)
    2. Semantic: Match skills by relevance to the current task
    3. Keyword: Fallback matching by word overlap
    """

    def __init__(
        self,
        loader: SkillLoader,
        token_budget: int = DEFAULT_SKILL_BUDGET,
    ) -> None:
        self.loader = loader
        self.token_budget = token_budget

    def inject(
        self,
        task: str,
        explicit_skills: list[str] | None = None,
        max_skills: int = 5,
    ) -> str:
        """Build a skill injection string for the system prompt.

        Args:
            task: The user's current task/message.
            explicit_skills: Skill names the user explicitly requested.
            max_skills: Maximum number of skills to inject.

        Returns:
            Formatted string to append to the system prompt, or empty string.
        """
        selected: list[Skill] = []
        budget_remaining = self.token_budget

        # 1. Add explicitly requested skills first (always included)
        if explicit_skills:
            for name in explicit_skills:
                skill = self.loader.get(name)
                if skill and skill.token_estimate <= budget_remaining:
                    selected.append(skill)
                    budget_remaining -= skill.token_estimate

        # 2. Auto-select relevant skills by keyword matching
        if len(selected) < max_skills:
            candidates = self._rank_by_relevance(task, exclude=[s.name for s in selected])
            for skill in candidates:
                if len(selected) >= max_skills:
                    break
                if skill.token_estimate > budget_remaining:
                    continue
                selected.append(skill)
                budget_remaining -= skill.token_estimate

        if not selected:
            return ""

        # Format the skills section
        return self._format_skills(selected)

    def extract_skill_mentions(self, text: str) -> list[str]:
        """Extract @skill-name mentions from user text.

        Returns list of skill names (without the @ prefix).
        """
        mentions = re.findall(r"@([\w-]+)", text)
        # Filter to only valid skill names
        return [m for m in mentions if self.loader.get(m) is not None]

    def _rank_by_relevance(
        self,
        task: str,
        exclude: list[str] | None = None,
    ) -> list[Skill]:
        """Rank skills by relevance to the task using keyword overlap.

        Falls back to simple word matching. Could be enhanced with
        embeddings for semantic matching.
        """
        exclude_set = set(exclude or [])
        task_words = set(task.lower().split())

        # Remove very common words
        stop_words = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been",
            "being", "have", "has", "had", "do", "does", "did", "will",
            "would", "could", "should", "may", "might", "can", "shall",
            "to", "of", "in", "for", "on", "with", "at", "by", "from",
            "as", "into", "through", "during", "before", "after", "above",
            "below", "between", "out", "off", "over", "under", "again",
            "further", "then", "once", "here", "there", "when", "where",
            "why", "how", "all", "each", "every", "both", "few", "more",
            "most", "other", "some", "such", "no", "nor", "not", "only",
            "own", "same", "so", "than", "too", "very", "just", "because",
            "but", "and", "or", "if", "while", "about", "up", "my", "me",
            "i", "you", "it", "this", "that", "what", "which", "who",
            "please", "help", "want", "need", "make", "create", "build",
        }
        task_words -= stop_words

        scored: list[tuple[float, Skill]] = []
        for skill in self.loader.list_skills():
            if skill.name in exclude_set:
                continue

            # Score = how many task words appear in skill name + description + tags
            skill_text = f"{skill.name} {skill.description} {' '.join(skill.tags)}".lower()
            skill_words = set(skill_text.split())

            overlap = len(task_words & skill_words)
            if overlap > 0:
                # Bonus for name match
                name_bonus = 3 if any(w in skill.name.lower() for w in task_words) else 0
                # Bonus for tag match
                tag_bonus = sum(
                    2 for tag in skill.tags
                    if tag.lower() in task_words
                )
                score = overlap + name_bonus + tag_bonus
                scored.append((score, skill))

        # Sort by score descending
        scored.sort(key=lambda x: x[0], reverse=True)
        return [skill for _, skill in scored]

    def _format_skills(self, skills: list[Skill]) -> str:
        """Format selected skills into a prompt injection section."""
        if not skills:
            return ""

        parts = [
            "\n## Active Skills",
            f"The following {len(skills)} skill(s) provide specialized knowledge for this task:\n",
        ]

        for skill in skills:
            parts.append(f"### {skill.name}")
            if skill.description:
                parts.append(f"*{skill.description}*\n")
            parts.append(skill.content)
            parts.append("")  # blank line separator

        return "\n".join(parts)
