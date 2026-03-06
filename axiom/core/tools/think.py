"""Think tool -- forces explicit chain-of-thought reasoning.

Zero side effects. The LLM externalizes its reasoning into a visible,
logged trace step before taking action. This is the most important tool
in Axiom's arsenal -- it prevents shallow single-attempt actions.
"""

from __future__ import annotations
from typing import Any
from axiom.core.tools.base import AxiomTool


class ThinkTool(AxiomTool):
    """Pure reasoning tool with zero side effects.

    The agent MUST use this before complex tasks to plan its approach,
    consider what could go wrong, and prepare fallback strategies.
    """

    name = "think"
    description = (
        "Reason explicitly before acting. Use this to plan your approach, "
        "consider alternatives, and prepare fallback strategies. "
        "MANDATORY before any multi-step task or search operation. "
        "Has zero side effects -- just returns your reasoning for the trace."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "reasoning": {
                "type": "string",
                "description": (
                    "Your explicit chain-of-thought. Include: "
                    "1) What am I trying to accomplish? "
                    "2) What's the simplest path? "
                    "3) What could go wrong? "
                    "4) What's my Plan B if this fails?"
                ),
            },
        },
        "required": ["reasoning"],
    }
    risk_level = "low"

    async def execute(self, **kwargs: Any) -> str:
        reasoning = kwargs.get("reasoning", "")
        if not reasoning:
            return "[Think] No reasoning provided. State your plan before acting."
        return f"[Reasoning]\n{reasoning}\n\n[Ready to act on this reasoning.]"
