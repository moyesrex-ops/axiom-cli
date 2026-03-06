"""Memory tool -- allows the agent to search and store persistent memories."""

from __future__ import annotations

import json
from typing import Any

from axiom.core.tools.base import AxiomTool


class MemorySearchTool(AxiomTool):
    name = "memory_search"
    description = (
        "Search the agent's persistent memory for relevant past conversations, "
        "facts, and decisions. Use this to recall previous context."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Natural language search query for memory retrieval.",
            },
            "k": {
                "type": "integer",
                "description": "Number of results to return (default 5).",
                "default": 5,
            },
        },
        "required": ["query"],
    }
    risk_level = "low"

    def __init__(self, memory_manager=None):
        self._memory = memory_manager

    async def execute(self, **kwargs: Any) -> str:
        if self._memory is None:
            return "Memory system not initialized."

        query = kwargs.get("query", "")
        k = kwargs.get("k", 5)

        results = self._memory.search(query, k=k)
        if not results:
            return f"No memories found matching: {query}"

        output = [f"Found {len(results)} relevant memories:\n"]
        for i, r in enumerate(results, 1):
            meta = r.get("metadata", {})
            output.append(
                f"{i}. [{meta.get('type', 'unknown')}] "
                f"(similarity: {1 - r.get('distance', 0):.2f})\n"
                f"   {r['text'][:500]}\n"
            )

        return "\n".join(output)


class MemorySaveTool(AxiomTool):
    name = "memory_save"
    description = (
        "Save an important fact, decision, or learning to persistent memory. "
        "Use this when you learn something important that should be remembered."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "content": {
                "type": "string",
                "description": "The fact or information to remember.",
            },
            "tags": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional tags for categorization (e.g., ['project', 'preference']).",
                "default": [],
            },
        },
        "required": ["content"],
    }
    risk_level = "low"

    def __init__(self, memory_manager=None):
        self._memory = memory_manager

    async def execute(self, **kwargs: Any) -> str:
        if self._memory is None:
            return "Memory system not initialized."

        content = kwargs.get("content", "")
        tags = kwargs.get("tags", [])

        if not content:
            return "No content provided to save."

        self._memory.store_fact(content, tags=tags)
        return f"Saved to memory: {content[:100]}{'...' if len(content) > 100 else ''}"
