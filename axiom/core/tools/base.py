"""Base tool interface and error type for all Axiom tools.

Every tool implements the AxiomTool ABC, providing a name, description,
JSON-schema parameters, risk level, and an async execute() method.

The ToolRegistry lives in ``axiom.core.tools.registry``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class ToolError(Exception):
    """Raised when a tool encounters a recoverable error during execution.

    The agent loop catches these and feeds the error message back into
    the LLM's observation so it can retry or adjust its approach.
    """

    def __init__(self, message: str, tool_name: str = "", recoverable: bool = True):
        self.tool_name = tool_name
        self.recoverable = recoverable
        super().__init__(message)


class AxiomTool(ABC):
    """Abstract base class for all Axiom tools.

    Subclass this and fill in the class-level attributes, then implement
    the async ``execute(**kwargs)`` method.

    Attributes:
        name:              Short snake_case identifier (e.g. ``"code_execute"``).
        description:       One-line description shown to the LLM.
        parameters_schema: JSON Schema ``properties`` dict describing accepted kwargs.
        risk_level:        ``"low"`` / ``"medium"`` / ``"high"`` -- controls the
                           CLI approval flow.  ``"high"`` tools always prompt for
                           user confirmation before running.
    """

    name: str = ""
    description: str = ""
    parameters_schema: dict[str, Any] = {}
    risk_level: str = "low"  # "low" | "medium" | "high"

    @abstractmethod
    async def execute(self, **kwargs: Any) -> str:
        """Run the tool and return a string result for the LLM.

        Raises:
            ToolError: On recoverable failures (fed back to the LLM).
            Exception: On unrecoverable failures (bubbles up to the agent loop).
        """
        ...

    # ── LLM function-calling schema ───────────────────────────────

    def to_llm_schema(self) -> dict[str, Any]:
        """Return an OpenAI-compatible function/tool schema for this tool.

        The schema is suitable for passing in the ``tools`` parameter of
        LiteLLM / OpenAI ``chat.completions.create()``.

        Supports parameters_schema as either:
          - A full JSON Schema object (with ``type``, ``properties``, ``required``)
          - A flat properties dict (legacy format, auto-wrapped)
        """
        schema = self.parameters_schema
        # If the schema already has "type": "object", use it directly
        if schema.get("type") == "object":
            params = schema
        else:
            # Legacy flat-properties format: wrap it
            params = {
                "type": "object",
                "properties": schema,
                "required": [
                    k
                    for k, v in schema.items()
                    if isinstance(v, dict) and v.get("required", False)
                ],
            }
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": params,
            },
        }

    # ── Display helpers ───────────────────────────────────────────

    def __repr__(self) -> str:
        return f"<AxiomTool {self.name!r} risk={self.risk_level}>"

    def __str__(self) -> str:
        return f"{self.name}: {self.description}"
