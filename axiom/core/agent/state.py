"""Agent state machine -- statuses, modes, traces, and the main AgentState.

Ported from axiom-box with SSE removed and CLI-specific fields added.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


# ── Enums ─────────────────────────────────────────────────────────────


class AgentStatus(str, Enum):
    """Current phase of the agent execution loop."""

    PLANNING = "planning"
    EXECUTING = "executing"
    OBSERVING = "observing"
    REASONING = "reasoning"
    REFLECTING = "reflecting"
    COMPLETE = "complete"
    ERROR = "error"
    AWAITING_USER = "awaiting_user"

    @property
    def is_terminal(self) -> bool:
        return self in (AgentStatus.COMPLETE, AgentStatus.ERROR)


class AgentMode(str, Enum):
    """Which execution strategy the agent is using."""

    PLAN = "plan"
    REACT = "react"
    COUNCIL = "council"


# ── Dataclasses ───────────────────────────────────────────────────────


@dataclass
class PlanStep:
    """A single step in a PLAN-mode execution plan."""

    description: str
    tool_name: Optional[str] = None
    tool_args: dict[str, Any] = field(default_factory=dict)
    status: str = "pending"  # pending | running | done | failed | skipped
    result: Optional[str] = None
    reasoning: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "description": self.description,
            "tool_name": self.tool_name,
            "tool_args": self.tool_args,
            "status": self.status,
            "result": self.result,
            "reasoning": self.reasoning,
        }


@dataclass
class ReActTrace:
    """One iteration of the ReAct (Thought -> Action -> Observation) loop."""

    iteration: int
    thought: str = ""
    action: Optional[str] = None
    action_input: dict[str, Any] = field(default_factory=dict)
    observation: Optional[str] = None
    success: bool = True
    duration_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "iteration": self.iteration,
            "thought": self.thought,
            "action": self.action,
            "action_input": self.action_input,
            "observation": self.observation,
            "success": self.success,
            "duration_ms": self.duration_ms,
        }


@dataclass
class ToolCallRecord:
    """Record of a single tool invocation (used in both PLAN and REACT)."""

    tool_name: str
    args: dict[str, Any] = field(default_factory=dict)
    result: Optional[str] = None
    success: bool = True
    duration_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "args": self.args,
            "result": self.result,
            "success": self.success,
            "duration_ms": self.duration_ms,
        }


# ── Main Agent State ──────────────────────────────────────────────────


@dataclass
class AgentState:
    """Full mutable state for a single agent run.

    Tracks the execution mode, plan steps, ReAct traces, tool calls,
    iterations, and CLI-specific fields like model/cost/token tracking.
    """

    # Identity
    session_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])

    # Mode and status
    mode: AgentMode = AgentMode.PLAN
    status: AgentStatus = AgentStatus.PLANNING
    confidence: float = 0.0

    # PLAN mode state
    plan: list[PlanStep] = field(default_factory=list)
    current_step_index: int = 0
    replans: int = 0
    max_replans: int = 8

    # REACT mode state
    react_trace: list[ReActTrace] = field(default_factory=list)

    # COUNCIL mode state
    council_result: Optional[dict[str, Any]] = None

    # Shared execution tracking
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    iterations: int = 0
    max_iterations: int = 50
    consecutive_failures: int = 0
    max_consecutive_failures: int = 10

    # Timing
    started_at: float = field(default_factory=time.time)
    finished_at: Optional[float] = None

    # Messages (user + assistant + tool results accumulated during the run)
    messages: list[dict[str, Any]] = field(default_factory=list)

    # Final answer
    final_answer: Optional[str] = None
    error: Optional[str] = None

    # ── CLI-specific fields ───────────────────────────────────────
    active_model: str = ""
    cost_usd: float = 0.0
    tokens_in: int = 0
    tokens_out: int = 0

    # ── Properties ────────────────────────────────────────────────

    @property
    def is_terminal(self) -> bool:
        """Whether the agent has reached a final state."""
        return self.status.is_terminal

    @property
    def current_step(self) -> Optional[PlanStep]:
        """The currently-executing plan step (PLAN mode only)."""
        if 0 <= self.current_step_index < len(self.plan):
            return self.plan[self.current_step_index]
        return None

    @property
    def last_trace(self) -> Optional[ReActTrace]:
        """Most recent ReAct trace entry."""
        return self.react_trace[-1] if self.react_trace else None

    @property
    def elapsed_seconds(self) -> float:
        """Seconds since the agent run started."""
        end = self.finished_at or time.time()
        return round(end - self.started_at, 2)

    @property
    def total_tokens(self) -> int:
        return self.tokens_in + self.tokens_out

    # ── Mutators ──────────────────────────────────────────────────

    def add_trace(self, trace: ReActTrace) -> None:
        """Append a ReAct trace and bump the iteration counter."""
        self.react_trace.append(trace)
        self.iterations += 1

    def add_tool_call(self, record: ToolCallRecord) -> None:
        """Record a tool invocation."""
        self.tool_calls.append(record)
        if not record.success:
            self.consecutive_failures += 1
        else:
            self.consecutive_failures = 0

    def mark_complete(self, answer: str) -> None:
        """Transition to COMPLETE with a final answer."""
        self.status = AgentStatus.COMPLETE
        self.final_answer = answer
        self.finished_at = time.time()

    def mark_error(self, error: str) -> None:
        """Transition to ERROR."""
        self.status = AgentStatus.ERROR
        self.error = error
        self.finished_at = time.time()

    def add_usage(self, tokens_in: int = 0, tokens_out: int = 0, cost: float = 0.0) -> None:
        """Accumulate token usage and cost from an LLM call."""
        self.tokens_in += tokens_in
        self.tokens_out += tokens_out
        self.cost_usd += cost

    # ── Serialization ─────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        """Full state snapshot -- no truncation (CLI has no payload limits)."""
        return {
            "session_id": self.session_id,
            "mode": self.mode.value,
            "status": self.status.value,
            "confidence": self.confidence,
            "plan": [s.to_dict() for s in self.plan],
            "current_step_index": self.current_step_index,
            "replans": self.replans,
            "react_trace": [t.to_dict() for t in self.react_trace],
            "council_result": self.council_result,
            "tool_calls": [tc.to_dict() for tc in self.tool_calls],
            "iterations": self.iterations,
            "max_iterations": self.max_iterations,
            "consecutive_failures": self.consecutive_failures,
            "elapsed_seconds": self.elapsed_seconds,
            "final_answer": self.final_answer,
            "error": self.error,
            "active_model": self.active_model,
            "cost_usd": round(self.cost_usd, 6),
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "total_tokens": self.total_tokens,
        }
