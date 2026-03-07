"""Tri-mode agent orchestrator -- PLAN, REACT, COUNCIL.

This is the core agent execution engine. It:
1. Auto-selects the best mode for the task (via auto_selector)
2. Runs the selected mode's execution loop
3. Uses observer for self-correction (Reflexion pattern)
4. Integrates AgentTracer for full observability
5. Calls Learner for self-improvement after successful runs
6. Supports COUNCIL mode for multi-LLM consensus
7. Yields events for the CLI renderer
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, AsyncGenerator, Optional

from axiom.core.agent.state import AgentState, AgentMode, AgentStatus
from axiom.core.agent.nodes.planner import generate_plan, generate_react_thought
from axiom.core.agent.nodes.executor import execute_tool
from axiom.core.agent.nodes.observer import observe_progress
from axiom.core.agent.nodes.synthesizer import synthesize_answer
from axiom.core.agent.auto_selector import auto_select_mode as _keyword_select_mode
from axiom.core.agent.tracer import AgentTracer
from axiom.core.agent.nodes.learner import extract_learnings, should_learn

logger = logging.getLogger(__name__)

MAX_PLAN_ITERATIONS = 50
MAX_REACT_ITERATIONS = 30
MAX_REPLANS = 8
MAX_CONSECUTIVE_FAILURES = 10


class EventType(str, Enum):
    """Event types yielded by the agent graph."""
    THINKING = "thinking"
    PLAN_CREATED = "plan_created"
    STEP_START = "step_start"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    OBSERVATION = "observation"
    REPLAN = "replan"
    ANSWER = "answer"
    ERROR = "error"
    TRACE = "trace"
    COUNCIL_START = "council_start"
    COUNCIL_MEMBER = "council_member"
    COUNCIL_SYNTHESIS = "council_synthesis"
    LEARNING = "learning"


@dataclass
class AgentEvent:
    """Event yielded by the agent during execution."""
    type: EventType
    data: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


def _select_mode(user_message: str) -> AgentMode:
    """Map the keyword-based auto_select_mode string to an AgentMode enum.

    Uses the full auto_selector module which supports PLAN, REACT,
    COUNCIL, and chat modes with 60+ keyword patterns.
    """
    mode_str = _keyword_select_mode(
        user_message,
        available_modes=["plan", "react", "council", "chat"],
    )
    _map = {
        "plan": AgentMode.PLAN,
        "react": AgentMode.REACT,
        "council": AgentMode.COUNCIL,
        "chat": AgentMode.REACT,  # chat falls back to lightweight REACT
    }
    return _map.get(mode_str, AgentMode.REACT)


async def run_agent(
    router: Any,
    registry: Any,
    messages: list[dict[str, str]],
    mode: Optional[AgentMode] = None,
    renderer: Any = None,
    tracer: Optional[AgentTracer] = None,
    skills_dir: Optional[Path] = None,
) -> AsyncGenerator[AgentEvent, None]:
    """Run the agent graph and yield events.

    This is the main entry point for agent execution.

    Args:
        router: UniversalRouter for LLM calls
        registry: ToolRegistry for tool execution
        messages: Conversation messages
        mode: Optional forced mode (auto-selects if None)
        renderer: Optional renderer for real-time display
        tracer: Optional AgentTracer for observability logging
        skills_dir: Optional path for saving learned skills

    Yields:
        AgentEvent objects for each step of execution
    """
    # Create tracer if not provided
    if tracer is None:
        tracer = AgentTracer(enabled=True)

    user_message = ""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            user_message = msg.get("content", "")
            break

    # Auto-select mode if not specified
    if mode is None:
        mode = _select_mode(user_message)
        tracer.log_mode_selection(mode.value, f"auto-selected for: {user_message[:80]}")
    else:
        tracer.log_mode_selection(mode.value, "explicitly set by user/app")

    yield AgentEvent(type=EventType.THINKING, data={"mode": mode.value})

    tool_names = registry.list_names() if registry else []
    all_tool_results: list[dict[str, Any]] = []

    # ── Dispatch to the appropriate mode runner ────────────────
    if mode == AgentMode.COUNCIL:
        async for event in _run_council_mode(
            router, registry, messages, tool_names, tracer
        ):
            yield event
        # Council mode doesn't produce tool_results for learner
        return

    elif mode == AgentMode.PLAN:
        async for event in _run_plan_mode(
            router, registry, messages, tool_names, tracer
        ):
            # Collect tool results for learner
            if event.type == EventType.TOOL_RESULT:
                all_tool_results.append(event.data)
            yield event

    elif mode == AgentMode.REACT:
        async for event in _run_react_mode(
            router, registry, messages, tool_names, tracer
        ):
            if event.type == EventType.TOOL_RESULT:
                all_tool_results.append(event.data)
            yield event

    else:
        # Fallback to REACT
        async for event in _run_react_mode(
            router, registry, messages, tool_names, tracer
        ):
            if event.type == EventType.TOOL_RESULT:
                all_tool_results.append(event.data)
            yield event

    # ── Post-execution: Self-improvement via Learner ───────────
    try:
        if await should_learn(all_tool_results):
            tracer.log("LEARN", "Learner triggered — extracting patterns from successful run")
            # Extract the final answer from the last ANSWER event
            final_answer = ""
            # We'll extract learnings in background to not block the user
            save_dir = skills_dir or Path("memory/skills")
            learning = await extract_learnings(
                router=router,
                task=user_message,
                tool_results=all_tool_results,
                final_answer=user_message,  # placeholder — real answer set by mode
                save_dir=save_dir,
            )
            if learning:
                yield AgentEvent(
                    type=EventType.LEARNING,
                    data={"skill": learning[:200], "saved": bool(skills_dir)},
                )
                tracer.log("LEARN", "Learner: skill extracted and saved")
    except Exception as exc:
        logger.debug("Learner post-execution failed: %s", exc)


async def _run_plan_mode(
    router: Any,
    registry: Any,
    messages: list[dict[str, str]],
    tool_names: list[str],
    tracer: Optional[AgentTracer] = None,
) -> AsyncGenerator[AgentEvent, None]:
    """Execute in PLAN mode: generate plan -> execute steps -> observe -> replan if needed."""

    all_tool_results: list[dict[str, Any]] = []
    replan_count = 0
    consecutive_failures = 0

    # Generate initial plan
    yield AgentEvent(type=EventType.THINKING, data={"phase": "planning"})
    t0 = time.time()
    plan = await generate_plan(router, messages, tool_names)
    if tracer:
        tracer.log_model_call(
            model="planner",
            tokens_in=0,
            tokens_out=0,
            duration_ms=int((time.time() - t0) * 1000),
        )
        tracer.log("PLAN", f"Plan generated: {len(plan)} steps")
    yield AgentEvent(type=EventType.PLAN_CREATED, data={"plan": plan})

    iteration = 0
    while iteration < MAX_PLAN_ITERATIONS:
        for step_idx, step in enumerate(plan):
            iteration += 1
            if iteration > MAX_PLAN_ITERATIONS:
                break

            action = step.get("action", "think")
            args = step.get("args", {})
            desc = step.get("description", "")

            yield AgentEvent(
                type=EventType.STEP_START,
                data={"step": step_idx + 1, "total": len(plan), "description": desc},
            )

            # Execute the step
            if action not in ("think", "finish", "reasoning"):
                yield AgentEvent(
                    type=EventType.TOOL_CALL,
                    data={"tool": action, "args": args},
                )

                t1 = time.time()
                result = await execute_tool(registry, action, args)
                dur = int((time.time() - t1) * 1000)
                all_tool_results.append(result)

                if tracer:
                    tracer.log_tool_call(
                        tool_name=action,
                        args=args,
                        success=result["success"],
                        duration_ms=dur,
                    )

                yield AgentEvent(
                    type=EventType.TOOL_RESULT,
                    data={
                        "tool": action,
                        "success": result["success"],
                        "result": result["result"][:2000],
                        "duration_ms": result["duration_ms"],
                    },
                )

                if not result["success"]:
                    consecutive_failures += 1
                    # ── Retry hint injection ────────────────────────
                    # Tell the LLM to try a DIFFERENT tool or args
                    # Log retry hint for trace only — don't pollute permanent messages
                    hint = (
                        f"[RETRY HINT] Tool '{action}' failed "
                        f"({consecutive_failures}/{MAX_CONSECUTIVE_FAILURES}). "
                        f"Try a DIFFERENT tool or arguments."
                    )
                    if tracer:
                        tracer.log("RETRY", hint)
                else:
                    consecutive_failures = 0

                # Check failure threshold
                if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                    if tracer:
                        tracer.log("ERROR", f"Aborting: {consecutive_failures} consecutive failures")
                    yield AgentEvent(
                        type=EventType.ERROR,
                        data={"message": f"Too many consecutive failures ({consecutive_failures}). Tried {iteration} iterations."},
                    )
                    break

        # Observe progress after executing the plan
        observation = await observe_progress(
            router, messages, all_tool_results, iteration, MAX_PLAN_ITERATIONS
        )

        if tracer:
            tracer.log_observation(
                status=observation["status"],
                confidence=observation.get("confidence", 0),
                reason=observation.get("reason", ""),
            )

        yield AgentEvent(type=EventType.OBSERVATION, data=observation)

        if observation["status"] == "complete":
            break
        elif observation["status"] == "replan" and replan_count < MAX_REPLANS:
            replan_count += 1
            yield AgentEvent(
                type=EventType.REPLAN,
                data={"attempt": replan_count, "reason": observation["reason"]},
            )

            # Generate new plan with context of what failed
            context_msg = {
                "role": "user",
                "content": f"Previous plan had issues: {observation['reason']}. "
                           f"Results so far: {[r['result'][:200] for r in all_tool_results[-3:]]}. "
                           f"Create an improved plan.",
            }
            plan = await generate_plan(router, messages + [context_msg], tool_names)
            yield AgentEvent(type=EventType.PLAN_CREATED, data={"plan": plan})
            continue
        else:
            break

    # Synthesize final answer
    yield AgentEvent(type=EventType.THINKING, data={"phase": "synthesizing"})
    answer = await synthesize_answer(router, messages, all_tool_results)
    yield AgentEvent(type=EventType.ANSWER, data={"answer": answer})


async def _run_react_mode(
    router: Any,
    registry: Any,
    messages: list[dict[str, str]],
    tool_names: list[str],
    tracer: Optional[AgentTracer] = None,
) -> AsyncGenerator[AgentEvent, None]:
    """Execute in REACT mode: interleaved Thought -> Action -> Observation loop."""

    react_trace: list[dict[str, Any]] = []
    all_tool_results: list[dict[str, Any]] = []
    consecutive_failures = 0

    for iteration in range(MAX_REACT_ITERATIONS):
        # Generate next thought + action
        yield AgentEvent(type=EventType.THINKING, data={"iteration": iteration + 1})

        thought = await generate_react_thought(router, messages, react_trace, tool_names)

        action = thought.get("action", "finish")
        args = thought.get("args", {})
        thought_text = thought.get("thought", "")

        yield AgentEvent(
            type=EventType.TRACE,
            data={"thought": thought_text, "action": action},
        )

        # Check for finish
        if action == "finish":
            answer = args.get("answer", thought_text)
            yield AgentEvent(type=EventType.ANSWER, data={"answer": answer})
            return

        # Execute the action
        yield AgentEvent(type=EventType.TOOL_CALL, data={"tool": action, "args": args})

        t1 = time.time()
        result = await execute_tool(registry, action, args)
        dur = int((time.time() - t1) * 1000)
        all_tool_results.append(result)

        if tracer:
            tracer.log_tool_call(
                tool_name=action, args=args,
                success=result["success"], duration_ms=dur,
            )

        yield AgentEvent(
            type=EventType.TOOL_RESULT,
            data={
                "tool": action,
                "success": result["success"],
                "result": result["result"][:2000],
                "duration_ms": result["duration_ms"],
            },
        )

        # Update trace
        react_trace.append({
            "thought": thought_text,
            "action": action,
            "args": args,
            "observation": result["result"][:1000],
            "success": result["success"],
        })

        if not result["success"]:
            consecutive_failures += 1
            # Log retry hint for trace only — don't pollute permanent messages
            hint = (
                f"[RETRY HINT] Tool '{action}' failed "
                f"({consecutive_failures}/{MAX_CONSECUTIVE_FAILURES}). "
                f"Try a DIFFERENT tool or arguments."
            )
            if tracer:
                tracer.log("RETRY", hint)
        else:
            consecutive_failures = 0

        if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
            yield AgentEvent(
                type=EventType.ERROR,
                data={"message": f"Too many consecutive failures ({consecutive_failures}). Tried {iteration + 1} iterations."},
            )
            break

    # If we exhausted iterations, synthesize what we have
    answer = await synthesize_answer(router, messages, all_tool_results)
    yield AgentEvent(type=EventType.ANSWER, data={"answer": answer})


async def _run_council_mode(
    router: Any,
    registry: Any,
    messages: list[dict[str, str]],
    tool_names: list[str],
    tracer: Optional[AgentTracer] = None,
) -> AsyncGenerator[AgentEvent, None]:
    """Execute in COUNCIL mode: multi-LLM consensus deliberation.

    Phase 1: Query N models in parallel
    Phase 2: Circular peer review
    Phase 3: Chairman synthesizes best answer

    Falls back to PLAN mode if the task requires tool execution.
    """
    try:
        from axiom.core.llm.council import LLMCouncil
    except ImportError:
        logger.warning("LLMCouncil not available, falling back to REACT")
        async for event in _run_react_mode(router, registry, messages, tool_names, tracer):
            yield event
        return

    yield AgentEvent(
        type=EventType.COUNCIL_START,
        data={"phase": "deliberation"},
    )

    if tracer:
        tracer.log("COUNCIL", "COUNCIL mode: starting multi-LLM deliberation")

    council = LLMCouncil(router)

    try:
        t0 = time.time()
        result = await council.deliberate(messages=messages)
        total_ms = int((time.time() - t0) * 1000)

        if tracer:
            tracer.log(
                "COUNCIL",
                f"Council deliberation complete: {len(result.members)} members, "
                f"consensus={result.consensus_score:.2f}, {total_ms}ms",
            )

        # Yield individual member responses for trace mode
        for member in result.members:
            yield AgentEvent(
                type=EventType.COUNCIL_MEMBER,
                data={
                    "model": member.model,
                    "score": member.score,
                    "latency_ms": member.latency_ms,
                    "error": member.error,
                    "response_preview": (member.response or "")[:300],
                },
            )

        # Yield the synthesized answer
        yield AgentEvent(
            type=EventType.COUNCIL_SYNTHESIS,
            data={
                "consensus_score": result.consensus_score,
                "member_count": len(result.members),
                "chairman": result.chairman_model,
                "total_time_ms": result.total_time_ms,
            },
        )

        # Final answer is the council synthesis
        answer = result.synthesis or "Council could not reach a synthesis."
        yield AgentEvent(type=EventType.ANSWER, data={"answer": answer})

    except Exception as exc:
        logger.error("Council deliberation failed: %s", exc)
        if tracer:
            tracer.log("ERROR", f"Council failed: {exc}, falling back to REACT")

        yield AgentEvent(
            type=EventType.ERROR,
            data={"message": f"Council failed ({exc}), falling back to REACT mode"},
        )

        # Fallback to REACT
        async for event in _run_react_mode(router, registry, messages, tool_names, tracer):
            yield event
