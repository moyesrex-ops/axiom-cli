"""Sub-agent spawning tool — create and run specialized sub-agents.

Enables the main agent to delegate tasks to sub-agents that run
in parallel with their own tool subsets and potentially cheaper models.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Optional

from axiom.core.tools.base import AxiomTool, ToolError

logger = logging.getLogger(__name__)

# Track active sub-agents
_active_agents: dict[str, dict[str, Any]] = {}


class SpawnAgentTool(AxiomTool):
    """Create and run specialized sub-agents for parallel task execution.

    Sub-agents get their own execution context with a subset of tools
    and can optionally use cheaper/faster models for simple tasks.
    """

    name = "spawn_agent"
    description = (
        "Create a specialized sub-agent to handle a specific task. "
        "Sub-agents run in parallel with their own tool access. "
        "Useful for: parallel research, concurrent file operations, "
        "or delegating subtasks while continuing main work."
    )
    risk_level = "medium"
    parameters_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "description": "Agent action: spawn (create new), status (check), list (all agents), result (get output)",
                "enum": ["spawn", "status", "list", "result"],
            },
            "name": {
                "type": "string",
                "description": "Name for the sub-agent (e.g. 'researcher', 'coder')",
            },
            "task": {
                "type": "string",
                "description": "Task description for the sub-agent",
            },
            "tools": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Tool names the sub-agent can use (defaults to all)",
            },
            "model": {
                "type": "string",
                "description": "LLM model override (e.g. 'groq' for speed, 'opus' for quality)",
            },
        },
        "required": ["action"],
    }

    def __init__(self, router: Any = None, registry: Any = None):
        self._router = router
        self._registry = registry

    async def execute(self, **kwargs: Any) -> str:
        action = kwargs.get("action", "")

        if action == "spawn":
            return await self._spawn(kwargs)
        elif action == "status":
            return self._status(kwargs)
        elif action == "list":
            return self._list_agents()
        elif action == "result":
            return self._get_result(kwargs)
        else:
            raise ToolError(f"Unknown agent action: {action}", tool_name=self.name)

    async def _spawn(self, kwargs: dict) -> str:
        """Spawn a new sub-agent."""
        name = kwargs.get("name", f"agent_{len(_active_agents) + 1}")
        task = kwargs.get("task", "")
        model_override = kwargs.get("model")

        if not task:
            raise ToolError("Task description required to spawn agent", tool_name=self.name)

        if self._router is None or self._registry is None:
            raise ToolError("Router and registry required for agent spawning", tool_name=self.name)

        # Create the agent task as a background coroutine
        agent_state: dict[str, Any] = {
            "name": name,
            "task": task,
            "status": "running",
            "started_at": time.time(),
            "result": None,
            "error": None,
        }

        async def _run_agent():
            try:
                from axiom.core.agent.graph import run_agent, AgentEvent, EventType

                messages = [
                    {"role": "system", "content": f"You are a sub-agent named '{name}'. Complete this task efficiently."},
                    {"role": "user", "content": task},
                ]

                final_answer = ""
                async for event in run_agent(
                    router=self._router,
                    registry=self._registry,
                    messages=messages,
                ):
                    if event.type == EventType.ANSWER:
                        final_answer = event.data.get("answer", "")
                    elif event.type == EventType.ERROR:
                        agent_state["error"] = event.data.get("message", "Unknown error")

                agent_state["result"] = final_answer or "Task completed (no explicit answer)"
                agent_state["status"] = "completed"

            except Exception as e:
                agent_state["error"] = str(e)
                agent_state["status"] = "failed"
            finally:
                agent_state["finished_at"] = time.time()

        # Launch as background task
        asyncio.create_task(_run_agent())
        _active_agents[name] = agent_state

        return (
            f"Sub-agent '{name}' spawned.\n"
            f"Task: {task[:200]}\n"
            f"Use action='status' with name='{name}' to check progress,\n"
            f"or action='result' to get the output when done."
        )

    def _status(self, kwargs: dict) -> str:
        """Check status of a sub-agent."""
        name = kwargs.get("name", "")
        if not name:
            return self._list_agents()

        agent = _active_agents.get(name)
        if not agent:
            return f"No agent found with name: {name}"

        elapsed = time.time() - agent["started_at"]
        status = agent["status"]
        lines = [
            f"Agent: {name}",
            f"Status: {status}",
            f"Task: {agent['task'][:100]}",
            f"Elapsed: {elapsed:.1f}s",
        ]

        if agent.get("error"):
            lines.append(f"Error: {agent['error']}")
        if agent.get("result") and status == "completed":
            lines.append(f"Result preview: {agent['result'][:200]}...")

        return "\n".join(lines)

    def _list_agents(self) -> str:
        """List all active and completed sub-agents."""
        if not _active_agents:
            return "No sub-agents have been spawned."

        lines = [f"Sub-agents ({len(_active_agents)}):"]
        for name, agent in _active_agents.items():
            status = agent["status"]
            elapsed = time.time() - agent["started_at"]
            icon = {"running": "🔄", "completed": "✅", "failed": "❌"}.get(status, "❓")
            lines.append(f"  {icon} {name}: {status} ({elapsed:.1f}s) - {agent['task'][:60]}")

        return "\n".join(lines)

    def _get_result(self, kwargs: dict) -> str:
        """Get the result from a completed sub-agent."""
        name = kwargs.get("name", "")
        if not name:
            # Return the first completed agent's result
            for n, a in _active_agents.items():
                if a["status"] == "completed":
                    name = n
                    break
            if not name:
                return "No completed agents. Use action='list' to check status."

        agent = _active_agents.get(name)
        if not agent:
            return f"No agent found with name: {name}"

        if agent["status"] == "running":
            return f"Agent '{name}' is still running. Check back later."

        if agent["status"] == "failed":
            return f"Agent '{name}' failed: {agent.get('error', 'Unknown error')}"

        return agent.get("result", "No result available")
