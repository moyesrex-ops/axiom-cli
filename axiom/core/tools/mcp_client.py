"""MCP client tool — connect to any MCP server and use its tools.

Uses the Model Context Protocol (JSON-RPC 2.0 over stdio) to connect to
external tool servers.  Each MCP server runs as a subprocess communicating
via stdin/stdout.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any, Optional

from axiom.core.tools.base import AxiomTool, ToolError

logger = logging.getLogger(__name__)

# Active MCP server connections
_mcp_connections: dict[str, dict[str, Any]] = {}

# Default config path
_MCP_CONFIG = Path.home() / ".axiom" / "mcp_servers" / "servers.json"

# JSON-RPC 2.0 request counter
_request_id = 0


def _next_id() -> int:
    global _request_id
    _request_id += 1
    return _request_id


class MCPClientTool(AxiomTool):
    """Connect to MCP servers and invoke their tools.

    Supports connecting to any MCP-compatible server via stdio transport,
    listing available tools, and calling them.
    """

    name = "mcp_client"
    description = (
        "Connect to MCP (Model Context Protocol) servers and use their tools. "
        "Actions: connect (start server), list_tools (discover), call_tool (invoke), "
        "disconnect (stop server), list_servers (show config)."
    )
    risk_level = "medium"
    parameters_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "description": "MCP action: connect, list_tools, call_tool, disconnect, list_servers",
                "enum": ["connect", "list_tools", "call_tool", "disconnect", "list_servers"],
            },
            "server_name": {
                "type": "string",
                "description": "Name of the MCP server",
            },
            "command": {
                "type": "string",
                "description": "Command to start the MCP server (for connect)",
            },
            "args": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Arguments for the server command (for connect)",
            },
            "tool_name": {
                "type": "string",
                "description": "Name of the MCP tool to call (for call_tool)",
            },
            "tool_args": {
                "type": "object",
                "description": "Arguments to pass to the MCP tool (for call_tool)",
            },
        },
        "required": ["action"],
    }

    async def execute(self, **kwargs: Any) -> str:
        action = kwargs.get("action", "")

        if action == "connect":
            return await self._connect(kwargs)
        elif action == "list_tools":
            return await self._list_tools(kwargs)
        elif action == "call_tool":
            return await self._call_tool(kwargs)
        elif action == "disconnect":
            return await self._disconnect(kwargs)
        elif action == "list_servers":
            return self._list_servers()
        else:
            raise ToolError(f"Unknown MCP action: {action}", tool_name=self.name)

    async def _connect(self, kwargs: dict) -> str:
        """Start an MCP server subprocess and initialize connection."""
        server_name = kwargs.get("server_name", "")
        command = kwargs.get("command", "")
        args = kwargs.get("args", [])

        if not server_name:
            raise ToolError("server_name required", tool_name=self.name)

        if server_name in _mcp_connections:
            return f"Server '{server_name}' is already connected."

        # Try loading from config if command not provided
        if not command:
            config = self._load_server_config(server_name)
            if config:
                command = config.get("command", "")
                args = config.get("args", [])
            else:
                raise ToolError(
                    f"No command provided and server '{server_name}' not found in config",
                    tool_name=self.name,
                )

        try:
            # Start the server process
            proc = await asyncio.create_subprocess_exec(
                command,
                *args,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            conn: dict[str, Any] = {
                "process": proc,
                "command": command,
                "args": args,
                "tools": {},
            }
            _mcp_connections[server_name] = conn

            # Send initialize request
            init_result = await self._send_request(
                server_name,
                "initialize",
                {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {
                        "name": "axiom-cli",
                        "version": "1.0.0",
                    },
                },
            )

            # Send initialized notification
            await self._send_notification(server_name, "notifications/initialized", {})

            return (
                f"Connected to MCP server '{server_name}'.\n"
                f"Server info: {json.dumps(init_result.get('serverInfo', {}), indent=2)}\n"
                f"Use action='list_tools' to discover available tools."
            )

        except FileNotFoundError:
            raise ToolError(
                f"Command '{command}' not found. Make sure it's installed.",
                tool_name=self.name,
            )
        except Exception as e:
            # Clean up on failure
            _mcp_connections.pop(server_name, None)
            raise ToolError(
                f"Failed to connect to '{server_name}': {e}",
                tool_name=self.name,
            )

    async def _list_tools(self, kwargs: dict) -> str:
        """List available tools from a connected MCP server."""
        server_name = kwargs.get("server_name", "")
        if not server_name:
            if _mcp_connections:
                server_name = next(iter(_mcp_connections))
            else:
                return "No MCP servers connected. Use action='connect' first."

        if server_name not in _mcp_connections:
            return f"Server '{server_name}' not connected."

        result = await self._send_request(server_name, "tools/list", {})
        tools = result.get("tools", [])

        # Cache tool info
        conn = _mcp_connections[server_name]
        for t in tools:
            conn["tools"][t["name"]] = t

        if not tools:
            return f"Server '{server_name}' has no tools."

        lines = [f"MCP tools from '{server_name}' ({len(tools)}):"]
        for t in tools:
            name = t.get("name", "?")
            desc = t.get("description", "No description")[:80]
            lines.append(f"  * {name}: {desc}")

        return "\n".join(lines)

    async def _call_tool(self, kwargs: dict) -> str:
        """Call a tool on a connected MCP server."""
        server_name = kwargs.get("server_name", "")
        tool_name = kwargs.get("tool_name", "")
        tool_args = kwargs.get("tool_args", {})

        if not tool_name:
            raise ToolError("tool_name required for call_tool", tool_name=self.name)

        # Find which server has this tool
        if not server_name:
            for sname, conn in _mcp_connections.items():
                if tool_name in conn.get("tools", {}):
                    server_name = sname
                    break
            if not server_name:
                raise ToolError(
                    f"Tool '{tool_name}' not found on any connected server",
                    tool_name=self.name,
                )

        if server_name not in _mcp_connections:
            return f"Server '{server_name}' not connected."

        result = await self._send_request(
            server_name,
            "tools/call",
            {"name": tool_name, "arguments": tool_args},
        )

        # Format result
        content = result.get("content", [])
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict):
                    if item.get("type") == "text":
                        parts.append(item.get("text", ""))
                    else:
                        parts.append(json.dumps(item))
                else:
                    parts.append(str(item))
            return "\n".join(parts) or "(empty result)"
        return str(content)

    async def _disconnect(self, kwargs: dict) -> str:
        """Disconnect from an MCP server."""
        server_name = kwargs.get("server_name", "")
        if not server_name:
            raise ToolError("server_name required", tool_name=self.name)

        conn = _mcp_connections.pop(server_name, None)
        if not conn:
            return f"Server '{server_name}' was not connected."

        proc = conn.get("process")
        if proc and proc.returncode is None:
            try:
                proc.terminate()
                await asyncio.wait_for(proc.wait(), timeout=5.0)
            except (asyncio.TimeoutError, ProcessLookupError):
                proc.kill()

        return f"Disconnected from MCP server '{server_name}'."

    def _list_servers(self) -> str:
        """List configured and connected MCP servers."""
        lines = []

        # Connected servers
        if _mcp_connections:
            lines.append(f"Connected ({len(_mcp_connections)}):")
            for name, conn in _mcp_connections.items():
                tool_count = len(conn.get("tools", {}))
                lines.append(f"  * {name}: {tool_count} tools available")
        else:
            lines.append("No servers currently connected.")

        # Configured servers from config file
        configs = self._load_all_configs()
        if configs:
            lines.append(f"\nConfigured ({len(configs)}):")
            for name, cfg in configs.items():
                cmd = cfg.get("command", "?")
                connected = "connected" if name in _mcp_connections else "disconnected"
                lines.append(f"  * {name}: {cmd} [{connected}]")

        return "\n".join(lines)

    # ── JSON-RPC helpers ──────────────────────────────────────────

    async def _send_request(
        self, server_name: str, method: str, params: dict
    ) -> dict:
        """Send a JSON-RPC 2.0 request and wait for response."""
        conn = _mcp_connections.get(server_name)
        if not conn:
            raise ToolError(f"Server '{server_name}' not connected", tool_name=self.name)

        proc = conn["process"]
        if proc.stdin is None or proc.stdout is None:
            raise ToolError("Server process has no stdin/stdout", tool_name=self.name)

        req_id = _next_id()
        request = {
            "jsonrpc": "2.0",
            "id": req_id,
            "method": method,
            "params": params,
        }

        msg = json.dumps(request) + "\n"
        proc.stdin.write(msg.encode("utf-8"))
        await proc.stdin.drain()

        # Read response line
        try:
            line = await asyncio.wait_for(
                proc.stdout.readline(), timeout=30.0
            )
        except asyncio.TimeoutError:
            raise ToolError(
                f"Timeout waiting for response from '{server_name}'",
                tool_name=self.name,
            )

        if not line:
            raise ToolError(
                f"Server '{server_name}' closed connection",
                tool_name=self.name,
            )

        try:
            response = json.loads(line.decode("utf-8"))
        except json.JSONDecodeError:
            raise ToolError(
                f"Invalid JSON from server: {line[:200]}",
                tool_name=self.name,
            )

        if "error" in response:
            error = response["error"]
            raise ToolError(
                f"MCP error: {error.get('message', error)}",
                tool_name=self.name,
            )

        return response.get("result", {})

    async def _send_notification(
        self, server_name: str, method: str, params: dict
    ) -> None:
        """Send a JSON-RPC 2.0 notification (no response expected)."""
        conn = _mcp_connections.get(server_name)
        if not conn:
            return

        proc = conn["process"]
        if proc.stdin is None:
            return

        notification = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
        }

        msg = json.dumps(notification) + "\n"
        proc.stdin.write(msg.encode("utf-8"))
        await proc.stdin.drain()

    # ── Config helpers ────────────────────────────────────────────

    def _load_server_config(self, server_name: str) -> Optional[dict]:
        """Load a single server config by name."""
        configs = self._load_all_configs()
        return configs.get(server_name)

    def _load_all_configs(self) -> dict[str, dict]:
        """Load all server configs from disk."""
        if not _MCP_CONFIG.exists():
            return {}
        try:
            data = json.loads(_MCP_CONFIG.read_text(encoding="utf-8"))
            return data.get("mcpServers", data.get("servers", {}))
        except Exception:
            return {}
