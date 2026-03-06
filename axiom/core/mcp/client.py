"""MCP client -- connect to Model Context Protocol servers.

Manages MCP server lifecycle (start, communicate, stop) using
JSON-RPC 2.0 over stdio transport. Each server runs as a child
subprocess with stdin/stdout pipes.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class MCPTool:
    """A tool exposed by an MCP server."""
    name: str
    description: str = ""
    input_schema: dict[str, Any] = field(default_factory=dict)
    server_name: str = ""


@dataclass
class MCPServer:
    """A connected MCP server instance."""
    name: str
    command: str
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    process: Optional[asyncio.subprocess.Process] = field(default=None, repr=False)
    tools: list[MCPTool] = field(default_factory=list)
    _request_id: int = field(default=0, repr=False)
    connected: bool = False

    def next_id(self) -> int:
        self._request_id += 1
        return self._request_id


class MCPClient:
    """Manages connections to multiple MCP servers.

    Uses JSON-RPC 2.0 over stdio transport (same protocol as Claude Code).
    Each server runs as a subprocess with stdin/stdout communication.
    """

    def __init__(self) -> None:
        self._servers: dict[str, MCPServer] = {}

    async def connect(
        self,
        name: str,
        command: str,
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
    ) -> MCPServer:
        """Start an MCP server and perform initialization handshake.

        Args:
            name: Unique name for this server connection.
            command: Executable to run (e.g., "npx", "python").
            args: Command arguments (e.g., ["-y", "@modelcontextprotocol/server-filesystem"]).
            env: Additional environment variables.

        Returns:
            Connected MCPServer instance.
        """
        if name in self._servers and self._servers[name].connected:
            logger.info("Server '%s' already connected", name)
            return self._servers[name]

        import os
        full_env = {**os.environ, **(env or {})}

        try:
            process = await asyncio.create_subprocess_exec(
                command,
                *(args or []),
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=full_env,
            )
        except FileNotFoundError:
            raise ConnectionError(f"Command not found: {command}")
        except Exception as exc:
            raise ConnectionError(f"Failed to start MCP server '{name}': {exc}")

        server = MCPServer(
            name=name,
            command=command,
            args=args or [],
            env=env or {},
            process=process,
        )
        self._servers[name] = server

        # Perform MCP initialize handshake
        try:
            init_result = await self._send_request(
                server,
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
            await self._send_notification(server, "notifications/initialized", {})

            server.connected = True
            logger.info("Connected to MCP server '%s'", name)

            # Discover available tools
            await self._discover_tools(server)

            return server

        except Exception as exc:
            await self.disconnect(name)
            raise ConnectionError(f"MCP handshake failed for '{name}': {exc}")

    async def disconnect(self, name: str) -> None:
        """Disconnect from an MCP server."""
        server = self._servers.get(name)
        if not server:
            return

        if server.process and server.process.returncode is None:
            try:
                server.process.terminate()
                await asyncio.wait_for(server.process.wait(), timeout=5.0)
            except (asyncio.TimeoutError, ProcessLookupError):
                try:
                    server.process.kill()
                except ProcessLookupError:
                    pass

        server.connected = False
        del self._servers[name]
        logger.info("Disconnected from MCP server '%s'", name)

    async def disconnect_all(self) -> None:
        """Disconnect from all servers."""
        names = list(self._servers.keys())
        for name in names:
            await self.disconnect(name)

    async def call_tool(
        self,
        server_name: str,
        tool_name: str,
        arguments: dict[str, Any] | None = None,
    ) -> Any:
        """Call a tool on an MCP server.

        Args:
            server_name: Name of the connected server.
            tool_name: Name of the tool to call.
            arguments: Tool arguments.

        Returns:
            Tool result content.
        """
        server = self._servers.get(server_name)
        if not server or not server.connected:
            raise ConnectionError(f"Server '{server_name}' not connected")

        result = await self._send_request(
            server,
            "tools/call",
            {
                "name": tool_name,
                "arguments": arguments or {},
            },
        )

        # Extract text content from result
        content = result.get("content", [])
        if isinstance(content, list):
            texts = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    texts.append(item.get("text", ""))
            return "\n".join(texts) if texts else str(content)
        return str(content)

    def get_server(self, name: str) -> Optional[MCPServer]:
        """Get a connected server by name."""
        return self._servers.get(name)

    def list_servers(self) -> list[MCPServer]:
        """List all connected servers."""
        return list(self._servers.values())

    def get_all_tools(self) -> list[MCPTool]:
        """Get all tools across all connected servers."""
        tools = []
        for server in self._servers.values():
            tools.extend(server.tools)
        return tools

    # -- Internal Protocol Methods ------------------------------------

    async def _discover_tools(self, server: MCPServer) -> None:
        """Discover tools available on an MCP server."""
        try:
            result = await self._send_request(server, "tools/list", {})
            tools_data = result.get("tools", [])

            server.tools = [
                MCPTool(
                    name=t.get("name", ""),
                    description=t.get("description", ""),
                    input_schema=t.get("inputSchema", {}),
                    server_name=server.name,
                )
                for t in tools_data
                if t.get("name")
            ]
            logger.info(
                "Discovered %d tools on server '%s'",
                len(server.tools), server.name
            )
        except Exception as exc:
            logger.warning("Tool discovery failed for '%s': %s", server.name, exc)

    async def _send_request(
        self,
        server: MCPServer,
        method: str,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        """Send a JSON-RPC 2.0 request and wait for response."""
        if not server.process or server.process.returncode is not None:
            raise ConnectionError(f"Server '{server.name}' process not running")

        request_id = server.next_id()
        request = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
            "params": params,
        }

        request_line = json.dumps(request) + "\n"
        server.process.stdin.write(request_line.encode("utf-8"))
        await server.process.stdin.drain()

        # Read response (may need to skip notifications)
        while True:
            try:
                line = await asyncio.wait_for(
                    server.process.stdout.readline(),
                    timeout=30.0,
                )
            except asyncio.TimeoutError:
                raise TimeoutError(
                    f"MCP server '{server.name}' did not respond within 30s"
                )

            if not line:
                raise ConnectionError(
                    f"MCP server '{server.name}' closed connection"
                )

            try:
                response = json.loads(line.decode("utf-8").strip())
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue

            # Skip notifications (no "id" field)
            if "id" not in response:
                continue

            if response.get("id") != request_id:
                continue

            if "error" in response:
                error = response["error"]
                raise RuntimeError(
                    f"MCP error ({error.get('code', '?')}): "
                    f"{error.get('message', 'Unknown error')}"
                )

            return response.get("result", {})

    async def _send_notification(
        self,
        server: MCPServer,
        method: str,
        params: dict[str, Any],
    ) -> None:
        """Send a JSON-RPC 2.0 notification (no response expected)."""
        if not server.process or server.process.returncode is not None:
            return

        notification = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
        }

        line = json.dumps(notification) + "\n"
        server.process.stdin.write(line.encode("utf-8"))
        await server.process.stdin.drain()
