"""MCP bridge -- integrate MCP server tools into the Axiom tool registry.

Wraps MCP tools as AxiomTool instances so they appear alongside
native tools in the agent's tool list.
"""

from __future__ import annotations

import logging
from typing import Any

from axiom.core.mcp.client import MCPClient, MCPTool
from axiom.core.tools.base import AxiomTool, ToolError

logger = logging.getLogger(__name__)


class MCPBridgeTool(AxiomTool):
    """An AxiomTool wrapper around an MCP server tool.

    Delegates execution to the MCP client, making MCP tools
    appear as native Axiom tools to the agent.
    """

    risk_level = "medium"  # MCP tools are external, default to medium risk

    def __init__(
        self,
        mcp_tool: MCPTool,
        mcp_client: MCPClient,
    ) -> None:
        self._mcp_tool = mcp_tool
        self._mcp_client = mcp_client

        # Set AxiomTool attributes from MCP tool metadata
        self.name = f"mcp_{mcp_tool.server_name}_{mcp_tool.name}"
        self.description = (
            mcp_tool.description
            or f"MCP tool '{mcp_tool.name}' from server '{mcp_tool.server_name}'"
        )

        # Convert MCP input schema to AxiomTool parameters_schema
        self.parameters_schema = mcp_tool.input_schema or {
            "type": "object",
            "properties": {},
        }

    async def execute(self, **kwargs: Any) -> str:
        """Execute the MCP tool via the client."""
        try:
            result = await self._mcp_client.call_tool(
                server_name=self._mcp_tool.server_name,
                tool_name=self._mcp_tool.name,
                arguments=kwargs,
            )
            return str(result)
        except ConnectionError as exc:
            raise ToolError(
                f"MCP server '{self._mcp_tool.server_name}' disconnected: {exc}",
                tool_name=self.name,
            )
        except Exception as exc:
            raise ToolError(
                f"MCP tool '{self._mcp_tool.name}' failed: {exc}",
                tool_name=self.name,
            )


class MCPBridge:
    """Bridge between MCP servers and the Axiom tool registry.

    Manages the lifecycle of MCP connections and keeps the
    tool registry in sync with available MCP tools.
    """

    def __init__(self, mcp_client: MCPClient) -> None:
        self._client = mcp_client
        self._bridged_tools: dict[str, MCPBridgeTool] = {}

    def bridge_server(self, server_name: str) -> list[MCPBridgeTool]:
        """Create AxiomTool wrappers for all tools on a server.

        Returns list of bridge tools that can be registered.
        """
        server = self._client.get_server(server_name)
        if not server:
            return []

        tools = []
        for mcp_tool in server.tools:
            bridge = MCPBridgeTool(mcp_tool, self._client)
            self._bridged_tools[bridge.name] = bridge
            tools.append(bridge)
            logger.debug("Bridged MCP tool: %s", bridge.name)

        logger.info(
            "Bridged %d tools from MCP server '%s'",
            len(tools), server_name
        )
        return tools

    def bridge_all(self) -> list[MCPBridgeTool]:
        """Bridge tools from all connected servers."""
        all_tools = []
        for server in self._client.list_servers():
            all_tools.extend(self.bridge_server(server.name))
        return all_tools

    def get_bridged_tools(self) -> list[MCPBridgeTool]:
        """Get all currently bridged tools."""
        return list(self._bridged_tools.values())

    @property
    def count(self) -> int:
        return len(self._bridged_tools)
