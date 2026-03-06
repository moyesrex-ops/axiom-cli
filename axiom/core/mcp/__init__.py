"""MCP integration -- connect to Model Context Protocol servers."""

from axiom.core.mcp.client import MCPClient, MCPServer, MCPTool
from axiom.core.mcp.discovery import MCPDiscovery, ServerConfig
from axiom.core.mcp.bridge import MCPBridge, MCPBridgeTool

__all__ = [
    "MCPClient", "MCPServer", "MCPTool",
    "MCPDiscovery", "ServerConfig",
    "MCPBridge", "MCPBridgeTool",
]
