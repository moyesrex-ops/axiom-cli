"""MCP discovery -- auto-discover and load MCP server configurations.

Reads server definitions from a JSON config file and provides
methods to list available, connect, and manage MCP servers.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Default config locations (searched in order)
DEFAULT_CONFIG_PATHS: list[Path] = [
    Path("mcp_servers") / "servers.json",             # Project-level
    Path.home() / ".axiom" / "mcp_servers.json",      # User-level
]


@dataclass
class ServerConfig:
    """Configuration for an MCP server."""
    name: str
    command: str
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    description: str = ""
    auto_connect: bool = False


class MCPDiscovery:
    """Discover and manage MCP server configurations.

    Reads from servers.json config files and provides a registry
    of available MCP servers that can be connected on demand.
    """

    def __init__(self, config_paths: list[Path] | None = None) -> None:
        self._paths = config_paths or list(DEFAULT_CONFIG_PATHS)
        self._configs: dict[str, ServerConfig] = {}
        self._loaded = False

    def load_configs(self) -> int:
        """Load server configurations from all config files.

        Returns number of server configs loaded.
        """
        self._configs.clear()

        for config_path in self._paths:
            if config_path.exists() and config_path.is_file():
                try:
                    self._load_config_file(config_path)
                except Exception as exc:
                    logger.warning("Failed to load MCP config %s: %s", config_path, exc)

        self._loaded = True
        logger.info("Loaded %d MCP server configurations", len(self._configs))
        return len(self._configs)

    def _load_config_file(self, path: Path) -> None:
        """Load a single servers.json config file."""
        text = path.read_text(encoding="utf-8")
        data = json.loads(text)

        servers = data.get("servers", data.get("mcpServers", {}))

        if isinstance(servers, dict):
            # Format: {"server-name": {"command": "...", "args": [...]}}
            for name, config in servers.items():
                if not isinstance(config, dict):
                    continue
                self._configs[name] = ServerConfig(
                    name=name,
                    command=config.get("command", ""),
                    args=config.get("args", []),
                    env=config.get("env", {}),
                    description=config.get("description", ""),
                    auto_connect=config.get("autoConnect", False),
                )
        elif isinstance(servers, list):
            # Format: [{"name": "...", "command": "...", ...}]
            for config in servers:
                if not isinstance(config, dict) or "name" not in config:
                    continue
                name = config["name"]
                self._configs[name] = ServerConfig(
                    name=name,
                    command=config.get("command", ""),
                    args=config.get("args", []),
                    env=config.get("env", {}),
                    description=config.get("description", ""),
                    auto_connect=config.get("autoConnect", False),
                )

    def get(self, name: str) -> Optional[ServerConfig]:
        """Get a server config by name."""
        if not self._loaded:
            self.load_configs()
        return self._configs.get(name)

    def list_configs(self) -> list[ServerConfig]:
        """List all available server configurations."""
        if not self._loaded:
            self.load_configs()
        return sorted(self._configs.values(), key=lambda c: c.name)

    def get_auto_connect(self) -> list[ServerConfig]:
        """Get servers configured for auto-connection."""
        if not self._loaded:
            self.load_configs()
        return [c for c in self._configs.values() if c.auto_connect]

    @property
    def count(self) -> int:
        if not self._loaded:
            self.load_configs()
        return len(self._configs)
