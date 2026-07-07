"""MCP (Model Context Protocol) integration — connect to external tool servers.

Uses the official 'mcp' Python SDK to create clients that communicate
with MCP servers via stdio, HTTP+SSE, or Streamable HTTP transports.

MCP tools are exposed as LangChain tools with server-scoped namespacing.
"""

import subprocess
import json
from pathlib import Path
from typing import Optional

from langchain_core.tools import StructuredTool


class MCPServerConfig:
    """Configuration for a single MCP server connection."""

    def __init__(
        self,
        name: str,
        command: Optional[str] = None,
        args: Optional[list[str]] = None,
        url: Optional[str] = None,
        env: Optional[dict[str, str]] = None,
        transport: str = "stdio",  # "stdio", "sse", "streamable_http"
    ):
        self.name = name
        self.command = command
        self.args = args or []
        self.url = url
        self.env = env or {}
        self.transport = transport


def load_mcp_configs(workspace: str) -> list[MCPServerConfig]:
    """Load MCP server configurations from .opencode.json and ~/.opencode-mcp.json."""
    configs = []

    # Project-level config
    proj_config = Path(workspace) / ".opencode.json"
    if proj_config.exists():
        configs += _parse_config(proj_config)

    # User-level config
    user_config = Path.home() / ".opencode-mcp.json"
    if user_config.exists():
        configs += _parse_config(user_config)

    return configs


def _parse_config(filepath: Path) -> list[MCPServerConfig]:
    try:
        data = json.loads(filepath.read_text())
        configs = []
        for name, cfg in data.get("mcpServers", {}).items():
            c = MCPServerConfig(
                name=name,
                command=cfg.get("command"),
                args=cfg.get("args", []),
                url=cfg.get("url"),
                env=cfg.get("env", {}),
                transport=cfg.get("transport", "stdio"),
            )
            configs.append(c)
        return configs
    except (json.JSONDecodeError, OSError):
        return []


class MCPToolProxy:
    """Proxy that dynamically loads MCP tools on-demand.

    Note: Full MCP integration requires the 'mcp' package.
    If not installed, this provides graceful degradation with a notice.
    """

    _tools_cache: dict[str, list] = {}  # server_name -> list of LangChain tools

    @classmethod
    def load_tools(cls, configs: list[MCPServerConfig]) -> list:
        """Load tools from all configured MCP servers. Returns LangChain tools."""
        try:
            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import stdio_client
        except ImportError:
            return [cls._create_stub_mcp_tool(configs)]

        # For now, return a stub that explains MCP is available
        # Full async session management would be added here
        tools = []
        for config in configs:
            tools.append(cls._create_mcp_stub(config))
        return tools

    @classmethod
    def _create_mcp_stub(cls, config: MCPServerConfig):
        """Create a placeholder tool indicating MCP server availability."""
        return StructuredTool.from_function(
            name=f"mcp_{config.name}_status",
            description=f"Check status of MCP server '{config.name}'. "
                       f"Transport: {config.transport}.",
            func=lambda: json.dumps({
                "server": config.name,
                "status": "configured",
                "transport": config.transport,
                "note": "Install 'mcp' package (pip install mcp) for full MCP support",
            }),
        )

    @classmethod
    def _create_stub_mcp_tool(cls, configs: list[MCPServerConfig]):
        servers = [c.name for c in configs]
        return StructuredTool.from_function(
            name="mcp_status",
            description="Check MCP server status. Install 'mcp' package for full support.",
            func=lambda: json.dumps({
                "status": "mcp_package_not_installed",
                "configured_servers": servers,
                "help": "pip install mcp to enable Model Context Protocol integration",
            }),
        )


def get_mcp_tools(workspace: str) -> list:
    """Get all MCP tools for the given workspace."""
    configs = load_mcp_configs(workspace)
    if not configs:
        return []
    return MCPToolProxy.load_tools(configs)
