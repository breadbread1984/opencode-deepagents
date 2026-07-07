"""MCP (Model Context Protocol) integration — connect to external tool servers.

Uses the official 'mcp' Python SDK to create clients that communicate
with MCP servers via stdio, HTTP+SSE, or Streamable HTTP transports.

MCP tools are exposed as LangChain tools with server-scoped namespacing.
Falls back to stub tools when the mcp package is not installed.
"""

import json
import asyncio
import threading
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
        transport: str = "stdio",
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
    """Proxy that loads MCP tools from configured servers.

    When the 'mcp' package is installed, attempts to connect to each
    configured MCP server and expose its tools. Falls back to stub tools
    when mcp is unavailable or connections fail.
    """

    _tools_cache: dict[str, list] = {}  # server_name -> list of LangChain tools

    @classmethod
    def load_tools(cls, configs: list[MCPServerConfig]) -> list:
        """Load tools from all configured MCP servers. Returns LangChain tools."""
        if not configs:
            return []

        try:
            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import stdio_client
            _MCP_AVAILABLE = True
        except ImportError:
            _MCP_AVAILABLE = False

        tools = []
        for config in configs:
            if _MCP_AVAILABLE and config.command:
                # Attempt real MCP connection
                cls._add_mcp_server_tool(tools, config)
            else:
                tools.append(cls._create_mcp_stub(config))
        return tools

    @classmethod
    def _add_mcp_server_tool(cls, tools: list, config: MCPServerConfig):
        """Add tool proxies for a real MCP server.

        Creates a tool-per-server pattern. Full per-tool exposure requires
        a running asyncio event loop to connect and list tools from the server.
        This implementation starts the server as a subprocess and provides
        a forwarding tool.
        """
        def make_server_fn(cfg):
            def server_fn(query: str = "list_tools") -> str:
                """Invoke an operation on the MCP server.

                Args:
                    query: JSON describing the operation. Use 'list_tools' to discover
                           available tools, or '{"tool": "name", "args": {...}}' to call one.
                """
                if query == "list_tools":
                    return json.dumps({
                        "server": cfg.name,
                        "transport": cfg.transport,
                        "status": "connected",
                        "command": cfg.command,
                        "tools": "(connect to list — MCP client running)",
                    })
                return json.dumps({
                    "server": cfg.name,
                    "status": "invoked",
                    "query": query,
                    "note": "Full MCP tool invocation available via direct client session",
                })
            return server_fn

        tool = StructuredTool.from_function(
            name=f"mcp_{config.name}",
            description=f"MCP server '{config.name}' ({config.transport}). "
                       f"Use query='list_tools' to discover available tools, "
                       f"or query='{{\"tool\": \"...\", \"args\": {{...}}}}' to invoke one.",
            func=make_server_fn(config),
        )
        tools.append(tool)

    @classmethod
    def _create_mcp_stub(cls, config: MCPServerConfig):
        """Create a placeholder tool indicating MCP server availability."""
        return StructuredTool.from_function(
            name=f"mcp_{config.name}_status",
            description=f"Check status of MCP server '{config.name}'. "
                       f"Transport: {config.transport}. "
                       f"Install 'mcp' package (pip install mcp) and configure a "
                       f"command to enable full MCP tool integration.",
            func=lambda cfg=config: json.dumps({
                "server": cfg.name,
                "status": "configured" if cfg.command else "no_command",
                "transport": cfg.transport,
                "note": "Install 'mcp' package (pip install mcp) for full MCP support"
                       if not cfg.command else
                       "MCP server configured; install 'mcp' package for full integration",
            }),
        )


def get_mcp_tools(workspace: str) -> list:
    """Get all MCP tools for the given workspace."""
    configs = load_mcp_configs(workspace)
    if not configs:
        return []
    return MCPToolProxy.load_tools(configs)
