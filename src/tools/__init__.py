"""Opencode DeepAgents tools package.

deepagents built-in: ls, read_file, write_file, edit_file, glob, grep,
execute (shell), todo. We add web, patch, question, skill, LSP, and MCP tools.
"""

from src.tools.web import web_fetch, web_search
from src.tools.apply_patch import apply_patch, create_apply_patch_tool
from src.tools.question import question, create_question_tool
from src.tools.skill import skill, create_skill_tool
from src.tools.lsp import lsp_definition, lsp_references, lsp_hover, lsp_diagnostics, LSP_TOOLS
from src.tools.mcp import get_mcp_tools, load_mcp_configs, MCPServerConfig

__all__ = [
    "web_fetch",
    "web_search",
    "apply_patch",
    "question",
    "skill",
    "lsp_definition",
    "lsp_references",
    "lsp_hover",
    "lsp_diagnostics",
    "LSP_TOOLS",
    "get_mcp_tools",
    "load_mcp_configs",
    "get_custom_tools",
    "get_all_custom_tools",
]


def get_custom_tools() -> list:
    """Get project-specific tools not provided by deepagents (non-LSP, non-MCP)."""
    return [web_fetch, web_search, apply_patch, question, skill]


def get_all_custom_tools(workspace: str = ".") -> list:
    """Get all custom tools including LSP and MCP."""
    tools = get_custom_tools()
    tools.extend(LSP_TOOLS)
    tools.extend(get_mcp_tools(workspace))
    return tools
