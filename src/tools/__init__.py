"""Opencode DeepAgents tools package.

deepagents provides built-in: ls, read_file, write_file, edit_file, glob,
grep, execute (shell), and todo. We only add web tools on top.
"""

from src.tools.web import web_fetch, web_search

__all__ = [
    "web_fetch",
    "web_search",
    "get_custom_tools",
]


def get_custom_tools() -> list:
    """Get project-specific tools not provided by deepagents."""
    return [web_fetch, web_search]
