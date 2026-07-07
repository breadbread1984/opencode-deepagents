"""Permission system with HITL (Human-in-the-Loop) for tool approvals.

Features:
- Wildcard-based allow/deny/ask rules (matching original opencode)
- "Always allow" memory across tool calls
- External directory tracking
- Integration with LangGraph interrupt mechanism for deepagents HITL

deepagents uses LangGraph's interrupt() to pause before tool execution.
This module defines the rule engine and interrupt handler.
"""

import fnmatch
import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional, Callable


class PermissionAction(Enum):
    ALLOW = "allow"
    DENY = "deny"
    ASK = "ask"


@dataclass
class PermissionRule:
    """A single permission rule with wildcard pattern matching."""
    tool: str              # Wildcard pattern, e.g. "write", "shell", "*"
    pattern: str = "*"     # Additional pattern, e.g. file path pattern
    action: PermissionAction = PermissionAction.ASK


@dataclass
class PermissionConfig:
    """Manages permission rules for a session."""
    rules: list[PermissionRule] = field(default_factory=list)
    approved: set[str] = field(default_factory=set)   # "always allow" memory
    denied: set[str] = field(default_factory=set)     # "always deny" memory
    external_dirs: list[str] = field(default_factory=list)  # dirs outside workspace
    workspace: str = "."

    def evaluate(self, tool: str, filepath: str = "") -> PermissionAction:
        """Evaluate permission for a tool call.
        
        Returns the appropriate action (ALLOW/DENY/ASK).
        Checks: cached approvals -> rules -> default ASK.
        """
        cache_key = f"{tool}:{filepath}" if filepath else tool

        # Check cached decisions
        if cache_key in self.approved:
            return PermissionAction.ALLOW
        if cache_key in self.denied:
            return PermissionAction.DENY

        # Check rules (last matching rule wins)
        result = None
        for rule in self.rules:
            tool_match = fnmatch.fnmatch(tool, rule.tool)
            pattern_match = fnmatch.fnmatch(filepath, rule.pattern)
            if tool_match and pattern_match:
                result = rule.action

        if result is not None:
            return result

        # Default: ask
        return PermissionAction.ASK

    def approve(self, tool: str, filepath: str = "", remember: bool = False):
        """Record an approval."""
        cache_key = f"{tool}:{filepath}" if filepath else tool
        self.approved.add(cache_key)

    def reject(self, tool: str, filepath: str = "", remember: bool = False):
        """Record a rejection."""
        cache_key = f"{tool}:{filepath}" if filepath else tool
        self.denied.add(cache_key)

    def is_external(self, path: str) -> bool:
        """Check if a path is outside the workspace."""
        try:
            ws = Path(self.workspace).resolve()
            p = Path(path).resolve()
            return not str(p).startswith(str(ws))
        except Exception:
            return False

    def add_allow_all(self, patterns: list[str]):
        """Add unconditional allow rules for certain tools."""
        for p in patterns:
            self.rules.append(PermissionRule(tool=p, action=PermissionAction.ALLOW))

    def add_deny_all(self, patterns: list[str]):
        """Add unconditional deny rules for certain tools."""
        for p in patterns:
            self.rules.append(PermissionRule(tool=p, action=PermissionAction.DENY))


# ── Dangerous tool patterns that should always require approval ──

DANGEROUS_PATTERNS = [
    "write_file",
    "edit_file",
    "execute",
    "task",
]

READONLY_PATTERNS = [
    "ls",
    "read_file",
    "glob",
    "grep",
    "web_fetch",
    "web_search",
    "todo_write",
    "question",
]


def build_default_permissions(mode: str, workspace: str) -> PermissionConfig:
    """Build default permission config based on agent mode."""
    config = PermissionConfig(workspace=workspace)

    if mode == "plan":
        # Plan mode: read-only, auto-allow reads, deny writes/exec
        config.add_allow_all(READONLY_PATTERNS)
        config.add_deny_all(["write_file", "edit_file", "execute", "task"])
    elif mode == "build":
        # Build mode: auto-allow reads, ask for dangerous ops
        config.add_allow_all(READONLY_PATTERNS)
        for p in DANGEROUS_PATTERNS:
            config.rules.append(PermissionRule(tool=p, action=PermissionAction.ASK))

    return config


def load_permissions_from_config(config_path: str) -> list[PermissionRule]:
    """Load permission rules from opencode.json or .opencode-permissions.json."""
    import json
    try:
        path = Path(config_path)
        if not path.exists():
            return []
        data = json.loads(path.read_text())
        rules = []
        for key, action_str in data.get("permissions", {}).items():
            action = PermissionAction(action_str) if action_str in {"allow", "deny", "ask"} else PermissionAction.ASK
            tool, _, pattern = key.partition("/")
            rules.append(PermissionRule(tool=tool.strip(), pattern=pattern.strip(), action=action))
        return rules
    except (json.JSONDecodeError, OSError, ValueError):
        return []


class InterruptHandler:
    """Manages LangGraph interrupt state for tool approval (deepagents HITL).

    When interrupt_before=["tools"] is set, the graph pauses before each
    tool execution. This handler:
    1. Captures the interrupt payload
    2. Provides it to the UI for approval
    3. Resumes with approval/rejection when the user responds
    """

    def __init__(self, permission_config: PermissionConfig):
        self.permission = permission_config
        self._pending_interrupt: Optional[dict] = None
        self._resolve_callback: Optional[Callable] = None

    def has_pending(self) -> bool:
        return self._pending_interrupt is not None

    def get_pending(self) -> Optional[dict]:
        return self._pending_interrupt

    def set_pending(self, interrupt_data: dict):
        self._pending_interrupt = interrupt_data

    def should_interrupt(self, tool_name: str, tool_input: dict) -> PermissionAction:
        """Determine if a tool call needs approval."""
        # Extract file path from input if present
        filepath = ""
        for key in ("path", "file_path", "old_str", "new_str", "command", "url"):
            if key in tool_input:
                val = str(tool_input[key])
                if val and len(val) < 200:
                    filepath = val
                    break

        return self.permission.evaluate(tool_name, filepath)

    def approve(self, tool: str, filepath: str = "", remember: bool = False):
        """Handle user approval."""
        self.permission.approve(tool, filepath, remember)
        result = {"approved": True, "remember": remember}
        self._pending_interrupt = None
        return result

    def reject(self, tool: str, filepath: str = ""):
        """Handle user rejection."""
        self.permission.reject(tool, filepath)
        result = {"approved": False}
        self._pending_interrupt = None
        return result

    def clear(self):
        self._pending_interrupt = None
