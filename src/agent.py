"""Deep Agents-based AI coding agent — built on deepagents' harness.

Integrates: HITL permissions, git snapshots, LSP tools, MCP tools,
multi-file patch tool, question tool, skill loader, and background sub-agents.
"""

import asyncio
from pathlib import Path
from typing import Optional, AsyncIterator

from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend, LocalShellBackend
from deepagents.middleware.subagents import SubAgent

from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver

from src.config import (
    ModelConfig, AgentModeConfig, AGENT_MODES,
    load_model_config, PLAN_SUBAGENT_PROMPT, EXPLORE_SUBAGENT_PROMPT,
    DEFAULT_WORKSPACE, DEFAULT_AGENT_MODE, MAX_TOOL_ITERATIONS,
    DANGEROUS_TOOLS, READONLY_TOOLS, CHECKPOINT_DB,
)
from src.tools import get_all_custom_tools, get_custom_tools
from src.permission import (
    PermissionConfig, InterruptHandler, build_default_permissions,
    load_permissions_from_config,
)
from src.snapshot import SnapshotStore


class CodingAgent:
    """AI coding agent powered by deepagents with full opencode features.

    Uses deepagents' built-in harness plus:
    - HITL permission system (approve/reject tool calls before execution)
    - Git-based filesystem snapshots (safe undo/restore)
    - LSP tools (go-to-def, references, diagnostics)
    - MCP integration (external tool servers)
    - Multi-file apply_patch tool
    - Interactive question tool
    - Skill loader tool
    - Background sub-agents (code-explorer + plan-analyze)

    Two modes:
    - **build**: full filesystem + shell access, HITL approvals for dangerous ops
    - **plan**: read-only filesystem, no shell, analysis-only
    """

    def __init__(
        self,
        model_config: Optional[ModelConfig] = None,
        workspace: str = DEFAULT_WORKSPACE,
        agent_mode: str = DEFAULT_AGENT_MODE,
        checkpoint_db_path: Optional[str] = None,
        permission_config: Optional[PermissionConfig] = None,
    ):
        self.model_config = model_config or load_model_config()
        self.workspace = str(Path(workspace).resolve())
        self.agent_mode = agent_mode
        self.mode_config = AGENT_MODES.get(agent_mode, AGENT_MODES["build"])
        self.checkpoint_db_path = checkpoint_db_path or CHECKPOINT_DB

        # Permission system with HITL
        self.permission = permission_config or build_default_permissions(agent_mode, self.workspace)
        self.interrupt_handler = InterruptHandler(self.permission)

        # Load project-level permission config
        proj_config = Path(self.workspace) / ".opencode.json"
        if proj_config.exists():
            for rule in load_permissions_from_config(str(proj_config)):
                self.permission.rules.append(rule)

        # Snapshot store for file recovery
        self.snapshot = SnapshotStore(self.workspace)

        # Background sub-agent tracking
        self._bg_tasks: dict[str, "BackgroundTask"] = {}

        # Build the graph
        self.graph = self._build_graph()

    def _build_graph(self):
        """Build the deep agent with mode-appropriate configuration and HITL."""
        mode = self.mode_config
        tools = get_all_custom_tools(self.workspace)

        # ── Filesystem Backend ──
        if mode.allow_shell:
            backend = LocalShellBackend(root_dir=self.workspace)
        else:
            backend = FilesystemBackend(root_dir=self.workspace)

        # ── Sub-agents (multiple types) ──
        sub_agents = [
            SubAgent(
                name="plan-analyze",
                description=(
                    "Deep analysis sub-agent. Use for complex codebase analysis, "
                    "architecture review, researching approaches, or generating detailed plans. "
                    "This sub-agent is read-only and cannot modify files."
                ),
                system_prompt=PLAN_SUBAGENT_PROMPT,
                tools=get_custom_tools(),  # web + skill + question only
                model=self.model_config.to_model_string(),
            ),
            SubAgent(
                name="code-explorer",
                description=(
                    "Large-scale codebase exploration sub-agent. Use for finding where "
                    "features are implemented, tracing data flows, mapping module "
                    "dependencies, or understanding project structure. Read-only."
                ),
                system_prompt=EXPLORE_SUBAGENT_PROMPT,
                tools=get_custom_tools(),
                model=self.model_config.to_model_string(),
            ),
        ]

        # ── Checkpointer (persistent state across turns) ──
        if self.checkpoint_db_path == ":memory:":
            checkpointer = MemorySaver()
        else:
            db_path = Path(self.checkpoint_db_path)
            db_path.parent.mkdir(parents=True, exist_ok=True)
            checkpointer = SqliteSaver.from_conn_string(str(db_path))

        # ── System prompt with model-specific overrides ──
        provider = getattr(self.model_config, "provider", "openai")
        system_prompt = mode.format_prompt(self.workspace, provider)

        # ── Create the deep agent with HITL ──
        agent = create_deep_agent(
            model=self.model_config.to_model_string(),
            tools=tools,
            system_prompt=system_prompt,
            sub_agents=sub_agents,
            backend=backend,
            checkpointer=checkpointer,
            # HITL: interrupt before tool execution for permission checks
            interrupt_before=["tools"],
        )

        return agent

    async def astream(
        self, user_message: str, thread_id: str = "default",
        approve_all: bool = False,
    ) -> AsyncIterator[dict]:
        """Stream agent execution with HITL permission checks.

        When a tool needs approval, yields:
            {"type": "approval_needed", "tool": "...", "input": "..."}

        The caller must call resume() to approve or reject.

        Args:
            user_message: User's input message
            thread_id: LangGraph thread ID for state isolation
            approve_all: Auto-approve all tool calls (skip HITL)

        Yields:
            {"type": "token", "content": "..."}
            {"type": "approval_needed", "tool": "...", "input": "...", "tool_call_id": "..."}
            {"type": "tool_start", "name": "...", "input": "..."}
            {"type": "tool_end", "name": "...", "output": "..."}
            {"type": "done"}
        """
        config = {"configurable": {"thread_id": thread_id}}

        last_tool_name = None
        pending_approval = None
        tool_inputs = {}  # tool_call_id -> input preview

        # Reset interrupt handler for this turn
        self.interrupt_handler = InterruptHandler(self.permission)
        # Re-apply saved approvals
        self.interrupt_handler.permission.approved = self.permission.approved
        self.interrupt_handler.permission.denied = self.permission.denied

        # Snapshot before starting
        self.snapshot.track(f"pre-{thread_id}")

        async for event in self.graph.astream_events(
            {"messages": [{"role": "user", "content": user_message}]},
            config=config,
            version="v2",
        ):
            kind = event.get("event", "")

            if kind == "on_chat_model_stream":
                chunk = event["data"]["chunk"]
                delta = getattr(chunk, "content", "")
                if delta:
                    yield {"type": "token", "content": delta}

            elif kind == "on_tool_start":
                name = event.get("name", "unknown")
                inp = event["data"].get("input", {})
                safe_input = _format_tool_input(inp)
                tool_call_id = event.get("run_id", str(hash(str(inp))))
                tool_inputs[tool_call_id] = inp

                # Permission check via HITL
                if not approve_all and self.interrupt_handler.should_interrupt(name, inp) == "ask":
                    # Take snapshot before dangerous operations
                    if name in DANGEROUS_TOOLS:
                        self.snapshot.track(f"pre-{name}")

                    yield {
                        "type": "approval_needed",
                        "tool": name,
                        "input": safe_input,
                        "tool_call_id": tool_call_id,
                    }
                    # Don't yield tool_start yet — wait for approval
                    pending_approval = (name, inp, tool_call_id)
                    continue

                last_tool_name = name
                yield {
                    "type": "tool_start",
                    "name": name,
                    "input": safe_input,
                }

            elif kind == "on_tool_end":
                if pending_approval and event.get("run_id") == pending_approval[2]:
                    # This tool was approved — now emit start + end together
                    pname, pinp, pid = pending_approval
                    yield {
                        "type": "tool_start",
                        "name": pname,
                        "input": _format_tool_input(pinp),
                    }
                    pending_approval = None

                output = event["data"].get("output", "")
                yield {
                    "type": "tool_end",
                    "name": event.get("name", last_tool_name or "unknown"),
                    "output": str(output)[:2000],
                }

        # Persist approval state
        self.permission.approved.update(self.interrupt_handler.permission.approved)
        self.permission.denied.update(self.interrupt_handler.permission.denied)

        yield {"type": "done"}

    async def resume(
        self, thread_id: str, approved: bool, tool_call_id: str,
    ) -> AsyncIterator[dict]:
        """Resume agent execution after HITL approval decision.

        Args:
            thread_id: LangGraph thread ID
            approved: Whether the user approved the tool call
            tool_call_id: The tool call ID to approve/reject
        """
        config = {"configurable": {"thread_id": thread_id}}
        approved_val = "approved" if approved else "rejected"
        resume_input = {"decisions": {tool_call_id: approved_val}}

        last_tool_name = None

        async for event in self.graph.astream_events(
            resume_input, config=config, version="v2",
        ):
            kind = event.get("event", "")

            if kind == "on_chat_model_stream":
                chunk = event["data"]["chunk"]
                delta = getattr(chunk, "content", "")
                if delta:
                    yield {"type": "token", "content": delta}

            elif kind == "on_tool_start":
                name = event.get("name", "unknown")
                inp = event["data"].get("input", {})
                last_tool_name = name
                yield {
                    "type": "tool_start",
                    "name": name,
                    "input": _format_tool_input(inp),
                }

            elif kind == "on_tool_end":
                output = event["data"].get("output", "")
                yield {
                    "type": "tool_end",
                    "name": event.get("name", last_tool_name or "unknown"),
                    "output": str(output)[:2000],
                }

        yield {"type": "done"}

    def invoke(self, user_message: str, thread_id: str = "default") -> dict:
        """Synchronous invoke — returns final state."""
        config = {"configurable": {"thread_id": thread_id}}
        self.snapshot.track(f"pre-invoke-{thread_id}")
        return self.graph.invoke(
            {"messages": [{"role": "user", "content": user_message}]},
            config=config,
        )

    def get_state(self, thread_id: str = "default"):
        """Get LangGraph state for a thread."""
        config = {"configurable": {"thread_id": thread_id}}
        return self.graph.get_state(config)

    def get_messages(self, thread_id: str = "default") -> list:
        """Get messages from the conversation state."""
        state = self.get_state(thread_id)
        if state and state.values:
            return state.values.get("messages", [])
        return []

    def undo_last_action(self, thread_id: str = "default") -> str:
        """Restore filesystem to the last snapshot."""
        snapshots = self.snapshot.list_snapshots(2)
        if len(snapshots) >= 2:
            prev = snapshots[1]["hash"]
            if self.snapshot.restore(prev):
                return f"Restored filesystem to snapshot {prev}"
        return "No previous snapshot available for undo"

    def set_mode(self, mode: str):
        """Switch agent mode and rebuild."""
        if mode not in AGENT_MODES:
            raise ValueError(f"Unknown agent mode: {mode}. Use {list(AGENT_MODES.keys())}")
        self.agent_mode = mode
        self.mode_config = AGENT_MODES[mode]
        self.permission = build_default_permissions(mode, self.workspace)
        self.graph = self._build_graph()

    def set_workspace(self, workspace: str):
        """Change workspace and rebuild."""
        self.workspace = str(Path(workspace).resolve())
        self.snapshot = SnapshotStore(self.workspace)
        self.permission = build_default_permissions(self.agent_mode, self.workspace)
        self.graph = self._build_graph()

    def set_model(self, model_config: ModelConfig):
        """Change model and rebuild."""
        self.model_config = model_config
        self.graph = self._build_graph()

    def list_snapshots(self, limit: int = 10) -> list[dict]:
        """List recent filesystem snapshots."""
        return self.snapshot.list_snapshots(limit)

    def restore_snapshot(self, snapshot_hash: str) -> bool:
        """Restore to a specific snapshot."""
        return self.snapshot.restore(snapshot_hash)

    # ── Background Task Support ──

    def start_background_task(self, task_id: str, prompt: str):
        """Start a background sub-agent task. Returns immediately."""
        task = BackgroundTask(task_id, prompt, self)
        self._bg_tasks[task_id] = task
        asyncio.create_task(task.run())
        return task

    def get_background_task(self, task_id: str) -> Optional["BackgroundTask"]:
        return self._bg_tasks.get(task_id)

    def list_background_tasks(self) -> list[dict]:
        return [t.status() for t in self._bg_tasks.values()]


class BackgroundTask:
    """Runs a sub-agent asynchronously in the background."""

    def __init__(self, task_id: str, prompt: str, agent: CodingAgent):
        self.task_id = task_id
        self.prompt = prompt
        self.agent = agent
        self.result: Optional[str] = None
        self.error: Optional[str] = None
        self.done = asyncio.Event()

    async def run(self):
        try:
            messages = []
            async for event in self.agent.astream(self.prompt, thread_id=f"bg-{self.task_id}"):
                if event["type"] == "done":
                    break
                elif event["type"] == "token":
                    messages.append(event.get("content", ""))
            self.result = "".join(messages) if messages else "(empty response)"
        except Exception as e:
            self.error = str(e)
        finally:
            self.done.set()

    def status(self) -> dict:
        return {
            "task_id": self.task_id,
            "prompt": self.prompt[:100],
            "running": not self.done.is_set(),
            "result": self.result[:500] if self.result else None,
            "error": self.error,
        }

    async def wait(self, timeout: Optional[float] = None) -> Optional[str]:
        try:
            await asyncio.wait_for(self.done.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            return None
        return self.result


def _format_tool_input(inp: dict, max_len: int = 500) -> str:
    """Format tool input dict for display, truncating long values."""
    import json as _json
    safe = {}
    for k, v in inp.items():
        s = str(v)
        if len(s) > max_len:
            s = s[:max_len] + f"... ({len(s)} total chars)"
        safe[k] = s
    return _json.dumps(safe, indent=2, ensure_ascii=False)
