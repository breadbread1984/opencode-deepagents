"""Deep Agents-based AI coding agent — built on deepagents' harness.

Integrates: HITL permissions, git snapshots, LSP tools, MCP tools,
multi-file patch tool, question tool, skill loader, and background sub-agents.
"""

import asyncio
import json
import os
import threading
from pathlib import Path
from typing import Optional, AsyncIterator

# Load .env BEFORE any LangChain imports so LangSmith / API keys are picked up
from dotenv import load_dotenv
load_dotenv()

from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend, LocalShellBackend
from deepagents.middleware.subagents import SubAgent

from langchain_openai import ChatOpenAI

from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command

from src.config import (
    ModelConfig, AgentModeConfig, AGENT_MODES,
    load_model_config, PLAN_SUBAGENT_PROMPT, EXPLORE_SUBAGENT_PROMPT,
    DEFAULT_WORKSPACE, DEFAULT_AGENT_MODE, MAX_TOOL_ITERATIONS,
    DANGEROUS_TOOLS, CHECKPOINT_DB,
)
from src.tools import get_all_custom_tools, get_custom_tools
from src.permission import (
    PermissionConfig, InterruptHandler, PermissionAction,
    build_default_permissions, load_permissions_from_config,
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

        # Track checkpointer for proper cleanup on rebuild
        self._checkpointer = None
        self._checkpointer_ctx = None

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

        # Graph is lazily initialized on first astream() call
        self.graph = None

    async def _ensure_graph(self):
        """Lazily build the agent graph with async checkpointer setup.
        Called from astream() / resume() to handle the running event loop properly.
        """
        if self.graph is not None:
            return

        # Close old checkpointer if re-building
        if self._checkpointer is not None:
            if self._checkpointer_ctx is not None:
                try:
                    await self._checkpointer_ctx.__aexit__(None, None, None)
                except Exception:
                    pass
                self._checkpointer_ctx = None
            elif hasattr(self._checkpointer, "close"):
                try:
                    self._checkpointer.close()
                except Exception:
                    pass
            self._checkpointer = None

        # Create checkpointer with proper async handling
        if self.checkpoint_db_path == ":memory:":
            self._checkpointer = MemorySaver()
        else:
            db_path = Path(self.checkpoint_db_path)
            db_path.parent.mkdir(parents=True, exist_ok=True)
            self._checkpointer_ctx = AsyncSqliteSaver.from_conn_string(str(db_path))
            self._checkpointer = await self._checkpointer_ctx.__aenter__()

        self.graph = self._build_graph()

    def _build_chat_model(self):
        """Create a ChatOpenAI instance for OpenAI-compatible endpoints.

        stream_usage=False prevents the OpenAI SDK from adding
        ``stream_options={"include_usage": true}`` to streaming requests,
        which DashScope and other compatible APIs do not support.
        """
        kwargs: dict = {
            "model": self.model_config.model,
            "temperature": self.model_config.temperature,
            "stream_usage": False,
        }
        if self.model_config.api_key:
            kwargs["openai_api_key"] = self.model_config.api_key
            # ChatOpenAI / the OpenAI SDK also checks OPENAI_API_KEY env var
            # internally — set it so the key is found regardless of code path
            os.environ["OPENAI_API_KEY"] = self.model_config.api_key
        if self.model_config.base_url:
            kwargs["openai_api_base"] = self.model_config.base_url
        return ChatOpenAI(**kwargs)

    def _build_graph(self):
        """Build the deep agent with mode-appropriate configuration and HITL."""
        mode = self.mode_config
        tools = get_all_custom_tools(self.workspace)

        # Filesystem Backend
        if mode.allow_shell:
            backend = LocalShellBackend(root_dir=self.workspace)
        else:
            backend = FilesystemBackend(root_dir=self.workspace)

        # Build ChatOpenAI instance (with stream_usage=False for DashScope compat)
        model_instance = self._build_chat_model()

        # Sub-agents (multiple types) — keep using string-based model for sub-agents
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

        # System prompt with model-specific overrides
        provider = getattr(self.model_config, "provider", "dashscope")
        system_prompt = mode.format_prompt(self.workspace, provider)

        # HITL: interrupt before tool execution for permission checks.
        # When the graph pauses at the tools node, astream() detects this,
        # checks permissions against pending tool calls, and yields
        # approval_needed events. resume() sends Command(resume=...) to continue.
        agent = create_deep_agent(
            model=model_instance,
            tools=tools,
            system_prompt=system_prompt,
            subagents=sub_agents,
            backend=backend,
            checkpointer=self._checkpointer,
            interrupt_on={"tools": True},
        )

        return agent

    async def astream(
        self, user_message: str, thread_id: str = "default",
        approve_all: bool = False,
    ) -> AsyncIterator[dict]:
        """Stream agent execution with HITL permission checks.

        When a tool needs approval, yields:
            {"type": "approval_needed", "tool": "...", "input": "...", "tool_call_id": "..."}

        The caller must call resume() to approve or reject, then call astream()
        again with the same thread_id to continue after approval.

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
        await self._ensure_graph()

        config = {"configurable": {"thread_id": thread_id}}

        last_tool_name = None

        # Reset interrupt handler for this turn
        self.interrupt_handler = InterruptHandler(self.permission)
        # Re-apply saved approvals/denials from previous turns
        self.interrupt_handler.permission.approved = self.permission.approved
        self.interrupt_handler.permission.denied = self.permission.denied

        # Snapshot before starting
        self.snapshot.track(f"pre-{thread_id}")

        interrupted = False
        had_approval = False

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
                inp = _safe_tool_input(event["data"].get("input"))
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

        # After the stream finishes (may be interrupted), check if the graph
        # is paused at the tools node (interrupt_on triggered).
        state = await self.graph.aget_state(config)
        is_interrupted = bool(state and state.next and "tools" in state.next)

        if is_interrupted and not approve_all:
            # Extract pending tool calls from the saved state
            pending_tools = _extract_pending_tool_calls(state)
            for tc in pending_tools:
                name = tc["name"]
                args = tc["args"]
                action = self.interrupt_handler.should_interrupt(name, args)

                if action == PermissionAction.DENY:
                    # Auto-reject — inject rejection and resume automatically
                    await self.graph.ainvoke(
                        Command(resume={"decision": "rejected", "tool_call_id": tc["id"]}),
                        config=config,
                    )
                    yield {
                        "type": "tool_start",
                        "name": name,
                        "input": _format_tool_input(args),
                    }
                    yield {
                        "type": "tool_end",
                        "name": name,
                        "output": f"Tool '{name}' was denied by permission rules.",
                    }

                elif action == PermissionAction.ASK:
                    # Take snapshot before dangerous operations
                    if name in DANGEROUS_TOOLS:
                        self.snapshot.track(f"pre-{name}")

                    had_approval = True
                    yield {
                        "type": "approval_needed",
                        "tool": name,
                        "input": _format_tool_input(args),
                        "tool_call_id": tc["id"],
                    }

                # action == ALLOW: the resume() call with approval will let it through

        if not had_approval and is_interrupted and not approve_all:
            # All pending tools are auto-allowed — resume the graph silently
            await self.graph.ainvoke(
                Command(resume={"decision": "approved_all"}),
                config=config,
            )

        # Persist approval state for future turns
        self.permission.approved.update(self.interrupt_handler.permission.approved)
        self.permission.denied.update(self.interrupt_handler.permission.denied)

        yield {"type": "done"}

    async def resume(
        self, thread_id: str, approved: bool, tool_call_id: str,
        remember: bool = False,
    ) -> AsyncIterator[dict]:
        """Resume agent execution after HITL approval decision.

        Sends Command(resume=...) to LangGraph so the interrupted tools node
        proceeds with the user's decision.

        Args:
            thread_id: LangGraph thread ID
            approved: Whether the user approved the tool call
            tool_call_id: The tool call ID to approve/reject
            remember: Whether to save this decision permanently
        """
        await self._ensure_graph()

        config = {"configurable": {"thread_id": thread_id}}
        decision = "approved" if approved else "rejected"

        # Record the decision if user chose "remember"
        if remember and approved:
            state = await self.graph.aget_state(config)
            pending_tools = _extract_pending_tool_calls(state)
            for tc in pending_tools:
                if tc["id"] == tool_call_id:
                    filepath = ""
                    for key in ("path", "file_path", "command"):
                        if key in tc["args"]:
                            filepath = str(tc["args"][key])[:200]
                            break
                    self.permission.approve(tc["name"], filepath, remember=True)
                    break

        cmd = Command(resume={"decision": decision, "tool_call_id": tool_call_id})

        last_tool_name = None

        async for event in self.graph.astream_events(
            cmd, config=config, version="v2",
        ):
            kind = event.get("event", "")

            if kind == "on_chat_model_stream":
                chunk = event["data"]["chunk"]
                delta = getattr(chunk, "content", "")
                if delta:
                    yield {"type": "token", "content": delta}

            elif kind == "on_tool_start":
                name = event.get("name", "unknown")
                inp = _safe_tool_input(event["data"].get("input"))
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

        # After resume completes, check for further interrupts (multi-tool turns)
        state = await self.graph.aget_state(config)
        is_interrupted = bool(state and state.next and "tools" in state.next)
        if is_interrupted:
            # Auto-resume with approval for remaining auto-allowed tools
            await self.graph.ainvoke(
                Command(resume={"decision": "approved_all"}),
                config=config,
            )

        yield {"type": "done"}

    def invoke(self, user_message: str, thread_id: str = "default") -> dict:
        """Synchronous invoke — returns final state (no HITL)."""
        config = {"configurable": {"thread_id": thread_id}}
        self.snapshot.track(f"pre-invoke-{thread_id}")
        return asyncio.run(self.graph.ainvoke(
            {"messages": [{"role": "user", "content": user_message}]},
            config=config,
        ))

    def get_state(self, thread_id: str = "default"):
        """Get LangGraph state for a thread."""
        config = {"configurable": {"thread_id": thread_id}}
        return asyncio.run(self.graph.aget_state(config))

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
        """Switch agent mode. Graph rebuilds lazily on next astream()."""
        if mode not in AGENT_MODES:
            raise ValueError(f"Unknown agent mode: {mode}. Use {list(AGENT_MODES.keys())}")
        self.agent_mode = mode
        self.mode_config = AGENT_MODES[mode]
        self.permission = build_default_permissions(mode, self.workspace)
        self.graph = None  # invalidate — rebuilds lazily

    def set_workspace(self, workspace: str):
        """Change workspace. Graph rebuilds lazily on next astream()."""
        self.workspace = str(Path(workspace).resolve())
        self.snapshot = SnapshotStore(self.workspace)
        self.permission = build_default_permissions(self.agent_mode, self.workspace)
        self.graph = None  # invalidate — rebuilds lazily

    def set_model(self, model_config: ModelConfig):
        """Change model. Graph rebuilds lazily on next astream()."""
        self.model_config = model_config
        self.graph = None  # invalidate — rebuilds lazily

    def list_snapshots(self, limit: int = 10) -> list[dict]:
        """List recent filesystem snapshots."""
        return self.snapshot.list_snapshots(limit)

    def restore_snapshot(self, snapshot_hash: str) -> bool:
        """Restore to a specific snapshot."""
        return self.snapshot.restore(snapshot_hash)

    # ── Background Task Support ──

    def start_background_task(self, task_id: str, prompt: str):
        """Start a background sub-agent task. Returns immediately.

        Handles both async (event loop available) and sync (no event loop) contexts.
        """
        task = BackgroundTask(task_id, prompt, self)
        self._bg_tasks[task_id] = task
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(task.run())
        except RuntimeError:
            # No running event loop — use a daemon thread
            t = threading.Thread(target=lambda: asyncio.run(task.run()), daemon=True)
            t.start()
        return task

    def get_background_task(self, task_id: str) -> Optional["BackgroundTask"]:
        return self._bg_tasks.get(task_id)

    def list_background_tasks(self) -> list[dict]:
        return [t.status() for t in self._bg_tasks.values()]

    async def close(self):
        """Release resources (checkpointer connections)."""
        if self._checkpointer_ctx is not None:
            try:
                await self._checkpointer_ctx.__aexit__(None, None, None)
            except Exception:
                pass
            self._checkpointer_ctx = None
        elif self._checkpointer is not None and hasattr(self._checkpointer, "close"):
            try:
                self._checkpointer.close()
            except Exception:
                pass
        self._checkpointer = None


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
                if event["type"] == "approval_needed":
                    # Auto-approve background tasks
                    async for _ in self.agent.resume(
                        f"bg-{self.task_id}", approved=True,
                        tool_call_id=event["tool_call_id"],
                    ):
                        pass
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


# ── Helpers ──

def _extract_pending_tool_calls(state) -> list[dict]:
    """Extract pending tool call info from an interrupted LangGraph state."""
    if not state or not state.values:
        return []
    messages = state.values.get("messages", [])
    if not messages:
        return []
    last_msg = messages[-1]
    tool_calls = getattr(last_msg, "tool_calls", None)
    if not tool_calls:
        # May be a list of dicts or structured objects
        additional = getattr(last_msg, "additional_kwargs", {})
        raw_calls = additional.get("tool_calls", [])
        tool_calls = []
        for rc in raw_calls:
            fn = rc.get("function", {})
            try:
                args = json.loads(fn.get("arguments", "{}"))
            except (json.JSONDecodeError, TypeError):
                args = {}
            tool_calls.append({
                "id": rc.get("id", ""),
                "name": fn.get("name", "unknown"),
                "args": args,
            })
    result = []
    for tc in tool_calls:
        if isinstance(tc, dict):
            result.append({
                "id": tc.get("id", ""),
                "name": tc.get("name", "unknown"),
                "args": tc.get("args", {}),
            })
        else:
            result.append({
                "id": getattr(tc, "id", ""),
                "name": getattr(tc, "name", "unknown"),
                "args": getattr(tc, "args", {}),
            })
    return result


def _safe_tool_input(inp) -> dict:
    """Normalize tool input to a dict, handling non-dict types."""
    if isinstance(inp, dict):
        return inp
    if isinstance(inp, str):
        try:
            return json.loads(inp)
        except (json.JSONDecodeError, TypeError):
            return {"input": inp}
    if inp is None:
        return {}
    return {"input": str(inp)}


def _format_tool_input(inp: dict, max_len: int = 500) -> str:
    """Format tool input dict for display, truncating long values."""
    safe = {}
    for k, v in inp.items():
        s = str(v)
        if len(s) > max_len:
            s = s[:max_len] + f"... ({len(s)} total chars)"
        safe[k] = s
    return json.dumps(safe, indent=2, ensure_ascii=False)
