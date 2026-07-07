"""Deep Agents-based AI coding agent — built on deepagents' harness."""

from pathlib import Path
from typing import Optional, AsyncIterator

from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend, LocalShellBackend
from deepagents.middleware.subagents import SubAgent

from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver

from src.config import (
    ModelConfig, AgentModeConfig, AGENT_MODES,
    load_model_config, PLAN_SUBAGENT_PROMPT,
    DEFAULT_WORKSPACE, DEFAULT_AGENT_MODE, MAX_TOOL_ITERATIONS,
)
from src.tools import get_custom_tools


class CodingAgent:
    """AI coding agent powered by deepagents.

    Uses deepagents' built-in harness for:
    - Filesystem tools (ls, read_file, write_file, edit_file, glob, grep)
    - Shell execution (via LocalShellBackend)
    - Sub-agent delegation (task tool)
    - Todo list management
    - Context summarization for long conversations

    Two modes:
    - **build**: full filesystem + shell access
    - **plan**: read-only filesystem, no shell, analysis-only
    """

    def __init__(
        self,
        model_config: Optional[ModelConfig] = None,
        workspace: str = DEFAULT_WORKSPACE,
        agent_mode: str = DEFAULT_AGENT_MODE,
        checkpoint_db_path: Optional[str] = None,
    ):
        self.model_config = model_config or load_model_config()
        self.workspace = str(Path(workspace).resolve())
        self.agent_mode = agent_mode
        self.mode_config = AGENT_MODES.get(agent_mode, AGENT_MODES["build"])
        self.checkpoint_db_path = checkpoint_db_path or ":memory:"
        self.graph = self._build_graph()

    def _build_graph(self):
        """Build the deep agent with mode-appropriate configuration."""
        mode = self.mode_config
        tools = get_custom_tools()  # web_fetch, web_search — deepagents provides the rest

        # ── Filesystem Backend ──
        if mode.allow_shell:
            backend = LocalShellBackend(root_dir=self.workspace)
        else:
            backend = FilesystemBackend(root_dir=self.workspace)

        # ── Sub-agents ──
        sub_agents = [
            SubAgent(
                name="plan-analyze",
                description=(
                    "Deep analysis sub-agent. Use for complex codebase analysis, "
                    "architecture review, researching approaches, or generating detailed plans. "
                    "This sub-agent is read-only and cannot modify files."
                ),
                system_prompt=PLAN_SUBAGENT_PROMPT,
                tools=tools,  # web tools only, no file write/execute
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

        # ── Create the deep agent ──
        system_prompt = mode.format_prompt(self.workspace)

        agent = create_deep_agent(
            model=self.model_config.to_model_string(),
            tools=tools,
            system_prompt=system_prompt,
            sub_agents=sub_agents,
            backend=backend,
            checkpointer=checkpointer,
        )

        return agent

    async def astream(
        self, user_message: str, thread_id: str = "default"
    ) -> AsyncIterator[dict]:
        """Stream agent execution with typed events.

        Yields:
            {"type": "token", "content": "..."}   — streaming text chunk
            {"type": "tool_start", "name": "...", "input": "..."}  — tool invocation beginning
            {"type": "tool_end", "name": "...", "output": "..."}   — tool result
            {"type": "done"}                                       — completion signal
        """
        config = {"configurable": {"thread_id": thread_id}}

        last_tool_name = None
        current_content = []

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
                    current_content.append(delta)
                    yield {"type": "token", "content": delta}

            elif kind == "on_tool_start":
                name = event.get("name", "unknown")
                inp = event["data"].get("input", {})
                safe_input = _format_tool_input(inp)
                last_tool_name = name
                yield {
                    "type": "tool_start",
                    "name": name,
                    "input": safe_input,
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
        return self.graph.invoke(
            {"messages": [{"role": "user", "content": user_message}]},
            config=config,
        )

    def get_state(self, thread_id: str = "default"):
        """Get LangGraph state for a thread."""
        config = {"configurable": {"thread_id": thread_id}}
        return self.graph.get_state(config)

    def set_mode(self, mode: str):
        """Switch agent mode and rebuild. State resets per new graph."""
        if mode not in AGENT_MODES:
            raise ValueError(f"Unknown agent mode: {mode}. Use {list(AGENT_MODES.keys())}")
        self.agent_mode = mode
        self.mode_config = AGENT_MODES[mode]
        self.graph = self._build_graph()

    def set_workspace(self, workspace: str):
        """Change workspace and rebuild."""
        self.workspace = str(Path(workspace).resolve())
        self.graph = self._build_graph()

    def set_model(self, model_config: ModelConfig):
        """Change model and rebuild."""
        self.model_config = model_config
        self.graph = self._build_graph()


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
