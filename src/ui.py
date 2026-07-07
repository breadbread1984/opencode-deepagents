"""Gradio UI for OpenCode DeepAgents — full opencode feature set.

Supports: HITL permission approvals, snapshot/undo controls, background tasks,
MCP config, question display, LSP tools, and multi-file apply_patch.
"""

import asyncio
from pathlib import Path
from typing import Optional

import gradio as gr

from src.agent import CodingAgent, BackgroundTask
from src.config import (
    load_model_config, ModelConfig, AGENT_MODES,
    DEFAULT_AGENT_MODE, DEFAULT_WORKSPACE,
)
from src.session import (
    create_session, list_sessions, get_session, delete_session,
    get_thread_id, update_session,
)
from src.tools.mcp import load_mcp_configs

CSS = """
::root {
    --bg-primary: #0d1117;
    --bg-secondary: #161b22;
    --bg-tertiary: #21262d;
    --border-color: #30363d;
    --text-primary: #e6edf3;
    --text-secondary: #8b949e;
    --accent: #58a6ff;
    --accent-green: #3fb950;
    --accent-orange: #d2991d;
    --accent-purple: #a371f7;
    --accent-red: #f85149;
}
body { background: var(--bg-primary) !important; }
.gradio-container { max-width: 1600px !important; margin: 0 auto !important; }
.main-container { display: flex; height: calc(100vh - 60px); }
.sidebar {
    width: 340px; background: var(--bg-secondary);
    border-right: 1px solid var(--border-color);
    padding: 16px; overflow-y: auto;
}
.chat-area { flex: 1; display: flex; flex-direction: column; min-width: 0; }
footer { display: none !important; }
.header {
    background: var(--bg-secondary); border-bottom: 1px solid var(--border-color);
    padding: 12px 20px; display: flex; align-items: center; justify-content: space-between;
}
.header h1 { margin: 0; font-size: 18px; color: var(--text-primary); }
.tool-msg {
    border-left: 3px solid var(--accent-orange);
    padding-left: 12px; margin: 8px 0;
    color: var(--text-secondary); font-size: 0.9em;
}
.tool-msg .name { color: var(--accent-orange); font-weight: bold; }
.tool-msg.subagent { border-left-color: var(--accent-purple); }
.tool-msg.subagent .name { color: var(--accent-purple); }
.tool-msg.patch { border-left-color: var(--accent-green); }
.tool-msg.patch .name { color: var(--accent-green); }
.tool-msg.lsp { border-left-color: var(--accent); }
.tool-msg.lsp .name { color: var(--accent); }
.tool-output {
    background: var(--bg-tertiary); border-radius: 6px;
    padding: 10px; margin: 6px 0; max-height: 300px; overflow-y: auto;
    font-family: 'SF Mono', 'Monaco', 'Menlo', monospace; font-size: 0.85em;
    white-space: pre-wrap; word-break: break-word;
}
.approval-box {
    border: 2px solid var(--accent-orange);
    background: rgba(210, 153, 29, 0.1);
    border-radius: 8px; padding: 16px; margin: 12px 0;
}
.approval-box .tool-name { font-weight: bold; color: var(--accent-orange); font-size: 1.1em; }
.approval-box .btn-row { display: flex; gap: 8px; margin-top: 12px; }
.approve-btn { background: var(--accent-green) !important; color: #000 !important; }
.reject-btn { background: var(--accent-red) !important; color: #fff !important; }
.remember-btn { background: var(--accent) !important; color: #000 !important; }
.mode-badge {
    display: inline-block; padding: 2px 8px; border-radius: 4px;
    font-size: 0.75em; font-weight: bold; margin-left: 8px;
}
.mode-build { background: var(--accent-green); color: #000; }
.mode-plan { background: var(--accent-orange); color: #000; }
.snapshot-list { font-size: 0.85em; max-height: 200px; overflow-y: auto; }
"""

THEME = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="slate",
    neutral_hue="slate",
).set(
    body_background_fill="*neutral_950",
    block_background_fill="*neutral_900",
    block_border_color="*neutral_700",
    input_background_fill="*neutral_800",
    button_primary_background_fill="*primary_500",
    button_primary_background_fill_hover="*primary_400",
)


class GradioApp:
    """Gradio application wrapping deepagents CodingAgent with full HITL support."""

    def __init__(self):
        self.agent: Optional[CodingAgent] = None
        self.current_session_id: Optional[str] = None
        self._pending_thread_id: Optional[str] = None
        self._pending_tool_call_id: Optional[str] = None
        self._pending_tool_name: Optional[str] = None
        self._submit_lock = False

    def build(self) -> gr.Blocks:
        with gr.Blocks(title="OpenCode DeepAgents", fill_height=True) as app:
            # Hidden state for HITL approval tracking
            self._hitl_state = gr.State({
                "pending": False, "tool": "", "input": "",
                "tool_call_id": "", "thread_id": "",
            })

            self._build_header()
            with gr.Row(equal_height=False, elem_classes="main-container"):
                with gr.Column(scale=1, elem_classes="sidebar"):
                    self._build_sidebar()
                with gr.Column(scale=3, elem_classes="chat-area"):
                    self._build_chat()
            self._wire_events()
        return app

    def _build_header(self):
        gr.HTML(
            '<div class="header">'
            '<h1>OpenCode DeepAgents</h1>'
            '<span style="color: var(--text-secondary); font-size: 0.85em">'
            'Powered by deepagents · LangGraph · Gradio</span>'
            '</div>'
        )

    def _build_sidebar(self):
        gr.Markdown("### Sessions")
        self.session_list = gr.Dropdown(
            choices=[], value=None, label="Current Session",
            interactive=True, allow_custom_value=False,
        )
        with gr.Row():
            self.new_session_btn = gr.Button("+ New", size="sm", variant="secondary")
            self.delete_session_btn = gr.Button("Delete", size="sm", variant="stop")
        self.reload_sessions_btn = gr.Button("Refresh", size="sm")

        gr.Markdown("---")
        gr.Markdown("### Settings")

        self.agent_mode = gr.Dropdown(
            choices=["build", "plan"], value=DEFAULT_AGENT_MODE, label="Agent Mode",
        )
        self.model_provider = gr.Dropdown(
            choices=["dashscope", "openai", "anthropic", "ollama"], value="dashscope", label="Provider",
        )
        self.model_name = gr.Textbox(value="qwen3.6-plus", label="Model")
        self.api_key = gr.Textbox(label="API Key", type="password", placeholder="sk-...")
        self.api_base = gr.Textbox(label="Base URL (optional)", placeholder="https://dashscope-intl.aliyuncs.com/compatible-mode/v1")
        self.workspace_path = gr.Textbox(value=DEFAULT_WORKSPACE, label="Workspace")
        self.approve_all_cb = gr.Checkbox(value=False, label="Auto-approve all tools (skip HITL)")
        with gr.Row():
            self.apply_settings_btn = gr.Button("Apply Settings", variant="primary", size="sm")

        gr.Markdown("---")
        gr.Markdown("### Undo & Snapshots")
        with gr.Row():
            self.undo_btn = gr.Button("Undo Last", size="sm", variant="secondary")
            self.snapshot_list_btn = gr.Button("List Snapshots", size="sm")
        self.snapshot_display = gr.HTML(value="", elem_classes="snapshot-list")

        gr.Markdown("---")
        gr.Markdown("### MCP Servers")
        self.mcp_status = gr.HTML(value=self._render_mcp_status())
        self.refresh_mcp_btn = gr.Button("Refresh MCP", size="sm")

        gr.Markdown("---")
        gr.Markdown("### Files")
        with gr.Accordion("Workspace Browser", open=False):
            self.file_explorer = gr.FileExplorer(
                root_dir=DEFAULT_WORKSPACE, label="", file_count="multiple",
            )

    def _build_chat(self):
        self.chatbot = gr.Chatbot(
            label="", height=600,
        )
        with gr.Row():
            self.msg_input = gr.Textbox(
                placeholder="Ask anything... (type /help for commands)",
                scale=9, lines=2, max_lines=10,
                show_label=False, container=False,
            )
            self.send_btn = gr.Button("Send", scale=1, variant="primary")
            self.approve_btn = gr.Button("Approve", scale=1, variant="primary", visible=False)
            self.reject_btn = gr.Button("Reject", scale=1, variant="stop", visible=False)
            self.remember_btn = gr.Button("Always Allow", scale=1, variant="secondary", visible=False)

    def _wire_events(self):
        # Sessions
        self.new_session_btn.click(fn=self._on_new_session, outputs=[self.session_list, self.chatbot])
        self.new_session_btn.click(fn=self._on_refresh_sessions, outputs=[self.session_list])
        self.delete_session_btn.click(fn=self._on_delete_session, inputs=[self.session_list],
                                       outputs=[self.session_list, self.chatbot])
        self.reload_sessions_btn.click(fn=self._on_refresh_sessions, outputs=[self.session_list])
        self.session_list.change(fn=self._on_session_change, inputs=[self.session_list],
                                  outputs=[self.agent_mode, self.workspace_path, self.model_name])

        # Chat — only one event source triggers send (use send_btn, not both)
        chat_inputs = [self.msg_input, self.chatbot, self.session_list,
                        self.agent_mode, self.workspace_path,
                        self.model_provider, self.model_name, self.api_key, self.api_base,
                        self.approve_all_cb]
        self.send_btn.click(fn=self._on_user_message, inputs=chat_inputs,
                            outputs=[self.msg_input, self.chatbot, self.approve_btn,
                                     self.reject_btn, self.remember_btn],
                            queue=True)
        self.msg_input.submit(fn=self._on_user_message, inputs=chat_inputs,
                              outputs=[self.msg_input, self.chatbot, self.approve_btn,
                                       self.reject_btn, self.remember_btn],
                              queue=True)

        # HITL approval buttons
        self.approve_btn.click(
            fn=self._on_approve,
            inputs=[self.chatbot, self.session_list, self._hitl_state, self.approve_all_cb],
            outputs=[self.chatbot, self.approve_btn, self.reject_btn, self.remember_btn],
            queue=True,
        )
        self.reject_btn.click(
            fn=self._on_reject,
            inputs=[self.chatbot, self.session_list, self._hitl_state],
            outputs=[self.chatbot, self.approve_btn, self.reject_btn, self.remember_btn],
            queue=True,
        )
        self.remember_btn.click(
            fn=self._on_remember_approve,
            inputs=[self.chatbot, self.session_list, self._hitl_state],
            outputs=[self.chatbot, self.approve_btn, self.reject_btn, self.remember_btn],
            queue=True,
        )

        # Settings
        self.apply_settings_btn.click(fn=self._on_apply_settings,
            inputs=[self.model_provider, self.model_name, self.api_key, self.api_base,
                    self.agent_mode, self.workspace_path, self.session_list, self.approve_all_cb],
            outputs=[self.chatbot])

        # Undo & Snapshots
        self.undo_btn.click(fn=self._on_undo, inputs=[self.session_list],
                            outputs=[self.chatbot, self.snapshot_display])
        self.snapshot_list_btn.click(fn=self._on_list_snapshots, outputs=[self.snapshot_display])
        self.refresh_mcp_btn.click(fn=lambda: self._render_mcp_status(), outputs=[self.mcp_status])

    # ── Session Handlers ──

    def _on_refresh_sessions(self):
        sessions = list_sessions()
        choices = [(f"{s['name']} [{s['agent_mode']}] ({s['id'][:8]})", s["id"]) for s in sessions]
        return gr.update(choices=choices)

    def _on_new_session(self):
        sid = create_session()
        sessions = list_sessions()
        choices = [(f"{s['name']} [{s['agent_mode']}] ({s['id'][:8]})", s["id"]) for s in sessions]
        return gr.update(choices=choices, value=sid), []

    def _on_delete_session(self, session_id: str):
        if not session_id:
            return self._on_refresh_sessions(), []
        delete_session(session_id)
        self.current_session_id = None
        self.agent = None
        return self._on_refresh_sessions(), []

    def _on_session_change(self, session_id: str):
        if not session_id:
            return DEFAULT_AGENT_MODE, DEFAULT_WORKSPACE, "qwen3.6-plus"
        session = get_session(session_id)
        if not session:
            return DEFAULT_AGENT_MODE, DEFAULT_WORKSPACE, "qwen3.6-plus"
        self.current_session_id = session_id
        self.agent = None
        return (
            session.get("agent_mode", DEFAULT_AGENT_MODE),
            session.get("workspace", DEFAULT_WORKSPACE),
            session.get("model", "qwen3.6-plus"),
        )

    # ── Chat Handlers ──

    async def _on_user_message(
        self, message, history, session_id,
        agent_mode, workspace, model_provider, model_name, api_key, api_base,
        approve_all,
    ):
        if not message or not message.strip():
            yield "", history, gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)
            return

        # Handle /commands
        if message.startswith("/"):
            result = self._handle_command(message, history)
            yield "", result, gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)
            return

        session_id = self._ensure_session(session_id, message)
        model_config = self._build_model_config(model_provider, model_name, api_key, api_base)
        await self._ensure_agent(model_config, workspace, agent_mode)

        history.append({"role": "user", "content": message})
        yield "", history, gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)

        parts = []
        token_buf = ""
        subagent_active = False

        try:
            thread_id = get_thread_id(session_id) or f"session-{session_id}"
            self._pending_thread_id = thread_id

            async for event in self.agent.astream(message, thread_id=thread_id, approve_all=bool(approve_all)):
                etype = event["type"]

                if etype == "token":
                    token_buf += event.get("content", "")

                elif etype == "approval_needed":
                    # HITL: flush token buffer, show approval box, and STOP
                    if token_buf:
                        parts.append(("text", token_buf))
                        token_buf = ""

                    self._pending_tool_call_id = event["tool_call_id"]
                    self._pending_tool_name = event["tool"]

                    parts.append(("approval_needed", {
                        "tool": event["tool"],
                        "input": event["input"],
                        "tool_call_id": event["tool_call_id"],
                    }))

                    rendered = self._render_parts(parts, token_buf)
                    history = self._set_last_assistant(history, rendered)

                    yield "", history, gr.update(visible=True), gr.update(visible=True), gr.update(visible=True)
                    return  # Stop here — wait for user approval

                elif etype == "tool_start":
                    if token_buf:
                        parts.append(("text", token_buf))
                        token_buf = ""
                    name = event["name"]
                    if name == "task":
                        subagent_active = True
                    cls = self._tool_class(name)
                    parts.append(("tool_start", {
                        "name": name, "input": event.get("input", ""),
                        "css_class": cls, "subagent": name == "task",
                    }))

                elif etype == "tool_end":
                    parts.append(("tool_end", {
                        "name": event.get("name", ""),
                        "output": event.get("output", ""),
                        "subagent": subagent_active,
                    }))
                    subagent_active = False

                elif etype == "done":
                    if token_buf:
                        parts.append(("text", token_buf))
                        token_buf = ""

                rendered = self._render_parts(parts, token_buf)
                history = self._set_last_assistant(history, rendered)
                yield "", history, gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)

            rendered = self._render_parts(parts, "")
            history = self._set_last_assistant(history, rendered)
            update_session(session_id, agent_mode=agent_mode, workspace=workspace, model=model_name)

        except Exception as e:
            history.append({"role": "assistant", "content": f"Error: {e}"})

        yield "", history, gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)

    async def _on_approve(self, history, session_id, hitl_state, approve_all):
        """Handle user approval of a pending tool call."""
        if not self.agent or not self._pending_tool_call_id:
            yield history, gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)
            return

        thread_id = get_thread_id(session_id) or f"session-{session_id}"
        parts = self._parse_existing_parts(history)
        token_buf = ""

        try:
            async for event in self.agent.resume(thread_id, approved=True, tool_call_id=self._pending_tool_call_id):
                etype = event["type"]

                if etype == "token":
                    token_buf += event.get("content", "")

                elif etype == "tool_start":
                    if token_buf:
                        parts.append(("text", token_buf))
                        token_buf = ""
                    name = event["name"]
                    cls = self._tool_class(name)
                    parts.append(("tool_start", {
                        "name": name, "input": event.get("input", ""),
                        "css_class": cls, "subagent": name == "task",
                    }))

                elif etype == "tool_end":
                    parts.append(("tool_end", {
                        "name": event.get("name", ""),
                        "output": event.get("output", ""),
                        "subagent": False,
                    }))

                elif etype == "done":
                    if token_buf:
                        parts.append(("text", token_buf))
                        token_buf = ""

                rendered = self._render_parts(parts, token_buf)
                history = self._set_last_assistant(history, rendered)
                yield history, gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)

            rendered = self._render_parts(parts, "")
            history = self._set_last_assistant(history, rendered)

        except Exception as e:
            history.append({"role": "assistant", "content": f"Error: {e}"})

        self._pending_tool_call_id = None
        self._pending_tool_name = None
        yield history, gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)

    async def _on_reject(self, history, session_id, hitl_state):
        """Handle user rejection of a pending tool call."""
        if not self.agent or not self._pending_tool_call_id:
            yield history, gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)
            return

        thread_id = get_thread_id(session_id) or f"session-{session_id}"
        parts = self._parse_existing_parts(history)
        token_buf = ""

        # Replace the approval box with rejection note
        parts.append(("text", f"\n\n*Tool call **{self._pending_tool_name or 'unknown'}** was rejected.*\n"))

        try:
            async for event in self.agent.resume(thread_id, approved=False, tool_call_id=self._pending_tool_call_id):
                etype = event["type"]

                if etype == "token":
                    token_buf += event.get("content", "")

                elif etype == "tool_start":
                    if token_buf:
                        parts.append(("text", token_buf))
                        token_buf = ""
                    name = event["name"]
                    cls = self._tool_class(name)
                    parts.append(("tool_start", {
                        "name": name, "input": event.get("input", ""),
                        "css_class": cls, "subagent": name == "task",
                    }))

                elif etype == "tool_end":
                    parts.append(("tool_end", {
                        "name": event.get("name", ""),
                        "output": event.get("output", ""),
                        "subagent": False,
                    }))

                elif etype == "done":
                    if token_buf:
                        parts.append(("text", token_buf))
                        token_buf = ""

                rendered = self._render_parts(parts, token_buf)
                history = self._set_last_assistant(history, rendered)
                yield history, gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)

            rendered = self._render_parts(parts, "")
            history = self._set_last_assistant(history, rendered)

        except Exception as e:
            history.append({"role": "assistant", "content": f"Error: {e}"})

        self._pending_tool_call_id = None
        self._pending_tool_name = None
        yield history, gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)

    async def _on_remember_approve(self, history, session_id, hitl_state):
        """Approve AND remember this decision permanently."""
        if not self.agent or not self._pending_tool_call_id:
            yield history, gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)
            return

        thread_id = get_thread_id(session_id) or f"session-{session_id}"
        parts = self._parse_existing_parts(history)
        token_buf = ""

        try:
            async for event in self.agent.resume(
                thread_id, approved=True, tool_call_id=self._pending_tool_call_id,
                remember=True,
            ):
                etype = event["type"]

                if etype == "token":
                    token_buf += event.get("content", "")

                elif etype == "tool_start":
                    if token_buf:
                        parts.append(("text", token_buf))
                        token_buf = ""
                    name = event["name"]
                    cls = self._tool_class(name)
                    parts.append(("tool_start", {
                        "name": name, "input": event.get("input", ""),
                        "css_class": cls, "subagent": name == "task",
                    }))

                elif etype == "tool_end":
                    parts.append(("tool_end", {
                        "name": event.get("name", ""),
                        "output": event.get("output", ""),
                        "subagent": False,
                    }))

                elif etype == "done":
                    if token_buf:
                        parts.append(("text", token_buf))
                        token_buf = ""

                rendered = self._render_parts(parts, token_buf)
                history = self._set_last_assistant(history, rendered)
                yield history, gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)

            rendered = self._render_parts(parts, "")
            history = self._set_last_assistant(history, rendered)

        except Exception as e:
            history.append({"role": "assistant", "content": f"Error: {e}"})

        self._pending_tool_call_id = None
        self._pending_tool_name = None
        yield history, gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)

    def _parse_existing_parts(self, history) -> list:
        """Parse the last assistant message back into render parts for continuation."""
        if not history:
            return []
        last = history[-1] if history else None
        if not last or last.get("role") != "assistant":
            return []
        content = last.get("content", "")
        if not content:
            return []
        # Return text part containing the full rendered content as base
        return [("text", content)]

    def _handle_command(self, message: str, history: list) -> list:
        cmd = message.strip().lower()
        resp = ""

        if cmd in ("/help", "/?"):
            resp = """**Available Commands**
- `/undo` — Restore filesystem to before the last action
- `/snapshots` — List recent filesystem snapshots
- `/mode build` — Switch to build mode (code + shell)
- `/mode plan` — Switch to plan mode (read-only analysis)
- `/help` — Show this help message"""
        elif cmd == "/undo":
            if self.agent:
                r = self.agent.undo_last_action()
                resp = f"**Undo**: {r}"
            else:
                resp = "No active agent to undo."
        elif cmd == "/snapshots":
            if self.agent:
                snaps = self.agent.list_snapshots(5)
                if snaps:
                    rows = "\n".join(f"- `{s['hash']}` {s['message']} ({s['date']})" for s in snaps)
                    resp = f"**Snapshots**:\n{rows}"
                else:
                    resp = "No snapshots available."
            else:
                resp = "No active agent."
        elif cmd.startswith("/mode "):
            mode = cmd.split(" ", 1)[1].strip()
            if mode in AGENT_MODES and self.agent:
                self.agent.set_mode(mode)
                resp = f"Switched to **{mode}** mode."
            else:
                resp = f"Unknown mode: {mode}. Use `build` or `plan`."
        else:
            resp = f"Unknown command: {cmd}. Try `/help`."

        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": resp})
        return history

    def _ensure_session(self, session_id: str, message: str) -> str:
        if not session_id:
            session_id = create_session(name=message.strip()[:40])
            self.current_session_id = session_id
        else:
            self.current_session_id = session_id
        return session_id

    async def _ensure_agent(self, model_config, workspace, agent_mode):
        ws_resolved = str(Path(workspace).resolve())
        if self.agent is None:
            self.agent = CodingAgent(model_config=model_config, workspace=workspace, agent_mode=agent_mode)
        elif self.agent.agent_mode != agent_mode or self.agent.workspace != ws_resolved:
            # Close old agent to free DB connections
            await self.agent.close()
            self.agent = CodingAgent(model_config=model_config, workspace=workspace, agent_mode=agent_mode)

    def _tool_class(self, name: str) -> str:
        if name in ("write_file", "edit_file", "apply_patch"):
            return "patch"
        if name.startswith("lsp_"):
            return "lsp"
        if name.startswith("mcp_"):
            return "subagent"
        return ""

    def _render_parts(self, parts: list, current_token: str = "") -> str:
        result = []
        open_tool = None

        for ptype, pdata in parts:
            if ptype == "text":
                if open_tool:
                    result.append("</div>")
                    open_tool = None
                result.append(pdata)

            elif ptype == "approval_needed":
                if open_tool:
                    result.append("</div>")
                    open_tool = None
                tool = pdata["tool"]
                inp = str(pdata["input"])[:1000]
                result.append(
                    f'<div class="approval-box">'
                    f'<span class="tool-name">Approve tool call?</span>'
                    f'<div><strong>{tool}</strong></div>'
                    f'<div class="tool-output">{inp}</div>'
                    f'</div>'
                )

            elif ptype == "tool_start":
                if open_tool:
                    result.append("</div>")
                css_cls = pdata.get("css_class", "")
                cls_parts = ["tool-msg"]
                if pdata.get("subagent"):
                    cls_parts.append("subagent")
                if css_cls:
                    cls_parts.append(css_cls)
                cls = " ".join(cls_parts)

                icons = {"task": "", "code-explorer": "", "plan-analyze": ""}
                icon = icons.get(pdata["name"], "")
                if not icon:
                    icon = "🔧" if not pdata.get("subagent") else ""

                inp = str(pdata["input"])[:500]
                result.append(
                    f'<div class="{cls}">'
                    f'<span class="name">{icon} {pdata["name"]}</span>'
                    f'<div class="tool-output">{inp}</div>'
                )
                open_tool = pdata["name"]

            elif ptype == "tool_end":
                out = str(pdata.get("output", ""))[:800]
                result.append(f'<div class="tool-output" style="max-height:200px">{out}</div>')

        if open_tool:
            result.append("</div>")

        if current_token:
            result.append(current_token)

        return "\n".join(result)

    def _set_last_assistant(self, history, content):
        if history and history[-1].get("role") == "assistant":
            history[-1]["content"] = content
        else:
            history.append({"role": "assistant", "content": content})
        return history

    def _build_model_config(self, provider, model_name, api_key, api_base):
        config = load_model_config()
        if provider: config.provider = provider
        if model_name: config.model = model_name
        if api_key: config.api_key = api_key
        if api_base: config.base_url = api_base
        return config

    # ── Undo & Snapshots ──

    def _on_undo(self, session_id: str):
        if not self.agent:
            return [{"role": "assistant", "content": "No active agent."}], ""
        result = self.agent.undo_last_action()
        snaps = self._format_snapshots()
        return [{"role": "assistant", "content": f"**Undo**: {result}"}], snaps

    def _on_list_snapshots(self):
        return self._format_snapshots()

    def _format_snapshots(self) -> str:
        if not self.agent:
            return "No active agent."
        snaps = self.agent.list_snapshots(10)
        if not snaps:
            return "No snapshots available."
        rows = "<br>".join(
            f"<code>{s['hash']}</code> {s['message']} <span style='color:var(--text-secondary)'>({s['date']})</span>"
            for s in snaps
        )
        return f"<strong>Recent Snapshots</strong><br>{rows}"

    # ── MCP Status ──

    def _render_mcp_status(self) -> str:
        ws = str(Path(self.workspace_path.value).resolve()) if hasattr(self, 'workspace_path') else DEFAULT_WORKSPACE
        configs = load_mcp_configs(ws)
        if not configs:
            return '<span style="color:var(--text-secondary)">No MCP servers configured.</span><br>' \
                   '<small>Add servers to <code>.opencode.json</code> or <code>~/.opencode-mcp.json</code></small>'
        rows = ""
        for c in configs:
            transport_icon = {"stdio": "", "sse": "", "streamable_http": ""}.get(c.transport, "")
            rows += f"<div><strong>{c.name}</strong> <span style='color:var(--text-secondary)'>({c.transport})</span>{transport_icon}</div>"
        return rows

    # ── Settings ──

    def _on_apply_settings(self, provider, model_name, api_key, api_base, agent_mode, workspace, session_id, approve_all):
        config = self._build_model_config(provider, model_name, api_key, api_base)
        if self.agent:
            self.agent.close()
        self.agent = CodingAgent(model_config=config, workspace=workspace, agent_mode=agent_mode)
        if session_id:
            update_session(session_id, model=model_name, agent_mode=agent_mode, workspace=workspace)

        mode_badge = '<span class="mode-badge mode-build">BUILD</span>' if agent_mode == "build" else '<span class="mode-badge mode-plan">PLAN</span>'
        hitl_status = "Off (auto-approve)" if approve_all else "On (ask before dangerous tools)"
        return [{
            "role": "assistant",
            "content": f"Settings applied{mode_badge}\n\n"
                      f"- Mode: **{agent_mode}**\n"
                      f"- Model: **{model_name}**\n"
                      f"- Workspace: `{workspace}`\n"
                      f"- HITL: **{hitl_status}**\n"
                      f"- Backend: {'LocalShellBackend' if agent_mode == 'build' else 'FilesystemBackend (read-only)'}"
        }]


def create_app() -> gr.Blocks:
    return GradioApp().build()
