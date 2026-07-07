"""Gradio UI for OpenCode DeepAgents — powered by deepagents and LangGraph."""

import json
from pathlib import Path
from typing import Optional

import gradio as gr

from src.agent import CodingAgent
from src.config import (
    load_model_config, ModelConfig, AGENT_MODES,
    DEFAULT_AGENT_MODE, DEFAULT_WORKSPACE,
)
from src.session import (
    create_session, list_sessions, get_session, delete_session,
    get_thread_id, update_session,
)

CSS = """
:root {
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
.gradio-container { max-width: 1400px !important; margin: 0 auto !important; }
.main-container { display: flex; height: calc(100vh - 60px); }
.sidebar {
    width: 320px; background: var(--bg-secondary);
    border-right: 1px solid var(--border-color);
    padding: 16px; overflow-y: auto;
}
.chat-area { flex: 1; display: flex; flex-direction: column; }
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
.tool-output {
    background: var(--bg-tertiary); border-radius: 6px;
    padding: 10px; margin: 6px 0; max-height: 300px; overflow-y: auto;
    font-family: 'SF Mono', 'Monaco', 'Menlo', monospace; font-size: 0.85em;
    white-space: pre-wrap; word-break: break-word;
}
.mode-badge {
    display: inline-block; padding: 2px 8px; border-radius: 4px;
    font-size: 0.75em; font-weight: bold; margin-left: 8px;
}
.mode-build { background: var(--accent-green); color: #000; }
.mode-plan { background: var(--accent-orange); color: #000; }
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
    """Gradio application wrapping deepagents CodingAgent."""

    def __init__(self):
        self.agent: Optional[CodingAgent] = None
        self.current_session_id: Optional[str] = None

    def build(self) -> gr.Blocks:
        with gr.Blocks(theme=THEME, css=CSS, title="OpenCode DeepAgents", fill_height=True) as app:
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
            '<h1>🤖 OpenCode DeepAgents</h1>'
            '<span style="color: var(--text-secondary); font-size: 0.85em">'
            'Powered by deepagents · LangGraph · Gradio</span>'
            '</div>'
        )

    def _build_sidebar(self):
        gr.Markdown("### 📁 Sessions")
        self.session_list = gr.Dropdown(
            choices=[], value=None, label="Current Session",
            interactive=True, allow_custom_value=False,
        )
        with gr.Row():
            self.new_session_btn = gr.Button("+ New", size="sm", variant="secondary")
            self.delete_session_btn = gr.Button("🗑 Delete", size="sm", variant="stop")
        self.reload_sessions_btn = gr.Button("↻ Refresh", size="sm")

        gr.Markdown("---")
        gr.Markdown("### ⚙️ Settings")

        self.agent_mode = gr.Dropdown(
            choices=["build", "plan"],
            value=DEFAULT_AGENT_MODE,
            label="Agent Mode",
            info="build: code + shell | plan: read-only analysis",
        )

        self.model_provider = gr.Dropdown(
            choices=["openai", "anthropic", "ollama"],
            value="openai", label="Provider",
        )
        self.model_name = gr.Textbox(
            value="gpt-4o", label="Model", placeholder="gpt-4o, claude-sonnet-4-20250514..."
        )
        self.api_key = gr.Textbox(
            label="API Key", type="password",
            placeholder="sk-... (uses env var if empty)",
        )
        self.api_base = gr.Textbox(
            label="Base URL (optional)",
            placeholder="https://api.openai.com/v1",
        )
        self.workspace_path = gr.Textbox(
            value=DEFAULT_WORKSPACE, label="Workspace",
        )

        with gr.Row():
            self.apply_settings_btn = gr.Button("Apply", variant="primary", size="sm")

        gr.Markdown("---")
        gr.Markdown("### 📂 Files")
        with gr.Accordion("Workspace Browser", open=False):
            self.file_explorer = gr.FileExplorer(
                root_dir=DEFAULT_WORKSPACE, label="", file_count="multiple",
            )

    def _build_chat(self):
        self.chatbot = gr.Chatbot(
            label="", height=600, type="messages",
            bubble_full_width=False,
            render_markdown=True,
        )
        with gr.Row():
            self.msg_input = gr.Textbox(
                placeholder="Ask anything...",
                scale=9, lines=2, max_lines=10,
                show_label=False, container=False,
            )
            self.send_btn = gr.Button("▶ Send", scale=1, variant="primary")
        self._agent_state = gr.State({"mode": DEFAULT_AGENT_MODE, "workspace": DEFAULT_WORKSPACE})

    def _wire_events(self):
        self._wire_session_events()
        self._wire_chat_events()
        self._wire_settings_events()

    def _wire_session_events(self):
        self.new_session_btn.click(
            fn=self._on_new_session,
            outputs=[self.session_list, self.chatbot],
        ).then(fn=self._on_refresh_sessions, outputs=[self.session_list])

        self.delete_session_btn.click(
            fn=self._on_delete_session,
            inputs=[self.session_list],
            outputs=[self.session_list, self.chatbot],
        )

        self.reload_sessions_btn.click(
            fn=self._on_refresh_sessions,
            outputs=[self.session_list],
        )

        self.session_list.change(
            fn=self._on_session_change,
            inputs=[self.session_list],
            outputs=[self.agent_mode, self.workspace_path, self.model_name],
        )

    def _wire_chat_events(self):
        self.send_btn.click(
            fn=self._on_user_message,
            inputs=[self.msg_input, self.chatbot, self.session_list,
                    self.agent_mode, self.workspace_path,
                    self.model_provider, self.model_name, self.api_key, self.api_base],
            outputs=[self.msg_input, self.chatbot],
            queue=True,
        ).then(fn=self._on_refresh_sessions, outputs=[self.session_list])

        self.msg_input.submit(
            fn=self._on_user_message,
            inputs=[self.msg_input, self.chatbot, self.session_list,
                    self.agent_mode, self.workspace_path,
                    self.model_provider, self.model_name, self.api_key, self.api_base],
            outputs=[self.msg_input, self.chatbot],
            queue=True,
        ).then(fn=self._on_refresh_sessions, outputs=[self.session_list])

    def _wire_settings_events(self):
        self.apply_settings_btn.click(
            fn=self._on_apply_settings,
            inputs=[self.model_provider, self.model_name, self.api_key, self.api_base,
                    self.agent_mode, self.workspace_path, self.session_list],
            outputs=[self.chatbot],
        )

    # ── Session Handlers ──

    def _on_refresh_sessions(self):
        sessions = list_sessions()
        choices = [
            (f"{s['name']} [{s['agent_mode']}] ({s['id'][:8]}...)", s["id"])
            for s in sessions
        ]
        return gr.update(choices=choices)

    def _on_new_session(self):
        sid = create_session()
        sessions = list_sessions()
        choices = [
            (f"{s['name']} [{s['agent_mode']}] ({s['id'][:8]}...)", s["id"])
            for s in sessions
        ]
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
            return DEFAULT_AGENT_MODE, DEFAULT_WORKSPACE, "gpt-4o"
        session = get_session(session_id)
        if not session:
            return DEFAULT_AGENT_MODE, DEFAULT_WORKSPACE, "gpt-4o"
        self.current_session_id = session_id
        # Reset agent so next message picks up session config
        self.agent = None
        return (
            session.get("agent_mode", DEFAULT_AGENT_MODE),
            session.get("workspace", DEFAULT_WORKSPACE),
            session.get("model", "gpt-4o"),
        )

    # ── Chat Handlers ──

    async def _on_user_message(
        self, message, history, session_id,
        agent_mode, workspace, model_provider, model_name, api_key, api_base,
    ):
        if not message or not message.strip():
            yield "", history
            return

        # Ensure session
        if not session_id:
            session_id = create_session(name=message.strip()[:40])
            self.current_session_id = session_id
        else:
            self.current_session_id = session_id

        # Build model config
        model_config = self._build_model_config(model_provider, model_name, api_key, api_base)

        # Get or create agent
        ws_resolved = str(Path(workspace).resolve())
        if self.agent is None:
            self.agent = CodingAgent(
                model_config=model_config, workspace=workspace, agent_mode=agent_mode,
            )
        elif self.agent.agent_mode != agent_mode or self.agent.workspace != ws_resolved:
            self.agent = CodingAgent(
                model_config=model_config, workspace=workspace, agent_mode=agent_mode,
            )

        # Add user message
        history.append({"role": "user", "content": message})
        yield "", history

        # Stream response
        parts = []  # Ordered (type, data) tuples
        token_buf = ""

        try:
            thread_id = get_thread_id(session_id) or f"session-{session_id}"
            subagent_active = False

            async for event in self.agent.astream(message, thread_id=thread_id):
                etype = event["type"]

                if etype == "token":
                    token_buf += event.get("content", "")

                elif etype == "tool_start":
                    if token_buf:
                        parts.append(("text", token_buf))
                        token_buf = ""
                    name = event["name"]
                    if name == "task":
                        subagent_active = True
                    parts.append(("tool_start", {
                        "name": name, "input": event.get("input", ""),
                        "subagent": name == "task",
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
                if history and history[-1]["role"] == "assistant":
                    history[-1]["content"] = rendered
                else:
                    history.append({"role": "assistant", "content": rendered})
                yield "", history

            # Finalize
            rendered = self._render_parts(parts, "")
            if history and history[-1]["role"] == "assistant":
                history[-1]["content"] = rendered
            elif parts:
                history.append({"role": "assistant", "content": rendered})

            # Update session metadata
            update_session(session_id, agent_mode=agent_mode, workspace=workspace, model=model_name)

        except Exception as e:
            error_msg = f"❌ Error: {e}"
            history.append({"role": "assistant", "content": error_msg})

        yield "", history

    def _build_model_config(self, provider, model_name, api_key, api_base):
        config = load_model_config()
        if provider:
            config.provider = provider
        if model_name:
            config.model = model_name
        if api_key:
            config.api_key = api_key
        if api_base:
            config.base_url = api_base
        return config

    def _render_parts(self, parts: list, current_token: str = "") -> str:
        """Render streaming parts into HTML for the Gradio Chatbot."""
        result = []
        open_tool = None

        for ptype, pdata in parts:
            if ptype == "text":
                if open_tool:
                    result.append("</div>")
                    open_tool = None
                result.append(pdata)

            elif ptype == "tool_start":
                if open_tool:
                    result.append("</div>")
                cls = "tool-msg subagent" if pdata.get("subagent") else "tool-msg"
                icon = "🤖" if pdata.get("subagent") else "🔧"
                result.append(
                    f'<div class="{cls}">'
                    f'<span class="name">{icon} {pdata["name"]}</span>'
                    f'<div class="tool-output">{pdata["input"][:500]}</div>'
                )
                open_tool = pdata["name"]

            elif ptype == "tool_end":
                result.append(
                    f'<div class="tool-output" style="max-height:200px">'
                    f'{pdata["output"][:800]}</div>'
                )

        if open_tool:
            result.append("</div>")

        if current_token:
            result.append(current_token)

        return "\n".join(result)

    # ── Settings Handlers ──

    def _on_apply_settings(self, provider, model_name, api_key, api_base, agent_mode, workspace, session_id):
        config = self._build_model_config(provider, model_name, api_key, api_base)
        self.agent = CodingAgent(
            model_config=config, workspace=workspace, agent_mode=agent_mode,
        )
        if session_id:
            update_session(session_id, model=model_name, agent_mode=agent_mode, workspace=workspace)

        mode_badge = '<span class="mode-badge mode-build">BUILD</span>' if agent_mode == "build" else '<span class="mode-badge mode-plan">PLAN</span>'
        return [{
            "role": "assistant",
            "content": f"✅ Settings applied{mode_badge}\n\n"
                      f"- Mode: **{agent_mode}**\n"
                      f"- Model: **{model_name}**\n"
                      f"- Workspace: `{workspace}`\n"
                      f"- Backend: {'LocalShellBackend' if agent_mode == 'build' else 'FilesystemBackend (read-only)'}"
        }]


def create_app() -> gr.Blocks:
    return GradioApp().build()
