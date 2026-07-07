# OpenCode DeepAgents

AI Coding Agent built with **[deepagents](https://github.com/langchain-ai/deepagents)** (LangChain's production-ready agent harness), **LangGraph**, and **Gradio**.

A Python reimplementation of the [OpenCode](https://github.com/anomalyco/opencode) AI coding assistant, leveraging `deepagents` for its built-in sub-agent delegation, filesystem tools, shell execution, todo management, and context summarization.

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│                     Gradio UI                             │
│  ┌──────────┐  ┌─────────────┐  ┌───────────────────┐   │
│  │ Sessions │  │    Chat     │  │  Settings & Files  │   │
│  └──────────┘  └─────────────┘  └───────────────────┘   │
├──────────────────────────────────────────────────────────┤
│              deepagents Harness                           │
│  ┌──────────────────────────────────────────────────┐    │
│  │  create_deep_agent(                              │    │
│  │    model, tools, system_prompt,                  │    │
│  │    sub_agents=[plan-analyze],                    │    │
│  │    backend=LocalShellBackend | FilesystemBackend │    │
│  │  )                                                │    │
│  └──────────────────────────────────────────────────┘    │
│                                                          │
│  Built-in Middleware:                                    │
│  ├─ FilesystemMiddleware → ls, read_file, write_file,   │
│  │   edit_file, glob, grep, execute                      │
│  ├─ SubAgentMiddleware  → task (delegation)              │
│  ├─ SummarizationMiddleware → auto context compaction    │
│  ├─ TodoMiddleware      → built-in task tracking         │
│  └─ SkillsMiddleware    → loadable skill behaviors       │
├──────────────────────────────────────────────────────────┤
│              Custom Tools                                 │
│  web_fetch · web_search (DuckDuckGo)                      │
├──────────────────────────────────────────────────────────┤
│         Session Store (SQLite) + LangGraph Checkpointer  │
└──────────────────────────────────────────────────────────┘
```

## Features

- **Two Agent Modes**: `build` (LocalShellBackend, full access) and `plan` (FilesystemBackend, read-only)
- **Sub-Agent Delegation**: `plan-analyze` sub-agent for deep code analysis via `task` tool
- **11 Built-in Tools**: ls, read_file, write_file, edit_file, glob, grep, execute, task, todo + web_fetch, web_search
- **Streaming Chat**: Real-time token streaming with tool call and sub-agent visualization
- **Session Management**: Metadata in SQLite, conversation state in LangGraph checkpointer
- **Multi-Provider**: OpenAI, Anthropic, Ollama (any LangChain-compatible model)
- **Auto Summarization**: Long conversations automatically compacted
- **Project Config**: Per-project `.opencode.json` configuration

## Quick Start

### Prerequisites

- Python 3.11+
- pip or uv

### Installation

```bash
cd opencode-deepagents
pip install -r requirements.txt
# or: uv pip install -r requirements.txt
```

### Configuration

```bash
cp .env.example .env
# Edit .env with your API keys
```

```env
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o
OPENAI_API_KEY=sk-your-key
```

### Run

```bash
python app.py                     # http://127.0.0.1:7860
python app.py --port 8080         # Custom port
python app.py --share             # Public link
python app.py --mode plan         # Read-only analysis mode
```

## Agent Modes

### Build Mode (`LocalShellBackend`)
Full filesystem access with unrestricted shell command execution.
- Create, read, edit, and delete files
- Run build commands, tests, git operations
- Full code search and web research
- Delegate to sub-agents for complex analysis

### Plan Mode (`FilesystemBackend`, read-only)
Analysis and planning without modification capability.
- Read files and search codebases
- Web research and documentation lookup
- Structured analysis reports
- `execute` tool unavailable, `write_file`/`edit_file` denied

## What deepagents Provides

| Feature | deepagents Built-in |
|---------|-------------------|
| File read/write/edit | `FilesystemMiddleware` |
| Code search (grep/glob) | `FilesystemMiddleware` |
| Shell execution | `LocalShellBackend` / `execute` tool |
| Sub-agent spawning | `SubAgentMiddleware` / `task` tool |
| Todo management | Built-in todo tools |
| Context summarization | `SummarizationMiddleware` |
| Skills loading | `SkillsMiddleware` |
| Persistent memory | `AGENTS.md` via `MemoryMiddleware` |
| Streaming | LangGraph native |

We only add `web_fetch` and `web_search` as custom tools.

## Project Structure

```
opencode-deepagents/
├── app.py                    # Entry point
├── requirements.txt          # Python dependencies
├── .env.example              # Environment config
├── README.md
└── src/
    ├── agent.py              # CodingAgent using create_deep_agent()
    ├── config.py             # System prompts, mode configs
    ├── session.py            # SQLite session metadata store
    ├── ui.py                 # Gradio UI (chat + sidebar)
    └── tools/
        ├── __init__.py       # Tool exports
        └── web.py            # web_fetch + web_search (custom)
```

## Comparison

| | Original OpenCode | OpenCode DeepAgents |
|---|---|---|
| Runtime | Bun / TypeScript | Python 3.11+ |
| Agent Framework | Custom (Effect-TS) | deepagents (LangGraph) |
| UI | SolidJS / TUI / Electron | Gradio |
| File tools | Hand-written (12 tools) | deepagents built-in middleware |
| Sub-agents | Custom `@general` | deepagents `SubAgentMiddleware` |
| Context mgmt | Manual | deepagents auto-summarization |
| Human-in-loop | Custom permission system | LangGraph interrupts |
| Streaming | Effect Streams | LangGraph astream_events |

## License

MIT
