# OpenCode DeepAgents

**AI Coding Agent** — reimplementation of [OpenCode](https://github.com/anthropics/opencode) using LangChain `deepagents` and Gradio.

## Architecture

```
opencode-deepagents/
├── app.py                     # CLI entry point (--port, --share, --workspace, --mode)
├── requirements.txt           # Python dependencies
├── .env.example               # API key configuration
├── .dockerignore
├── README.md
├── docker/
│   ├── Dockerfile               # Docker image definition
│   └── docker-compose.yml       # Docker Compose config
└── src/
    ├── __init__.py
    ├── agent.py               # CodingAgent with deepagents harness + HITL + snapshots
    ├── config.py              # Model config, modes, system prompts, skills listing
    ├── permission.py          # HITL permission system (wildcard rules, approve/deny/ask)
    ├── session.py             # SQLite session metadata + permission cache + snapshot log
    ├── snapshot.py            # Git-based filesystem snapshots for safe undo/restore
    ├── ui.py                  # Gradio web UI with approval dialogs and snapshot controls
    └── tools/
        ├── __init__.py        # Tool registry
        ├── web.py             # web_fetch, web_search
        ├── apply_patch.py     # Multi-file unified diff patch tool
        ├── question.py        # Interactive question tool (HITL)
        ├── skill.py           # Skill loader with multi-source discovery
        ├── lsp.py             # LSP tools: go-to-def, references, hover, diagnostics
        └── mcp.py             # MCP client integration (external tool servers)
```

## Feature Parity with OpenCode

| Feature | How It's Implemented |
|---|---|
| **File read/write/edit** | `deepagents` `FilesystemMiddleware` (ls, read_file, write_file, edit_file) |
| **Code search (grep/glob)** | `deepagents` `FilesystemMiddleware` |
| **Shell execution** | `deepagents` `LocalShellBackend` |
| **Multi-file apply_patch** | Custom tool with structured `*** Add/Update/Delete File:` format |
| **Web fetch & search** | `web_fetch` (httpx + BeautifulSoup), `web_search` (DuckDuckGo) |
| **Todo management** | `deepagents` `TodoMiddleware` |
| **Sub-agents** | `deepagents` `SubAgentMiddleware` (plan-analyze + code-explorer) |
| **Background tasks** | `BackgroundTask` class with asyncio-based async execution |
| **HITL Permissions** | LangGraph `interrupt_before=["tools"]` with wildcard allow/deny/ask rules |
| **Filesystem Snapshots** | Git-based snapshot store (track, restore, diff, undo) |
| **LSP Integration** | `lsp_definition`, `lsp_references`, `lsp_hover`, `lsp_diagnostics` |
| **MCP Integration** | `mcp` Python SDK integration with config loading from `.opencode.json` |
| **Question tool** | Interactive Q&A with HITL interrupt for approval |
| **Skill loader** | Multi-source discovery (`~/.claude/skills/`, `.agents/skills/`, project dirs) |
| **Model-specific prompts** | Per-model family overrides (gpt, claude, gemini) |
| **Permission cache** | "Always allow" memory persisted in SQLite |
| **Snapshot log** | Snapshot records linked to sessions in SQLite |

## Key Features

### Human-in-the-Loop (HITL) Permissions

Uses LangGraph's `interrupt_before=["tools"]` to pause execution before destructive tool calls (write_file, edit_file, execute, apply_patch, task). The UI shows an approval box where you can:

- **[A]pprove** — allow this single call
- **[R]eject** — deny this call  
- **[Y] Always allow** — remember the decision
- Toggle **Auto-approve** in settings to skip HITL entirely

Permission rules support wildcard patterns (`write`, `shell/*`, `*`) with three actions:
`allow`, `deny`, `ask`.

### Filesystem Snapshots & Undo

Every dangerous operation is preceded by a git-based snapshot. `/undo` restores files to the previous state. `/snapshots` lists recent checkpoints. Fully separate from conversation state.

### Sub-Agents

- **plan-analyze** — deep codebase analysis, architecture review, planning (read-only)
- **code-explorer** — large-scale exploration, feature tracing, module mapping (read-only)

Both run as deepagents `SubAgent` instances with their own tool sets. Plus support for `BackgroundTask` async execution.

### MCP (Model Context Protocol)

Configure MCP servers in `.opencode.json`:
```json
{
  "mcpServers": {
    "my-db": {
      "command": "my-mcp-server",
      "args": ["--db-url", "..."],
      "transport": "stdio"
    }
  }
}
```

### Skills

Place `SKILL.md` files in `.claude/skills/<name>/`, `.agents/skills/<name>/`, or globally in `~/.claude/skills/<name>/`. Skills auto-discover and appear in the system prompt. Use the `skill` tool to load full content on demand.

## Getting Started

```bash
# Clone and setup
cd opencode-deepagents
cp .env.example .env
# Edit .env with your API keys

# Install dependencies
pip install -r requirements.txt

# Launch
python app.py                        # http://127.0.0.1:7860
python app.py --port 8080 --share    # Public URL
python app.py --workspace ~/my-project --mode plan
```

## Docker Compose Deployment

```bash
# 1. Configure environment
cp .env.example .env
# Edit .env with your API keys

# 2. Build and start
cd docker
docker compose up -d

# 3. Open http://localhost:7860
```

### Docker Compose options

| Variable | Default | Description |
|---|---|---|
| `HOST_PORT` | `7860` | Host port to bind |
| `DEFAULT_AGENT_MODE` | `build` | Agent mode: `build` or `plan` |

```bash
# Custom port
HOST_PORT=8080 docker compose up -d

# Plan mode (read-only)
DEFAULT_AGENT_MODE=plan docker compose up -d

# Mount a host directory as workspace
# Edit docker-compose.yml, uncomment the volume mount under services.opencode-deepagents.volumes
```

### Commands

Type these in the chat:
- `/help` — show available commands
- `/undo` — restore filesystem to last snapshot
- `/snapshots` — list recent filesystem snapshots
- `/mode build` — switch to build mode (code + shell)
- `/mode plan` — switch to plan mode (read-only)
- `/mode ask` — switch to ask mode

## Configuration

### Environment Variables (.env)

| Variable | Default | Description |
|---|---|---|
| `LLM_PROVIDER` | `openai` | Model provider |
| `LLM_MODEL` | `gpt-4o` | Model name |
| `OPENAI_API_KEY` | — | API key |
| `ANTHROPIC_API_KEY` | — | Anthropic API key |
| `OPENAI_BASE_URL` | — | Custom API endpoint |
| `MAX_TOOL_ITERATIONS` | `50` | Max tool calls per turn |
| `TOOL_TIMEOUT_SECONDS` | `120` | Tool timeout |
| `DEFAULT_AGENT_MODE` | `build` | Default agent mode |
| `CHECKPOINT_DB` | `~/.opencode-deepagents/checkpoints.db` | LangGraph checkpoint DB |

### Project Config (.opencode.json)

```json
{
  "system_prompt": "Additional project-level instructions...",
  "permissions": {
    "write/*": "allow",
    "execute/rm *": "ask"
  },
  "mcpServers": { ... }
}
```

## Dependencies

- **deepagents** — AI agent harness (filesystem, shell, todo, sub-agents, HITL)
- **LangGraph** — agent runtime (state graph, checkpoints, interrupts)
- **Gradio** — web UI framework
- **SQLite** — session and permission storage
- **httpx + BeautifulSoup** — web tools
- **mcp** (optional) — Model Context Protocol integration
- **python-lsp-server / ruff** (optional) — LSP diagnostics
