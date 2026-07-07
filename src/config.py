"""
OpenCode DeepAgents - Configuration module.

Defines agent modes (build/plan), system prompts, permissions,
skills discovery, model-specific prompts, and model configuration.
"""

import os
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

from dotenv import load_dotenv

load_dotenv()


@dataclass
class ModelConfig:
    provider: str = "openai"
    model: str = "gpt-4o"
    temperature: float = 0.0
    api_key: str = ""
    base_url: str = ""

    def to_model_string(self) -> str:
        """Format for deepagents model parameter: 'provider:model'."""
        return f"{self.provider}:{self.model}"


# ── System Prompts ──────────────────────────────────────────────────

BUILD_SYSTEM_PROMPT = """You are an expert AI coding assistant with full access to the workspace.

## Capabilities
- **Read, write, and edit files** in the workspace
- **Apply multi-file patches** for coordinated changes across files
- **Execute shell commands** to build, test, and debug code
- **Search codebases** with grep and glob patterns
- **Navigate code** with go-to-definition and find-references (LSP)
- **Fetch web content** and search the web for documentation
- **Manage todo lists** to track complex multi-step work
- **Delegate to sub-agents** for complex research or analysis tasks
- **Load project skills** for domain-specific guidance
- **Ask questions** when you need user clarification
- **Use MCP tools** from connected external servers

## Guidelines
1. Read files before editing them — never guess content
2. Use edit_file for surgical changes; use apply_patch for multi-file coordinated changes
3. After making changes, verify by reading back or running tests
4. Search the codebase thoroughly before making assumptions
5. Write clean, idiomatic code following project conventions
6. Explain reasoning briefly before taking action
7. Use the todo system to plan complex multi-step tasks
8. Run diagnostics after file edits to catch errors early

## Workspace
Current working directory: {workspace}
"""

PLAN_SYSTEM_PROMPT = """You are an expert AI coding architect in **PLAN MODE** — analysis and planning only.

## Capabilities (READ-ONLY)
- Read files and understand codebases
- Search code with grep and glob patterns
- Navigate code definitions and references (LSP)
- Fetch web content and search for documentation
- Manage todo lists for planning
- Load project skills for context
- Ask questions for clarification

## Guidelines
1. Provide thorough analysis and clear, actionable plans
2. Explain trade-offs between different approaches
3. Outline step-by-step implementation plans with file paths
4. Identify potential risks, edge cases, and dependencies
5. Do NOT request to write, edit, or execute — you are read-only planning
6. When the user is ready, they will switch to build mode for implementation

## Workspace
Current working directory: {workspace}
"""

PLAN_SUBAGENT_PROMPT = """You are a specialized planning and analysis sub-agent.

Your task is to deeply analyze a specific aspect of the codebase or problem.
Provide a detailed, structured report with:
1. Current state analysis
2. Relevant code references with file paths and line numbers
3. Proposed changes with rationale
4. Risk assessment and edge cases
5. Recommended implementation order

Do NOT suggest writing or editing files — this is analysis only.
"""

EXPLORE_SUBAGENT_PROMPT = """You are a code-explorer sub-agent specializing in large-scale codebase exploration.

Your task is to efficiently explore the codebase to answer specific questions:
1. Map out relevant directories and files
2. Trace data flows, function calls, and module dependencies
3. Find where specific features or behaviors are implemented
4. Identify patterns, conventions, and architecture decisions

Use grep, glob, and LSP tools to navigate. Be thorough but efficient.
Report findings with specific file paths and line references.
"""

# ── Model-Specific Prompt Adjustments ────────────────────────────────

MODEL_PROMPT_OVERRIDES = {
    "claude": """
## Formatting
Use Markdown for responses. Place code blocks in triple backticks with language identifiers.

## Tool Usage
Be concise and direct with tool calls. Combine related operations when possible.
""",
    "gemini": """
## Formatting
Use Markdown for responses. You work best with clear, structured instructions.

## Tool Usage
Make one tool call at a time and wait for results before proceeding.
""",
    "gpt": """
## Formatting
You are an autonomous agent. You can make multiple tool calls in sequence.
Take initiative to explore and understand the codebase before making changes.
""",
}


# ── Agent Mode Configuration ────────────────────────────────────────

@dataclass
class AgentModeConfig:
    name: str           # "build" | "plan"
    system_prompt: str
    allow_shell: bool = False
    allow_file_writes: bool = False

    def format_prompt(self, workspace: str, model_provider: str = "") -> str:
        """Format the system prompt with workspace, model-specific overrides, and skills."""
        provider_key = model_provider.lower()
        if "claude" in provider_key:
            provider_key = "claude"
        elif "gemini" in provider_key:
            provider_key = "gemini"
        else:
            provider_key = "gpt"

        prompt = self.system_prompt.format(workspace=workspace)

        # Add model-specific overrides
        if provider_key in MODEL_PROMPT_OVERRIDES:
            prompt += "\n" + MODEL_PROMPT_OVERRIDES[provider_key]

        # Merge project-level .opencode.json config
        config_path = Path(workspace) / ".opencode.json"
        if config_path.exists():
            try:
                data = json.loads(config_path.read_text())
                project_prompt = data.get("system_prompt", "")
                if project_prompt:
                    prompt = f"{project_prompt}\n\n{prompt}"
            except (json.JSONDecodeError, OSError):
                pass

        # Append available skills
        prompt = _append_skills_listing(prompt, workspace)
        # Append workspace file listing
        prompt = _append_workspace_listing(prompt, workspace)
        return prompt


BUILD_MODE = AgentModeConfig(
    name="build",
    system_prompt=BUILD_SYSTEM_PROMPT,
    allow_shell=True,
    allow_file_writes=True,
)

PLAN_MODE = AgentModeConfig(
    name="plan",
    system_prompt=PLAN_SYSTEM_PROMPT,
    allow_shell=False,
    allow_file_writes=False,
)

AGENT_MODES: dict[str, AgentModeConfig] = {
    "build": BUILD_MODE,
    "plan": PLAN_MODE,
}


# ── Helpers ─────────────────────────────────────────────────────────

def _append_workspace_listing(prompt: str, workspace: str) -> str:
    """Append top-level workspace file listing to prompt."""
    try:
        ws_path = Path(workspace).resolve()
        entries = sorted(ws_path.iterdir(), key=lambda e: (not e.is_dir(), e.name.lower()))
        visible = [e for e in entries if not e.name.startswith(".")][:30]
        if visible:
            listing = "\n".join(
                f"  {e.name}{'/' if e.is_dir() else ''}" for e in visible
            )
            prompt += f"\n\n## Workspace Contents\n```\n{listing}\n```"
    except Exception:
        pass
    return prompt


def _append_skills_listing(prompt: str, workspace: str) -> str:
    """Append available skills to the prompt."""
    from src.tools.skill import _discover_skills, _parse_skill
    skills_map = _discover_skills(workspace)
    if not skills_map:
        return prompt

    lines = ["## Available Skills"]
    for name, path in sorted(skills_map.items()):
        info = _parse_skill(path)
        if info:
            desc = info.get("description", "")
            lines.append(f"- **{name}**: {desc}")
    prompt += "\n\n" + "\n".join(lines) + "\nUse the `skill` tool to load a skill's full content."
    return prompt


def load_model_config() -> ModelConfig:
    """Load model configuration from environment variables."""
    return ModelConfig(
        provider=os.getenv("LLM_PROVIDER", "openai"),
        model=os.getenv("LLM_MODEL", "gpt-4o"),
        temperature=float(os.getenv("LLM_TEMPERATURE", "0.0")),
        api_key=os.getenv("OPENAI_API_KEY", os.getenv("ANTHROPIC_API_KEY", "")),
        base_url=os.getenv("OPENAI_BASE_URL", ""),
    )


# ── Permission Defaults ─────────────────────────────────────────────

DANGEROUS_TOOLS = ["write_file", "edit_file", "execute", "apply_patch", "task"]
READONLY_TOOLS = ["ls", "read_file", "glob", "grep", "web_fetch", "web_search",
                  "todo_write", "question", "skill",
                  "lsp_definition", "lsp_references", "lsp_hover", "lsp_diagnostics"]


# ── Global Settings ─────────────────────────────────────────────────

MAX_TOOL_ITERATIONS = int(os.getenv("MAX_TOOL_ITERATIONS", "50"))
TOOL_TIMEOUT_SECONDS = int(os.getenv("TOOL_TIMEOUT_SECONDS", "120"))
DEFAULT_WORKSPACE = os.getenv("DEFAULT_WORKSPACE", str(Path.cwd()))
DEFAULT_AGENT_MODE = os.getenv("DEFAULT_AGENT_MODE", "build")
CHECKPOINT_DB = os.getenv("CHECKPOINT_DB", str(Path.home() / ".opencode-deepagents" / "checkpoints.db"))
