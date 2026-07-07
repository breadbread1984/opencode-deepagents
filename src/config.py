"""
OpenCode DeepAgents - Configuration module.

Defines agent modes (build/plan), system prompts, and model configuration.
Uses deepagents' SubAgent system for plan mode delegation.
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
- **Execute shell commands** to build, test, and debug code
- **Search codebases** with grep and glob patterns
- **Fetch web content** and search the web for documentation
- **Manage todo lists** to track complex multi-step work
- **Delegate to sub-agents** for complex research or analysis tasks

## Guidelines
1. Read files before editing them — never guess content
2. Use edit_file for surgical changes; prefer it over rewriting entire files
3. After making changes, verify by reading back or running tests
4. Search the codebase thoroughly before making assumptions
5. Write clean, idiomatic code following project conventions
6. Explain reasoning briefly before taking action
7. Use the todo system to plan complex multi-step tasks

## Workspace
Current working directory: {workspace}
"""

PLAN_SYSTEM_PROMPT = """You are an expert AI coding architect in **PLAN MODE** — analysis and planning only.

## Capabilities (READ-ONLY)
- Read files and understand codebases
- Search code with grep and glob patterns
- Fetch web content and search for documentation
- Manage todo lists for planning

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


# ── Agent Mode Configuration ────────────────────────────────────────

@dataclass
class AgentModeConfig:
    name: str           # "build" | "plan"
    system_prompt: str
    allow_shell: bool = False
    allow_file_writes: bool = False

    def format_prompt(self, workspace: str) -> str:
        """Format the system prompt with workspace path."""
        prompt = self.system_prompt.format(workspace=workspace)

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


def load_model_config() -> ModelConfig:
    """Load model configuration from environment variables."""
    return ModelConfig(
        provider=os.getenv("LLM_PROVIDER", "openai"),
        model=os.getenv("LLM_MODEL", "gpt-4o"),
        temperature=float(os.getenv("LLM_TEMPERATURE", "0.0")),
        api_key=os.getenv("OPENAI_API_KEY", os.getenv("ANTHROPIC_API_KEY", "")),
        base_url=os.getenv("OPENAI_BASE_URL", ""),
    )


# ── Global Settings ─────────────────────────────────────────────────

MAX_TOOL_ITERATIONS = int(os.getenv("MAX_TOOL_ITERATIONS", "30"))
TOOL_TIMEOUT_SECONDS = int(os.getenv("TOOL_TIMEOUT_SECONDS", "120"))
DEFAULT_WORKSPACE = os.getenv("DEFAULT_WORKSPACE", str(Path.cwd()))
DEFAULT_AGENT_MODE = os.getenv("DEFAULT_AGENT_MODE", "build")
