"""Skill loading tool — equivalent to opencode's skill tool.

Loads skill content from skill files (SKILL.md with YAML frontmatter)
from multiple discovery sources: project .claude/skills/, .agents/skills/,
and the global ~/.claude/skills/ directory.
"""

import re
from pathlib import Path
from typing import Optional

from langchain_core.tools import tool


# Skill discovery paths
SKILL_SOURCES = [
    ".claude/skills",
    ".agents/skills",
    str(Path.home() / ".claude" / "skills"),
    str(Path.home() / ".agents" / "skills"),
]


def _discover_skills(workspace: str) -> dict[str, Path]:
    """Scan all skill sources and return {skill_name: path_to_SKILL.md}."""
    skills: dict[str, Path] = {}
    seen: set[str] = set()

    root = Path(workspace).resolve()

    for source in SKILL_SOURCES:
        sk_dir = root / source if not Path(source).is_absolute() else Path(source)
        if not sk_dir.exists():
            continue
        for skill_path in sorted(sk_dir.rglob("SKILL.md")):
            name = skill_path.parent.name
            if name in seen:
                continue
            seen.add(name)
            skills[name] = skill_path

    return skills


def _parse_skill(filepath: Path) -> Optional[dict]:
    """Parse a SKILL.md file with optional YAML frontmatter."""
    try:
        content = filepath.read_text(encoding="utf-8")
    except Exception:
        return None

    name = filepath.parent.name
    description = ""
    body = content

    # Parse YAML frontmatter if present
    if content.startswith("---"):
        parts = content.split("---", 2)
        if len(parts) >= 3:
            fm_text = parts[1]
            body = parts[2]
            for line in fm_text.strip().split("\n"):
                line = line.strip()
                if line.startswith("name:"):
                    name = line.split(":", 1)[1].strip().strip('"').strip("'")
                elif line.startswith("description:"):
                    description = line.split(":", 1)[1].strip().strip('"').strip("'")

    return {
        "name": name,
        "description": description,
        "path": str(filepath),
        "base_dir": str(filepath.parent),
        "content": body.strip(),
        "files": _list_skill_files(filepath.parent),
    }


def _list_skill_files(base_dir: Path) -> str:
    """List relevant files in the skill directory (excluding SKILL.md)."""
    try:
        files = []
        for f in sorted(base_dir.rglob("*")):
            if f.is_file() and f.name != "SKILL.md" and not f.name.startswith("."):
                rel = f.relative_to(base_dir)
                files.append(str(rel))
        return "\n".join(files[:50]) if files else "(no additional files)"
    except Exception:
        return "(error listing files)"


@tool
def skill(name: str, workspace: str = ".") -> str:
    """Load and display a skill by name.

    Skills provide domain-specific guidance, conventions, or instructions.
    Use this when you need to load a project-specific skill for context.

    Args:
        name: The skill name to load (directory name containing SKILL.md).
              Pass "*" to list all available skills.
        workspace: Root directory for skill discovery (defaults to ".")

    Returns:
        Skill content with metadata, or a list of available skills
    """
    skills_map = _discover_skills(workspace)

    if name == "*" or name == "list":
        if not skills_map:
            return "(no skills found)"
        listing = ["Available skills:"]
        for skill_name, skill_path in skills_map.items():
            info = _parse_skill(skill_path)
            desc = info["description"] if info else "(no description)"
            listing.append(f"  - {skill_name}: {desc}")
        return "\n".join(listing)

    if name not in skills_map:
        # Case-insensitive search
        matches = [k for k in skills_map if k.lower() == name.lower()]
        if not matches:
            # Fuzzy: search description
            for skill_name, skill_path in skills_map.items():
                info = _parse_skill(skill_path)
                if info and name.lower() in info.get("description", "").lower():
                    matches.append(skill_name)
                    break
        if not matches:
            names = ", ".join(sorted(skills_map.keys()))
            return f"Skill '{name}' not found. Available: {names}"
        name = matches[0]

    info = _parse_skill(skills_map[name])
    if not info:
        return f"Error loading skill: {name}"

    return f"""# Skill: {info['name']}
**Description**: {info['description']}
**Base Directory**: {info['base_dir']}

{info['content']}

## Skill Files
{info['files']}
"""


def create_skill_tool():
    return skill
