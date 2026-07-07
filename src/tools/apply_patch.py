"""Multi-file unified diff patch tool — equivalent to opencode's apply_patch.

Handles a structured patch format with:
- *** Add File: <path>  / *** Update File: <path> / *** Delete File: <path>
- Block-based updates with context matching
- Atomic application (all or nothing per file)
"""

import re
from pathlib import Path
from langchain_core.tools import tool


# Use a more specific marker to reduce false matches in code content
_MARKER_PREFIX = "@@OPCD@@ "
_ADD_MARKER = f"{_MARKER_PREFIX}Add File:"
_UPDATE_MARKER = f"{_MARKER_PREFIX}Update File:"
_DELETE_MARKER = f"{_MARKER_PREFIX}Delete File:"
_OLD_MARKER = "@@OPCD@@ old"
_NEW_MARKER = "@@OPCD@@ new"


@tool
def apply_patch(input_text: str, workspace: str = ".") -> str:
    """Apply a multi-file structured patch to the workspace.

    Format:
      @@OPCD@@ Add File: path/to/file
      <full file contents>
      @@OPCD@@ Update File: path/to/file
      @@OPCD@@ old
      <lines to replace>
      @@OPCD@@ new
      <replacement lines>
      @@OPCD@@ Delete File: path/to/file

    Use this for making changes across multiple files in one operation.
    Each file change is applied atomically.

    Args:
        input_text: The structured patch text
        workspace: Root directory for relative paths (defaults to ".")

    Returns:
        Result summary per file or error description
    """
    root = Path(workspace).resolve()
    results = []

    # Split into file sections using unique markers to avoid false matches
    pattern = re.escape(_MARKER_PREFIX) + r"(Add|Update|Delete) File: (.+)"
    sections = re.split(pattern, input_text)

    # sections[0] is anything before the first marker
    idx = 1
    while idx < len(sections):
        if idx + 2 > len(sections):
            break
        action = sections[idx].strip()        # "Add", "Update", "Delete"
        filepath_str = sections[idx + 1].strip()  # file path
        content_block = sections[idx + 2] if idx + 2 < len(sections) else ""

        filepath = (root / filepath_str).resolve()
        # Enforce workspace containment
        if not str(filepath).startswith(str(root)):
            results.append(f"[DENIED] {filepath_str}: path escapes workspace")
            idx += 3
            continue

        try:
            if action == "Add":
                _apply_add(filepath, content_block)
                results.append(f"[ADDED] {filepath_str}")

            elif action == "Update":
                _apply_update(filepath, content_block)
                results.append(f"[UPDATED] {filepath_str}")

            elif action == "Delete":
                _apply_delete(filepath)
                results.append(f"[DELETED] {filepath_str}")

        except FileNotFoundError:
            results.append(f"[FAILED] {filepath_str}: file not found for Update/Delete")
        except PermissionError:
            results.append(f"[FAILED] {filepath_str}: permission denied")
        except Exception as e:
            results.append(f"[FAILED] {filepath_str}: {e}")

        idx += 3

    return "\n".join(results) if results else "(empty patch)"


def _apply_add(filepath: Path, content: str):
    """Add a new file with given content."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    filepath.write_text(content.strip(), encoding="utf-8")


def _apply_update(filepath: Path, content: str):
    """Update a file using old/new blocks with unique markers."""
    if not filepath.exists():
        raise FileNotFoundError(str(filepath))

    text = filepath.read_text(encoding="utf-8")

    # Use unique markers to avoid collisions with file content
    pattern = re.escape(_OLD_MARKER) + r"\s*\n(.*?)\n\s*" + re.escape(_NEW_MARKER)
    matches = list(re.finditer(pattern, content, re.DOTALL))

    if not matches:
        # Fallback: try plain replace of entire content block
        stripped = content.strip()
        if stripped:
            filepath.write_text(stripped, encoding="utf-8")
            return
        raise ValueError("no old/new blocks found in update section")

    for match in matches:
        old_text = match.group(1).strip()
        # Find new_text after the match (everything after "@@ new" until next marker or end)
        new_start = match.end()
        rest = content[new_start:]
        # Stop at next @@OPCD@@ marker or end of string
        next_marker = rest.find(_MARKER_PREFIX)
        if next_marker >= 0:
            new_text = rest[:next_marker].strip()
        else:
            new_text = rest.strip()

        if old_text in text:
            text = text.replace(old_text, new_text, 1)
        else:
            raise ValueError(f"old block not found in file: {old_text[:80]}...")

    filepath.write_text(text, encoding="utf-8")


def _apply_delete(filepath: Path):
    """Delete a file."""
    if not filepath.exists():
        raise FileNotFoundError(str(filepath))
    filepath.unlink()


def create_apply_patch_tool():
    return apply_patch
