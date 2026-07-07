"""Multi-file unified diff patch tool — equivalent to opencode's apply_patch.

Handles a structured patch format with:
- *** Add File: <path>  / *** Update File: <path> / *** Delete File: <path>
- Block-based updates with context matching
- Atomic application (all or nothing per file)
"""

import re
from pathlib import Path
from langchain_core.tools import tool


@tool
def apply_patch(input_text: str, workspace: str = ".") -> str:
    """Apply a multi-file structured patch to the workspace.

    Format:
      *** Add File: path/to/file
      <full file contents>
      *** Update File: path/to/file
      @@ old @@
      <lines to replace>
      @@ new @@
      <replacement lines>
      *** Delete File: path/to/file

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

    # Split into file sections
    pattern = r"\*\*\* (Add|Update|Delete) File: (.+)"
    sections = re.split(pattern, input_text)

    # sections[0] is anything before the first ***
    idx = 1
    while idx < len(sections):
        if idx + 2 > len(sections):
            break
        action = sections[idx].strip()    # "Add", "Update", "Delete"
        path_str = sections[idx + 1].strip()  # file path
        content_block = sections[idx + 2] if idx + 2 < len(sections) else ""

        filepath = (root / path_str).resolve()
        # Enforce workspace containment
        if not str(filepath).startswith(str(root)):
            results.append(f"[DENIED] {path_str}: path escapes workspace")
            idx += 3
            continue

        try:
            if action == "Add":
                _apply_add(filepath, content_block)
                results.append(f"[ADDED] {path_str}")

            elif action == "Update":
                _apply_update(filepath, content_block)
                results.append(f"[UPDATED] {path_str}")

            elif action == "Delete":
                _apply_delete(filepath)
                results.append(f"[DELETED] {path_str}")

        except FileNotFoundError:
            results.append(f"[FAILED] {path_str}: file not found for Update/Delete")
        except PermissionError:
            results.append(f"[FAILED] {path_str}: permission denied")
        except Exception as e:
            results.append(f"[FAILED] {path_str}: {e}")

        idx += 3

    return "\n".join(results) if results else "(empty patch)"


def _apply_add(filepath: Path, content: str):
    """Add a new file with given content."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    filepath.write_text(content.strip(), encoding="utf-8")


def _apply_update(filepath: Path, content: str):
    """Update a file using old/new blocks."""
    if not filepath.exists():
        raise FileNotFoundError(str(filepath))

    text = filepath.read_text(encoding="utf-8")

    # Parse @@ old / @@ new blocks
    blocks = re.split(r"@@ (old|new) @@", content)
    # blocks structure: [leading, "old", old_content, "new", new_content, ...]
    idx = 1
    while idx < len(blocks):
        if idx + 2 > len(blocks):
            break
        marker = blocks[idx].strip()
        block_text = blocks[idx + 1].strip()
        if marker == "old":
            old_text = block_text
            # Next should be "new"
            if idx + 3 < len(blocks) and blocks[idx + 2].strip() == "new":
                new_text = blocks[idx + 3].strip()
                if old_text in text:
                    text = text.replace(old_text, new_text, 1)
                else:
                    raise ValueError(f"old block not found in file: {old_text[:80]}...")
                idx += 4
                continue
        idx += 2

    filepath.write_text(text, encoding="utf-8")


def _apply_delete(filepath: Path):
    """Delete a file."""
    if not filepath.exists():
        raise FileNotFoundError(str(filepath))
    filepath.unlink()


def create_apply_patch_tool():
    return apply_patch
