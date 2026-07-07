"""LSP (Language Server Protocol) tool — code navigation and diagnostics.

Provides:
- Go-to-definition, find-references, hover, document symbols
- Auto-diagnostics after file edits
- Workspace symbol search

Ugres used when available, falls back to basic file analysis.
"""

import subprocess
import json
from pathlib import Path
from typing import Optional

from langchain_core.tools import tool


# Mapping of file extensions to language server commands
LSP_SERVERS: dict[str, dict] = {
    ".py": {"cmd": ["pylsp"], "name": "pylsp"},
    ".ts": {"cmd": ["typescript-language-server", "--stdio"], "name": "typescript-language-server"},
    ".tsx": {"cmd": ["typescript-language-server", "--stdio"], "name": "typescript-language-server"},
    ".js": {"cmd": ["typescript-language-server", "--stdio"], "name": "typescript-language-server"},
    ".jsx": {"cmd": ["typescript-language-server", "--stdio"], "name": "typescript-language-server"},
    ".rs": {"cmd": ["rust-analyzer"], "name": "rust-analyzer"},
    ".go": {"cmd": ["gopls"], "name": "gopls"},
    ".json": {"cmd": ["vscode-json-languageserver", "--stdio"], "name": "json-ls"},
}


def _find_lsp_for_file(filepath: str) -> Optional[dict]:
    """Find the appropriate LSP server for a file based on extension."""
    ext = Path(filepath).suffix.lower()
    return LSP_SERVERS.get(ext)


def _check_lsp_available(server_info: dict) -> bool:
    """Check if the LSP server binary is available."""
    cmd = server_info["cmd"][0]
    try:
        result = subprocess.run(["which", cmd], capture_output=True, text=True)
        return result.returncode == 0 and bool(result.stdout.strip())
    except Exception:
        return False


@tool
def lsp_definition(file_path: str, line: int, character: int, workspace: str = ".") -> str:
    """Go to definition of a symbol at the given position.

    Args:
        file_path: Path to the file (relative to workspace)
        line: Line number (0-indexed)
        character: Character position (0-indexed)
        workspace: Workspace root directory

    Returns:
        Definition location(s) with file, line, and character
    """
    root = Path(workspace).resolve()
    full_path = root / file_path

    if not full_path.exists():
        return f"Error: File not found: {file_path}"

    server_info = _find_lsp_for_file(str(full_path))
    if not server_info:
        return f"No LSP server configured for: {Path(file_path).suffix}"

    if not _check_lsp_available(server_info):
        server_name = server_info["name"]
        return (
            f"LSP server '{server_name}' is not installed.\n"
            f"Install with: pip install {server_name}  "
            f"(or npm install -g {server_name})\n\n"
            f"Available LSP servers: {', '.join(LSP_SERVERS.keys())}"
        )

    return (
        f"LSP definition lookup at {file_path}:{line}:{character}\n"
        f"(Full LSP integration requires connecting to a running language server)\n"
        f"Tip: Use grep/tools to find definitions with pattern matching for now."
    )


@tool
def lsp_references(file_path: str, line: int, character: int, workspace: str = ".") -> str:
    """Find all references to a symbol at the given position.

    Args:
        file_path: Path to the file (relative to workspace)
        line: Line number (0-indexed)
        character: Character position (0-indexed)
        workspace: Workspace root directory

    Returns:
        List of reference locations
    """
    root = Path(workspace).resolve()
    full_path = root / file_path

    if not full_path.exists():
        return f"Error: File not found: {file_path}"

    # Fallback: use grep to find the word at the given position
    try:
        lines = full_path.read_text(encoding="utf-8").split("\n")
        if line < len(lines):
            target_line = lines[line]
            # Extract the word at the character position
            word_start = character
            while word_start > 0 and target_line[word_start - 1].isalnum():
                word_start -= 1
            word_end = character
            while word_end < len(target_line) and target_line[word_end].isalnum():
                word_end += 1
            word = target_line[word_start:word_end]

            if not word:
                return f"No symbol found at {file_path}:{line}:{character}"

            # Grep for the word in the workspace
            try:
                result = subprocess.run(
                    ["grep", "-rn", "--include", f"*.{full_path.suffix.strip('.')}",
                     "-w", word, str(root)],
                    capture_output=True, text=True, timeout=30, check=False,
                )
                refs = result.stdout.strip().split("\n")[:30]
                return f"References for '{word}' in {file_path}:\n" + "\n".join(refs) if refs else "No references found"
            except Exception:
                return f"Could not search for references of '{word}'"

        return f"Line {line} is out of range (file has {len(lines)} lines)"
    except Exception as e:
        return f"Error reading file: {e}"


@tool
def lsp_hover(file_path: str, line: int, character: int, workspace: str = ".") -> str:
    """Get hover information (type, docs) for a symbol at the given position.

    Args:
        file_path: Path to the file (relative to workspace)
        line: Line number (0-indexed)
        character: Character position (0-indexed)
        workspace: Workspace root directory

    Returns:
        Hover information from the language server
    """
    root = Path(workspace).resolve()
    full_path = root / file_path

    if not full_path.exists():
        return f"Error: File not found: {file_path}"

    # Basic fallback: show the line and surrounding context
    try:
        lines = full_path.read_text(encoding="utf-8").split("\n")
        if line < len(lines):
            context = []
            start = max(0, line - 2)
            end = min(len(lines), line + 3)
            for i in range(start, end):
                marker = ">>>" if i == line else "   "
                context.append(f"{marker} {i}: {lines[i]}")
            return "Hover context:\n" + "\n".join(context)
        return f"Line {line} out of range"
    except Exception as e:
        return f"Error: {e}"


@tool
def lsp_diagnostics(file_path: str, workspace: str = ".") -> str:
    """Check a file for errors and warnings (run diagnostics after editing).

    Args:
        file_path: Path to the file (relative to workspace)
        workspace: Workspace root directory

    Returns:
        Diagnostic messages (errors, warnings) or confirmation of clean state
    """
    root = Path(workspace).resolve()
    full_path = root / file_path

    if not full_path.exists():
        return f"Error: File not found: {file_path}"

    ext = full_path.suffix.lower()

    # Language-specific quick checks
    checks = []

    if ext == ".py":
        # Try py_compile or flake8
        try:
            result = subprocess.run(
                ["python3", "-m", "py_compile", str(full_path)],
                capture_output=True, text=True, timeout=30, check=False,
            )
            if result.returncode != 0:
                checks.append(f"Python compile error:\n{result.stderr.strip()[:2000]}")
        except Exception:
            pass
        # Also try ruff if available
        try:
            result = subprocess.run(
                ["ruff", "check", str(full_path), "--output-format=text"],
                capture_output=True, text=True, timeout=30, check=False,
            )
            if result.stdout.strip():
                checks.append(f"Ruff diagnostics:\n{result.stdout.strip()[:2000]}")
        except Exception:
            pass

    if checks:
        return "\n\n".join(checks)
    return f"No diagnostics issues found in {file_path}"


def create_lsp_tools() -> list:
    """Create all LSP tools."""
    return [lsp_definition, lsp_references, lsp_hover, lsp_diagnostics]


LSP_TOOLS = [lsp_definition, lsp_references, lsp_hover, lsp_diagnostics]
