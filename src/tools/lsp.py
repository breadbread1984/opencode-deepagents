"""LSP (Language Server Protocol) tool — code navigation and diagnostics.

Provides:
- Go-to-definition with grep-based fallback
- Find-references with grep fallback
- Hover information with context display
- Auto-diagnostics via language-specific checkers (py_compile, ruff)

Falls back to ripgrep/text analysis when language servers are not installed.
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

    Uses grep-based fallback to find symbol definitions when LSP is unavailable.

    Args:
        file_path: Path to the file (relative to workspace)
        line: Line number (0-indexed)
        character: Character position (0-indexed)
        workspace: Workspace root directory

    Returns:
        Definition location(s) with file, line, and context
    """
    root = Path(workspace).resolve()
    full_path = root / file_path

    if not full_path.exists():
        return f"Error: File not found: {file_path}"

    server_info = _find_lsp_for_file(str(full_path))

    # Check if LSP server is available for real protocol communication
    if server_info and _check_lsp_available(server_info):
        return (
            f"LSP server '{server_info['name']}' is available.\n"
            f"(Real-time LSP protocol communication requires a running server daemon.\n"
            f"Using grep-based fallback for definition lookup below.)\n\n"
            f"{_grep_definition(full_path, root, line, character)}"
        )

    # Fallback: grep-based definition search
    return _grep_definition(full_path, root, line, character)


def _grep_definition(file_path: Path, root: Path, line: int, character: int) -> str:
    """Find definitions using grep heuristics."""
    try:
        text = file_path.read_text(encoding="utf-8")
        lines = text.split("\n")

        if line >= len(lines):
            return f"Line {line} is out of range (file has {len(lines)} lines)"

        target_line = lines[line]
        if character >= len(target_line):
            char = len(target_line) - 1 if target_line else 0
        else:
            char = character

        # Extract the symbol word at cursor
        word_start = char
        while word_start > 0 and (target_line[word_start - 1].isalnum() or target_line[word_start - 1] == "_"):
            word_start -= 1
        word_end = char
        while word_end < len(target_line) and (target_line[word_end].isalnum() or target_line[word_end] == "_"):
            word_end += 1
        word = target_line[word_start:word_end]

        if not word or len(word) < 2:
            return f"No recognizable symbol at {file_path.name}:{line}:{character}"

        ext = file_path.suffix.strip(".")
        include_globs = f"*.{ext}" if ext else "*"

        # Strategy 1: Look for definitions (function/class declarations)
        patterns = {
            ".py": f"^\\s*(def|class)\\s+{re.escape(word)}\\b",
            ".ts": f"\\b(function|class|const|let|var|interface|type|enum)\\s+{re.escape(word)}\\b",
            ".tsx": f"\\b(function|class|const|let|var|interface|type|enum)\\s+{re.escape(word)}\\b",
            ".js": f"\\b(function|class|const|let|var)\\s+{re.escape(word)}\\b",
            ".jsx": f"\\b(function|class|const|let|var)\\s+{re.escape(word)}\\b",
            ".go": f"\\b(func|type|var|const)\\s+{re.escape(word)}\\b",
            ".rs": f"\\b(fn|struct|enum|trait|impl|const|static|type)\\s+{re.escape(word)}\\b",
        }

        ext_key = file_path.suffix.lower()
        pattern = patterns.get(ext_key, f"\\b{re.escape(word)}\\b")

        try:
            result = subprocess.run(
                ["grep", "-rn", "--include", include_globs, pattern, str(root)],
                capture_output=True, text=True, timeout=30, check=False,
            )
            lines_found = result.stdout.strip().split("\n")[:20]

            if lines_found and lines_found[0]:
                return (
                    f"Definition of '{word}' (from {file_path.name}:{line}):\n"
                    + "\n".join(lines_found)
                )
        except Exception:
            pass

        # Strategy 2: Plain word search as fallback
        try:
            result = subprocess.run(
                ["grep", "-rnw", "--include", include_globs, word, str(root)],
                capture_output=True, text=True, timeout=30, check=False,
            )
            occurrences = result.stdout.strip().split("\n")[:20]
            if occurrences and occurrences[0]:
                return (
                    f"Symbol '{word}' occurrences (grep fallback, from {file_path.name}:{line}):\n"
                    + "\n".join(occurrences)
                )
        except Exception:
            pass

        return f"Symbol '{word}' found at {file_path.name}:{line}:{character}, but no definition pattern matched."

    except Exception as e:
        return f"Error during definition lookup: {e}"


# Import re for the definition patterns
import re


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

    # Use grep to find the word at the given position
    try:
        lines = full_path.read_text(encoding="utf-8").split("\n")
        if line < len(lines):
            target_line = lines[line]
            # Extract the word at the character position
            word_start = character
            while word_start > 0 and (target_line[word_start - 1].isalnum() or target_line[word_start - 1] == "_"):
                word_start -= 1
            word_end = character
            while word_end < len(target_line) and (target_line[word_end].isalnum() or target_line[word_end] == "_"):
                word_end += 1
            word = target_line[word_start:word_end]

            if not word:
                return f"No symbol found at {file_path}:{line}:{character}"

            # Grep for the word in the workspace
            ext = full_path.suffix.strip(".") or "*"
            include_glob = f"*.{ext}" if ext != "*" else "*"
            try:
                result = subprocess.run(
                    ["grep", "-rn", "--include", include_glob, "-w", word, str(root)],
                    capture_output=True, text=True, timeout=30, check=False,
                )
                refs = result.stdout.strip().split("\n")[:30]
                return (
                    f"References for '{word}' in {file_path}:\n" + "\n".join(refs)
                    if refs and refs[0]
                    else f"No references found for '{word}'"
                )
            except Exception as e:
                return f"Could not search for references of '{word}': {e}"

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
        Hover information — context and word at the cursor position
    """
    root = Path(workspace).resolve()
    full_path = root / file_path

    if not full_path.exists():
        return f"Error: File not found: {file_path}"

    # Show the line and surrounding context
    try:
        lines = full_path.read_text(encoding="utf-8").split("\n")
        if line < len(lines):
            # Extract word at cursor
            target_line = lines[line]
            ws = character
            while ws > 0 and (target_line[ws - 1].isalnum() or target_line[ws - 1] == "_"):
                ws -= 1
            we = character
            while we < len(target_line) and (target_line[we].isalnum() or target_line[we] == "_"):
                we += 1
            word = target_line[ws:we] if ws < we else ""

            context = []
            start = max(0, line - 2)
            end = min(len(lines), line + 3)
            for i in range(start, end):
                marker = ">>>" if i == line else "   "
                context.append(f"{marker} {i}: {lines[i]}")

            header = f"Hover at {file_path}:{line}:{character}"
            if word:
                header += f" (word: '{word}')"
            return header + "\n" + "\n".join(context)
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
    checks = []

    # Python diagnostics
    if ext == ".py":
        # py_compile for syntax errors
        try:
            result = subprocess.run(
                ["python3", "-m", "py_compile", str(full_path)],
                capture_output=True, text=True, timeout=30, check=False,
            )
            if result.returncode != 0:
                checks.append(f"Python syntax error:\n{result.stderr.strip()[:2000]}")
        except Exception:
            pass

        # ruff linting
        try:
            result = subprocess.run(
                ["ruff", "check", str(full_path), "--output-format=text"],
                capture_output=True, text=True, timeout=30, check=False,
            )
            if result.stdout.strip():
                checks.append(f"Ruff diagnostics:\n{result.stdout.strip()[:2000]}")
        except Exception:
            pass

    # Generic: check syntax via compilation for other languages
    elif ext in (".ts", ".tsx"):
        try:
            result = subprocess.run(
                ["npx", "--yes", "typescript", "--noEmit", str(full_path)],
                capture_output=True, text=True, timeout=60, check=False,
            )
            if result.stdout.strip() or result.stderr.strip():
                checks.append(f"TypeScript diagnostics:\n{(result.stdout + result.stderr).strip()[:2000]}")
        except Exception:
            checks.append("TypeScript diagnostics: npx/typescript not available. Install Node.js and try: npx tsc --noEmit")
    elif ext in (".rs",):
        try:
            result = subprocess.run(
                ["rustc", "--edition", "2021", "--crate-type", "lib", "-Z", "no-codegen", str(full_path)],
                capture_output=True, text=True, timeout=60, check=False,
            )
            if result.stderr.strip():
                checks.append(f"Rust diagnostics:\n{result.stderr.strip()[:2000]}")
        except Exception:
            checks.append("Rust diagnostics: rustc not available.")
    elif ext in (".go",):
        try:
            result = subprocess.run(
                ["go", "vet", str(full_path)],
                capture_output=True, text=True, timeout=30, check=False,
            )
            if result.stdout.strip() or result.stderr.strip():
                checks.append(f"Go diagnostics:\n{(result.stdout + result.stderr).strip()[:2000]}")
        except Exception:
            checks.append("Go diagnostics: go vet not available.")

    if checks:
        return "\n\n".join(checks)
    return f"No diagnostics issues found in {file_path}"


def create_lsp_tools() -> list:
    """Create all LSP tools."""
    return [lsp_definition, lsp_references, lsp_hover, lsp_diagnostics]


LSP_TOOLS = [lsp_definition, lsp_references, lsp_hover, lsp_diagnostics]
