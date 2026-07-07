"""Git-based filesystem snapshot/restore system for safe undo.

Mimics original opencode's git-snapshot approach:
- track()  — record current filesystem state as a commit
- restore() — revert filesystem to a previous snapshot
- diff()   — show what changed since a snapshot
- cleanup() — remove old snapshots to keep disk usage bounded
"""

import subprocess
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


SNAPSHOT_DIR = Path.home() / ".opencode-deepagents" / "snapshots"


class SnapshotStore:
    """Manages git-based snapshots for file recovery."""

    def __init__(self, workspace: str):
        self.workspace = str(Path(workspace).resolve())
        self.snapshot_dir = SNAPSHOT_DIR / _hash_path(self.workspace)
        self._initialized = False

    def _ensure_init(self):
        if self._initialized:
            return
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        if not (self.snapshot_dir / ".git").exists():
            subprocess.run(
                ["git", "init", "--quiet", str(self.snapshot_dir)],
                check=False, capture_output=True,
            )
        self._initialized = True

    def _git(self, *args: str) -> subprocess.CompletedProcess:
        self._ensure_init()
        return subprocess.run(
            ["git", "-C", str(self.snapshot_dir)] + list(args),
            capture_output=True, text=True, check=False,
        )

    def _copy_workspace_to_snapshot(self):
        """Copy current workspace files into snapshot dir (respecting .gitignore)."""
        import shutil
        ws = Path(self.workspace)
        for item in ws.iterdir():
            if item.name == ".git" or item.name.startswith("."):
                continue
            dest = self.snapshot_dir / item.name
            if item.is_dir():
                if dest.exists():
                    shutil.rmtree(dest)
                shutil.copytree(item, dest, ignore=shutil.ignore_patterns(
                    ".git", "__pycache__", "*.pyc", "node_modules",
                ))
            else:
                shutil.copy2(item, dest)

    def _copy_snapshot_to_workspace(self):
        """Restore files from snapshot dir back to workspace."""
        import shutil
        ws = Path(self.workspace)
        for item in self.snapshot_dir.iterdir():
            if item.name == ".git" or item.name.startswith("."):
                continue
            dest = ws / item.name
            if item.is_dir():
                if dest.exists():
                    shutil.rmtree(dest)
                shutil.copytree(item, dest)
            else:
                shutil.copy2(item, dest)

    def track(self, label: str = "") -> str:
        """Create a snapshot of the current workspace state. Returns snapshot hash."""
        self._ensure_init()
        try:
            self._copy_workspace_to_snapshot()
            self._git("add", "-A")
            commit_msg = label or datetime.now(timezone.utc).strftime("snapshot-%Y%m%dT%H%M%SZ")
            result = self._git("commit", "--allow-empty", "--quiet", "-m", commit_msg)
            if result.returncode == 0:
                hash_result = self._git("rev-parse", "HEAD")
                return hash_result.stdout.strip()[:8] if hash_result.stdout.strip() else "empty"
            return "error"
        except Exception as e:
            return f"error: {e}"

    def restore(self, snapshot_hash: str) -> bool:
        """Restore workspace files to the given snapshot."""
        try:
            self._git("checkout", snapshot_hash, "--", ".")
            self._copy_snapshot_to_workspace()
            return True
        except Exception:
            return False

    def restore_latest(self) -> bool:
        """Restore to the most recent snapshot."""
        result = self._git("log", "--format=%H", "-n", "1")
        sha = result.stdout.strip()
        if not sha:
            return False
        return self.restore(sha)

    def diff(self, from_hash: Optional[str] = None) -> str:
        """Show diff between current state and a snapshot (or HEAD~1)."""
        ref = from_hash or "HEAD~1"
        result = self._git("diff", ref, "HEAD", "--stat")
        return result.stdout.strip() or "(no changes)"

    def diff_full(self, from_hash: Optional[str] = None) -> str:
        """Show full unified diff."""
        ref = from_hash or "HEAD~1"
        result = self._git("diff", ref, "HEAD")
        return result.stdout.strip()[:5000] or "(no changes)"

    def list_snapshots(self, limit: int = 10) -> list[dict]:
        """List recent snapshots."""
        result = self._git("log", f"--format=%H|%s|%ai", f"-n{limit}")
        snapshots = []
        for line in result.stdout.strip().split("\n"):
            if not line:
                continue
            parts = line.split("|", 2)
            if len(parts) == 3:
                snapshots.append({
                    "hash": parts[0][:8],
                    "message": parts[1],
                    "date": parts[2],
                })
        return snapshots

    def cleanup(self, keep: int = 50):
        """Remove old snapshots, keeping the most recent N.

        Uses git rebase to squash old commits into the earliest kept one,
        then garbage-collects to reclaim disk space.
        """
        try:
            all_hashes = [
                h for h in self._git("log", "--format=%H").stdout.strip().split("\n") if h
            ]
            total = len(all_hashes)
            if total <= keep:
                return

            # Keep the keep-most-recent commits, squash older ones
            oldest_keep = all_hashes[keep - 1] if keep > 0 else all_hashes[0]
            oldest_keep_parent = all_hashes[keep] if keep < total else None

            if oldest_keep_parent:
                # Create a replacement tree from the oldest kept commit
                self._git("checkout", "--orphan", "_snapshot_cleanup")
                # Copy the tree from oldest_keep
                self._git("reset", "--hard", oldest_keep)
                # Rebase remaining newer commits on top
                newer = list(reversed(all_hashes[:keep - 1])) if keep > 1 else []
                for sha in newer:
                    self._git("cherry-pick", sha)

                # Replace the branch
                self._git("branch", "-M", "_snapshot_cleanup", "main")
                self._git("gc", "--prune=now", "--aggressive")
        except Exception:
            pass

    def snapshot_count(self) -> int:
        """Return the number of snapshots stored."""
        result = self._git("rev-list", "--count", "HEAD")
        try:
            return int(result.stdout.strip())
        except (ValueError, TypeError):
            return 0


def _hash_path(path: str) -> str:
    """Produce a short hash of a path for directory naming."""
    return hashlib.sha256(path.encode()).hexdigest()[:16]
