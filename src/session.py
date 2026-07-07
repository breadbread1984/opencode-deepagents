"""Lightweight session metadata store (SQLite).

deepagents handles conversation state via its LangGraph checkpointer.
This module stores session metadata: name, workspace, model, mode, thread_id,
permission settings, and snapshot references.
"""

import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

DB_PATH = Path.home() / ".opencode-deepagents" / "sessions.db"


def _get_db() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    _migrate(conn)
    return conn


def _migrate(conn: sqlite3.Connection):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            id TEXT PRIMARY KEY,
            thread_id TEXT NOT NULL UNIQUE,
            name TEXT NOT NULL DEFAULT 'Untitled',
            workspace TEXT NOT NULL DEFAULT '.',
            model TEXT NOT NULL DEFAULT 'gpt-4o',
            agent_mode TEXT NOT NULL DEFAULT 'build',
            hitl_enabled INTEGER NOT NULL DEFAULT 1,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
    """)

    # Add hitl_enabled column if missing (migration from old schema)
    cols = [c[1] for c in conn.execute("PRAGMA table_info(sessions)").fetchall()]
    if "hitl_enabled" not in cols:
        conn.execute("ALTER TABLE sessions ADD COLUMN hitl_enabled INTEGER NOT NULL DEFAULT 1")

    # Permission cache table
    conn.execute("""
        CREATE TABLE IF NOT EXISTS permission_cache (
            session_id TEXT NOT NULL,
            cache_key TEXT NOT NULL,
            action TEXT NOT NULL,  -- 'approve' or 'deny'
            created_at TEXT NOT NULL,
            PRIMARY KEY (session_id, cache_key),
            FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
        )
    """)

    # Snapshot log table
    conn.execute("""
        CREATE TABLE IF NOT EXISTS snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            snapshot_hash TEXT NOT NULL,
            label TEXT NOT NULL,
            message_index INTEGER,
            created_at TEXT NOT NULL,
            FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
        )
    """)

    conn.commit()


# ── Session CRUD ──

def create_session(
    name: str = "Untitled",
    workspace: str = ".",
    model: str = "qwen3.6-plus",
    agent_mode: str = "build",
) -> str:
    """Create a new session. Returns session ID."""
    session_id = uuid.uuid4().hex[:12]
    thread_id = f"session-{session_id}"
    now = datetime.now(timezone.utc).isoformat()

    conn = _get_db()
    try:
        conn.execute(
            """INSERT INTO sessions (id, thread_id, name, workspace, model, agent_mode, hitl_enabled, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, 1, ?, ?)""",
            (session_id, thread_id, name, workspace, model, agent_mode, now, now),
        )
        conn.commit()
        return session_id
    finally:
        conn.close()


def get_session(session_id: str) -> Optional[dict]:
    """Get session metadata by ID."""
    conn = _get_db()
    try:
        row = conn.execute(
            "SELECT * FROM sessions WHERE id = ?", (session_id,)
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def list_sessions() -> list[dict]:
    """List all sessions, most recently updated first."""
    conn = _get_db()
    try:
        rows = conn.execute(
            "SELECT * FROM sessions ORDER BY updated_at DESC"
        ).fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


def get_thread_id(session_id: str) -> Optional[str]:
    """Get the LangGraph thread_id for a session."""
    session = get_session(session_id)
    return session["thread_id"] if session else None


def update_session(session_id: str, **kwargs):
    """Update session metadata fields."""
    allowed = {"name", "model", "agent_mode", "workspace", "hitl_enabled"}
    updates = {k: v for k, v in kwargs.items() if k in allowed}
    if not updates:
        return

    updates["updated_at"] = datetime.now(timezone.utc).isoformat()
    set_clause = ", ".join(f"{k} = ?" for k in updates)
    values = list(updates.values()) + [session_id]

    conn = _get_db()
    try:
        conn.execute(f"UPDATE sessions SET {set_clause} WHERE id = ?", values)
        conn.commit()
    finally:
        conn.close()


def delete_session(session_id: str):
    """Delete a session and its cached permissions/snapshots."""
    conn = _get_db()
    try:
        conn.execute("DELETE FROM permission_cache WHERE session_id = ?", (session_id,))
        conn.execute("DELETE FROM snapshots WHERE session_id = ?", (session_id,))
        conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
        conn.commit()
    finally:
        conn.close()


# ── Permission Cache ──

def save_permission_cache(session_id: str, key: str, action: str):
    """Save a permission decision (approve/deny) for a tool+pattern."""
    conn = _get_db()
    try:
        now = datetime.now(timezone.utc).isoformat()
        conn.execute(
            """INSERT OR REPLACE INTO permission_cache (session_id, cache_key, action, created_at)
               VALUES (?, ?, ?, ?)""",
            (session_id, key, action, now),
        )
        conn.commit()
    finally:
        conn.close()


def load_permission_cache(session_id: str) -> dict[str, str]:
    """Load all cached permission decisions for a session."""
    conn = _get_db()
    try:
        rows = conn.execute(
            "SELECT cache_key, action FROM permission_cache WHERE session_id = ?",
            (session_id,),
        ).fetchall()
        return {r["cache_key"]: r["action"] for r in rows}
    finally:
        conn.close()


def clear_permission_cache(session_id: str):
    """Clear all cached permission decisions for a session."""
    conn = _get_db()
    try:
        conn.execute("DELETE FROM permission_cache WHERE session_id = ?", (session_id,))
        conn.commit()
    finally:
        conn.close()


# ── Snapshot Log ──

def log_snapshot(session_id: str, snapshot_hash: str, label: str, message_index: Optional[int] = None):
    """Record a filesystem snapshot for tracking."""
    conn = _get_db()
    try:
        now = datetime.now(timezone.utc).isoformat()
        conn.execute(
            """INSERT INTO snapshots (session_id, snapshot_hash, label, message_index, created_at)
               VALUES (?, ?, ?, ?, ?)""",
            (session_id, snapshot_hash, label, message_index, now),
        )
        conn.commit()
    finally:
        conn.close()
