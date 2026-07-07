"""Lightweight session metadata store (SQLite).

deepagents handles conversation state via its LangGraph checkpointer.
This module stores session metadata: name, workspace, model, mode, thread_id.
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
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
    """)
    conn.commit()


def create_session(
    name: str = "Untitled",
    workspace: str = ".",
    model: str = "gpt-4o",
    agent_mode: str = "build",
) -> str:
    """Create a new session. Returns session ID."""
    session_id = uuid.uuid4().hex[:12]
    thread_id = f"session-{session_id}"
    now = datetime.now(timezone.utc).isoformat()

    conn = _get_db()
    try:
        conn.execute(
            """INSERT INTO sessions (id, thread_id, name, workspace, model, agent_mode, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
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
    allowed = {"name", "model", "agent_mode", "workspace"}
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
    """Delete a session."""
    conn = _get_db()
    try:
        conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
        conn.commit()
    finally:
        conn.close()
