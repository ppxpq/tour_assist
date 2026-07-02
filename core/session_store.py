from __future__ import annotations

import sqlite3
import time
import uuid
from contextlib import closing
from pathlib import Path
from typing import Any

from utils import config


SESSION_DB_FILE = Path(config.BASE_DIR) / "data" / "sessions.sqlite3"


def _now_text() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _normalize_message_row(row: sqlite3.Row) -> dict[str, str]:
    return {
        "role": str(row["role"]),
        "content": str(row["content"] or ""),
        "created_at": str(row["created_at"] or ""),
    }


class SessionStore:
    def __init__(self, db_path: Path = SESSION_DB_FILE) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=5)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA busy_timeout = 5000")
        conn.execute("PRAGMA journal_mode = WAL")
        return conn

    def _init_schema(self) -> None:
        with closing(self._connect()) as conn:
            with conn:
                conn.executescript(
                    """
                    CREATE TABLE IF NOT EXISTS sessions (
                        id TEXT PRIMARY KEY,
                        title TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    );

                    CREATE TABLE IF NOT EXISTS messages (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT NOT NULL,
                        role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
                        content TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
                    );

                    CREATE INDEX IF NOT EXISTS idx_messages_session_id
                        ON messages(session_id, id);

                    CREATE TABLE IF NOT EXISTS session_metadata (
                        key TEXT PRIMARY KEY,
                        value TEXT NOT NULL
                    );
                    """
                )

    def create_session(self, title: str | None = None, session_id: str | None = None) -> dict[str, Any]:
        session_id = session_id or uuid.uuid4().hex
        now = _now_text()

        with closing(self._connect()) as conn:
            with conn:
                resolved_title = title or "新会话"
                conn.execute(
                    """
                    INSERT INTO sessions(id, title, created_at, updated_at)
                    VALUES(?, ?, ?, ?)
                    """,
                    (session_id, resolved_title, now, now),
                )
                self._set_metadata(conn, "current_session", session_id)

        return {
            "id": session_id,
            "title": resolved_title,
            "message_count": 0,
            "created_at": now,
            "updated_at": now,
        }

    def ensure_default_session(self) -> dict[str, Any]:
        first = self.first_session_id()
        if first:
            return self.get_session(first)
        return self.create_session("新会话")

    def list_sessions(self) -> list[dict[str, Any]]:
        with closing(self._connect()) as conn:
            rows = conn.execute(
                """
                SELECT
                    s.id,
                    s.title,
                    s.created_at,
                    s.updated_at,
                    COUNT(m.id) AS message_count
                FROM sessions s
                LEFT JOIN messages m ON m.session_id = s.id
                GROUP BY s.id
                ORDER BY s.created_at ASC, s.rowid ASC
                """
            ).fetchall()
        return [self._summary_from_row(row) for row in rows]

    def get_session(self, session_id: str) -> dict[str, Any]:
        summary = self.get_session_summary(session_id)
        if summary is None:
            raise KeyError(session_id)
        return {
            **summary,
            "messages": self.get_session_messages(session_id),
        }

    def get_session_summary(self, session_id: str) -> dict[str, Any] | None:
        with closing(self._connect()) as conn:
            row = conn.execute(
                """
                SELECT
                    s.id,
                    s.title,
                    s.created_at,
                    s.updated_at,
                    COUNT(m.id) AS message_count
                FROM sessions s
                LEFT JOIN messages m ON m.session_id = s.id
                WHERE s.id = ?
                GROUP BY s.id
                """,
                (session_id,),
            ).fetchone()
        return self._summary_from_row(row) if row else None

    def get_session_messages(self, session_id: str) -> list[dict[str, str]]:
        with closing(self._connect()) as conn:
            if not self._session_exists(conn, session_id):
                raise KeyError(session_id)
            rows = conn.execute(
                """
                SELECT role, content, created_at
                FROM messages
                WHERE session_id = ?
                ORDER BY id ASC
                """,
                (session_id,),
            ).fetchall()
        return [_normalize_message_row(row) for row in rows]

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        created_at: str | None = None,
    ) -> dict[str, Any]:
        if role not in {"user", "assistant"}:
            raise ValueError(f"unsupported message role: {role}")

        created_at = created_at or _now_text()
        with closing(self._connect()) as conn:
            with conn:
                if not self._session_exists(conn, session_id):
                    raise KeyError(session_id)
                cursor = conn.execute(
                    """
                    INSERT INTO messages(session_id, role, content, created_at)
                    VALUES(?, ?, ?, ?)
                    """,
                    (session_id, role, content, created_at),
                )
                conn.execute(
                    "UPDATE sessions SET updated_at = ? WHERE id = ?",
                    (created_at, session_id),
                )
                self._set_metadata(conn, "current_session", session_id)
                message_id = int(cursor.lastrowid)

        return {
            "id": message_id,
            "session_id": session_id,
            "role": role,
            "content": content,
            "created_at": created_at,
        }

    def update_session(
        self,
        session_id: str,
        *,
        title: str | None = None,
        updated_at: str | None = None,
    ) -> dict[str, Any]:
        fields: list[str] = []
        values: list[Any] = []
        if title is not None:
            fields.append("title = ?")
            values.append(title)
        if updated_at is not None:
            fields.append("updated_at = ?")
            values.append(updated_at)

        if fields:
            with closing(self._connect()) as conn:
                with conn:
                    if not self._session_exists(conn, session_id):
                        raise KeyError(session_id)
                    values.append(session_id)
                    conn.execute(
                        f"UPDATE sessions SET {', '.join(fields)} WHERE id = ?",
                        values,
                    )

        summary = self.get_session_summary(session_id)
        if summary is None:
            raise KeyError(session_id)
        return summary

    def delete_session(self, session_id: str) -> dict[str, Any]:
        with closing(self._connect()) as conn:
            with conn:
                if not self._session_exists(conn, session_id):
                    raise KeyError(session_id)
                conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
                current_session_id = self._get_metadata(conn, "current_session")
                if current_session_id == session_id:
                    self._set_metadata(conn, "current_session", self._first_session_id(conn))

        next_session_id = self.get_current_session_id()
        if next_session_id:
            return {"current_session": next_session_id}
        return {"current_session": ""}

    def clear_session_messages(self, session_id: str) -> dict[str, Any]:
        now = _now_text()
        with closing(self._connect()) as conn:
            with conn:
                if not self._session_exists(conn, session_id):
                    raise KeyError(session_id)
                conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
                conn.execute(
                    "UPDATE sessions SET updated_at = ? WHERE id = ?",
                    (now, session_id),
                )
                self._set_metadata(conn, "current_session", session_id)
        return self.get_session(session_id)

    def first_session_id(self) -> str:
        with closing(self._connect()) as conn:
            return self._first_session_id(conn)

    def get_current_session_id(self) -> str:
        with closing(self._connect()) as conn:
            session_id = self._get_metadata(conn, "current_session")
            if session_id and self._session_exists(conn, session_id):
                return session_id
            return self._first_session_id(conn)

    def set_current_session(self, session_id: str) -> None:
        with closing(self._connect()) as conn:
            with conn:
                if not self._session_exists(conn, session_id):
                    raise KeyError(session_id)
                self._set_metadata(conn, "current_session", session_id)

    @staticmethod
    def _summary_from_row(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "id": str(row["id"]),
            "title": str(row["title"] or "新会话"),
            "message_count": int(row["message_count"] or 0),
            "created_at": str(row["created_at"] or ""),
            "updated_at": str(row["updated_at"] or ""),
        }

    @staticmethod
    def _session_exists(conn: sqlite3.Connection, session_id: str) -> bool:
        row = conn.execute(
            "SELECT 1 FROM sessions WHERE id = ?",
            (session_id,),
        ).fetchone()
        return row is not None

    @staticmethod
    def _first_session_id(conn: sqlite3.Connection) -> str:
        row = conn.execute(
            """
            SELECT id
            FROM sessions
            ORDER BY created_at ASC, rowid ASC
            LIMIT 1
            """
        ).fetchone()
        return str(row["id"]) if row else ""

    @staticmethod
    def _get_metadata(conn: sqlite3.Connection, key: str) -> str:
        row = conn.execute(
            "SELECT value FROM session_metadata WHERE key = ?",
            (key,),
        ).fetchone()
        return str(row["value"]) if row else ""

    @staticmethod
    def _set_metadata(conn: sqlite3.Connection, key: str, value: str) -> None:
        conn.execute(
            """
            INSERT INTO session_metadata(key, value)
            VALUES(?, ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value
            """,
            (key, value),
        )
