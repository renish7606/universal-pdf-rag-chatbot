"""Local SQLite storage for PDF chat conversations."""

import json
import sqlite3
import uuid
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

DB_PATH = Path("data/chat_history.db")


def _connect(db_path: Path = DB_PATH) -> sqlite3.Connection:
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def initialize(db_path: Path = DB_PATH) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with closing(_connect(db_path)) as connection, connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS chats (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                pdf_name TEXT NOT NULL,
                pdf_hash TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chat_id TEXT NOT NULL REFERENCES chats(id) ON DELETE CASCADE,
                role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
                content TEXT NOT NULL,
                sources_json TEXT NOT NULL DEFAULT '[]',
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            CREATE INDEX IF NOT EXISTS messages_by_chat
                ON messages(chat_id, id);
            CREATE INDEX IF NOT EXISTS chats_by_recent
                ON chats(updated_at DESC);
            """
        )


def create_chat(pdf_name: str, pdf_hash: str, db_path: Path = DB_PATH) -> Dict[str, Any]:
    chat = {
        "id": uuid.uuid4().hex,
        "title": pdf_name,
        "pdf_name": pdf_name,
        "pdf_hash": pdf_hash,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="microseconds"),
    }
    with closing(_connect(db_path)) as connection, connection:
        connection.execute(
            """
            INSERT INTO chats (id, title, pdf_name, pdf_hash, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                chat["id"],
                chat["title"],
                pdf_name,
                pdf_hash,
                chat["created_at"],
                chat["created_at"],
            ),
        )
    return chat


def list_chats(db_path: Path = DB_PATH) -> List[Dict[str, Any]]:
    with closing(_connect(db_path)) as connection, connection:
        rows = connection.execute(
            """
            SELECT id, title, pdf_name, pdf_hash, created_at, updated_at
            FROM chats
            ORDER BY updated_at DESC, id DESC
            """
        ).fetchall()
    return [dict(row) for row in rows]


def get_chat(chat_id: str, db_path: Path = DB_PATH) -> Optional[Dict[str, Any]]:
    with closing(_connect(db_path)) as connection, connection:
        row = connection.execute(
            """
            SELECT id, title, pdf_name, pdf_hash, created_at, updated_at
            FROM chats WHERE id = ?
            """,
            (chat_id,),
        ).fetchone()
    return dict(row) if row is not None else None


def get_messages(chat_id: str, db_path: Path = DB_PATH) -> List[Dict[str, Any]]:
    with closing(_connect(db_path)) as connection, connection:
        rows = connection.execute(
            """
            SELECT role, content, sources_json
            FROM messages WHERE chat_id = ? ORDER BY id
            """,
            (chat_id,),
        ).fetchall()
    return [
        {
            "role": row["role"],
            "content": row["content"],
            "sources": json.loads(row["sources_json"]),
        }
        for row in rows
    ]


def add_message(
    chat_id: str,
    role: str,
    content: str,
    sources: Optional[List[Any]] = None,
    db_path: Path = DB_PATH,
) -> None:
    updated_at = datetime.now(timezone.utc).isoformat(timespec="microseconds")
    with closing(_connect(db_path)) as connection, connection:
        chat = connection.execute(
            "SELECT title FROM chats WHERE id = ?", (chat_id,)
        ).fetchone()
        if chat is None:
            raise ValueError(f"Chat '{chat_id}' does not exist.")
        connection.execute(
            """
            INSERT INTO messages (chat_id, role, content, sources_json)
            VALUES (?, ?, ?, ?)
            """,
            (chat_id, role, content, json.dumps(sources or [])),
        )
        if role == "user":
            user_count = connection.execute(
                "SELECT COUNT(*) FROM messages WHERE chat_id = ? AND role = 'user'",
                (chat_id,),
            ).fetchone()[0]
            if user_count == 1:
                title = content.strip() or chat["title"]
                if len(title) > 52:
                    title = title[:49].rstrip() + "..."
                connection.execute(
                    "UPDATE chats SET title = ? WHERE id = ?", (title, chat_id)
                )
        connection.execute(
            "UPDATE chats SET updated_at = ? WHERE id = ?",
            (updated_at, chat_id),
        )
