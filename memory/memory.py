import os
import sqlite3


BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

DATA_DIR = os.path.join(
    BASE_DIR,
    "data"
)

DB_PATH = os.path.join(
    DATA_DIR,
    "memory.db"
)


class Memory:

    def __init__(self):

        os.makedirs(
            DATA_DIR,
            exist_ok=True
        )

        self.conn = sqlite3.connect(
            DB_PATH
        )

        self._create_tables()

    def _create_tables(self):

        cursor = self.conn.cursor()

        # 会话消息
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )

        # 长期记忆
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                key TEXT NOT NULL UNIQUE,
                value TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )

        self.conn.commit()

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str
    ):

        cursor = self.conn.cursor()

        cursor.execute(
            """
            INSERT INTO messages (
                session_id,
                role,
                content
            )
            VALUES (?, ?, ?)
            """,
            (
                session_id,
                role,
                content
            )
        )

        self.conn.commit()

    def get_messages(
        self,
        session_id: str
    ):

        cursor = self.conn.cursor()

        cursor.execute(
            """
            SELECT role, content
            FROM messages
            WHERE session_id = ?
            ORDER BY id ASC
            """,
            (session_id,)
        )

        rows = cursor.fetchall()

        return [
            {
                "role": role,
                "content": content
            }
            for role, content in rows
        ]

    def save_memory(
        self,
        key: str,
        value: str
    ):

        cursor = self.conn.cursor()

        cursor.execute(
            """
            INSERT INTO memories (
                key,
                value
            )
            VALUES (?, ?)

            ON CONFLICT(key)
            DO UPDATE SET
                value = excluded.value,
                updated_at = CURRENT_TIMESTAMP
            """,
            (
                key,
                value
            )
        )

        self.conn.commit()

    def get_memories(self):

        cursor = self.conn.cursor()

        cursor.execute(
            """
            SELECT key, value
            FROM memories
            ORDER BY id ASC
            """
        )

        rows = cursor.fetchall()

        return {
            key: value
            for key, value in rows
        }