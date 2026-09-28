import os
import sqlite3


DB_PATH = "data/memory.db"


class Memory:

    def __init__(self):

        os.makedirs(
            "data",
            exist_ok=True
        )

        self.conn = sqlite3.connect(
            DB_PATH
        )

        self._create_table()

    def _create_table(self):

        cursor = self.conn.cursor()

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )

        self.conn.commit()

    def add_message(
        self,
        role: str,
        content: str
    ):

        cursor = self.conn.cursor()

        cursor.execute(
            """
            INSERT INTO messages (
                role,
                content
            )
            VALUES (?, ?)
            """,
            (
                role,
                content
            )
        )

        self.conn.commit()

    def get_messages(self):

        cursor = self.conn.cursor()

        cursor.execute(
            """
            SELECT role, content
            FROM messages
            ORDER BY id ASC
            """
        )

        rows = cursor.fetchall()

        return [
            {
                "role": role,
                "content": content
            }
            for role, content in rows
        ]

    def clear(self):

        cursor = self.conn.cursor()

        cursor.execute(
            "DELETE FROM messages"
        )

        self.conn.commit()