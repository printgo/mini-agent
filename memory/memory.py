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

os.makedirs(
    DATA_DIR,
    exist_ok=True
)


class Memory:

    def __init__(self):
        self.conn = sqlite3.connect(
            DB_PATH
        )

        self._create_tables()

    def _create_tables(self):
        cursor = self.conn.cursor()

        # 当前 Session 的历史消息
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
                category TEXT NOT NULL,
                key TEXT NOT NULL,
                value TEXT NOT NULL,
                source_session_id TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(category, key)
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
        """
        保存当前 Session 的聊天消息。
        """

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
    ) -> list[dict]:
        """
        恢复指定 Session 的聊天历史。
        """

        cursor = self.conn.cursor()

        cursor.execute(
            """
            SELECT role, content
            FROM messages
            WHERE session_id = ?
            ORDER BY id ASC
            """,
            (
                session_id,
            )
        )

        rows = cursor.fetchall()

        return [
            {
                "role": role,
                "content": content
            }
            for role, content in rows
        ]

    def get_memory(
        self,
        category: str,
        key: str
    ) -> dict | None:
        """
        查询一条长期记忆。
        """

        cursor = self.conn.cursor()

        cursor.execute(
            """
            SELECT
                category,
                key,
                value,
                source_session_id
            FROM memories
            WHERE category = ?
              AND key = ?
            LIMIT 1
            """,
            (
                category,
                key
            )
        )

        row = cursor.fetchone()

        if row is None:
            return None

        return {
            "category": row[0],
            "key": row[1],
            "value": row[2],
            "source_session_id": row[3]
        }

    def add_memory(
        self,
        category: str,
        key: str,
        value: str,
        source_session_id: str | None = None
    ):
        """
        新增长期记忆。
        """

        cursor = self.conn.cursor()

        cursor.execute(
            """
            INSERT INTO memories (
                category,
                key,
                value,
                source_session_id
            )
            VALUES (?, ?, ?, ?)
            """,
            (
                category,
                key,
                value,
                source_session_id
            )
        )

        self.conn.commit()

    def update_memory(
        self,
        category: str,
        key: str,
        value: str,
        source_session_id: str | None = None
    ):
        """
        更新已有长期记忆。
        """

        cursor = self.conn.cursor()

        cursor.execute(
            """
            UPDATE memories
            SET
                value = ?,
                source_session_id = ?,
                updated_at = CURRENT_TIMESTAMP
            WHERE category = ?
              AND key = ?
            """,
            (
                value,
                source_session_id,
                category,
                key
            )
        )

        self.conn.commit()

    def delete_memory(
        self,
        category: str,
        key: str
    ):
        """
        删除一条长期记忆。
        """

        cursor = self.conn.cursor()

        cursor.execute(
            """
            DELETE FROM memories
            WHERE category = ?
              AND key = ?
            """,
            (
                category,
                key
            )
        )

        self.conn.commit()

    def get_memories(self) -> dict[str, str]:
        """
        获取全部长期记忆。

        返回示例：

        {
            "profile.name": "Tom",
            "career.direction": "Java后端开发"
        }
        """

        cursor = self.conn.cursor()

        cursor.execute(
            """
            SELECT
                category,
                key,
                value
            FROM memories
            ORDER BY id ASC
            """
        )

        rows = cursor.fetchall()

        return {
            f"{category}.{key}": value
            for category, key, value in rows
        }