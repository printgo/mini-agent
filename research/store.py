import json
import sqlite3
from datetime import datetime, timezone
from uuid import uuid4
from urllib.parse import urlsplit


class EvidenceStore:
    """研究任务和证据使用独立表，复用调用方的 SQLite 连接。"""

    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS research_tasks (
                id TEXT PRIMARY KEY,
                question TEXT NOT NULL,
                status TEXT NOT NULL,
                queries TEXT NOT NULL DEFAULT '[]',
                report TEXT NOT NULL DEFAULT '',
                errors TEXT NOT NULL DEFAULT '[]',
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS research_evidence (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_id TEXT NOT NULL REFERENCES research_tasks(id),
                title TEXT NOT NULL,
                url TEXT NOT NULL,
                content TEXT NOT NULL,
                search_query TEXT NOT NULL,
                source_type TEXT NOT NULL DEFAULT 'search_summary',
                retrieved_at TEXT NOT NULL,
                UNIQUE(task_id, url)
            );
        """)

    def create_task(self, question: str) -> str:
        task_id = uuid4().hex
        with self.conn:
            self.conn.execute(
                "INSERT INTO research_tasks "
                "(id, question, status, created_at) VALUES (?, ?, ?, ?)",
                (
                    task_id,
                    question,
                    "planning",
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
        return task_id

    def get_task(self, task_id: str) -> dict:
        row = self.conn.execute(
            "SELECT id, question, status, queries, report, errors, created_at "
            "FROM research_tasks WHERE id = ?",
            (task_id,),
        ).fetchone()

        if row is None:
            raise ValueError("研究任务不存在")

        task = dict(zip(
            [
                "id", "question", "status", "queries",
                "report", "errors", "created_at",
            ],
            row,
        ))
        task["queries"] = json.loads(task["queries"])
        task["errors"] = json.loads(task["errors"])
        return task

    def update_task(
        self,
        task_id: str,
        status: str,
        *,
        queries: list[str | dict] | None = None,
        report: str | None = None,
        errors: list[str] | None = None,
    ) -> None:
        task = self.get_task(task_id)

        with self.conn:
            self.conn.execute(
                "UPDATE research_tasks "
                "SET status = ?, queries = ?, report = ?, errors = ? WHERE id = ?",
                (
                    status,
                    json.dumps(
                        task["queries"] if queries is None else queries,
                        ensure_ascii=False,
                    ),
                    task["report"] if report is None else report,
                    json.dumps(
                        task["errors"] if errors is None else errors,
                        ensure_ascii=False,
                    ),
                    task_id,
                ),
            )

    def add(self, task_id: str, query: str, item: dict) -> bool:
        """接受有效摘要；同一任务的重复 URL 保留原记录。"""
        if not isinstance(item, dict):
            return False

        url = item.get("url")
        content = item.get("content")
        if not isinstance(url, str) or not isinstance(content, str):
            return False

        url, content = url.strip(), content.strip()

        try:
            parsed = urlsplit(url)
            if parsed.scheme not in {"http", "https"} or not parsed.hostname:
                return False
        except ValueError:
            return False

        if not content:
            return False

        title = item.get("title")
        title = title.strip() if isinstance(title, str) else ""

        with self.conn:
            self.conn.execute(
                "INSERT INTO research_evidence "
                "(task_id, title, url, content, search_query, retrieved_at) "
                "VALUES (?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(task_id, url) DO NOTHING",
                (
                    task_id,
                    (title or url)[:200],
                    url,
                    content[:1500],
                    query,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
        return True

    def list_evidence(self, task_id: str) -> list[dict]:
        rows = self.conn.execute(
            "SELECT title, url, content, search_query, source_type, retrieved_at "
            "FROM research_evidence WHERE task_id = ? ORDER BY id",
            (task_id,),
        ).fetchall()

        columns = [
            "title", "url", "content",
            "search_query", "source_type", "retrieved_at",
        ]
        return [
            {"evidence_id": f"E{number}", **dict(zip(columns, row))}
            for number, row in enumerate(rows, 1)
        ]

    def list_tasks(self) -> list[dict]:
        rows = self.conn.execute(
            "SELECT id, question, status FROM research_tasks "
            "ORDER BY created_at DESC LIMIT 10"
        ).fetchall()
        return [
            dict(zip(["id", "question", "status"], row))
            for row in rows
        ]