import json
from datetime import datetime, timezone
from urllib.parse import urlsplit

from research.rag import split_text
from research.store import EvidenceStore


class DocumentStore(EvidenceStore):
    """保留 V9 表，新增文档快照与固定编号的分块。"""

    def __init__(self, conn):
        super().__init__(conn)

        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS research_documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_id TEXT NOT NULL REFERENCES research_tasks(id),
                title TEXT NOT NULL,
                url TEXT NOT NULL,
                content TEXT NOT NULL,
                source_type TEXT NOT NULL,
                metadata TEXT NOT NULL,
                retrieved_at TEXT NOT NULL,
                UNIQUE(task_id, url, source_type)
            );

            CREATE TABLE IF NOT EXISTS research_chunks (
                document_id INTEGER NOT NULL
                    REFERENCES research_documents(id),
                chunk_index INTEGER NOT NULL,
                start_char INTEGER NOT NULL,
                end_char INTEGER NOT NULL,
                content TEXT NOT NULL,
                PRIMARY KEY(document_id, chunk_index)
            );

            CREATE INDEX IF NOT EXISTS idx_documents_task
            ON research_documents(task_id);
        """)

    def add_document(self, task_id: str, item: dict) -> int:
        title = item["title"].strip()
        url = item["url"].strip()
        content = item["content"].strip()
        parsed = urlsplit(url)

        if (
            not title
            or not content
            or parsed.scheme not in {"http", "https"}
            or not parsed.hostname
        ):
            raise ValueError("文档标题、URL 或正文无效")

        metadata = {
            **item.get("metadata", {}),
            "truncated": len(content) > 12000,
        }
        content = content[:12000]

        # 文档与其全部分块必须同时保存成功。
        with self.conn:
            cursor = self.conn.execute(
                "INSERT INTO research_documents "
                "(task_id, title, url, content, source_type, "
                "metadata, retrieved_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(task_id, url, source_type) DO NOTHING",
                (
                    task_id,
                    title[:200],
                    url,
                    content,
                    item["source_type"],
                    json.dumps(metadata, ensure_ascii=False),
                    datetime.now(timezone.utc).isoformat(),
                ),
            )

            document_id = self.conn.execute(
                "SELECT id FROM research_documents "
                "WHERE task_id = ? AND url = ? AND source_type = ?",
                (task_id, url, item["source_type"]),
            ).fetchone()[0]

            if cursor.rowcount == 1:
                self.conn.executemany(
                    "INSERT INTO research_chunks "
                    "(document_id, chunk_index, start_char, end_char, content) "
                    "VALUES (?, ?, ?, ?, ?)",
                    [
                        (
                            document_id,
                            chunk["index"],
                            chunk["start"],
                            chunk["end"],
                            chunk["content"],
                        )
                        for chunk in split_text(content)
                    ],
                )

        return document_id

    def list_documents(self, task_id: str) -> list[dict]:
        rows = self.conn.execute(
            "SELECT id, title, url, source_type, metadata, retrieved_at "
            "FROM research_documents WHERE task_id = ? ORDER BY id",
            (task_id,),
        ).fetchall()

        result = []
        for row in rows:
            item = dict(zip(
                [
                    "id", "title", "url",
                    "source_type", "metadata", "retrieved_at",
                ],
                row,
            ))
            item["metadata"] = json.loads(item["metadata"])
            result.append(item)

        return result

    def list_chunks(self, task_id: str) -> list[dict]:
        rows = self.conn.execute(
            "SELECT d.id, c.chunk_index, d.title, d.url, c.content, "
            "d.source_type, d.metadata, d.retrieved_at, "
            "c.start_char, c.end_char "
            "FROM research_chunks c JOIN research_documents d "
            "ON d.id = c.document_id WHERE d.task_id = ? "
            "ORDER BY d.id, c.chunk_index",
            (task_id,),
        ).fetchall()

        result = []
        for row in rows:
            item = dict(zip(
                [
                    "document_id", "chunk_index", "title", "url", "content",
                    "source_type", "metadata", "retrieved_at", "start", "end",
                ],
                row,
            ))
            item["metadata"] = json.loads(item["metadata"])
            item["evidence_id"] = f"D{row[0]}C{row[1]}"
            result.append(item)

        return result