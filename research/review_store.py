import json
from datetime import datetime, timezone

from research.documents import DocumentStore


class ReviewStore(DocumentStore):
    def __init__(self, conn):
        super().__init__(conn)

        with self.conn:
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS research_reviews (
                    task_id TEXT NOT NULL REFERENCES research_tasks(id),
                    round_no INTEGER NOT NULL,
                    evidence_ids TEXT NOT NULL,
                    draft TEXT NOT NULL,
                    review TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY(task_id, round_no)
                )
            """)

    def save_review(self, task_id, round_no, evidence, draft, review):
        with self.conn:
            self.conn.execute(
                "INSERT INTO research_reviews "
                "(task_id, round_no, evidence_ids, draft, review, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(task_id, round_no) DO UPDATE SET "
                "evidence_ids=excluded.evidence_ids, draft=excluded.draft, "
                "review=excluded.review",
                (
                    task_id,
                    round_no,
                    json.dumps([
                        item["evidence_id"] for item in evidence
                    ]),
                    json.dumps(draft, ensure_ascii=False),
                    json.dumps(review, ensure_ascii=False),
                    datetime.now(timezone.utc).isoformat(),
                ),
            )

    def list_reviews(self, task_id):
        self.get_task(task_id)

        rows = self.conn.execute(
            "SELECT round_no, evidence_ids, draft, review, created_at "
            "FROM research_reviews WHERE task_id=? ORDER BY round_no",
            (task_id,),
        ).fetchall()

        result = []

        for row in rows:
            item = dict(zip(
                [
                    "round_no", "evidence_ids",
                    "draft", "review", "created_at",
                ],
                row,
            ))

            for key in ("evidence_ids", "draft", "review"):
                item[key] = json.loads(item[key])

            result.append(item)

        return result