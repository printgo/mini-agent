import json
import re

def _text(value, maximum=3000):
    if not isinstance(value, str) or not value.strip() or len(value) > maximum:
        raise ValueError("字段必须是规定长度内的非空字符串")
    return value.strip()

class KnowledgeStore:
    """独立研究记忆；用户命名空间隔离不等于身份认证。"""
    def __init__(self, conn):
        self.conn = conn
        with conn:
            conn.execute("""CREATE TABLE IF NOT EXISTS research_task_owners (
                task_id TEXT PRIMARY KEY REFERENCES research_tasks(id),
                user_id TEXT NOT NULL)""")
            conn.execute("""CREATE TABLE IF NOT EXISTS research_knowledge (
                user_id TEXT NOT NULL, kind TEXT NOT NULL, key TEXT NOT NULL,
                value TEXT NOT NULL, task_id TEXT, evidence_ids TEXT NOT NULL,
                confirmed INTEGER NOT NULL DEFAULT 0,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY(user_id,kind,key))""")
            conn.execute("""CREATE TABLE IF NOT EXISTS research_memory_migrations (
                name TEXT PRIMARY KEY,
                applied_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP)""")
        # 在 CLI 的 UPDATE/DELETE 之前完成旧数据迁移，避免首次打开聊天又恢复旧值。
        if conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='memories'"
        ).fetchone():
            self.seed_chat({
                category + "." + key: value
                for category, key, value in conn.execute(
                    "SELECT category,key,value FROM memories ORDER BY id"
                )
            })

    def seed_chat(self, memories):
        """只迁移一次 default 聊天记忆；迁移标记和插入在同一事务提交。"""
        with self.conn:
            inserted = self.conn.execute(
                "INSERT OR IGNORE INTO research_memory_migrations(name) VALUES (?)",
                ("chat-default-v19",),
            )
            if inserted.rowcount != 1:
                return
            for name, value in memories.items():
                category, _ = name.split(".", 1)
                kind = "topic" if category == "research" else "preference"
                self.conn.execute(
                    "INSERT OR IGNORE INTO research_knowledge "
                    "(user_id,kind,key,value,evidence_ids) VALUES ('default',?,?,?,'[]')",
                    (kind, name, str(value)),
                )

    def owner(self, task_id):
        row = self.conn.execute(
            "SELECT user_id FROM research_task_owners WHERE task_id=?", (task_id,)
        ).fetchone()
        return row[0] if row else None

    def assert_owner(self, user_id, task_id):
        if self.owner(task_id) != _text(user_id, 128):
            raise PermissionError("任务未绑定当前用户")
        return task_id

    def bind(self, user_id, task_id):
        user_id = _text(user_id, 128)
        if not self.conn.execute(
            "SELECT 1 FROM research_tasks WHERE id=?", (task_id,)
        ).fetchone():
            raise ValueError("任务不存在")
        with self.conn:
            self.conn.execute(
                "INSERT OR IGNORE INTO research_task_owners VALUES (?, ?)",
                (task_id, user_id),
            )
            self.assert_owner(user_id, task_id)

    def apply(self, user_id, action, kind, key, value="", *, task_id=None,
              evidence_ids=()):
        user_id, key = _text(user_id, 128), _text(key, 128)
        if kind not in {"preference", "topic", "conclusion"}:
            raise ValueError("kind 必须是 preference/topic/conclusion")
        if action not in {"ADD", "UPDATE", "KEEP", "DELETE"}:
            raise ValueError("不支持的记忆操作")
        if kind == "conclusion" and action in {"ADD", "UPDATE"}:
            raise ValueError("结论必须通过显式人工 confirm 写入")
        if task_id is not None:
            self.assert_owner(user_id, task_id)
        if evidence_ids:
            raise ValueError("仅人工确认结论允许保存证据引用")
        args = (user_id, kind, key)
        with self.conn:
            exists = self.conn.execute(
                "SELECT 1 FROM research_knowledge WHERE user_id=? AND kind=? AND key=?",
                args,
            ).fetchone()
            if action == "KEEP":
                return
            if action == "DELETE":
                self.conn.execute(
                    "DELETE FROM research_knowledge WHERE user_id=? AND kind=? AND key=?",
                    args,
                )
            elif action == "ADD":
                self.conn.execute(
                    "INSERT INTO research_knowledge "
                    "(user_id,kind,key,value,task_id,evidence_ids) VALUES (?,?,?,?,?,'[]')",
                    (*args, _text(value), task_id),
                )
            elif not exists:
                raise ValueError("UPDATE 目标不存在")
            else:
                self.conn.execute(
                    "UPDATE research_knowledge SET value=?,task_id=?,"
                    "updated_at=CURRENT_TIMESTAMP WHERE user_id=? AND kind=? AND key=?",
                    (_text(value), task_id, *args),
                )

    def confirm(self, user_id, task_id, key, text, evidence_ids):
        """只能由显式人工命令调用；不会从 Critic pass 自动触发。"""
        self.assert_owner(user_id, task_id)
        status = self.conn.execute(
            "SELECT status FROM research_tasks WHERE id=?", (task_id,)
        ).fetchone()
        if status is None or status[0] != "completed":
            raise ValueError("仅允许确认 completed 任务中的结论")
        if (not isinstance(evidence_ids, list) or not evidence_ids
                or any(not isinstance(x, str) for x in evidence_ids)):
            raise ValueError("确认结论必须提供非空证据编号列表")
        available = {row[0] for row in self.conn.execute(
            "SELECT 'D'||d.id||'C'||c.chunk_index FROM research_documents d "
            "JOIN research_chunks c ON c.document_id=d.id WHERE d.task_id=?",
            (task_id,),
        )}
        if not set(evidence_ids) <= available:
            raise ValueError("引用证据不属于当前任务")
        with self.conn:
            self.conn.execute(
                "INSERT INTO research_knowledge "
                "(user_id,kind,key,value,task_id,evidence_ids,confirmed) "
                "VALUES (?,'conclusion',?,?,?,?,1) "
                "ON CONFLICT(user_id,kind,key) DO UPDATE SET "
                "value=excluded.value,task_id=excluded.task_id,"
                "evidence_ids=excluded.evidence_ids,confirmed=1,"
                "updated_at=CURRENT_TIMESTAMP",
                (_text(user_id, 128), _text(key, 128), _text(text), task_id,
                 json.dumps(list(dict.fromkeys(evidence_ids)))),
            )

    def retrieve(self, user_id, query, limit=6):
        if type(limit) is not int or not 1 <= limit <= 20:
            raise ValueError("limit 必须为 1..20")
        terms = set(re.findall(r"[a-z0-9_]+|[\u4e00-\u9fff]", query.lower()))
        rows = []
        for row in self.conn.execute(
            "SELECT kind,key,value,task_id,evidence_ids,confirmed "
            "FROM research_knowledge WHERE user_id=? ORDER BY kind,key",
            (_text(user_id, 128),),
        ):
            kind, key, value, task, citations, confirmed = row
            score = len(terms & set(re.findall(
                r"[a-z0-9_]+|[\u4e00-\u9fff]", (key+" "+value).lower()
            )))
            if kind == "preference" or score:
                rows.append((score, {
                    "kind": kind, "key": key, "value": value,
                    "source_task_id": task, "evidence_ids": json.loads(citations),
                    "human_confirmed": bool(confirmed),
                }))
        rows.sort(key=lambda item: (-item[0], item[1]["kind"], item[1]["key"]))
        return [item for _, item in rows[:limit]]
