import json
from datetime import datetime, timezone
from urllib.parse import urlsplit

from research.review_store import ReviewStore
from research.sources import collect
from research.workflow import WorkflowResearchAgent


SOURCE_TYPES = {
    "web": {"web_text", "web_summary"},
    "paper": {"paper_abstract"},
    "github": {"github_readme", "github_metadata"},
}


class RecoveryStore(ReviewStore):
    """记录同一任务内各条查询的采集结果，供节点重放时复用。"""

    def __init__(self, conn):
        super().__init__(conn)
        with self.conn:
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS research_collections (
                    task_id TEXT NOT NULL REFERENCES research_tasks(id),
                    source TEXT NOT NULL,
                    query TEXT NOT NULL,
                    status TEXT NOT NULL
                        CHECK(status IN ('fetched', 'applied')),
                    documents TEXT NOT NULL,
                    warnings TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    applied_at TEXT,
                    PRIMARY KEY(task_id, source, query)
                )
            """)

    def get_collection(self, task_id, source, query):
        row = self.conn.execute(
            "SELECT status, documents, warnings, created_at, applied_at "
            "FROM research_collections "
            "WHERE task_id=? AND source=? AND query=?",
            (task_id, source, query),
        ).fetchone()
        if row is None:
            return None
        item = dict(zip(
            ["status", "documents", "warnings", "created_at", "applied_at"],
            row,
        ))
        item["documents"] = json.loads(item["documents"])
        item["warnings"] = json.loads(item["warnings"])
        return item

    def save_collection(self, task_id, source, query, documents, warnings):
        # 先校验整个结果，避免缓存无效文档导致恢复后反复失败。
        if source not in SOURCE_TYPES:
            raise ValueError("采集来源无效")
        if not isinstance(query, str) or not 1 <= len(query) <= 200:
            raise ValueError("采集查询无效")
        if not isinstance(documents, list):
            raise ValueError("采集 documents 必须为列表")
        if (
            not isinstance(warnings, list)
            or any(not isinstance(item, str) for item in warnings)
        ):
            raise ValueError("采集 warnings 必须为字符串列表")

        warnings = list(warnings)
        if not documents:
            warnings.append(f"{source}: 查询 {query} 没有返回有效文档")
        documents = documents[:3]

        for document in documents:
            if not isinstance(document, dict):
                raise ValueError("文档必须是对象")
            if any(
                not isinstance(document.get(key), str)
                or not document[key].strip()
                for key in ("title", "url", "content")
            ):
                raise ValueError("文档标题、URL 或正文无效")
            source_type = document.get("source_type")
            if (
                not isinstance(source_type, str)
                or source_type not in SOURCE_TYPES[source]
            ):
                raise ValueError("文档类型与采集来源不匹配")
            if not isinstance(document.get("metadata", {}), dict):
                raise ValueError("文档 metadata 必须是对象")
            parsed = urlsplit(document["url"].strip())
            if parsed.scheme not in {"http", "https"} or not parsed.hostname:
                raise ValueError("文档 URL 无效")

        payload = json.dumps(
            documents, ensure_ascii=False, allow_nan=False
        )
        warning_json = json.dumps(warnings, ensure_ascii=False)
        with self.conn:
            self.conn.execute(
                "INSERT INTO research_collections "
                "(task_id, source, query, status, documents, warnings, "
                "created_at) VALUES (?, ?, ?, 'fetched', ?, ?, ?) "
                "ON CONFLICT(task_id, source, query) DO NOTHING",
                (
                    task_id, source, query, payload, warning_json,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
        return self.get_collection(task_id, source, query)

    def mark_collection_applied(self, task_id, source, query):
        with self.conn:
            cursor = self.conn.execute(
                "UPDATE research_collections "
                "SET status='applied', applied_at=? "
                "WHERE task_id=? AND source=? AND query=? AND status='fetched'",
                (
                    datetime.now(timezone.utc).isoformat(),
                    task_id, source, query,
                ),
            )
        if cursor.rowcount != 1:
            raise RuntimeError("采集结果状态不符合预期")

    def list_collections(self, task_id):
        self.get_task(task_id)
        rows = self.conn.execute(
            "SELECT source, query, status, documents, warnings, "
            "created_at, applied_at FROM research_collections "
            "WHERE task_id=? ORDER BY created_at, source, query",
            (task_id,),
        ).fetchall()
        return [
            {
                "source": row[0],
                "query": row[1],
                "status": row[2],
                "document_count": len(json.loads(row[3])),
                "warnings": json.loads(row[4]),
                "created_at": row[5],
                "applied_at": row[6],
            }
            for row in rows
        ]


class RecoveryWorkflowAgent(WorkflowResearchAgent):
    VERSION = "13.0"

    def run(self, *args, **kwargs):
        raise ValueError("V13 请使用 start() / resume() 执行工作流")

    def resume(self, task_id, *, approved=None):
        if self.checkpoint_version(task_id) is None:
            raise ValueError("没有 checkpoint；planned 任务请先调用 start()")
        return super().resume(task_id, approved=approved)

    def checkpoint_version(self, task_id):
        # 版本路由前不能调用 inspect：父类会拒绝其他版本的状态。
        self.store.get_task(task_id)
        snapshot = self.graph.get_state(self._config(task_id))
        if not snapshot.values:
            return None
        version = snapshot.values.get("version")
        if not isinstance(version, str):
            raise ValueError("checkpoint 缺少有效的工作流版本")
        return version

    def _collect(self, task_id, queries, errors):
        for entry in queries:
            source, query = entry["source"], entry["query"]
            result = self.store.get_collection(task_id, source, query)

            if result is None:
                print(f"[Source] {source}: {query}")
                try:
                    documents, warnings = collect(source, query)
                    result = self.store.save_collection(
                        task_id, source, query, documents, warnings
                    )
                except Exception as error:
                    raise RuntimeError(
                        f"采集未完成（{source}: {query}）：{error}；"
                        f"使用 /resume {task_id} 重试"
                    ) from error
            else:
                print(f"[Cache] {source}: {query} | {result['status']}")

            if result["status"] == "fetched":
                # 每篇文档和分块由原 DocumentStore 原子保存。
                # 中途退出后可重放；原有唯一约束会保留已保存的文档。
                for document in result["documents"]:
                    self.store.add_document(task_id, document)
                self.store.mark_collection_applied(task_id, source, query)

            # 节点上次可能未完成，warnings 必须从缓存重新合并。
            for warning in result["warnings"]:
                if warning not in errors:
                    errors.append(warning)