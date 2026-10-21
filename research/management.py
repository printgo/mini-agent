from datetime import datetime, timedelta, timezone

from research.runtime import TaskFileLock, TaskStateError


class BudgetExceeded(TaskStateError):
    pass


class BudgetLedger:
    MODEL_LIMIT = 12
    QUERY_LIMIT = 8

    def __init__(self, conn):
        self.conn = conn
        with conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS research_budgets (
                    task_id TEXT PRIMARY KEY REFERENCES research_tasks(id),
                    model_limit INTEGER NOT NULL,
                    query_limit INTEGER NOT NULL,
                    model_used INTEGER NOT NULL DEFAULT 0,
                    query_used INTEGER NOT NULL DEFAULT 0,
                    CHECK(model_used >= 0 AND model_limit >= model_used),
                    CHECK(query_used >= 0 AND query_limit >= query_used)
                )
            """)

    def _ensure(self, task_id):
        self.conn.execute(
            "INSERT OR IGNORE INTO research_budgets "
            "(task_id, model_limit, query_limit) VALUES (?, ?, ?)",
            (task_id, self.MODEL_LIMIT, self.QUERY_LIMIT),
        )

    def get(self, task_id):
        row = self.conn.execute(
            "SELECT model_limit, query_limit, model_used, query_used "
            "FROM research_budgets WHERE task_id=?",
            (task_id,),
        ).fetchone()
        values = row or (self.MODEL_LIMIT, self.QUERY_LIMIT, 0, 0)
        return dict(zip(
            ["model_limit", "query_limit", "model_used", "query_used"], values
        ))

    def reserve(self, task_id, kind):
        if kind not in {"model", "query"}:
            raise ValueError("未知预算类型")
        with self.conn:
            self._ensure(task_id)
            cursor = self.conn.execute(
                f"UPDATE research_budgets SET {kind}_used={kind}_used+1 "
                f"WHERE task_id=? AND {kind}_used < {kind}_limit",
                (task_id,),
            )
            if cursor.rowcount != 1:
                raise BudgetExceeded(
                    f"{kind} 调用额度已用尽；"
                    f"使用 /limits {task_id} 模型上限 查询上限 调整后恢复"
                )
        # 预占成功并提交后才调用外部接口；失败或崩溃不退还。

    def set_limits(self, task_id, model_limit, query_limit):
        if any(
            type(value) is not int or not 0 <= value <= 1_000_000
            for value in (model_limit, query_limit)
        ):
            raise TaskStateError("调用上限必须为 0～1000000 的整数")
        with self.conn:
            self._ensure(task_id)
            cursor = self.conn.execute(
                "UPDATE research_budgets SET model_limit=?, query_limit=? "
                "WHERE task_id=? AND model_used<=? AND query_used<=?",
                (
                    model_limit, query_limit, task_id,
                    model_limit, query_limit,
                ),
            )
            if cursor.rowcount != 1:
                raise TaskStateError("新上限不能小于已使用次数")
        return self.get(task_id)


class TaskMaintenance:
    def __init__(self, controller, ledger):
        self.controller = controller
        self.store = controller.store
        self.conn = self.store.conn
        self.ledger = ledger

    def usage(self, task_id):
        self.store.get_task(task_id)
        version = self.controller.research.checkpoint_version(task_id)
        if version not in {None, "16.0", "19.0"}:
            return {"metered": False, "version": version, "note": "旧版本未计量"}
        return {
            "metered": True,
            **self.ledger.get(task_id),
            "scope": "写作/审查模型调用及采集查询尝试；规划和内部请求另计",
        }

    def set_limits(self, task_id, model_limit, query_limit):
        with TaskFileLock(self.controller.db_path, task_id):
            info = self.controller.inspect(task_id)
            if info["version"] not in {None, "16.0", "19.0"}:
                raise TaskStateError("只支持设置 V16/V19 任务的调用额度")
            if not info["resume_allowed"]:
                raise TaskStateError("只能设置 planned 或可恢复任务的额度")
            return self.ledger.set_limits(task_id, model_limit, query_limit)

    def cache_stats(self, task_id):
        self.store.get_task(task_id)
        result = {}
        for name, table, size in (
            (
                "collections", "research_collections",
                "length(CAST(documents AS BLOB))+length(CAST(warnings AS BLOB))",
            ),
            ("generations", "research_generations", "length(CAST(result AS BLOB))"),
        ):
            row = self.conn.execute(
                f"SELECT COUNT(*), COALESCE(SUM({size}), 0) "
                f"FROM {table} WHERE task_id=?",
                (task_id,),
            ).fetchone()
            result[name] = {"rows": row[0], "json_bytes": row[1]}
        return result

    def purge_cache(self, task_id, older_than_days=0):
        if (
            type(older_than_days) is not int
            or not 0 <= older_than_days <= 36500
        ):
            raise TaskStateError("清理天数必须为 0～36500 的整数")
        with TaskFileLock(self.controller.db_path, task_id):
            info = self.controller.inspect(task_id)
            finished = (
                info["status"] in {"completed", "partial", "cancelled", "failed"}
                and not info["next"]
            )
            if not (info["cancelled_at"] or finished):
                raise TaskStateError("可恢复任务的缓存不能整体清理")
            suffix, arguments = "", [task_id]
            if older_than_days:
                cutoff = datetime.now(timezone.utc) - timedelta(
                    days=older_than_days
                )
                suffix = " AND created_at < ?"
                arguments.append(cutoff.isoformat())
            deleted = {}
            with self.conn:
                for name, table in (
                    ("collections", "research_collections"),
                    ("generations", "research_generations"),
                ):
                    cursor = self.conn.execute(
                        f"DELETE FROM {table} WHERE task_id=?" + suffix,
                        arguments,
                    )
                    deleted[name] = cursor.rowcount
            return deleted

    def invalidate_current_generation(self, task_id):
        with TaskFileLock(self.controller.db_path, task_id):
            info = self.controller.inspect(task_id)
            if not info["resume_allowed"]:
                raise TaskStateError("任务不可恢复")
            selected = self.controller._select(task_id)
            snapshot = selected._snapshot(task_id)
            if (
                len(snapshot.next) != 1
                or snapshot.next[0] not in {"write", "review"}
                or not hasattr(selected, "_input_hash")
            ):
                raise TaskStateError("只能失效当前待恢复的 write/review 缓存")
            node, state = snapshot.next[0], snapshot.values
            number = state["round_no"] + (1 if node == "write" else 0)
            input_hash = selected._input_hash(node, state)
            with self.conn:
                cursor = self.conn.execute(
                    "DELETE FROM research_generations "
                    "WHERE task_id=? AND node=? AND round_no=? AND input_hash=?",
                    (task_id, node, number, input_hash),
                )
            return {
                "node": node, "round_no": number, "deleted": cursor.rowcount,
                "note": "下次恢复可能重新调用模型，已用额度保留",
            }
