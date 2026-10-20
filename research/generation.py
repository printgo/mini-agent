import hashlib
import json
from datetime import datetime, timezone

from llm import client as llm_client
from research.critic import validate_review
from research.multisource import _missing_sources
from research.recovery import RecoveryStore, RecoveryWorkflowAgent


def canonical_json(value):
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False,
    )


class GenerationStore(RecoveryStore):
    """保存已通过业务校验的 Writer/Critic 节点结果。"""

    def __init__(self, conn):
        super().__init__(conn)
        with self.conn:
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS research_generations (
                    task_id TEXT NOT NULL REFERENCES research_tasks(id),
                    node TEXT NOT NULL CHECK(node IN ('write', 'review')),
                    round_no INTEGER NOT NULL CHECK(round_no >= 1),
                    input_hash TEXT NOT NULL,
                    result TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY(task_id, node, round_no, input_hash)
                )
            """)

    def get_generation(self, task_id, node, round_no, input_hash):
        row = self.conn.execute(
            "SELECT result FROM research_generations "
            "WHERE task_id=? AND node=? AND round_no=? AND input_hash=?",
            (task_id, node, round_no, input_hash),
        ).fetchone()
        if row is None:
            return None
        result = json.loads(row[0])
        if not isinstance(result, dict):
            raise ValueError("模型结果缓存损坏：结果必须为对象")
        return result

    def save_generation(self, task_id, node, round_no, input_hash, result):
        payload = canonical_json(result)
        with self.conn:
            self.conn.execute(
                "INSERT INTO research_generations "
                "(task_id, node, round_no, input_hash, result, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(task_id, node, round_no, input_hash) DO NOTHING",
                (
                    task_id, node, round_no, input_hash, payload,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )

    def list_generations(self, task_id):
        self.get_task(task_id)
        rows = self.conn.execute(
            "SELECT node, round_no, input_hash, created_at "
            "FROM research_generations WHERE task_id=? "
            "ORDER BY round_no, "
            "CASE node WHEN 'write' THEN 0 ELSE 1 END, created_at",
            (task_id,),
        ).fetchall()
        return [
            dict(zip(
                ["node", "round_no", "input_hash", "created_at"], row
            ))
            for row in rows
        ]


class GenerationWorkflowAgent(RecoveryWorkflowAgent):
    VERSION = "14.0"
    # 修改提示词、校验、渲染或节点逻辑时，也需要更新此值。
    CACHE_REVISION = "writer-critic-v1"

    def run(self, *args, **kwargs):
        raise ValueError(
            f"V{self.VERSION} 请使用 start() / resume() 执行工作流"
        )

    def _input_hash(self, node, state):
        payload = {
            "version": self.VERSION,
            "revision": self.CACHE_REVISION,
            "node": node,
            "state": state,
            "base_url": llm_client.BASE_URL,
            "request_options": llm_client.request_options([]),
        }
        return hashlib.sha256(
            canonical_json(payload).encode("utf-8")
        ).hexdigest()

    def _validate_update(self, node, state, update):
        if not isinstance(update, dict):
            raise ValueError("模型节点结果必须为对象")

        if node == "write":
            if set(update) != {"round_no", "draft", "report", "review"}:
                raise ValueError("写作缓存字段无效")
            if (
                type(update["round_no"]) is not int
                or update["round_no"] != state["round_no"] + 1
                or update["round_no"] > state["max_rounds"]
                or update["review"] != {}
                or not isinstance(update["draft"], dict)
            ):
                raise ValueError("写作缓存轮次或内容无效")
            report = self._render(
                self._task(state), update["draft"],
                state["evidence"], state["errors"],
            )
            if report != update["report"]:
                raise ValueError("缓存报告与当前渲染结果不一致")
            return

        review = validate_review(
            update.get("review"), state["draft"],
            state["evidence"], state["plan"],
        )
        used_ids = {
            value
            for section in state["draft"]["sections"]
            for value in section["evidence_ids"]
        }
        missing = _missing_sources(
            state["plan"],
            [
                item for item in state["evidence"]
                if item["evidence_id"] in used_ids
            ],
        )
        if missing:
            problem = "正文引用缺少来源：" + ", ".join(missing)
            if (
                review["decision"] != "revise"
                or not any(
                    issue["problem"] == problem
                    for issue in review["issues"]
                )
            ):
                raise ValueError("审查缓存未保留来源覆盖校验")

        expected = {
            "review": review,
            "feedback": [item["problem"] for item in review["issues"]],
        }
        if review["decision"] == "pass":
            expected.update(outcome="passed", next_node="finish")
        elif state["round_no"] >= state["max_rounds"]:
            expected.update(self._stop("已达到写作与审查轮数上限"))
        else:
            expected["next_node"] = (
                "write" if review["decision"] == "revise" else "repair"
            )
        if update != expected:
            raise ValueError("审查缓存与当前决策或轮次规则不一致")

    def _restore_audit(self, node, state, update):
        # 命中缓存也要恢复业务表，不能只返回图状态。
        self.store.update_task(
            state["task_id"], "writing" if node == "write" else "reviewing"
        )
        if node == "write":
            self.store.save_review(
                state["task_id"], update["round_no"],
                state["evidence"], update["draft"],
                {"decision": "pending", "action": "reviewing"},
            )
        else:
            self.store.save_review(
                state["task_id"], state["round_no"],
                state["evidence"], state["draft"],
                {
                    **update["review"],
                    "action": "reviewed",
                    "approved": None,
                },
            )

    def _cached_node(self, node, state, generate):
        number = state["round_no"] + (1 if node == "write" else 0)
        if not 1 <= number <= state["max_rounds"]:
            raise ValueError("模型节点轮次超出范围")
        task_id = state["task_id"]
        input_hash = self._input_hash(node, state)
        update = self.store.get_generation(
            task_id, node, number, input_hash
        )
        if update is not None:
            self._validate_update(node, state, update)
            self._restore_audit(node, state, update)
            print(f"[ModelCache] {node} | round={number}")
            return update

        # 父类执行完整校验；抛出异常时不写成功缓存。
        update = generate(state)
        self._validate_update(node, state, update)
        self.store.save_generation(
            task_id, node, number, input_hash, update
        )
        return update

    def _node_write(self, state):
        return self._cached_node("write", state, super()._node_write)

    def _node_review(self, state):
        return self._cached_node("review", state, super()._node_review)