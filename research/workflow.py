from typing import TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from research.critic import review_draft
from research.deep import DeepResearchAgent, expand_context
from research.multisource import (
    _missing_sources,
    _select_evidence,
    normalize_plan,
)


class ResearchState(TypedDict):
    version: str
    task_id: str
    question: str
    plan: list[dict]
    all_queries: list[dict]
    pending: list[dict]
    kind: str
    round_no: int
    used_supplements: int
    max_rounds: int
    max_supplements: int
    evidence: list[dict]
    draft: dict
    review: dict
    report: str
    feedback: list[str]
    errors: list[str]
    outcome: str
    reason: str
    next_node: str


class WorkflowResearchAgent(DeepResearchAgent):
    VERSION = "12.0"

    def __init__(self, store, checkpointer):
        super().__init__(store)
        builder = StateGraph(ResearchState)
        names = (
            "queue", "approve", "collect", "retrieve",
            "write", "review", "repair", "finish",
        )
        for name in names:
            builder.add_node(name, getattr(self, f"_node_{name}"))

        builder.add_edge(START, "queue")
        builder.add_edge("queue", "approve")
        builder.add_edge("collect", "retrieve")
        builder.add_edge("write", "review")
        builder.add_edge("finish", END)

        routes = {
            "approve": ("collect", "finish"),
            "retrieve": ("write", "finish"),
            "review": ("write", "repair", "finish"),
            "repair": ("write", "queue", "finish"),
        }
        for name, targets in routes.items():
            builder.add_conditional_edges(
                name,
                lambda state: state["next_node"],
                {target: target for target in targets},
            )

        self.graph = builder.compile(checkpointer=checkpointer)

    def run(self, *args, **kwargs):
        raise ValueError("V12 请使用 start() / resume() 执行工作流")

    @staticmethod
    def _config(task_id):
        return {
            "configurable": {"thread_id": task_id},
            "recursion_limit": 50,
        }

    def _snapshot(self, task_id):
        self.store.get_task(task_id)
        snapshot = self.graph.get_state(self._config(task_id))
        values = snapshot.values
        if values and (
            values["task_id"] != task_id
            or values["version"] != self.VERSION
        ):
            raise ValueError("checkpoint 的任务或工作流版本不匹配")
        return snapshot

    def inspect(self, task_id):
        snapshot = self._snapshot(task_id)
        values = snapshot.values
        return {
            "exists": bool(values),
            "task_id": task_id,
            "status": self.store.get_task(task_id)["status"],
            "next": list(snapshot.next),
            "round_no": values.get("round_no", 0),
            "used_supplements": values.get("used_supplements", 0),
            "max_rounds": values.get("max_rounds"),
            "max_supplements": values.get("max_supplements"),
            "pending": [
                item.value
                for task in snapshot.tasks
                for item in task.interrupts
            ],
            "node_errors": [
                str(task.error)
                for task in snapshot.tasks
                if task.error is not None
            ],
        }

    def _stream(self, task_id, value):
        return self.graph.stream(
            value,
            self._config(task_id),
            stream_mode="updates",
            version="v1",
            durability="sync",
        )

    def start(self, task_id):
        task = self.store.get_task(task_id)
        if self._snapshot(task_id).values:
            raise ValueError("任务已有 checkpoint，请使用 /resume")
        if task["status"] != "planned":
            raise ValueError("只有 planned 任务可以启动")
        plan = normalize_plan(task["queries"])
        return self._stream(task_id, {
            "version": self.VERSION,
            "task_id": task_id,
            "question": task["question"],
            "plan": plan,
            "all_queries": list(plan),
            "pending": list(plan),
            "kind": "initial",
            "round_no": 0,
            "used_supplements": 0,
            "max_rounds": self.MAX_ROUNDS,
            "max_supplements": self.MAX_SUPPLEMENTS,
            "evidence": [],
            "draft": {},
            "review": {},
            "report": "",
            "feedback": [],
            "errors": [],
            "outcome": "",
            "reason": "",
            "next_node": "queue",
        })

    def resume(self, task_id, *, approved=None):
        info = self.inspect(task_id)
        if not info["exists"]:
            raise ValueError("没有 V12 checkpoint，不能恢复旧版本任务")
        if not info["next"]:
            raise ValueError("任务已经结束，不能重复执行")
        if info["pending"]:
            if type(approved) is not bool:
                raise ValueError("待审批任务必须明确传入 True 或 False")
            value = Command(resume=approved)
        else:
            if approved is not None:
                raise ValueError("当前没有待审批项，请直接恢复")
            value = None
        return self._stream(task_id, value)

    @staticmethod
    def _task(state):
        return {"id": state["task_id"], "question": state["question"]}

    def _audit(self, state, **changes):
        review = {**state["review"], **changes}
        self.store.save_review(
            state["task_id"], state["round_no"],
            state["evidence"], state["draft"], review,
        )
        return review

    @staticmethod
    def _stop(reason):
        return {
            "outcome": "partial",
            "reason": reason,
            "next_node": "finish",
        }

    def _node_queue(self, state):
        self.store.update_task(state["task_id"], "awaiting_approval")
        if state["kind"] == "supplement":
            self._audit(state, action="awaiting_approval")
        return {}

    def _node_approve(self, state):
        # 审批节点恢复时会重跑；这里不做网络请求或数据库写入。
        answer = interrupt({
            "kind": state["kind"],
            "question": state["question"],
            "queries": state["pending"],
            "round_no": state["round_no"],
            "used_supplements": state["used_supplements"],
            "max_rounds": state["max_rounds"],
            "max_supplements": state["max_supplements"],
        })
        approved = answer is True
        review = dict(state["review"])
        if state["kind"] == "supplement":
            review.update(
                approved=approved,
                approved_queries=state["pending"] if approved else [],
            )
        if not approved:
            return {
                "review": review,
                "outcome": (
                    "cancelled" if state["kind"] == "initial" else "partial"
                ),
                "reason": "检索未获得批准",
                "next_node": "finish",
            }

        extra = state["pending"] if state["kind"] == "supplement" else []
        return {
            "review": review,
            "all_queries": [*state["all_queries"], *extra],
            "used_supplements": state["used_supplements"] + len(extra),
            "next_node": "collect",
        }

    def _node_collect(self, state):
        self.store.update_task(state["task_id"], "collecting")
        review = state["review"]
        if state["kind"] == "supplement":
            review = self._audit(
                state,
                action="supplementing",
                attempted_queries=state["pending"],
            )
        errors = list(state["errors"])
        self._collect(state["task_id"], state["pending"], errors)
        return {"errors": errors, "review": review}

    def _node_retrieve(self, state):
        self.store.update_task(state["task_id"], "retrieving")
        chunks = self.store.list_chunks(state["task_id"])
        evidence = _select_evidence(
            state["question"], state["all_queries"], chunks
        )
        if state["kind"] == "supplement":
            evidence = expand_context(evidence, chunks)
            self._audit(state, action="supplemented")
            old_ids = {item["evidence_id"] for item in state["evidence"]}
            if not any(item["evidence_id"] not in old_ids for item in evidence):
                return self._stop("补充检索没有带来新的可用片段")

        missing = _missing_sources(state["plan"], evidence)
        if missing:
            return {
                **self._stop("缺少可用来源：" + ", ".join(missing)),
                "evidence": evidence,
            }
        return {"evidence": evidence, "next_node": "write"}

    def _node_write(self, state):
        number = state["round_no"] + 1
        if number > state["max_rounds"]:
            raise ValueError("写作轮数已达上限")
        self.store.update_task(state["task_id"], "writing")
        draft = {}
        try:
            self.store.save_review(
                state["task_id"], number, state["evidence"], draft,
                {"decision": "pending", "action": "writing"},
            )
            draft = self._write(
                self._task(state), state["evidence"],
                state["feedback"], state["plan"],
            )
            report = self._render(
                self._task(state), draft, state["evidence"], state["errors"]
            )
        except Exception as error:
            self.store.save_review(
                state["task_id"], number, state["evidence"], draft,
                {"decision": "error", "action": "stopped", "error": str(error)},
            )
            raise

        self.store.save_review(
            state["task_id"], number, state["evidence"], draft,
            {"decision": "pending", "action": "reviewing"},
        )
        return {
            "round_no": number, "draft": draft,
            "report": report, "review": {},
        }

    def _node_review(self, state):
        self.store.update_task(state["task_id"], "reviewing")
        try:
            review = review_draft(
                self._ask_json, state["question"], state["draft"],
                state["evidence"], state["plan"],
            )
        except Exception as error:
            self._audit(
                state, decision="error", action="stopped", error=str(error)
            )
            raise

        used_ids = {
            value
            for section in state["draft"]["sections"]
            for value in section["evidence_ids"]
        }
        missing = _missing_sources(
            state["plan"],
            [item for item in state["evidence"] if item["evidence_id"] in used_ids],
        )
        if missing:
            review = {
                "decision": "revise",
                "issues": [{
                    "section": 0,
                    "problem": "正文引用缺少来源：" + ", ".join(missing),
                    "evidence_ids": [],
                }, *review["issues"]][:5],
                "queries": [],
            }

        self.store.save_review(
            state["task_id"], state["round_no"],
            state["evidence"], state["draft"],
            {**review, "action": "reviewed", "approved": None},
        )
        update = {
            "review": review,
            "feedback": [item["problem"] for item in review["issues"]],
        }
        if review["decision"] == "pass":
            return {**update, "outcome": "passed", "next_node": "finish"}
        if state["round_no"] >= state["max_rounds"]:
            return {**update, **self._stop("已达到写作与审查轮数上限")}
        return {
            **update,
            "next_node": "write" if review["decision"] == "revise" else "repair",
        }

    def _node_repair(self, state):
        expanded = expand_context(
            state["evidence"], self.store.list_chunks(state["task_id"])
        )
        if len(expanded) > len(state["evidence"]):
            old_ids = {item["evidence_id"] for item in state["evidence"]}
            self._audit(
                state, action="expanded_context",
                added_context_ids=[
                    item["evidence_id"] for item in expanded
                    if item["evidence_id"] not in old_ids
                ],
            )
            return {"evidence": expanded, "next_node": "write"}

        seen = {
            (item["source"], item["query"].casefold())
            for item in state["all_queries"]
        }
        queries = []
        for item in state["review"]["queries"]:
            key = (item["source"], item["query"].casefold())
            if key not in seen:
                queries.append(item)
                seen.add(key)
        if not queries:
            return self._stop("没有新的上下文或补充查询")
        if state["used_supplements"] + len(queries) > state["max_supplements"]:
            return self._stop("补充检索预算不足")
        return {"pending": queries, "kind": "supplement", "next_node": "queue"}

    def _node_finish(self, state):
        task_id = state["task_id"]
        if state["outcome"] == "passed":
            self.store.update_task(
                task_id,
                "partial" if state["errors"] else "completed",
                report=state["report"], errors=state["errors"],
            )
        elif state["outcome"] == "cancelled":
            self.store.update_task(task_id, "cancelled")
        else:
            if state["round_no"]:
                self._audit(
                    state, action="stopped", stop_reason=state["reason"]
                )
            self._partial(
                self._task(state), state["evidence"],
                state["errors"], state["reason"], state["review"],
            )
        return {}