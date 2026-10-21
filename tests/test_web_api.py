"""Web 闭环回归：真实 HTTP/SQLite/LangGraph，外部模型和检索使用固定夹具。"""
import json
import os
import sqlite3
import tempfile
import threading
import time
import unittest
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from uuid import uuid4

from fastapi.testclient import TestClient
from langgraph.checkpoint.sqlite import SqliteSaver
from test_context_memory import test_encoding

with patch("dotenv.load_dotenv", return_value=False), patch.dict(os.environ, {
    "DEEPSEEK_API_KEY": "offline-test",
    "DEEPSEEK_BASE_URL": "https://example.invalid",
    "DEEPSEEK_MODEL": "offline-model",
    "DEEPSEEK_MEMORY_MODEL": "offline-model",
    "LLM_TIMEOUT_SECONDS": "30",
    "LLM_MAX_RETRIES": "0",
    "LLM_MAX_OUTPUT_TOKENS": "4096",
}):
    from api.app import create_app
    from research.context import TokenContext
    from research.engine import TeamController, TeamWorkflowAgent
    from research.generation import GenerationStore
    from research.limited import LimitedWorkflowAgent
    from research.management import BudgetLedger
    from research.runtime import TaskBusy, TaskFileLock
    from research.web_service import TaskService, WebError, WebSettings


class FixtureIndex:
    config = {"embedding": "web-fixture", "reranker": "web-fixture", "revision": "1"}

    def search(self, task_id, query, chunks, top_k=6):
        return chunks[:top_k]


class WebApiTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.settings = WebSettings(
            db_path=root / "memory.db", index_path=root / "index",
            model_cache=root / "models", context_limit=20000,
        )
        self.calls = Counter()
        self.counter_lock = threading.Lock()
        self.plan_entered, self.plan_release = threading.Event(), threading.Event()
        self.write_entered, self.write_release = threading.Event(), threading.Event()
        self.plan_release.set()
        self.write_release.set()
        self.supplement = False
        self.thread_violations = []
        self.factory_threads = set()
        for target, effect in (
            ("research.agent.complete", self.complete),
            ("requests.sessions.Session.request", AssertionError("network disabled")),
            ("httpx.HTTPTransport.handle_request", AssertionError("network disabled")),
        ):
            guard = patch(target, side_effect=effect)
            guard.start()
            self.addCleanup(guard.stop)
        self.client = None
        self.open_client()
        self.addCleanup(self.close_client)
        self.addCleanup(self.plan_release.set)
        self.addCleanup(self.write_release.set)

    @contextmanager
    def runtime(self, settings):
        owner = threading.get_ident()
        self.factory_threads.add(owner)
        # 业务连接保留默认线程检查；SqliteSaver 按其接口约定允许内部保存线程。
        conn = sqlite3.connect(settings.db_path, timeout=10)
        cp = sqlite3.connect(Path(settings.db_path).with_name("research_checkpoints.db"),
                             timeout=10, check_same_thread=False)
        def trace(statement):
            if threading.get_ident() != owner:
                self.thread_violations.append((owner, threading.get_ident()))
        conn.set_trace_callback(trace)
        try:
            store = GenerationStore(conn)
            ledger = BudgetLedger(conn)
            agent = TeamWorkflowAgent(
                store, SqliteSaver(cp), ledger, db_path=settings.db_path,
                context=TokenContext(20000, 4096, test_encoding()),
                index=FixtureIndex(), collect_fn=self.collect,
            )
            legacy = LimitedWorkflowAgent(store, agent.graph.checkpointer, ledger)
            controller = TeamController(
                store, agent, {"19.0": agent, "16.0": legacy}, settings.db_path
            )
            yield SimpleNamespace(store=store, agent=agent, controller=controller, ledger=ledger)
        finally:
            cp.close()
            conn.close()

    def open_client(self):
        self.service = TaskService(self.settings, runtime_factory=self.runtime)
        self.app = create_app(settings=self.settings, service=self.service)
        self.client = TestClient(self.app)
        self.client.__enter__()

    def close_client(self):
        if self.client is not None:
            current, self.client = self.client, None
            current.__exit__(None, None, None)

    def collect(self, source, query):
        with self.counter_lock:
            self.calls["query"] += 1
        return [{
            "title": "alpha", "url": "https://example.com/" + query.replace(" ", "-"),
            "content": "alpha evidence supports the documented scope.",
            "source_type": "web_text", "metadata": {},
        }], []

    def complete(self, messages):
        instruction = messages[0]["content"]
        data = json.loads(messages[-1]["content"])
        if "Planner Agent" in instruction:
            with self.counter_lock:
                self.calls["plan"] += 1
            self.plan_entered.set()
            if not self.plan_release.wait(10):
                raise TimeoutError("test did not release planning")
            result = {"queries": [{"source": "web", "query": "alpha"}]}
        elif "Writer" in instruction:
            with self.counter_lock:
                self.calls["write"] += 1
            self.write_entered.set()
            if not self.write_release.wait(10):
                raise TimeoutError("test did not release writing")
            result = {
                "sections": [{"heading": "Conclusion", "text": "alpha is documented.",
                              "evidence_ids": [item["evidence_id"] for item in data["evidence"]]}],
                "limitations": ["Snapshot only."],
            }
        elif "Critic" in instruction:
            with self.counter_lock:
                self.calls["review"] += 1
                number = self.calls["review"]
            result = {"decision": "pass", "issues": [], "queries": []}
            if self.supplement and number == 1:
                result = {
                    "decision": "research",
                    "issues": [{"section": 0, "problem": "Need alpha details", "evidence_ids": []}],
                    "queries": [{"source": "web", "query": "alpha details"}],
                }
        else:
            raise AssertionError(instruction)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
            content=json.dumps(result)
        ))])

    def post(self, url, body, expected=202):
        response = self.client.post(url, json=body)
        self.assertEqual(response.status_code, expected, response.text)
        return response.json()

    def create(self, question="alpha research", request_id=None):
        result = self.post("/api/tasks", {"question": question, "request_id": request_id or uuid4().hex})
        self.assertTrue(result["task_id"])
        self.assertTrue(result["job_id"])
        return result["task_id"]

    def wait_detail(self, task_id, predicate, timeout=10):
        deadline = time.monotonic() + timeout
        last = None
        while time.monotonic() < deadline:
            response = self.client.get("/api/tasks/" + task_id)
            self.assertEqual(response.status_code, 200, response.text)
            last = response.json()
            if predicate(last):
                return last
            time.sleep(0.025)
        self.fail("task did not reach expected state: " + repr(last))

    def pending(self, task_id, kind="initial"):
        return self.wait_detail(task_id, lambda value: (
            value["pending"] is not None and value["pending"]["kind"] == kind
            and value["active_job"] is None
        ))

    def completed(self, task_id):
        return self.wait_detail(task_id, lambda value: (
            value["status"] == "completed" and value["active_job"] is None
        ))

    def approve(self, task_id, pending, approved=True, request_id=None, expected=202):
        body = {"approval_token": pending["token"], "approved": approved,
                "request_id": request_id or uuid4().hex}
        return self.post("/api/tasks/" + task_id + "/approval", body, expected), body

    def test_health_validation_and_api_404(self):
        self.assertTrue(self.client.get("/api/health").json()["ok"])
        missing = self.client.get("/api/not-a-real-route")
        self.assertEqual(missing.status_code, 404)
        self.assertIn("application/json", missing.headers["content-type"])
        for question in ("", " " * 5, "x" * 1001):
            with self.subTest(question=question[:20]):
                self.post("/api/tasks", {"question": question, "request_id": uuid4().hex}, 422)
        self.assertEqual(self.client.get("/api/tasks/not-found").status_code, 404)
        self.assertEqual(self.client.get("/api/tasks?limit=0").status_code, 422)

    def test_creation_returns_before_planning_finishes_and_polling_is_read_only(self):
        self.plan_release.clear()
        task = self.create()
        self.assertTrue(self.plan_entered.wait(3))
        for _ in range(3):
            detail = self.client.get("/api/tasks/" + task).json()
            self.assertEqual(detail["status"], "planning")
            self.assertIsNotNone(detail["active_job"])
        self.assertEqual(self.calls, Counter(plan=1))
        self.plan_release.set()
        self.pending(task)
        self.assertEqual(self.calls, Counter(plan=1))

    def test_create_request_id_is_idempotent_and_different_payload_is_rejected(self):
        request_id = uuid4().hex
        body = {"question": "alpha research", "request_id": request_id}
        first = self.post("/api/tasks", body)
        self.assertEqual(self.post("/api/tasks", body), first)
        self.post("/api/tasks", {**body, "question": "different"}, 409)
        self.pending(first["task_id"])
        listing = self.client.get("/api/tasks").json()
        self.assertEqual(listing["total"], 1)
        self.assertEqual([item["id"] for item in listing["items"]], [first["task_id"]])
        self.assertEqual(self.calls, Counter(plan=1))

    def test_concurrent_create_with_same_request_id_commits_one_task(self):
        request_id = uuid4().hex
        with ThreadPoolExecutor(max_workers=4) as workers:
            replies = list(workers.map(
                lambda _: self.service.create("alpha research", request_id), range(8)
            ))
        self.assertTrue(all(reply == replies[0] for reply in replies))
        self.pending(replies[0]["task_id"])
        self.assertEqual(self.client.get("/api/tasks").json()["total"], 1)
        self.assertEqual(self.calls, Counter(plan=1))

    def test_approval_wait_releases_task_lock_and_finishes_with_report(self):
        task = self.create()
        detail = self.pending(task)
        with TaskFileLock(self.settings.db_path, task):
            pass
        self.assertEqual(self.calls, Counter(plan=1))
        self.approve(task, detail["pending"])
        finished = self.completed(task)
        self.assertIn("alpha is documented", finished["report"])
        self.assertEqual(self.calls, Counter(plan=1, query=1, write=1, review=1))
        self.assertEqual((finished["usage"]["model_used"], finished["usage"]["query_used"]), (3, 1))
        self.assertEqual(self.thread_violations, [])
        self.assertGreaterEqual(len(self.factory_threads), 2)

    def test_rejection_finishes_without_collection(self):
        task = self.create()
        detail = self.pending(task)
        self.approve(task, detail["pending"], approved=False)
        rejected = self.wait_detail(task, lambda value: (
            value["status"] == "cancelled" and value["active_job"] is None
        ))
        self.assertFalse(rejected["resume_allowed"])
        self.assertEqual(self.calls, Counter(plan=1))

    def test_replayed_initial_approval_cannot_approve_later_supplement(self):
        self.supplement = True
        task = self.create()
        initial = self.pending(task)["pending"]
        first, body = self.approve(task, initial)
        supplement = self.pending(task, "supplement")["pending"]
        self.assertNotEqual(initial["token"], supplement["token"])
        self.assertEqual(self.post("/api/tasks/" + task + "/approval", body), first)
        self.approve(task, initial, expected=409)
        self.assertEqual(self.calls, Counter(plan=1, query=1, write=1, review=1))
        self.assertEqual(self.pending(task, "supplement")["pending"]["token"], supplement["token"])
        self.approve(task, supplement)
        finished = self.completed(task)
        self.assertEqual(self.calls, Counter(plan=1, query=2, write=2, review=2))
        self.assertEqual((finished["usage"]["model_used"], finished["usage"]["query_used"]), (5, 2))

    def test_resume_requires_explicit_approval_when_pending(self):
        task = self.create()
        self.pending(task)
        self.post("/api/tasks/" + task + "/resume", {"request_id": uuid4().hex}, 409)
        self.assertEqual(self.calls, Counter(plan=1))

    def test_closing_request_client_does_not_stop_background_execution(self):
        task = self.create()
        pending = self.pending(task)["pending"]
        self.write_release.clear()
        observer = TestClient(self.app)
        try:
            response = observer.post("/api/tasks/" + task + "/approval", json={
                "approved": True, "approval_token": pending["token"], "request_id": uuid4().hex,
            })
            self.assertEqual(response.status_code, 202, response.text)
            self.assertTrue(self.write_entered.wait(3))
        finally:
            observer.close()
        self.write_release.set()
        self.completed(task)
        self.assertEqual(self.calls, Counter(plan=1, query=1, write=1, review=1))

    def test_service_reopen_preserves_pending_approval_without_model_replay(self):
        task = self.create()
        before = self.pending(task)
        self.close_client()
        self.open_client()
        after = self.pending(task)
        self.assertEqual(after["pending"], before["pending"])
        self.assertEqual(self.calls, Counter(plan=1))
        self.approve(task, after["pending"])
        self.completed(task)

    def test_committed_writer_failure_can_resume_without_writing_again(self):
        task = self.create()
        pending = self.pending(task)["pending"]
        original = GenerationStore.save_generation
        crashed = threading.Event()
        def crash(store, *args):
            result = original(store, *args)
            if args[1] == "write" and not crashed.is_set():
                crashed.set()
                raise RuntimeError("after web writer cache commit")
            return result
        with patch.object(GenerationStore, "save_generation", new=crash), self.assertLogs(
            "research.web_service", level="ERROR"
        ):
            self.approve(task, pending)
            failed = self.wait_detail(task, lambda value: (
                value["active_job"] is None and value["last_job"]["status"] == "failed"
            ))
        self.assertTrue(failed["resume_allowed"])
        self.assertIn("writer cache commit", failed["last_job"]["error"])
        self.post("/api/tasks/" + task + "/resume", {"request_id": uuid4().hex})
        self.completed(task)
        self.assertEqual(self.calls, Counter(plan=1, query=1, write=1, review=1))

    def test_worker_lease_prevents_two_services_on_same_database(self):
        other = TaskService(self.settings, runtime_factory=self.runtime)
        try:
            with self.assertRaises((TaskBusy, WebError)):
                other.start()
        finally:
            other.stop()

    def test_production_runtime_factory_runs_real_graph_with_mocked_external_boundaries(self):
        # 故意不注入 runtime_factory，保护生产连接和 Checkpointer 配置。
        self.close_client()
        with patch("research.engine.TokenContext", side_effect=lambda *args, **kwargs: (
            TokenContext(20000, 4096, test_encoding())
        )), patch("research.engine.HybridIndex", side_effect=lambda *args, **kwargs: FixtureIndex()), patch(
            "research.engine.sources.collect", side_effect=self.collect
        ):
            self.service = TaskService(self.settings)
            self.app = create_app(settings=self.settings, service=self.service)
            self.client = TestClient(self.app)
            self.client.__enter__()
            try:
                task = self.create()
                pending = self.pending(task)["pending"]
                self.approve(task, pending)
                finished = self.completed(task)
                self.assertIn("alpha is documented", finished["report"])
                self.assertEqual(self.calls, Counter(plan=1, query=1, write=1, review=1))
            finally:
                self.close_client()

    def test_previously_running_job_is_interrupted_and_requires_manual_resume(self):
        self.close_client()
        created = self.service.create("alpha research", uuid4().hex)
        with sqlite3.connect(self.settings.db_path) as conn:
            conn.execute("UPDATE web_jobs SET status='running' WHERE id=?", (created["job_id"],))
        conn.close()
        self.open_client()
        detail = self.client.get("/api/tasks/" + created["task_id"]).json()
        self.assertEqual(detail["last_job"]["status"], "interrupted")
        self.assertIsNone(detail["active_job"])
        self.assertTrue(detail["resume_allowed"])
        self.assertEqual(self.calls, Counter())
        self.post("/api/tasks/" + created["task_id"] + "/resume", {"request_id": uuid4().hex})
        self.pending(created["task_id"])
        self.assertEqual(self.calls, Counter(plan=1))
        self.assertEqual(self.client.get("/api/tasks").json()["total"], 1)

    def test_queued_job_survives_service_restart_and_waits_for_approval(self):
        self.close_client()
        created = self.service.create("alpha research", uuid4().hex)
        self.open_client()
        detail = self.pending(created["task_id"])
        self.assertEqual(detail["last_job"]["status"], "done")
        self.assertEqual(self.calls, Counter(plan=1))

    def test_queued_approval_is_revalidated_against_checkpoint_before_execution(self):
        self.supplement = True
        task = self.create()
        initial = self.pending(task)["pending"]
        self.close_client()
        self.service.approve(task, initial["token"], True, uuid4().hex)
        # 模拟服务关闭期间，另一个合法调用者已完成第一次审批并推进到补查。
        with self.runtime(self.settings) as rt:
            with rt.controller.session(task) as selected:
                list(selected.resume(task, approved=True))
        before = Counter(self.calls)
        self.assertEqual(before, Counter(plan=1, query=1, write=1, review=1))
        with self.assertLogs("research.web_service", level="ERROR"):
            self.open_client()
            detail = self.wait_detail(task, lambda value: (
                value["active_job"] is None and value["last_job"]["status"] == "failed"
            ))
        self.assertEqual(detail["pending"]["kind"], "supplement")
        self.assertNotEqual(detail["pending"]["token"], initial["token"])
        self.assertEqual(self.calls, before)


if __name__ == "__main__":
    unittest.main()
