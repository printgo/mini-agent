import json
import os
import sqlite3
import tempfile
import threading
import unittest
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from langgraph.checkpoint.sqlite import SqliteSaver
from research.management import BudgetExceeded, BudgetLedger, TaskMaintenance
from research.runtime import (
    TaskBusy, TaskCancelled, TaskController, TaskFileLock, TaskStateError,
)

# 导入业务模块时不读取 .env，测试仅使用占位配置。
with patch("dotenv.load_dotenv", return_value=False), patch.dict(os.environ, {
    "DEEPSEEK_API_KEY": "offline-test",
    "DEEPSEEK_BASE_URL": "https://example.invalid",
    "DEEPSEEK_MODEL": "offline-model",
    "DEEPSEEK_MEMORY_MODEL": "offline-model",
    "LLM_TIMEOUT_SECONDS": "30",
    "LLM_MAX_RETRIES": "0",
    "LLM_MAX_OUTPUT_TOKENS": "4096",
}):
    from research.generation import GenerationStore, GenerationWorkflowAgent
    from research.limited import LimitedWorkflowAgent
    from research.recovery import RecoveryWorkflowAgent
    from research.workflow import WorkflowResearchAgent


class Legacy15(GenerationWorkflowAgent):
    VERSION = "15.0"


class ManagementTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "memory.db"
        self.conn = sqlite3.connect(self.path, check_same_thread=False, timeout=30)
        self.addCleanup(self.conn.close)
        self.cp = sqlite3.connect(":memory:", check_same_thread=False)
        self.addCleanup(self.cp.close)
        self.store = GenerationStore(self.conn)
        self.ledger = BudgetLedger(self.conn)
        self.agent = LimitedWorkflowAgent(self.store, SqliteSaver(self.cp), self.ledger)
        self.versions = {"16.0": self.agent}
        self.controller = TaskController(
            self.store, self.agent, self.versions, self.path
        )
        self.manager = TaskMaintenance(self.controller, self.ledger)
        self.calls = Counter()
        self.query_failures = 0
        self.bad_writer = False
        self.bad_critic = False
        self.cache_crashes = set()
        original_save = self.store.save_generation

        def save(*args):
            original_save(*args)
            if args[1] in self.cache_crashes:
                self.cache_crashes.remove(args[1])
                raise RuntimeError("injected after cache commit")

        for target, replacement in (
            ("research.agent.complete", self.complete),
            ("research.recovery.collect", self.collect),
            ("research.deep.collect", self.collect),
            ("requests.sessions.Session.request", AssertionError("network disabled")),
        ):
            mocked = patch(target, side_effect=replacement)
            mocked.start()
            self.addCleanup(mocked.stop)
        mocked = patch.object(self.store, "save_generation", side_effect=save)
        mocked.start()
        self.addCleanup(mocked.stop)
        quiet = patch("builtins.print")
        quiet.start()
        self.addCleanup(quiet.stop)
        self.task = self.new_task()

    def new_task(self):
        task = self.store.create_task("alpha research")
        self.store.update_task(
            task, "planned", queries=[{"source": "web", "query": "alpha"}]
        )
        return task

    def document(self):
        return {
            "title": "alpha", "url": "https://example.com/alpha",
            "content": "alpha research evidence supports this conclusion.",
            "source_type": "web_text", "metadata": {},
        }

    def collect(self, source, query):
        self.calls["query"] += 1
        if self.query_failures:
            self.query_failures -= 1
            raise TimeoutError("injected timeout")
        return [self.document()], []

    def complete(self, messages):
        instruction = messages[0]["content"]
        data = json.loads(messages[1]["content"])
        if "Writer" in instruction:
            self.calls["write"] += 1
            ids = [item["evidence_id"] for item in data["evidence"]]
            result = {
                "sections": [{
                    "heading": "Conclusion", "text": "alpha is documented.",
                    "evidence_ids": ["D999C1"] if self.bad_writer else ids,
                }],
                "limitations": ["snapshot"],
            }
        elif "Critic" in instruction:
            self.calls["review"] += 1
            result = {"decision": "pass", "issues": [], "queries": []}
            if self.bad_critic:
                self.bad_critic = False
                result["extra"] = "invalid"
        else:
            self.calls["plan"] += 1
            result = {"queries": [{"source": "web", "query": "alpha"}]}
        return SimpleNamespace(choices=[
            SimpleNamespace(message=SimpleNamespace(content=json.dumps(result)))
        ])

    def start(self):
        with self.controller.session(self.task) as selected:
            list(selected.start(self.task))

    def resume(self, approved=None):
        with self.controller.session(self.task) as selected:
            list(selected.resume(self.task, approved=approved))

    def seed_cache(self, suffix="old", created_at=None):
        created_at = created_at or datetime.now(timezone.utc).isoformat()
        with self.conn:
            self.conn.execute(
                "INSERT INTO research_collections "
                "(task_id, source, query, status, documents, warnings, created_at) "
                "VALUES (?, 'web', ?, 'fetched', '{', '[]', ?)",
                (self.task, suffix, created_at),
            )
            self.conn.execute(
                "INSERT INTO research_generations "
                "(task_id, node, round_no, input_hash, result, created_at) "
                "VALUES (?, 'write', 1, ?, '{', ?)",
                (self.task, suffix, created_at),
            )

    def test_budget_blocks_call_then_increase_resumes(self):
        self.manager.set_limits(self.task, 0, 1)
        self.start()
        with self.assertRaises(BudgetExceeded):
            self.resume(True)
        self.assertEqual(self.calls["write"], 0)
        self.assertEqual(self.ledger.get(self.task)["model_used"], 0)
        self.assertEqual(self.controller.list_failures(self.task), [])
        state = self.agent._snapshot(self.task).values
        before = self.agent._input_hash("write", state)
        self.manager.set_limits(self.task, 2, 1)
        self.assertEqual(before, self.agent._input_hash("write", state))
        self.resume()
        self.assertEqual(self.store.get_task(self.task)["status"], "completed")
        self.assertEqual(self.ledger.get(self.task)["model_used"], 2)

    def test_query_failure_consumes_budget(self):
        self.query_failures = 1
        self.manager.set_limits(self.task, 12, 1)
        self.start()
        with self.assertRaises(RuntimeError):
            self.resume(True)
        with self.assertRaises(BudgetExceeded):
            self.resume()
        self.assertEqual(self.calls["query"], 1)
        self.manager.set_limits(self.task, 12, 2)
        self.resume()
        self.assertEqual(self.ledger.get(self.task)["query_used"], 2)

    def test_invalid_writer_counts_and_scope_is_reset(self):
        self.bad_writer = True
        self.start()
        with self.assertRaises(ValueError):
            self.resume(True)
        self.assertEqual(self.ledger.get(self.task)["model_used"], 1)
        self.assertEqual(self.store.list_generations(self.task), [])
        planned = self.agent.prepare("another question")
        self.assertEqual(planned["status"], "planned")
        self.assertEqual(self.ledger.get(self.task)["model_used"], 1)
        self.bad_writer = False
        self.resume()
        self.assertEqual(self.ledger.get(self.task)["model_used"], 3)

    def test_critic_retry_and_budget_exhaustion(self):
        self.bad_critic = True
        self.manager.set_limits(self.task, 2, 1)
        self.start()
        with self.assertRaises(BudgetExceeded):
            self.resume(True)
        self.assertEqual(self.calls["review"], 1)
        self.manager.set_limits(self.task, 3, 1)
        self.resume()
        self.assertEqual(self.calls["review"], 2)
        self.assertEqual(self.ledger.get(self.task)["model_used"], 3)

    def test_cached_nodes_work_with_exhausted_budget(self):
        self.cache_crashes = {"write", "review"}
        self.manager.set_limits(self.task, 1, 1)
        self.start()
        with self.assertRaises(RuntimeError):
            self.resume(True)
        with self.assertRaises(BudgetExceeded):
            self.resume()
        self.assertEqual(self.calls["write"], 1)
        self.manager.set_limits(self.task, 2, 1)
        with self.assertRaises(RuntimeError):
            self.resume()
        self.resume()
        self.assertEqual(self.calls, Counter(query=1, write=1, review=1))
        self.assertEqual(self.store.get_task(self.task)["status"], "completed")

    def test_fetched_and_applied_collection_do_not_consume_again(self):
        self.ledger.set_limits(self.task, 12, 1)
        self.ledger.reserve(self.task, "query")
        self.store.save_collection(
            self.task, "web", "alpha", [self.document()], []
        )
        query = [{"source": "web", "query": "alpha"}]
        self.agent._collect(self.task, query, [])
        self.agent._collect(self.task, query, [])
        self.assertEqual(self.calls["query"], 0)
        self.assertEqual(self.ledger.get(self.task)["query_used"], 1)

    def test_budget_persists_and_invalid_limits_are_atomic(self):
        self.ledger.reserve(self.task, "model")
        for limits in ((0, 8), (-1, 8), (True, 8), (1.5, 8), (12, -1)):
            with self.assertRaises(TaskStateError):
                self.ledger.set_limits(self.task, *limits)
        with closing(sqlite3.connect(self.path)) as other:
            ledger = BudgetLedger(other)
            self.assertEqual(ledger.get(self.task)["model_used"], 1)
            self.assertEqual(ledger.get(self.task)["model_limit"], 12)

    def test_two_connections_cannot_exceed_last_budget(self):
        self.ledger.set_limits(self.task, 1, 8)
        barrier = threading.Barrier(2)

        def attempt():
            with closing(sqlite3.connect(self.path, timeout=30)) as conn:
                ledger = BudgetLedger(conn)
                barrier.wait(timeout=10)
                try:
                    ledger.reserve(self.task, "model")
                    return True
                except BudgetExceeded:
                    return False

        with ThreadPoolExecutor(max_workers=2) as workers:
            results = list(workers.map(lambda _: attempt(), range(2)))
        self.assertEqual(sum(results), 1)
        self.assertEqual(self.ledger.get(self.task)["model_used"], 1)

    def test_active_cache_is_protected(self):
        self.start()
        self.seed_cache()
        for status in ("fetched", "applied"):
            with self.conn:
                self.conn.execute("UPDATE research_collections SET status=?", (status,))
            with self.assertRaises(TaskStateError):
                self.manager.purge_cache(self.task)
        self.assertEqual(self.manager.cache_stats(self.task)["collections"]["rows"], 1)

    def test_terminal_status_with_pending_finish_is_protected(self):
        self.start()
        self.agent.graph.update_state(
            self.agent._config(self.task),
            {"next_node": "finish", "outcome": "passed"}, as_node="review",
        )
        self.store.update_task(self.task, "completed")
        self.seed_cache()
        self.assertEqual(self.agent.inspect(self.task)["next"], ["finish"])
        with self.assertRaises(TaskStateError):
            self.manager.purge_cache(self.task)

    def test_purge_corrupt_cache_preserves_report_evidence_and_usage(self):
        self.seed_cache()
        self.store.add_document(self.task, self.document())
        self.store.save_review(self.task, 1, [], {}, {"action": "kept"})
        self.ledger.reserve(self.task, "query")
        self.store.update_task(self.task, "completed", report="KEEP")
        self.assertGreater(self.manager.cache_stats(self.task)["generations"]["json_bytes"], 0)
        self.assertEqual(
            self.manager.purge_cache(self.task),
            {"collections": 1, "generations": 1},
        )
        self.assertEqual(self.store.get_task(self.task)["report"], "KEEP")
        self.assertTrue(self.store.list_chunks(self.task))
        self.assertTrue(self.store.list_reviews(self.task))
        self.assertEqual(self.ledger.get(self.task)["query_used"], 1)

    def test_cancelled_task_can_purge_but_cannot_resume(self):
        self.start()
        self.controller.cancel(self.task)
        self.seed_cache()
        self.manager.purge_cache(self.task)
        with self.assertRaises(TaskCancelled):
            self.resume()
        self.assertIsNotNone(self.controller.inspect(self.task)["cancelled_at"])

    def test_age_filter_and_atomic_delete(self):
        self.store.update_task(self.task, "completed")
        self.seed_cache("old", "2000-01-01T00:00:00+00:00")
        self.seed_cache("new")
        self.assertEqual(
            self.manager.purge_cache(self.task, 1),
            {"collections": 1, "generations": 1},
        )
        self.conn.execute("""
            CREATE TRIGGER reject_cache_delete
            BEFORE DELETE ON research_generations
            BEGIN SELECT RAISE(ABORT, 'injected'); END
        """)
        with self.assertRaises(sqlite3.DatabaseError):
            self.manager.purge_cache(self.task)
        self.assertEqual(self.manager.cache_stats(self.task)["collections"]["rows"], 1)
        self.assertEqual(self.manager.cache_stats(self.task)["generations"]["rows"], 1)

    def test_busy_maintenance_has_no_effect(self):
        self.seed_cache()
        with TaskFileLock(self.path, self.task):
            for action in (
                lambda: self.manager.purge_cache(self.task),
                lambda: self.manager.set_limits(self.task, 20, 20),
                lambda: self.manager.invalidate_current_generation(self.task),
            ):
                with self.assertRaises(TaskBusy):
                    action()
        self.assertEqual(self.manager.cache_stats(self.task)["collections"]["rows"], 1)

    def test_invalidate_only_current_generation(self):
        self.cache_crashes = {"write"}
        self.start()
        with self.assertRaises(RuntimeError):
            self.resume(True)
        self.store.save_generation(self.task, "write", 2, "other", {})
        with self.conn:
            self.conn.execute(
                "UPDATE research_generations SET result='{' WHERE round_no=1"
            )
        result = self.manager.invalidate_current_generation(self.task)
        self.assertEqual(result["deleted"], 1)
        self.assertEqual(len(self.store.list_generations(self.task)), 1)
        self.assertEqual(self.ledger.get(self.task)["model_used"], 1)
        self.resume()
        self.assertEqual(self.ledger.get(self.task)["model_used"], 3)

    def test_cannot_invalidate_approval_or_clear_unknown_version(self):
        self.start()
        with self.assertRaises(TaskStateError):
            self.manager.invalidate_current_generation(self.task)
        self.agent.graph.update_state(
            self.agent._config(self.task), {"version": "99.0"}
        )
        with self.assertRaises(TaskStateError):
            self.manager.purge_cache(self.task)

    def test_legacy_versions_are_unmetered(self):
        for cls in (
            WorkflowResearchAgent, RecoveryWorkflowAgent,
            GenerationWorkflowAgent, Legacy15,
        ):
            old = cls(self.store, self.agent.graph.checkpointer)
            self.versions[old.VERSION] = old
            task_id = self.new_task()
            list(old.start(task_id))
            with self.controller.session(task_id) as selected:
                self.assertIs(selected, old)
                list(selected.resume(task_id, approved=True))
            self.assertFalse(self.manager.usage(task_id)["metered"])
            row = self.conn.execute(
                "SELECT 1 FROM research_budgets WHERE task_id=?", (task_id,)
            ).fetchone()
            self.assertIsNone(row)


if __name__ == "__main__":
    unittest.main()