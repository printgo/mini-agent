"""已有任务规划接口回归，保护队列身份、计量及任务状态。"""
import json
import os
import sqlite3
import tempfile
import unittest
from contextlib import ExitStack, closing
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from langgraph.checkpoint.sqlite import SqliteSaver
from test_context_memory import test_encoding

with patch("dotenv.load_dotenv", return_value=False), patch.dict(os.environ, {
    "DEEPSEEK_API_KEY": "offline-test", "DEEPSEEK_BASE_URL": "https://example.invalid",
    "DEEPSEEK_MODEL": "offline-model", "DEEPSEEK_MEMORY_MODEL": "offline-model",
    "LLM_TIMEOUT_SECONDS": "30", "LLM_MAX_RETRIES": "0", "LLM_MAX_OUTPUT_TOKENS": "4096",
}):
    from research.context import TokenContext
    from research.engine import TeamWorkflowAgent
    from research.generation import GenerationStore
    from research.management import BudgetLedger
    from research.runtime import TaskBusy, TaskFileLock


class PlanExistingTests(unittest.TestCase):
    def setUp(self):
        resources = ExitStack()
        self.addCleanup(resources.close)
        root = Path(resources.enter_context(tempfile.TemporaryDirectory()))
        self.path = root / "memory.db"
        conn = resources.enter_context(closing(sqlite3.connect(self.path, check_same_thread=False)))
        cp = resources.enter_context(closing(sqlite3.connect(root / "checkpoint.db", check_same_thread=False)))
        self.store = GenerationStore(conn)
        self.ledger = BudgetLedger(conn)
        self.agent = TeamWorkflowAgent(
            self.store, SqliteSaver(cp), self.ledger, db_path=self.path,
            context=TokenContext(20000, 4096, test_encoding()),
        )
        response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps({
            "queries": [{"source": "web", "query": "alpha"}],
        })))])
        self.model = resources.enter_context(patch("research.agent.complete", return_value=response))
        resources.enter_context(patch("requests.sessions.Session.request", side_effect=AssertionError("network disabled")))

    def create(self, status="planning"):
        task = self.store.create_task("alpha research")
        self.agent.knowledge.bind("default", task)
        if status != "planning":
            self.store.update_task(task, status)
        return task

    def test_plans_existing_task_and_repeated_planned_call_is_idempotent(self):
        task = self.create()
        planned = self.agent.plan_existing(task)
        self.assertEqual((planned["id"], planned["status"]), (task, "planned"))
        self.assertEqual(self.store.conn.execute("SELECT COUNT(*) FROM research_tasks").fetchone()[0], 1)
        self.assertEqual(self.model.call_count, 1)
        self.assertEqual(self.ledger.get(task)["model_used"], 1)
        frozen = self.agent._frozen(task)
        self.assertEqual(self.agent.plan_existing(task), planned)
        self.assertEqual(self.agent._frozen(task), frozen)
        self.assertEqual(self.model.call_count, 1)
        self.assertEqual(self.ledger.get(task)["model_used"], 1)

    def test_terminal_task_is_not_replanned_or_marked_failed(self):
        task = self.create("completed")
        before = self.store.get_task(task)
        with self.assertRaises(ValueError):
            self.agent.plan_existing(task)
        self.assertEqual(self.store.get_task(task), before)
        self.assertEqual(self.model.call_count, 0)
        self.assertEqual(self.ledger.get(task)["model_used"], 0)

    def test_busy_task_does_not_change_planning_state_or_consume_quota(self):
        task = self.create()
        before = self.store.get_task(task)
        with TaskFileLock(self.path, task):
            with self.assertRaises(TaskBusy):
                self.agent.plan_existing(task)
        self.assertEqual(self.store.get_task(task), before)
        self.assertEqual(self.model.call_count, 0)
        self.assertEqual(self.ledger.get(task)["model_used"], 0)
        self.assertEqual(self.store.conn.execute(
            "SELECT COUNT(*) FROM research_task_context WHERE task_id=?", (task,)
        ).fetchone()[0], 0)


if __name__ == "__main__":
    unittest.main()
