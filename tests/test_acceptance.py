import json
import os
import sqlite3
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from langgraph.checkpoint.sqlite import SqliteSaver
from research.management import BudgetLedger
from research.quality import audit_database
from research.runtime import TaskController

# 导入时不读取真实 .env；测试不访问模型或搜索服务。
with patch("dotenv.load_dotenv", return_value=False), patch.dict(os.environ, {
    "DEEPSEEK_API_KEY": "offline-test",
    "DEEPSEEK_BASE_URL": "https://example.invalid",
    "DEEPSEEK_MODEL": "offline-model",
    "DEEPSEEK_MEMORY_MODEL": "offline-model",
    "LLM_TIMEOUT_SECONDS": "30",
    "LLM_MAX_RETRIES": "0",
    "LLM_MAX_OUTPUT_TOKENS": "4096",
}):
    from research.generation import GenerationStore
    from research.limited import LimitedWorkflowAgent


class AcceptanceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "memory.db"
        self.checkpoint_path = Path(self.tmp.name) / "checkpoints.db"
        self.conn = self.cp = None
        self.addCleanup(self.close_connections)
        self.calls, self.writes, self.mode = Counter(), [], "normal"
        self.open_connections()
        for target, effect in (
            ("research.agent.complete", self.complete),
            ("research.recovery.collect", self.collect),
            ("requests.sessions.Session.request", AssertionError("network disabled")),
            ("httpx.Client.send", AssertionError("network disabled")),
        ):
            mocked = patch(target, side_effect=effect)
            mocked.start()
            self.addCleanup(mocked.stop)
        quiet = patch("builtins.print")
        quiet.start()
        self.addCleanup(quiet.stop)

    def close_connections(self):
        for name in ("cp", "conn"):
            connection = getattr(self, name, None)
            if connection is not None:
                connection.close()
                setattr(self, name, None)

    def open_connections(self):
        self.close_connections()
        self.conn = sqlite3.connect(self.path, check_same_thread=False, timeout=30)
        self.cp = sqlite3.connect(
            self.checkpoint_path, check_same_thread=False, timeout=30
        )
        self.store = GenerationStore(self.conn)
        self.ledger = BudgetLedger(self.conn)
        self.agent = LimitedWorkflowAgent(
            self.store, SqliteSaver(self.cp), self.ledger
        )
        self.controller = TaskController(
            self.store, self.agent, {self.agent.VERSION: self.agent}, self.path
        )

    def prepare_task(self, sources=("web",)):
        self.task = self.store.create_task("alpha research")
        plan = [{"source": source, "query": "alpha " + source} for source in sources]
        self.store.update_task(self.task, "planned", queries=plan)
        self.advance(start=True)
        self.assertEqual(self.calls, Counter())
        self.assertEqual(self.agent.inspect(self.task)["pending"][0]["kind"], "initial")

    def advance(self, *, start=False, approved=None):
        with self.controller.session(self.task) as selected:
            events = selected.start(self.task) if start else selected.resume(
                self.task, approved=approved
            )
            list(events)

    def collect(self, source, query):
        self.calls["query"] += 1
        if self.mode == "missing" and source == "paper":
            return [], []
        source_type = {
            "web": "web_text", "paper": "paper_abstract", "github": "github_readme"
        }[source]
        return [{
            "title": "alpha " + source,
            "url": "https://example.com/" + source + "/" + query.replace(" ", "-"),
            "content": "alpha evidence describes the documented scope of " + query,
            "source_type": source_type,
            "metadata": {},
        }], []

    def complete(self, messages):
        instruction = messages[0]["content"]
        data = json.loads(messages[1]["content"])
        if "Writer" in instruction:
            self.calls["write"] += 1
            self.writes.append(data)
            ids = [item["evidence_id"] for item in data["evidence"]]
            if self.mode == "coverage" and self.calls["write"] == 1:
                ids = ids[:1]
            result = {
                "sections": [{
                    "heading": "Conclusion", "text": "alpha is documented.",
                    "evidence_ids": ids,
                }],
                "limitations": ["Snapshots only."],
            }
        elif "Critic" in instruction:
            self.calls["review"] += 1
            result = {"decision": "pass", "issues": [], "queries": []}
            if self.calls["review"] == 1 and self.mode in {"revise", "supplement"}:
                revise = self.mode == "revise"
                result = {
                    "decision": "revise" if revise else "research",
                    "issues": [{
                        "section": 0,
                        "problem": "Clarify scope" if revise else "Need alpha details",
                        "evidence_ids": [],
                    }],
                    "queries": [] if revise else [
                        {"source": "web", "query": "alpha details"}
                    ],
                }
        else:
            raise AssertionError("Unexpected model operation")
        return SimpleNamespace(choices=[
            SimpleNamespace(message=SimpleNamespace(content=json.dumps(result)))
        ])

    def assert_finished(self, query, write, review, rounds, status="completed"):
        info = self.agent.inspect(self.task)
        self.assertEqual(info["next"], [])
        self.assertEqual(info["round_no"], rounds)
        self.assertEqual(self.store.get_task(self.task)["status"], status)
        self.assertEqual(self.calls, Counter(query=query, write=write, review=review))
        usage = self.ledger.get(self.task)
        self.assertEqual(usage["query_used"], query)
        self.assertEqual(usage["model_used"], write + review)
        self.assertEqual(
            audit_database(self.path, self.task)["accepted"], status == "completed"
        )

    def test_three_sources_are_retrieved_and_cited(self):
        self.prepare_task(("web", "paper", "github"))
        self.advance(approved=True)
        self.assert_finished(3, 1, 1, 1)
        draft = self.store.list_reviews(self.task)[-1]["draft"]
        cited = {value for section in draft["sections"] for value in section["evidence_ids"]}
        found = {
            item["source_type"] for item in self.writes[0]["evidence"]
            if item["evidence_id"] in cited
        }
        self.assertEqual(found, {"web_text", "paper_abstract", "github_readme"})

    def test_revise_feedback_reaches_second_writer(self):
        self.mode = "revise"
        self.prepare_task()
        self.advance(approved=True)
        self.assert_finished(1, 2, 2, 2)
        self.assertEqual(self.writes[1]["feedback"], ["Clarify scope"])
        self.assertEqual(self.store.list_reviews(self.task)[0]["review"]["decision"], "revise")

    def test_source_coverage_overrides_incorrect_critic_pass(self):
        self.mode = "coverage"
        self.prepare_task(("web", "paper", "github"))
        self.advance(approved=True)
        self.assert_finished(3, 2, 2, 2)
        self.assertEqual(
            self.writes[1]["feedback"], ["正文引用缺少来源：paper, github"]
        )
        self.assertEqual(self.store.list_reviews(self.task)[0]["review"]["decision"], "revise")

    def reach_supplement_approval(self):
        self.mode = "supplement"
        self.prepare_task()
        self.advance(approved=True)
        self.assertEqual(self.calls, Counter(query=1, write=1, review=1))
        pending = self.agent.inspect(self.task)["pending"][0]
        self.assertEqual(pending["kind"], "supplement")
        self.assertEqual(pending["queries"], [{"source": "web", "query": "alpha details"}])

    def test_reject_supplement_keeps_partial_without_extra_calls(self):
        self.reach_supplement_approval()
        self.advance(approved=False)
        self.assert_finished(1, 1, 1, 1, "partial")
        self.assertEqual(self.agent.inspect(self.task)["used_supplements"], 0)
        self.assertFalse(self.store.list_reviews(self.task)[0]["review"]["approved"])
        self.assertIn("没有生成通过审查的正式结论", self.store.get_task(self.task)["report"])

    def test_approve_supplement_after_reopening_connections(self):
        self.reach_supplement_approval()
        self.open_connections()
        self.advance(approved=True)
        self.assert_finished(2, 2, 2, 2)
        self.assertEqual(self.agent.inspect(self.task)["used_supplements"], 1)
        self.assertTrue(self.store.list_reviews(self.task)[0]["review"]["approved"])
        self.assertEqual(len(self.store.list_documents(self.task)), 2)

    def test_missing_required_source_prevents_model_generation(self):
        self.mode = "missing"
        self.prepare_task(("web", "paper", "github"))
        self.advance(approved=True)
        self.assert_finished(3, 0, 0, 0, "partial")
        self.assertIn("缺少可用来源：paper", self.store.get_task(self.task)["report"])

    def test_committed_writer_cache_survives_reopening_connections(self):
        self.prepare_task()
        original = self.store.save_generation

        def crash(*args):
            original(*args)
            if args[1] == "write":
                raise RuntimeError("injected after cache commit")

        with patch.object(self.store, "save_generation", side_effect=crash):
            with self.assertRaisesRegex(RuntimeError, "after cache commit"):
                self.advance(approved=True)
        self.assertEqual(self.agent.inspect(self.task)["next"], ["write"])
        self.assertEqual(len(self.store.list_generations(self.task)), 1)
        self.open_connections()
        self.advance()
        self.assert_finished(1, 1, 1, 1)


if __name__ == "__main__":
    unittest.main()