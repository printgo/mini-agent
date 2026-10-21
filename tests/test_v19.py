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
from test_context_memory import test_encoding
from research.context import TokenContext
from research.knowledge import KnowledgeStore
from research.management import BudgetExceeded, BudgetLedger, TaskMaintenance
from research.runtime import TaskStateError

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
    from research.quality import audit_database
    from research.engine import TeamController, TeamWorkflowAgent


class TestIndex:
    config = {"embedding": "fixture", "reranker": "fixture", "revision": "1"}

    def search(self, task_id, query, chunks, top_k=6):
        return chunks[:top_k]


class IntegrationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "memory.db"
        self.cp_path = Path(self.tmp.name) / "research_checkpoints.db"
        self.conn = self.cp = None
        self.addCleanup(self.close)
        self.calls, self.inputs = Counter(), []
        self.open()
        for target, effect in (
            ("research.agent.complete", self.complete),
            ("requests.sessions.Session.request", AssertionError("network disabled")),
            ("httpx.Client.send", AssertionError("network disabled")),
        ):
            mocked = patch(target, side_effect=effect)
            mocked.start()
            self.addCleanup(mocked.stop)
        quiet = patch("builtins.print")
        quiet.start()
        self.addCleanup(quiet.stop)

    def close(self):
        for name in ("conn", "cp"):
            conn = getattr(self, name, None)
            if conn is not None:
                conn.close()
                setattr(self, name, None)

    def open(self, user="default"):
        self.close()
        self.conn = sqlite3.connect(self.path, check_same_thread=False)
        self.cp = sqlite3.connect(self.cp_path, check_same_thread=False)
        self.store = GenerationStore(self.conn)
        self.ledger = BudgetLedger(self.conn)
        self.agent = TeamWorkflowAgent(
            self.store, SqliteSaver(self.cp), self.ledger,
            db_path=self.path, user_id=user,
            context=TokenContext(20000, 4096, test_encoding()),
            index=TestIndex(), collect_fn=self.collect,
        )
        self.old = LimitedWorkflowAgent(self.store, self.agent.graph.checkpointer, self.ledger)
        self.controller = TeamController(
            self.store, self.agent, {"16.0": self.old, "19.0": self.agent}, self.path
        )
        self.manager = TaskMaintenance(self.controller, self.ledger)

    def collect(self, source, query):
        self.calls["query"] += 1
        return [{
            "title": "alpha",
            "url": "https://example.com/" + (
                query.replace(" ", "-") if getattr(self, "supplement", False) else "alpha"
            ),
            "content": "alpha evidence supports the documented scope.",
            "source_type": "web_text", "metadata": {},
        }], []

    def complete(self, messages):
        instruction = messages[0]["content"]
        data = json.loads(messages[-1]["content"])
        self.inputs.append((instruction, data))
        if "Planner Agent" in instruction:
            self.calls["plan"] += 1
            result = {"queries": [{
                "source": "local" if data.get("local_available") else "web",
                "query": "alpha",
            }]}
        elif "Writer" in instruction:
            self.calls["write"] += 1
            result = {
                "sections": [{
                    "heading": "Conclusion", "text": "alpha is documented.",
                    "evidence_ids": [item["evidence_id"] for item in data["evidence"]],
                }],
                "limitations": ["Snapshot only."],
            }
        elif "Critic" in instruction:
            self.calls["review"] += 1
            result = {"decision": "pass", "issues": [], "queries": []}
            if getattr(self, "remote_review", False) or (
                getattr(self, "supplement", False) and self.calls["review"] == 1
            ):
                result = {
                    "decision": "research",
                    "issues": [{
                        "section": 0, "problem": "Need alpha details",
                        "evidence_ids": [],
                    }],
                    "queries": [{"source": "web", "query": "alpha details"}],
                }
        else:
            raise AssertionError(instruction)
        return SimpleNamespace(choices=[
            SimpleNamespace(message=SimpleNamespace(content=json.dumps(result)))
        ])

    def start(self, task):
        with self.controller.session(task) as selected:
            list(selected.start(task))

    def resume(self, task, approved=None):
        with self.controller.session(task) as selected:
            list(selected.resume(task, approved=approved))

    def test_remote_pipeline_meters_planner_and_uses_reviewer_agent(self):
        task = self.agent.prepare("alpha research")["id"]
        self.start(task)
        self.assertEqual(self.calls, Counter(plan=1))
        with patch.object(self.agent.reviewer, "review", wraps=self.agent.reviewer.review) as review:
            self.resume(task, True)
            self.assertEqual(review.call_count, 1)
        self.assertTrue(audit_database(self.path, task)["accepted"])
        self.assertEqual(self.ledger.get(task)["model_used"], 3)
        self.assertEqual(self.calls, Counter(plan=1, query=1, write=1, review=1))

    def test_local_document_pipeline_is_offline_and_traceable(self):
        document = Path(self.tmp.name) / "alpha.md"
        document.write_text("# Alpha\nalpha local evidence.\n", encoding="utf-8")
        task = self.agent.prepare("alpha", files=[document], local_only=True)["id"]
        self.start(task)
        self.resume(task, True)
        self.assertEqual(self.calls["query"], 0)
        self.assertEqual(self.ledger.get(task)["query_used"], 0)
        chunks = self.store.list_chunks(task)
        self.assertEqual(chunks[0]["metadata"]["section"], "Alpha")
        self.assertTrue(chunks[0]["url"].startswith("file:"))
        self.assertTrue(audit_database(self.path, task)["accepted"])

    def test_memory_freezes_per_task_and_confirmed_conclusion_reaches_next_task(self):
        knowledge = self.agent.knowledge
        knowledge.apply("default", "ADD", "preference", "style", "concise")
        task = self.agent.prepare("alpha")["id"]
        before = self.agent._frozen(task)
        knowledge.apply("default", "UPDATE", "preference", "style", "detailed")
        self.start(task)
        self.resume(task, True)
        writer = next(data for instruction, data in self.inputs if "Writer" in instruction)
        self.assertIn("concise", json.dumps(writer["memory"]))
        self.assertNotIn("detailed", json.dumps(writer["memory"]))
        self.assertEqual(self.agent._frozen(task), before)
        ident = self.store.list_chunks(task)[0]["evidence_id"]
        knowledge.confirm("default", task, "alpha", "alpha human confirmed", [ident])
        next_task = self.agent.prepare("alpha followup")["id"]
        memories = self.agent._frozen(next_task)
        self.assertTrue(any(item["human_confirmed"] for item in memories))
        self.assertIn("detailed", json.dumps(memories))

    def test_committed_writer_cache_survives_reopening_without_extra_calls(self):
        task = self.agent.prepare("alpha")["id"]
        self.start(task)
        original = self.store.save_generation

        def crash(*args):
            original(*args)
            if args[1] == "write":
                raise RuntimeError("after cache commit")

        with patch.object(self.store, "save_generation", side_effect=crash):
            with self.assertRaisesRegex(RuntimeError, "cache commit"):
                self.resume(task, True)
        self.assertEqual(self.agent.inspect(task)["next"], ["write"])
        self.open()
        self.resume(task)
        self.assertEqual(self.calls, Counter(plan=1, query=1, write=1, review=1))
        self.assertEqual(self.ledger.get(task)["model_used"], 3)

    def test_budget_can_be_increased_for_v19_without_rewriting(self):
        task = self.agent.prepare("alpha")["id"]
        self.start(task)
        self.manager.set_limits(task, 2, 1)
        with self.assertRaises(BudgetExceeded):
            self.resume(task, True)
        self.assertEqual(self.calls["write"], 1)
        self.assertEqual(self.calls["review"], 0)
        self.manager.set_limits(task, 3, 1)
        self.resume(task)
        self.assertEqual(self.calls["write"], 1)
        self.assertEqual(self.calls["review"], 1)

    def test_committed_reviewer_cache_survives_reopening_without_extra_calls(self):
        task = self.agent.prepare("alpha")["id"]
        self.start(task)
        original = self.store.save_generation

        def crash(*args):
            original(*args)
            if args[1] == "review":
                raise RuntimeError("after reviewer cache commit")

        with patch.object(self.store, "save_generation", side_effect=crash):
            with self.assertRaisesRegex(RuntimeError, "reviewer cache commit"):
                self.resume(task, True)
        self.assertEqual(self.agent.inspect(task)["next"], ["review"])
        self.assertEqual(len(self.store.list_generations(task)), 2)
        self.assertEqual(self.ledger.get(task)["model_used"], 3)
        self.open()
        self.resume(task)
        self.assertEqual(self.calls, Counter(plan=1, query=1, write=1, review=1))
        self.assertEqual(self.ledger.get(task)["model_used"], 3)
        self.assertEqual(self.agent.inspect(task)["next"], [])
        self.assertTrue(audit_database(self.path, task)["accepted"])

    def test_supplement_approval_survives_reopening_without_repeating_initial_calls(self):
        self.supplement = True
        task = self.agent.prepare("alpha")["id"]
        frozen = self.agent._frozen(task)
        self.start(task)
        self.resume(task, True)
        before = self.agent.inspect(task)
        self.assertEqual(before["pending"][0]["kind"], "supplement")
        self.assertEqual(
            before["pending"][0]["queries"],
            [{"source": "web", "query": "alpha details"}],
        )
        self.assertEqual(before["used_supplements"], 0)
        self.assertEqual(self.calls, Counter(plan=1, query=1, write=1, review=1))
        self.open()
        self.assertEqual(self.agent.inspect(task)["pending"], before["pending"])
        self.assertEqual(self.agent._frozen(task), frozen)
        self.resume(task, True)
        self.assertEqual(self.calls, Counter(plan=1, query=2, write=2, review=2))
        usage = self.ledger.get(task)
        self.assertEqual((usage["model_used"], usage["query_used"]), (5, 2))
        self.assertEqual(self.agent.inspect(task)["used_supplements"], 1)
        self.assertEqual(len(self.store.list_documents(task)), 2)
        first_review = self.store.list_reviews(task)[0]["review"]
        self.assertTrue(first_review["approved"])
        self.assertEqual(
            first_review["approved_queries"],
            [{"source": "web", "query": "alpha details"}],
        )
        self.assertTrue(audit_database(self.path, task)["accepted"])

    def test_local_only_reviewer_cannot_request_remote_supplement(self):
        self.remote_review = True
        document = Path(self.tmp.name) / "alpha.md"
        document.write_text("# Alpha\nalpha local evidence.\n", encoding="utf-8")
        task = self.agent.prepare("alpha", files=[document], local_only=True)["id"]
        self.start(task)
        with self.assertRaisesRegex(ValueError, "来源"):
            self.resume(task, True)
        self.assertEqual(self.calls, Counter(plan=1, write=1, review=1))
        usage = self.ledger.get(task)
        self.assertEqual((usage["model_used"], usage["query_used"]), (3, 0))
        self.assertEqual(self.store.list_collections(task), [])
        self.assertEqual(self.agent.inspect(task)["pending"], [])
        self.assertEqual(self.agent.inspect(task)["next"], ["review"])
        self.assertEqual(self.controller.list_failures(task)[0]["category"], "validation")

    def test_other_profile_cannot_resume_and_old_unstarted_tasks_route_v16(self):
        task = self.agent.prepare("alpha")["id"]
        self.start(task)
        self.open(user="other")
        with self.assertRaises(TaskStateError):
            with self.controller.session(task):
                self.fail("must not enter")
        self.assertEqual(self.controller.list_failures(task), [])
        self.open()
        old_task = self.store.create_task("alpha")
        self.store.update_task(
            old_task, "planned", queries=[{"source": "web", "query": "alpha"}]
        )
        self.assertIs(self.controller._select(old_task), self.old)

    def test_cli_memory_can_initialize_without_model_credentials(self):
        import main
        cold = Path(self.tmp.name) / "cold.db"
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(main.main([
                "--db", str(cold), "--memory", "ADD", "preference", "style", "concise"
            ]), 0)
            self.assertEqual(main.main([
                "--db", str(cold), "--memories", "style"
            ]), 0)
        other = sqlite3.connect(cold)
        try:
            self.assertEqual(KnowledgeStore(other).retrieve("default", "")[0]["value"], "concise")
        finally:
            other.close()
        self.assertEqual(self.calls, Counter())

    def test_confirmation_is_never_automatic(self):
        task = self.agent.prepare("alpha")["id"]
        self.start(task)
        self.resume(task, True)
        found = self.agent.knowledge.retrieve("default", "alpha")
        self.assertTrue(found)
        self.assertFalse(any(item["human_confirmed"] for item in found))


if __name__ == "__main__":
    unittest.main()
