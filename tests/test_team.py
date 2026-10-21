import os
import sqlite3
import threading
import unittest
from collections import Counter
from unittest.mock import patch

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
    from research.management import BudgetExceeded, BudgetLedger
    from research.team import PlannerAgent, ResearcherAgent, ReviewerAgent


class TeamTests(unittest.TestCase):
    def setUp(self):
        self.conn = sqlite3.connect(":memory:")
        self.addCleanup(self.conn.close)
        self.store = GenerationStore(self.conn)
        self.task = self.store.create_task("alpha")
        self.ledger = BudgetLedger(self.conn)
        self.calls = []

    def collect(self, source, query):
        self.calls.append((source, query))
        return [{
            "title": query,
            "url": "https://example.com/" + source + "/" + query,
            "content": "alpha body",
            "source_type": {
                "web": "web_text", "paper": "paper_abstract",
                "github": "github_readme",
            }[source],
            "metadata": {},
        }], [source + " warning"]

    def entries(self):
        return [
            {"source": "web", "query": "alpha"},
            {"source": "paper", "query": "alpha"},
        ]

    def test_planner_rejects_duplicates_and_unavailable_local(self):
        for result in (
            {"queries": [{"source": "local", "query": "alpha"}]},
            {"queries": [
                {"source": "web", "query": "alpha"},
                {"source": "web", "query": "ALPHA"},
            ]},
            {"queries": [{"source": "web", "query": "alpha", "depends_on": []}]},
        ):
            with self.subTest(result=result), self.assertRaises(ValueError):
                PlannerAgent(lambda *_: result).plan("alpha")

    def test_planner_local_and_memory(self):
        seen = []

        def ask(instruction, data):
            seen.append(data)
            return {"queries": [{"source": "local", "query": " alpha "}]}

        result = PlannerAgent(ask).plan(
            "alpha", [{"content": "preference"}], local_available=True
        )
        self.assertEqual(result, [{"source": "local", "query": "alpha"}])
        self.assertEqual(seen[0]["memories"], [{"content": "preference"}])

    def test_planner_local_only_rejects_remote_output(self):
        planner = PlannerAgent(lambda *_: {
            "queries": [{"source": "web", "query": "alpha"}]
        })
        with self.assertRaises(ValueError):
            planner.plan("alpha", local_available=True, local_only=True)

    def test_reviewer_uses_strict_schema(self):
        args = (
            "alpha",
            {"sections": [{
                "heading": "h", "text": "t", "evidence_ids": ["D1C1"],
            }], "limitations": []},
            [{"evidence_id": "D1C1"}],
            [{"source": "web", "query": "alpha"}],
        )
        valid = {"decision": "pass", "issues": [], "queries": []}
        self.assertEqual(ReviewerAgent(lambda *_: valid).review(*args), valid)
        with self.assertRaises(ValueError):
            ReviewerAgent(lambda *_: {**valid, "extra": True}).review(*args)

    def test_parallel_collection_keeps_database_on_main_thread(self):
        barrier = threading.Barrier(2)
        main_thread = threading.get_ident()
        worker_threads, sql_threads = [], []
        self.conn.set_trace_callback(lambda _: sql_threads.append(threading.get_ident()))

        def collect(source, query):
            worker_threads.append(threading.get_ident())
            self.assertNotEqual(threading.get_ident(), main_thread)
            barrier.wait(timeout=5)
            return self.collect(source, query)

        agent = ResearcherAgent(self.store, self.ledger, collect)
        errors = []
        results = agent.run(self.task, self.entries(), errors)
        self.assertEqual(len(set(worker_threads)), 2)
        self.assertTrue(sql_threads)
        self.assertEqual(set(sql_threads), {main_thread})
        self.assertEqual([item["source"] for item in results], ["web", "paper"])
        self.assertEqual(errors, ["web warning", "paper warning"])
        self.assertEqual(len(self.store.list_documents(self.task)), 2)
        self.assertEqual(self.ledger.get(self.task)["query_used"], 2)

    def test_partial_failure_is_recorded_and_resume_only_retries_failure(self):
        fail = [True]

        def collect(source, query):
            if source == "paper" and fail[0]:
                self.calls.append((source, query))
                raise TimeoutError("injected timeout")
            return self.collect(source, query)

        agent = ResearcherAgent(self.store, self.ledger, collect)
        with self.assertRaises(RuntimeError) as caught:
            agent.run(self.task, self.entries(), [])
        self.assertIsInstance(caught.exception.__cause__, TimeoutError)
        history = agent.list_results(self.task)
        self.assertEqual([item["status"] for item in history], ["succeeded", "failed"])
        self.assertTrue(history[0]["document_ids"])
        self.assertIn("injected timeout", history[1]["error"])
        self.assertEqual(
            self.store.get_collection(self.task, "web", "alpha")["status"], "applied"
        )
        self.assertIsNone(self.store.get_collection(self.task, "paper", "alpha"))
        fail[0] = False
        results = agent.run(self.task, self.entries(), [])
        self.assertEqual([item["status"] for item in results], ["cached", "succeeded"])
        self.assertEqual(Counter(self.calls), {
            ("web", "alpha"): 1, ("paper", "alpha"): 2,
        })
        self.assertEqual(self.ledger.get(self.task)["query_used"], 3)
        self.assertEqual(len(self.store.list_documents(self.task)), 2)

    def test_quota_exhaustion_preserves_already_submitted_success(self):
        self.ledger.set_limits(self.task, 12, 1)
        agent = ResearcherAgent(self.store, self.ledger, self.collect)
        with self.assertRaises(BudgetExceeded):
            agent.run(self.task, self.entries(), [])
        self.assertEqual(
            self.store.get_collection(self.task, "web", "alpha")["status"], "applied"
        )
        self.assertEqual(self.calls, [("web", "alpha")])
        self.ledger.set_limits(self.task, 12, 2)
        agent.run(self.task, self.entries(), [])
        self.assertEqual(Counter(self.calls), {
            ("web", "alpha"): 1, ("paper", "alpha"): 1,
        })
        self.assertEqual(self.ledger.get(self.task)["query_used"], 2)

    def test_local_uses_saved_documents_without_external_calls_or_quota(self):
        with self.conn:
            cursor = self.conn.execute(
                "INSERT INTO research_documents "
                "(task_id,title,url,content,source_type,metadata,retrieved_at) "
                "VALUES(?,?,?,?,?,?,?)",
                (self.task, "local", "local://alpha", "alpha text",
                 "local_document", "{}", "now"),
            )
            document_id = cursor.lastrowid
            self.conn.execute(
                "INSERT INTO research_chunks VALUES(?,?,?,?,?)",
                (document_id, 1, 0, 10, "alpha text"),
            )
        agent = ResearcherAgent(
            self.store, self.ledger, lambda *_: self.fail("external collection called")
        )
        result = agent.run(self.task, [{"source": "local", "query": "alpha"}], [])
        self.assertEqual(result[0]["document_ids"], [document_id])
        self.assertEqual(self.ledger.get(self.task)["query_used"], 0)
        self.assertEqual(self.store.list_collections(self.task), [])

    def test_missing_local_is_recorded_without_external_calls_or_quota(self):
        agent = ResearcherAgent(
            self.store, self.ledger, lambda *_: self.fail("external collection called")
        )
        with self.assertRaises(RuntimeError):
            agent.run(self.task, [{"source": "local", "query": "alpha"}], [])
        self.assertEqual(agent.list_results(self.task)[0]["status"], "failed")
        self.assertEqual(self.ledger.get(self.task)["query_used"], 0)


if __name__ == "__main__":
    unittest.main()
