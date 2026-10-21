import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import tiktoken

from research.context import ContextBudgetError, TokenContext
from research.knowledge import KnowledgeStore


def test_encoding():
    # 实际 tiktoken 引擎与离线测试词表；不是生产 cl100k_base。
    return tiktoken.Encoding(
        name="test-byte-bpe",
        pat_str=r"[\s\S]",
        mergeable_ranks={bytes([value]): value for value in range(256)},
        special_tokens={},
    )


class ContextTests(unittest.TestCase):
    def setUp(self):
        self.context = TokenContext(limit=2600, reserve=300, encoding=test_encoding())

    def pack(self, **kwargs):
        return self.context.pack("研究 alpha", system="Use supplied evidence.", **kwargs)

    @staticmethod
    def payload(result):
        return json.loads(result["messages"][-1]["content"])

    def test_real_tokenizer_counts_utf8_and_preserves_mandatory_fields(self):
        result = self.pack(
            facts=["关键事实"], unresolved=["尚未解决"],
            payload={"plan": [{"source": "web", "query": "alpha"}]},
        )
        data = self.payload(result)
        self.assertEqual(data["question"], "研究 alpha")
        self.assertEqual(data["context_summary"]["facts"], ["关键事实"])
        self.assertEqual(data["context_summary"]["unresolved"], ["尚未解决"])
        self.assertEqual(data["plan"][0]["query"], "alpha")
        self.assertGreater(self.context.count("研究"), self.context.count("ab"))
        self.assertLessEqual(
            result["estimated_input_tokens"] + result["reserved_output_tokens"], 2600
        )

    def test_oversized_goal_or_facts_cannot_be_silently_discarded(self):
        for keyword in ("goal", "facts", "unresolved"):
            with self.subTest(field=keyword), self.assertRaises(ContextBudgetError):
                if keyword == "goal":
                    self.context.pack("研究" * 2000)
                else:
                    self.pack(**{keyword: ["研究" * 2000]})

    def test_tool_schemas_are_included_in_budget(self):
        schema = [{"type": "function", "function": {
            "name": "long_tool", "description": "x" * 4000,
        }}]
        self.pack()
        with self.assertRaises(ContextBudgetError):
            self.pack(tools=schema)

    def test_production_tokenizer_cache_is_project_local_and_respects_override(self):
        expected = Path(__file__).resolve().parents[1] / "data" / "models" / "tiktoken"
        for configured in (None, "custom-tokenizer-cache"):
            with self.subTest(configured=configured), patch.dict("os.environ", {}, clear=True), patch(
                "tiktoken.get_encoding", return_value=test_encoding()
            ) as load:
                import os
                if configured is not None:
                    os.environ["TIKTOKEN_CACHE_DIR"] = configured
                TokenContext()
                self.assertEqual(os.environ["TIKTOKEN_CACHE_DIR"], configured or str(expected))
                load.assert_called_once_with("cl100k_base")

    def test_required_evidence_has_budget_priority_over_earlier_optional_item(self):
        context = TokenContext(limit=1700, reserve=300, encoding=test_encoding())
        evidence = [
            {"evidence_id": "D1C1", "content": "optional alpha", "title": "x" * 800},
            {"evidence_id": "D2C1", "content": "required alpha", "title": "required " * 60},
        ]
        result = context.pack("alpha", evidence=evidence, facts=["Keep D2C1"])
        self.assertIn("D2C1", [item["evidence_id"] for item in result["evidence"]])
        self.assertIn("D1C1", result["dropped_evidence_ids"])
        self.assertLessEqual(result["estimated_input_tokens"] + 300, 1700)

    def test_all_required_evidence_gets_minimum_budget_before_expanding_summaries(self):
        context = TokenContext(limit=1400, reserve=300, encoding=test_encoding())
        evidence = [
            {"evidence_id": f"D{index}C1", "content": "alpha " * 200}
            for index in range(1, 5)
        ]
        result = context.pack(
            "alpha", evidence=evidence, facts=["Keep D1C1 D2C1 D3C1 D4C1"]
        )
        self.assertEqual(
            [item["evidence_id"] for item in result["evidence"]],
            [item["evidence_id"] for item in evidence],
        )
        self.assertLessEqual(result["estimated_input_tokens"] + 300, 1400)

    @staticmethod
    def tool_history(content):
        return [
            {"role": "user", "content": "alpha"},
            {"role": "assistant", "content": "", "tool_calls": [{
                "id": "source-call", "type": "function",
                "function": {"name": "search_web", "arguments": "{}"},
            }]},
            {"role": "tool", "tool_call_id": "source-call", "content": content},
        ]

    def test_long_search_result_preserves_every_title_url_and_excerpt_association(self):
        sources = [
            {"title": "First alpha", "url": "https://example.com/first",
             "content": "alpha first evidence. " * 200},
            {"title": "Second alpha", "url": "https://example.com/second",
             "content": "alpha second evidence [D2C1]. " * 200},
        ]
        content = json.dumps({"success": True, "data": {
            "query": "alpha", "results": sources,
        }, "error": None})
        result = self.context.pack("alpha", history=self.tool_history(content))
        message = next(item for item in result["messages"] if item["role"] == "tool")
        summary = json.loads(message["content"])
        self.assertEqual(message["tool_call_id"], "source-call")
        self.assertTrue(summary["success"])
        self.assertEqual(summary["evidence_ids"], ["D2C1"])
        for original, compact in zip(sources, summary["sources"], strict=True):
            self.assertEqual((compact["title"], compact["url"]),
                             (original["title"], original["url"]))
            self.assertEqual(compact["extractive_summary"], "\n".join(
                original["content"][start:end] for start, end in compact["context_ranges"]
            ))
        self.assertLess(len(message["content"]), len(content))
        self.assertLessEqual(self.context.cost(result["messages"]) + 300, 2600)

    def test_long_webpage_keeps_its_source_url(self):
        content = json.dumps({"success": True, "data": {
            "url": "https://example.com/page", "content": "alpha page. " * 1000,
        }, "error": None})
        result = self.context.pack("alpha", history=self.tool_history(content))
        message = next(item for item in result["messages"] if item["role"] == "tool")
        summary = json.loads(message["content"])
        self.assertEqual(summary["sources"][0]["url"], "https://example.com/page")
        self.assertIn("alpha page.", summary["sources"][0]["extractive_summary"])

    def test_mandatory_source_metadata_over_budget_fails_instead_of_dropping_sources(self):
        content = json.dumps({"success": True, "data": {"results": [{
            "title": "x" * 5000, "url": "https://example.com/large", "content": "alpha",
        }]}, "error": None})
        with self.assertRaises(ContextBudgetError):
            self.context.pack("alpha", history=self.tool_history(content))

    def test_older_tool_summary_keeps_source_provenance_when_selected(self):
        content = json.dumps({"data": {"results": [{
            "title": "Historical alpha", "url": "https://example.com/history",
            "content": "alpha historical evidence. " * 200,
        }]}})
        history = self.tool_history(content) + [{"role": "user", "content": "alpha followup"}]
        result = self.context.pack("alpha followup", history=history)
        summary = next(item for item in result["summary"]["history"] if item["role"] == "tool")
        self.assertEqual(summary["sources"][0]["url"], "https://example.com/history")

    def test_latest_tool_pairs_and_all_tool_citations_are_preserved(self):
        references = [f"D1C{index}" for index in range(1, 21)]
        history = [
            {"role": "user", "content": "研究 alpha"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "call-a", "type": "function",
                 "function": {"name": "search", "arguments": "{}"}},
                {"id": "call-b", "type": "function",
                 "function": {"name": "read", "arguments": "{}"}},
            ]},
            {"role": "tool", "tool_call_id": "call-a",
             "content": "alpha " * 2000 + " ".join(references)},
            {"role": "tool", "tool_call_id": "call-b", "content": "short"},
        ]
        result = self.pack(history=history)
        messages = result["messages"]
        calls = next(item["tool_calls"] for item in messages if "tool_calls" in item)
        results = [item for item in messages if item["role"] == "tool"]
        self.assertEqual([item["id"] for item in calls], ["call-a", "call-b"])
        self.assertEqual([item["tool_call_id"] for item in results], ["call-a", "call-b"])
        summary = json.loads(results[0]["content"])
        self.assertEqual(set(summary["evidence_ids"]), set(references))
        self.assertLess(len(results[0]["content"]), len(history[2]["content"]))
        self.assertLessEqual(self.context.cost(messages) + 300, 2600)
        self.assertIn("alpha " * 1000, history[2]["content"])

    def test_orphan_or_incomplete_tool_pairs_are_rejected(self):
        for history in (
            [{"role": "tool", "tool_call_id": "missing", "content": "x"}],
            [{"role": "assistant", "content": "", "tool_calls": [{"id": "x"}]}],
        ):
            with self.subTest(history=history), self.assertRaises(ValueError):
                self.pack(history=history)

    def test_evidence_ranges_point_to_original_and_ids_are_stable(self):
        original = "alpha 恢复机制。没有足够的性能证据。" * 500
        evidence = {"evidence_id": "D7C3", "document_id": 7, "title": "设计",
                    "url": "https://example.com/design", "source_type": "web_text",
                    "chapter": "恢复", "content": original}
        result = self.pack(evidence=[evidence], facts=["参考 [D7C3]"])
        excerpt = result["evidence"][0]
        self.assertEqual(excerpt["evidence_id"], "D7C3")
        self.assertEqual(excerpt["chapter"], "恢复")
        self.assertEqual(excerpt["content"], "\n".join(
            original[start:end] for start, end in excerpt["context_ranges"]
        ))
        self.assertLess(len(excerpt["content"]), len(original))
        self.assertEqual(evidence["content"], original)

    def test_duplicates_are_removed_and_conflicting_ids_are_rejected(self):
        evidence = {"evidence_id": "D1C1", "source_type": "web_text",
                    "content": "alpha evidence."}
        result = self.pack(evidence=[
            evidence, dict(evidence), {**evidence, "evidence_id": "D2C1"},
        ])
        self.assertEqual(len(result["evidence"]), 1)
        self.assertIn("D2C1", result["dropped_evidence_ids"])
        with self.assertRaises(ValueError):
            self.pack(evidence=[evidence, {**evidence, "content": "different"}])

    def test_fact_and_unresolved_citations_cannot_be_dropped(self):
        evidence = {"evidence_id": "D1C1", "content": "alpha",
                    "title": "x" * 4000}
        for keyword in ("facts", "unresolved"):
            with self.subTest(field=keyword), self.assertRaises(ContextBudgetError):
                self.pack(evidence=[evidence], **{keyword: ["核验 [D1C1]"]})

    def test_feedback_mentioning_invalid_citation_does_not_require_it(self):
        result = self.pack(unresolved=["不要使用 D999C1"])
        self.assertEqual(self.payload(result)["context_summary"]["unresolved"], ["不要使用 D999C1"])

    def test_oversized_optional_memory_is_dropped_without_error(self):
        result = self.pack(memories=[{"value": "超长记忆" * 3000}])
        self.assertEqual(self.payload(result)["memory"], [])
        self.assertLessEqual(
            result["estimated_input_tokens"] + result["reserved_output_tokens"], 2600
        )

    def test_older_history_is_deduplicated_and_latest_user_is_kept(self):
        history = [
            {"role": "user", "content": "old alpha fact [D3C1]."},
            {"role": "assistant", "content": "old alpha fact [D3C1]."},
            {"role": "user", "content": "研究 alpha"},
        ]
        result = self.pack(history=history)
        summaries = result["summary"]["history"]
        self.assertEqual(len(summaries), 1)
        self.assertEqual(summaries[0]["evidence_ids"], ["D3C1"])
        self.assertEqual(result["messages"][1], history[-1])
        compressed = self.context.chat([
            {"role": "system", "content": "rules"}, *history,
        ])
        self.assertLessEqual(self.context.cost(compressed) + 300, 2600)


class KnowledgeTests(unittest.TestCase):
    def setUp(self):
        self.conn = sqlite3.connect(":memory:")
        self.addCleanup(self.conn.close)
        self.conn.executescript("""
            CREATE TABLE research_tasks(id TEXT PRIMARY KEY, status TEXT);
            INSERT INTO research_tasks VALUES
                ('a','completed'), ('b','completed'), ('pending','partial');
            CREATE TABLE research_documents(id INTEGER, task_id TEXT);
            INSERT INTO research_documents VALUES (1,'a'), (2,'b');
            CREATE TABLE research_chunks(document_id INTEGER, chunk_index INTEGER);
            INSERT INTO research_chunks VALUES (1,1), (2,1);
        """)
        self.memory = KnowledgeStore(self.conn)
        for user, task in (("alice", "a"), ("bob", "b"), ("alice", "pending")):
            self.memory.bind(user, task)

    def test_four_operations_and_missing_update(self):
        memory = self.memory
        memory.apply("alice", "ADD", "preference", "style", "concise")
        memory.apply("alice", "KEEP", "preference", "style", "ignored")
        self.assertEqual(memory.retrieve("alice", "")[0]["value"], "concise")
        memory.apply("alice", "UPDATE", "preference", "style", "detailed")
        self.assertEqual(memory.retrieve("alice", "")[0]["value"], "detailed")
        memory.apply("alice", "DELETE", "preference", "style")
        self.assertEqual(memory.retrieve("alice", ""), [])
        with self.assertRaises(ValueError):
            memory.apply("alice", "UPDATE", "topic", "missing", "alpha")

    def test_user_isolation_and_task_ownership(self):
        for user in ("alice", "bob"):
            self.memory.apply(user, "ADD", "preference", "style", user)
        self.assertEqual(self.memory.retrieve("alice", "")[0]["value"], "alice")
        self.assertEqual(self.memory.retrieve("bob", "")[0]["value"], "bob")
        self.assertEqual(self.memory.retrieve("unknown", "alpha"), [])
        with self.assertRaises(PermissionError):
            self.memory.bind("bob", "a")
        with self.assertRaises(PermissionError):
            self.memory.apply("bob", "ADD", "topic", "a", "alpha", task_id="a")
        self.assertEqual(self.memory.owner("a"), "alice")

    def test_model_cannot_automatically_confirm_conclusions(self):
        for action in ("ADD", "UPDATE"):
            with self.subTest(action=action), self.assertRaises(ValueError):
                self.memory.apply("alice", action, "conclusion", "claim", "alpha")
        self.assertEqual(self.memory.retrieve("alice", "alpha"), [])

    def test_manual_confirmation_checks_status_owner_and_citations(self):
        for user, task, citations, error in (
            ("alice", "pending", ["D1C1"], ValueError),
            ("bob", "a", ["D1C1"], PermissionError),
            ("alice", "a", ["D2C1"], ValueError),
            ("alice", "a", [], ValueError),
        ):
            with self.subTest(task=task, citations=citations), self.assertRaises(error):
                self.memory.confirm(user, task, "claim", "alpha conclusion", citations)
        self.memory.confirm("alice", "a", "claim", "alpha conclusion", ["D1C1", "D1C1"])
        result = self.memory.retrieve("alice", "alpha")[0]
        self.assertTrue(result["human_confirmed"])
        self.assertEqual(result["evidence_ids"], ["D1C1"])
        self.assertEqual(result["source_task_id"], "a")

    def test_retrieval_filters_irrelevant_topics(self):
        self.memory.apply("alice", "ADD", "topic", "alpha", "alpha research")
        self.memory.apply("alice", "ADD", "topic", "other", "unrelated database")
        found = self.memory.retrieve("alice", "alpha")
        self.assertEqual([item["key"] for item in found], ["alpha"])
        with self.assertRaises(ValueError):
            self.memory.retrieve("alice", "alpha", limit=True)

    def test_memory_and_owner_survive_new_connection(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "knowledge.db"
            disk = sqlite3.connect(path)
            try:
                self.conn.backup(disk)
                memory = KnowledgeStore(disk)
                memory.confirm("alice", "a", "claim", "alpha confirmed", ["D1C1"])
            finally:
                disk.close()
            reopened = sqlite3.connect(path)
            try:
                memory = KnowledgeStore(reopened)
                self.assertEqual(memory.owner("a"), "alice")
                self.assertTrue(memory.retrieve("alice", "alpha")[0]["human_confirmed"])
                self.assertEqual(memory.retrieve("bob", "alpha"), [])
            finally:
                reopened.close()


if __name__ == "__main__":
    unittest.main()
