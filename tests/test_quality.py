import io
import json
import sqlite3
import tempfile
import unittest
from contextlib import closing, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from research.quality import audit_database, evaluate, main


def example():
    chunks = [{
        "evidence_id": "D1C1", "title": "Alpha",
        "url": "https://example.com/alpha", "source_type": "web_text",
        "content": "Alpha has a public description.",
    }, {
        "evidence_id": "D1C2", "title": "Alpha",
        "url": "https://example.com/alpha", "source_type": "web_text",
        "content": "Another chunk from the same document.",
    }]
    task = {
        "id": "t", "question": "Alpha?", "status": "completed",
        "queries": [{"source": "web", "query": "alpha"}], "errors": [],
        "report": (
            "# 研究报告\n\n任务：t\n\n问题：Alpha?\n\n"
            "## 结论\n\nAlpha 有公开说明。 [D1C1]\n\n"
            "## 局限\n\n"
            "- 本报告依据检索片段生成，正文和 README 均为限长快照。"
            "论文来源仅为摘要，GitHub 资料不代表已审计代码；"
            "引用编号合法不代表结论已经得到语义核验。"
            "模型 Critic 审查不能替代人工核验或实验复现。\n\n"
            "- 仅为快照\n\n## 参考来源\n\n"
            "- [D1C1] Alpha — <https://example.com/alpha>"
        ),
    }
    audit = {
        "round_no": 1, "evidence_ids": ["D1C1"],
        "draft": {"sections": [{
            "heading": "结论", "text": "Alpha 有公开说明。",
            "evidence_ids": ["D1C1"],
        }], "limitations": ["仅为快照"]},
        "review": {
            "decision": "pass", "issues": [], "queries": [], "action": "reviewed",
        },
    }
    return task, audit, chunks


class QualityTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "质量 #1.db"

    def save(self, task, audit, chunks):
        with closing(sqlite3.connect(self.path)) as conn, conn:
            conn.executescript("""
                CREATE TABLE research_tasks (
                    id TEXT, question TEXT, status TEXT, queries TEXT,
                    report TEXT, errors TEXT);
                CREATE TABLE research_reviews (
                    task_id TEXT, round_no INTEGER, evidence_ids TEXT,
                    draft TEXT, review TEXT);
                CREATE TABLE research_documents (
                    id INTEGER, task_id TEXT, title TEXT, url TEXT,
                    source_type TEXT);
                CREATE TABLE research_chunks (
                    document_id INTEGER, chunk_index INTEGER, content TEXT);
            """)
            conn.execute(
                "INSERT INTO research_tasks VALUES (?, ?, ?, ?, ?, ?)",
                (task["id"], task["question"], task["status"],
                 json.dumps(task["queries"]), task["report"], json.dumps(task["errors"])),
            )
            if audit is not None:
                conn.execute(
                    "INSERT INTO research_reviews VALUES (?, ?, ?, ?, ?)",
                    (task["id"], audit["round_no"], json.dumps(audit["evidence_ids"]),
                     json.dumps(audit["draft"]), json.dumps(audit["review"])),
                )
            conn.execute(
                "INSERT INTO research_documents VALUES (1, ?, 'Alpha', "
                "'https://example.com/alpha', 'web_text')", (task["id"],),
            )
            conn.executemany(
                "INSERT INTO research_chunks VALUES (1, ?, ?)",
                [(index, chunk["content"]) for index, chunk in enumerate(chunks, 1)],
            )

    def assert_failed(self, data, code):
        result = evaluate(*data)
        self.assertFalse(result["accepted"])
        self.assertIn({"code": code, "passed": False}, result["checks"])

    def test_valid_snapshot_and_cli_exit_zero(self):
        self.save(*example())
        before = self.path.read_bytes()
        with redirect_stdout(io.StringIO()) as output:
            self.assertEqual(main(["--db", str(self.path), "--task", "t"]), 0)
        result = json.loads(output.getvalue())
        self.assertTrue(result["accepted"])
        self.assertEqual(result["metrics"]["unique_citations"], 1)
        self.assertEqual(before, self.path.read_bytes())

    def test_noncompleted_tasks_fail_even_with_pass(self):
        for status in ("partial", "cancelled", "failed", "writing", "awaiting_approval"):
            with self.subTest(status=status):
                data = example()
                data[0]["status"] = status
                self.assert_failed(data, "task_completed")

    def test_latest_pending_never_falls_back_to_previous_pass(self):
        self.save(*example())
        with closing(sqlite3.connect(self.path)) as conn, conn:
            conn.execute(
                "INSERT INTO research_reviews VALUES ('t', 2, '[]', '{}', "
                "'{\"decision\": \"pending\"}')"
            )
        self.assertFalse(audit_database(self.path, "t")["accepted"])

    def test_missing_audit_fails(self):
        task, _, chunks = example()
        self.assert_failed((task, None, chunks), "latest_audit_exists")

    def test_same_task_chunk_outside_current_round_fails(self):
        data = example()
        data[1]["draft"]["sections"][0]["evidence_ids"] = ["D1C2"]
        self.assert_failed(data, "citations_in_current_round")

    def test_other_task_chunk_cannot_be_used(self):
        data = example()
        data[1]["evidence_ids"] = ["D2C1"]
        data[1]["draft"]["sections"][0]["evidence_ids"] = ["D2C1"]
        self.save(*data)
        with closing(sqlite3.connect(self.path)) as conn, conn:
            conn.execute(
                "INSERT INTO research_documents VALUES "
                "(2, 'other', 'Other', 'https://example.com/other', 'web_text')"
            )
            conn.execute("INSERT INTO research_chunks VALUES (2, 1, 'other')")
        result = audit_database(self.path, "t")
        self.assertIn(
            {"code": "citations_in_current_round", "passed": False}, result["checks"]
        )

    def test_github_metadata_does_not_satisfy_readme_coverage(self):
        data = example()
        data[0]["queries"] = [{"source": "github", "query": "alpha"}]
        data[2][0]["source_type"] = "github_metadata"
        self.assert_failed(data, "planned_sources_cited")

    def test_reference_order_follows_round_evidence_order(self):
        data = example()
        data[1]["evidence_ids"] = ["D1C2", "D1C1"]
        data[1]["draft"]["sections"][0]["evidence_ids"] = ["D1C2", "D1C1"]
        data[0]["report"] = data[0]["report"].replace(
            "Alpha 有公开说明。 [D1C1]", "Alpha 有公开说明。 [D1C2] [D1C1]"
        ).replace(
            "- [D1C1] Alpha — <https://example.com/alpha>",
            "- [D1C2] Alpha — <https://example.com/alpha>\n\n"
            "- [D1C1] Alpha — <https://example.com/alpha>",
        )
        self.assertTrue(evaluate(*data)["accepted"])

    def test_report_replacement_is_detected(self):
        data = example()
        data[0]["report"] += "\nForged conclusion."
        self.assert_failed(data, "report_matches_latest_draft")

    def test_prompt_rules_are_now_audited(self):
        for text, code in (
            ("甲" * 801, "body_within_800_chars"),
            ("见 https://example.com", "no_manual_citations_or_urls"),
            ("已经证明 [D1C1]", "no_manual_citations_or_urls"),
        ):
            with self.subTest(code=code, text=text[:20]):
                data = example()
                data[1]["draft"]["sections"][0]["text"] = text
                self.assert_failed(data, code)

    def test_malformed_shapes_fail_without_crashing(self):
        for key, value, code in (
            ("draft", {"sections": [None], "limitations": []}, "draft_schema"),
            ("review", {"decision": "pass", "issues": [{}]}, "critic_pass"),
            ("evidence_ids", [{"bad": "id"}], "round_evidence_exists"),
        ):
            with self.subTest(key=key):
                data = example()
                data[1][key] = value
                self.assert_failed(data, code)

    def test_missing_database_exit_two_does_not_create_it(self):
        missing = self.path.with_name("missing.db")
        with redirect_stdout(io.StringIO()):
            self.assertEqual(main(["--db", str(missing), "--task", "t"]), 2)
        self.assertFalse(missing.exists())

    def test_rejected_snapshot_exit_one_and_corrupt_json_exit_two(self):
        data = example()
        data[0]["status"] = "partial"
        self.save(*data)
        with redirect_stdout(io.StringIO()):
            self.assertEqual(main(["--db", str(self.path), "--task", "t"]), 1)
        with closing(sqlite3.connect(self.path)) as conn, conn:
            conn.execute("UPDATE research_reviews SET draft='{'")
        with redirect_stdout(io.StringIO()):
            self.assertEqual(main(["--db", str(self.path), "--task", "t"]), 2)

    def test_readonly_connection_rejects_updates(self):
        self.save(*example())
        original = sqlite3.connect

        def checked(*args, **kwargs):
            conn = original(*args, **kwargs)
            with self.assertRaises(sqlite3.OperationalError):
                conn.execute("UPDATE research_tasks SET status='failed'")
            conn.rollback()
            return conn

        with patch("research.quality.sqlite3.connect", side_effect=checked):
            self.assertTrue(audit_database(self.path, "t")["accepted"])


if __name__ == "__main__":
    unittest.main()