import hashlib
import io
import json
import os
import sqlite3
import tempfile
import unittest
import zipfile
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing, redirect_stdout
from pathlib import Path
from unittest.mock import patch

import research.export as exporter
from test_quality import example


class ExportTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.db = self.root / "业务 #1.db"
        self.output = self.root / "研究 报告.zip"
        task, audit, chunks = example()
        with closing(sqlite3.connect(self.db)) as conn, conn:
            conn.executescript("""
                CREATE TABLE research_tasks (
                    id TEXT PRIMARY KEY, question TEXT, status TEXT,
                    queries TEXT, report TEXT, errors TEXT, created_at TEXT);
                CREATE TABLE research_reviews (
                    task_id TEXT, round_no INTEGER, evidence_ids TEXT,
                    draft TEXT, review TEXT, created_at TEXT);
                CREATE TABLE research_documents (
                    id INTEGER PRIMARY KEY, task_id TEXT, title TEXT, url TEXT,
                    source_type TEXT, metadata TEXT, retrieved_at TEXT);
                CREATE TABLE research_chunks (
                    document_id INTEGER, chunk_index INTEGER,
                    start_char INTEGER, end_char INTEGER, content TEXT);
            """)
            conn.execute(
                "INSERT INTO research_tasks VALUES (?, ?, ?, ?, ?, ?, 'time')",
                (task["id"], task["question"], task["status"],
                 json.dumps(task["queries"]), task["report"], json.dumps(task["errors"])),
            )
            conn.execute(
                "INSERT INTO research_reviews VALUES (?, ?, ?, ?, ?, 'time')",
                ("t", audit["round_no"], json.dumps(audit["evidence_ids"]),
                 json.dumps(audit["draft"]), json.dumps(audit["review"])),
            )
            conn.execute(
                "INSERT INTO research_documents VALUES "
                "(1, 't', 'Alpha', 'https://example.com/alpha', "
                "'web_text', '{\"truncated\":false}', 'time')"
            )
            conn.executemany(
                "INSERT INTO research_chunks VALUES (1, ?, ?, ?, ?)",
                [(i, (i - 1) * 700, (i - 1) * 700 + len(item["content"]),
                  item["content"]) for i, item in enumerate(chunks, 1)],
            )

    def update(self, sql, parameters=()):
        with closing(sqlite3.connect(self.db)) as conn, conn:
            conn.execute(sql, parameters)

    def run_cli(self, *extra, db=None, task="t"):
        with redirect_stdout(io.StringIO()):
            return exporter.main([
                "--db", str(db or self.db), "--task", task,
                "--output", str(self.output), *extra,
            ])

    def bundle(self):
        with zipfile.ZipFile(self.output) as archive:
            return {name: archive.read(name) for name in archive.namelist()}

    def assert_no_artifacts(self):
        self.assertFalse(self.output.exists())
        self.assertEqual(list(self.root.glob(".research-export-*.tmp")), [])

    def test_bundle_contains_utf8_evidence_and_valid_hashes(self):
        before = self.db.read_bytes()
        self.assertEqual(self.run_cli(), 0)
        files = self.bundle()
        self.assertEqual(set(files), {
            "report.md", "evidence.json", "quality.json", "manifest.json",
        })
        self.assertIn("Alpha 有公开说明。", files["report.md"].decode("utf-8"))
        evidence = json.loads(files["evidence.json"])
        self.assertEqual(evidence["scope"], "all_task_chunks")
        self.assertEqual(len(evidence["chunks"]), 2)
        self.assertEqual(evidence["chunks"][0]["metadata"], {"truncated": False})
        self.assertEqual(len(evidence["reviews"]), 1)
        manifest = json.loads(files["manifest.json"])
        self.assertEqual(manifest["artifact_kind"], "accepted_report")
        self.assertTrue(json.loads(files["quality.json"])["accepted"])
        self.assertEqual(set(manifest["files"]), set(files) - {"manifest.json"})
        for name, entry in manifest["files"].items():
            self.assertEqual(entry["bytes"], len(files[name]))
            self.assertEqual(entry["sha256"], hashlib.sha256(files[name]).hexdigest())
        self.assertEqual(before, self.db.read_bytes())

    def test_default_rejects_unaccepted_terminal_tasks(self):
        for status in ("partial", "cancelled", "failed"):
            with self.subTest(status=status):
                self.update("UPDATE research_tasks SET status=?", (status,))
                self.assertEqual(self.run_cli(), 1)
                self.assert_no_artifacts()

    def test_snapshot_is_explicitly_marked_for_all_terminal_states(self):
        for status in ("completed", "partial", "cancelled", "failed"):
            with self.subTest(status=status):
                self.update("UPDATE research_tasks SET status=?", (status,))
                self.assertEqual(self.run_cli("--snapshot"), 0)
                files = self.bundle()
                self.assertTrue(files["report.md"].decode().startswith(
                    "# 研究快照：不作为正式验收报告"
                ))
                manifest = json.loads(files["manifest.json"])
                self.assertEqual(manifest["artifact_kind"], "snapshot")
                self.assertEqual(manifest["task_status"], status)
                self.assertEqual(
                    json.loads(files["quality.json"])["accepted"], status == "completed"
                )
                self.output.unlink()

    def test_snapshot_also_rejects_running_tasks(self):
        for status in ("planning", "planned", "awaiting_approval", "writing", "reviewing"):
            with self.subTest(status=status):
                self.update("UPDATE research_tasks SET status=?", (status,))
                self.assertEqual(self.run_cli("--snapshot"), 1)
                self.assert_no_artifacts()

    def test_foreign_evidence_cannot_leak_into_snapshot(self):
        self.update(
            "INSERT INTO research_documents VALUES "
            "(2, 'other', 'SECRET', 'https://example.com/secret', 'web_text', '{}', 'time')"
        )
        self.update(
            "INSERT INTO research_chunks VALUES (2, 1, 0, 6, 'SECRET')"
        )
        _, audit, _ = example()
        audit["draft"]["sections"][0]["evidence_ids"] = ["D2C1"]
        self.update(
            "UPDATE research_reviews SET evidence_ids=?, draft=?",
            (json.dumps(["D2C1"]), json.dumps(audit["draft"])),
        )
        self.assertEqual(self.run_cli(), 1)
        self.assert_no_artifacts()
        self.assertEqual(self.run_cli("--snapshot"), 0)
        evidence = json.loads(self.bundle()["evidence.json"])
        self.assertEqual({x["document_id"] for x in evidence["chunks"]}, {1})
        self.assertNotIn("SECRET", json.dumps(evidence))

    def test_latest_pending_does_not_fall_back_to_older_pass(self):
        self.update(
            "INSERT INTO research_reviews VALUES ('t', 2, '[]', '{}', "
            "'{\"decision\":\"pending\"}', 'time')"
        )
        self.assertEqual(self.run_cli(), 1)
        self.assert_no_artifacts()
        self.assertEqual(self.run_cli("--snapshot"), 0)
        evidence = json.loads(self.bundle()["evidence.json"])
        self.assertEqual([x["round_no"] for x in evidence["reviews"]], [1, 2])
        self.assertFalse(json.loads(self.bundle()["quality.json"])["accepted"])

    def test_existing_file_or_directory_is_never_overwritten(self):
        self.output.write_bytes(b"KEEP")
        self.assertEqual(self.run_cli(), 2)
        self.assertEqual(self.output.read_bytes(), b"KEEP")
        self.output.unlink()
        self.output.mkdir()
        self.assertEqual(self.run_cli(), 2)
        self.assertTrue(self.output.is_dir())

    def test_concurrent_publication_has_exactly_one_winner(self):
        def attempt(_):
            try:
                exporter.export_task(self.db, "t", self.output)
                return True
            except FileExistsError:
                return False

        with ThreadPoolExecutor(max_workers=2) as pool:
            outcomes = list(pool.map(attempt, range(2)))
        self.assertEqual(sum(outcomes), 1)
        self.assertTrue(json.loads(self.bundle()["quality.json"])["accepted"])
        self.assertEqual(list(self.root.glob(".research-export-*.tmp")), [])

    def test_write_sync_or_publish_failure_leaves_no_output(self):
        for target in (
            "research.export.zipfile.ZipFile.writestr",
            "research.export.os.fsync",
            "research.export.os.rename" if os.name == "nt"
            else "research.export.os.link",
        ):
            with self.subTest(target=target), patch(
                target, side_effect=OSError("injected output failure")
            ):
                self.assertEqual(self.run_cli(), 2)
                self.assert_no_artifacts()

    def test_publish_race_preserves_other_writers_file(self):
        def competing_publish(*args):
            self.output.write_bytes(b"WINNER")
            raise FileExistsError("another writer won")

        publish = "os.rename" if os.name == "nt" else "os.link"
        with patch("research.export." + publish, side_effect=competing_publish):
            self.assertEqual(self.run_cli(), 2)
        self.assertEqual(self.output.read_bytes(), b"WINNER")
        self.assertEqual(list(self.root.glob(".research-export-*.tmp")), [])

    def test_missing_database_task_and_corrupt_json_exit_two(self):
        missing = self.root / "missing.db"
        self.assertEqual(self.run_cli(db=missing), 2)
        self.assertFalse(missing.exists())
        self.assertEqual(self.run_cli(task="unknown"), 2)
        self.update("UPDATE research_reviews SET draft='{'")
        self.assertEqual(self.run_cli(), 2)
        self.assert_no_artifacts()

    def test_wal_writer_during_read_cannot_mix_snapshots(self):
        original_connect = sqlite3.connect
        changed = []

        with closing(original_connect(self.db)) as conn:
            conn.execute("PRAGMA journal_mode=WAL")

        def connected(*args, **kwargs):
            conn = original_connect(*args, **kwargs)

            def trace(sql):
                if sql.startswith("SELECT round_no") and not changed:
                    with closing(original_connect(self.db)) as other, other:
                        other.execute("UPDATE research_tasks SET report='CHANGED'")
                        other.execute(
                            "UPDATE research_reviews SET review="
                            "'{\"decision\":\"pending\"}'"
                        )
                    changed.append(True)

            conn.set_trace_callback(trace)
            return conn

        with patch("research.export.sqlite3.connect", side_effect=connected):
            self.assertEqual(self.run_cli(), 0)
        self.assertEqual(changed, [True])
        evidence = json.loads(self.bundle()["evidence.json"])
        self.assertNotEqual(evidence["task"]["report"], "CHANGED")
        self.assertEqual(evidence["reviews"][-1]["review"]["decision"], "pass")

    def test_readonly_connection_rejects_updates(self):
        original_connect = sqlite3.connect

        def connected(*args, **kwargs):
            conn = original_connect(*args, **kwargs)
            with self.assertRaises(sqlite3.OperationalError):
                conn.execute("UPDATE research_tasks SET status='failed'")
            conn.rollback()
            return conn

        with patch("research.export.sqlite3.connect", side_effect=connected):
            self.assertEqual(self.run_cli(), 0)


if __name__ == "__main__":
    unittest.main()