"""冷进程运维回归：真实 SQLite/checkpoint，不需要模型凭据、词表或网络。"""
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from contextlib import closing
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
WORKER = r'''
import json
import sqlite3
import sys
from contextlib import ExitStack, closing
from pathlib import Path
from unittest.mock import patch

request = json.loads(sys.argv[1])
blocked = AssertionError("maintenance must not load tokenizer, model, or network")
with ExitStack() as guards:
    # 在任何业务导入前拦截，确保不读取开发者真实 .env。
    guards.enter_context(patch("dotenv.load_dotenv", return_value=False))
    for target in (
        "research.context.TokenContext", "tiktoken.get_encoding", "openai.OpenAI",
        "requests.sessions.Session.request", "httpx.Client.send",
    ):
        guards.enter_context(patch(target, side_effect=blocked))
    if request["mode"] == "seed":
        from langgraph.checkpoint.sqlite import SqliteSaver
        from research.generation import GenerationStore
        from research.management import BudgetLedger
        from research.engine import TeamWorkflowAgent

        root = Path(request["root"])
        with closing(sqlite3.connect(root / "memory.db", check_same_thread=False)) as conn, \
             closing(sqlite3.connect(root / "research_checkpoints.db", check_same_thread=False)) as checkpoint:
            store = GenerationStore(conn)
            ledger = BudgetLedger(conn)
            agent = TeamWorkflowAgent(
                store, SqliteSaver(checkpoint), ledger, db_path=root / "memory.db",
                index_path=root / "index", model_cache=root / "models",
            )
            tasks = {}
            for name in ("planned", "waiting"):
                task = store.create_task("alpha research")
                agent.knowledge.bind("default", task)
                with conn:
                    conn.execute("INSERT INTO research_task_context VALUES (?, '[]')", (task,))
                store.update_task(task, "planned", queries=[{"source": "web", "query": "alpha"}])
                ledger.reserve(task, "model")
                tasks[name] = task
                if name == "waiting":
                    list(agent.start(task))
            print(json.dumps(tasks))
    elif request["mode"] == "model":
        from llm.client import complete
        try:
            complete([{"role": "user", "content": "alpha"}])
        except ValueError as error:
            assert "DEEPSEEK_API_KEY" in str(error), str(error)
            print(json.dumps({"error": str(error)}))
        else:
            raise AssertionError("model call accepted missing API key")
    else:
        import main
        raise SystemExit(main.main(request["argv"]))
'''


def cold_process(request, cache):
    env = {
        name: value for name, value in os.environ.items()
        if not name.startswith(("DEEPSEEK_", "LLM_"))
    }
    env["PYTHONIOENCODING"] = "utf-8"
    env["TIKTOKEN_CACHE_DIR"] = str(cache)
    return subprocess.run(
        [sys.executable, "-B", "-c", WORKER, json.dumps(request)],
        cwd=PROJECT, env=env, capture_output=True, text=True, encoding="utf-8", timeout=45,
    )


class ColdCliTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.template = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.template.cleanup)
        cls.template_root = Path(cls.template.name)
        result = cold_process(
            {"mode": "seed", "root": str(cls.template_root)},
            cls.template_root / "absent-tokenizer",
        )
        if result.returncode != 0:
            raise AssertionError(result.stdout + result.stderr)
        cls.tasks = json.loads(result.stdout)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.db = self.root / "memory.db"
        self.cache = self.root / "absent-tokenizer"
        for name in ("memory.db", "research_checkpoints.db"):
            shutil.copyfile(self.template_root / name, self.root / name)

    def cli(self, *arguments, expected=0):
        result = cold_process({"mode": "cli", "argv": [
            "--db", str(self.db), "--models", str(self.root / "models"),
            "--index", str(self.root / "index"), *arguments,
        ]}, self.cache)
        self.assertEqual(result.returncode, expected, result.stdout + result.stderr)
        self.assertFalse(self.cache.exists(), "maintenance created a tokenizer cache")
        self.assertFalse((self.root / "index").exists(), "maintenance opened vector index")
        return json.loads(result.stdout)

    def test_planned_state_does_not_require_model_or_tokenizer(self):
        state = self.cli("--state", self.tasks["planned"])
        self.assertEqual(state["status"], "planned")
        self.assertFalse(state["exists"])
        self.assertTrue(state["resume_allowed"])

    def test_persistent_v19_checkpoint_state_is_available_offline(self):
        state = self.cli("--state", self.tasks["waiting"])
        self.assertEqual(state["version"], "19.0")
        self.assertEqual(state["status"], "awaiting_approval")
        self.assertEqual(state["pending"][0]["kind"], "initial")
        self.assertTrue(state["resume_allowed"])

    def test_usage_is_available_without_credentials_or_tokenizer(self):
        usage = self.cli("--usage", self.tasks["planned"])
        self.assertTrue(usage["metered"])
        self.assertEqual((usage["model_used"], usage["query_used"]), (1, 0))
        self.assertIn("规划", usage["scope"])

    def test_results_are_available_without_credentials_or_tokenizer(self):
        self.assertEqual(self.cli("--results", self.tasks["planned"]), [])

    def test_limits_persist_across_cold_processes(self):
        changed = self.cli("--limits", self.tasks["planned"], "16", "10")
        self.assertEqual((changed["model_limit"], changed["query_limit"]), (16, 10))
        usage = self.cli("--usage", self.tasks["planned"])
        self.assertEqual((usage["model_limit"], usage["query_limit"]), (16, 10))
        self.assertEqual(usage["model_used"], 1)

    def test_cancel_persists_without_credentials_or_tokenizer(self):
        task = self.tasks["planned"]
        self.assertEqual(self.cli("--cancel", task)["status"], "cancelled")
        state = self.cli("--state", task)
        self.assertIsNotNone(state["cancelled_at"])
        self.assertFalse(state["resume_allowed"])
        with closing(sqlite3.connect(self.db)) as conn:
            self.assertEqual(conn.execute(
                "SELECT COUNT(*) FROM research_cancellations WHERE task_id=?", (task,)
            ).fetchone()[0], 1)

    def test_actual_model_call_still_requires_api_key(self):
        result = cold_process({"mode": "model"}, self.cache)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("DEEPSEEK_API_KEY", json.loads(result.stdout)["error"])
        self.assertFalse(self.cache.exists())


if __name__ == "__main__":
    unittest.main()
