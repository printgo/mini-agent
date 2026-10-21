import json
import os
import sqlite3
import unittest
from unittest.mock import patch

from research.context import TokenContext
from test_context_memory import test_encoding

with patch("dotenv.load_dotenv", return_value=False), patch("os.makedirs"), patch.dict(os.environ, {
    "DEEPSEEK_API_KEY": "offline-test",
    "DEEPSEEK_BASE_URL": "https://example.invalid",
    "DEEPSEEK_MODEL": "offline-model",
    "DEEPSEEK_MEMORY_MODEL": "offline-model",
    "LLM_TIMEOUT_SECONDS": "30",
    "LLM_MAX_RETRIES": "0",
    "LLM_MAX_OUTPUT_TOKENS": "4096",
}):
    import agent.agent as app


class MemoryFixture:
    def __init__(self):
        self.conn = sqlite3.connect(":memory:")
        self.values = {"preference.response_style": "concise"}

    def get_messages(self, session):
        return []

    def get_memories(self):
        return dict(self.values)

    def get_memory(self, category, key):
        value = self.values.get(category + "." + key)
        return None if value is None else {"value": value}

    def add_memory(self, category, key, value, source_session_id=None):
        self.values[category + "." + key] = value

    update_memory = add_memory

    def delete_memory(self, category, key):
        self.values.pop(category + "." + key, None)


class ChatContextTests(unittest.TestCase):
    def setUp(self):
        self.memory = MemoryFixture()
        self.addCleanup(self.memory.conn.close)
        for target, kwargs in (
            ("Memory", {"return_value": self.memory}),
            ("TokenContext", {"side_effect": lambda **kwargs: TokenContext(
                limit=16000, reserve=4096, encoding=test_encoding()
            )}),
            ("get_tool_schemas", {"return_value": []}),
        ):
            mocked = patch.object(app, target, **kwargs)
            mocked.start()
            self.addCleanup(mocked.stop)
        quiet = patch("builtins.print")
        quiet.start()
        self.addCleanup(quiet.stop)
        self.agent = app.ResearchAgent("fixture")

    def test_existing_preferences_seed_research_profile_without_overwriting_it(self):
        knowledge = self.agent.knowledge
        self.assertEqual(knowledge.retrieve("default", "")[0]["value"], "concise")
        knowledge.apply(
            "default", "UPDATE", "preference", "preference.response_style", "detailed"
        )
        app.ResearchAgent("fixture")
        self.assertEqual(knowledge.retrieve("default", "")[0]["value"], "detailed")

    def test_chat_memory_actions_sync_to_research_profile(self):
        for action, value in (
            ("ADD", "alpha"), ("UPDATE", "beta"), ("KEEP", "gamma"), ("DELETE", "beta")
        ):
            with self.subTest(action=action), patch.object(
                app, "extract_memories", return_value=[{
                    "category": "research", "key": "direction", "value": value
                }]
            ), patch.object(app, "decide_memory_action", return_value={
                "action": action, "value": value, "reason": "fixture"
            }):
                self.agent._process_memories("删除" if action == "DELETE" else "更新方向")
        self.assertNotIn("research.direction", self.memory.values)
        rows = self.memory.conn.execute(
            "SELECT key FROM research_knowledge WHERE kind='topic'"
        ).fetchall()
        self.assertEqual(rows, [])

    def test_deleted_research_preference_does_not_reappear_when_chat_restarts(self):
        self.agent.knowledge.apply(
            "default", "DELETE", "preference", "preference.response_style"
        )
        self.assertEqual(self.memory.get_memories()["preference.response_style"], "concise")
        restarted = app.ResearchAgent("fixture")
        self.assertEqual(restarted.knowledge.retrieve("default", ""), [])

    def test_legacy_migration_precedes_first_cli_edit_and_survives_reopening(self):
        conn = sqlite3.connect(":memory:")
        self.addCleanup(conn.close)
        conn.executescript("""
            CREATE TABLE memories(id INTEGER PRIMARY KEY, category TEXT, key TEXT, value TEXT);
            INSERT INTO memories VALUES(1,'preference','language','Chinese');
        """)
        knowledge = app.KnowledgeStore(conn)
        self.assertEqual(knowledge.retrieve("default", "")[0]["value"], "Chinese")
        knowledge.apply("default", "DELETE", "preference", "preference.language")
        reopened = app.KnowledgeStore(conn)
        self.assertEqual(reopened.retrieve("default", ""), [])
        self.assertEqual(conn.execute("SELECT value FROM memories").fetchone()[0], "Chinese")

    def test_seed_marker_and_data_roll_back_together(self):
        conn = sqlite3.connect(":memory:")
        self.addCleanup(conn.close)
        knowledge = app.KnowledgeStore(conn)
        with self.assertRaises(ValueError):
            knowledge.seed_chat({"preference.language": "Chinese", "invalid": "x"})
        self.assertEqual(knowledge.retrieve("default", ""), [])
        knowledge.seed_chat({"preference.language": "English"})
        self.assertEqual(knowledge.retrieve("default", "")[0]["value"], "English")

    def test_chat_context_compresses_tools_and_retains_valid_call_pairs(self):
        self.agent.messages = [
            {"role": "user", "content": "研究 alpha"},
            {"role": "assistant", "content": "", "tool_calls": [{
                "id": "c1", "type": "function",
                "function": {"name": "search", "arguments": "{}"},
            }]},
            {"role": "tool", "tool_call_id": "c1",
             "content": "alpha. " * 2000 + " [D1C1]"},
        ]
        messages = self.agent.build_context("研究 alpha")
        tool = next(item for item in messages if item["role"] == "tool")
        self.assertEqual(tool["tool_call_id"], "c1")
        self.assertIn("D1C1", json.loads(tool["content"])["evidence_ids"])
        self.assertLessEqual(
            self.agent.context.cost(messages) + self.agent.context.reserve, 16000
        )


if __name__ == "__main__":
    unittest.main()
