"""离线契约测试：固定测试向量不代表真实 Embedding/Reranker 质量。"""
import math
import sqlite3
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch
from uuid import NAMESPACE_URL, uuid5

from research.documents import DocumentStore
from research.vector import HybridIndex, import_file


class FakeClient:
    stores, filters = {}, []

    def __init__(self, path):
        self.data = self.stores.setdefault(path, {})

    def close(self):
        pass

    def collection_exists(self, name):
        return name in self.data

    def create_collection(self, name, vectors_config):
        self.data[name] = {}

    def retrieve(self, name, ids, **kwargs):
        return [self.data[name][key] for key in ids if key in self.data[name]]

    def upsert(self, name, points, wait):
        assert wait is True
        self.data[name].update({point.id: point for point in points})

    def query_points(self, name, query, limit, query_filter, **kwargs):
        task = query_filter.must[0].match.value
        allowed = set(query_filter.must[1].has_id)
        self.filters.append((task, allowed))
        points = [point for point in self.data[name].values()
                  if point.payload["task_id"] == task and point.id in allowed]
        points.sort(key=lambda point: (
            -sum(a * b for a, b in zip(query, point.vector)), point.id))
        return NS(points=points[:limit])


class EmbeddingFixture:
    embedding_size = 2

    def __init__(self):
        self.batches = []

    def passage_embed(self, texts, **kwargs):
        self.batches.append(list(texts))
        return iter([[1., 0.] if "semantic" in text else [0., 1.]
                     for text in texts])

    def query_embed(self, texts, **kwargs):
        return iter([[1., 0.] for text in texts])


class RerankerFixture:
    def rerank(self, query, texts, **kwargs):
        return iter([10. if "promote" in text else 0. for text in texts])


class VectorTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.conn = sqlite3.connect(":memory:")
        self.addCleanup(self.conn.close)
        self.store = DocumentStore(self.conn)
        self.task = self.store.create_task("alpha")
        self.store.update_task(self.task, "planned")
        self.file = self.root / "资料.md"
        self.file.write_text("# First\nalpha\n# Second\nbeta", encoding="utf-8")
        FakeClient.stores, FakeClient.filters = {}, []
        fake = types.ModuleType("qdrant_client")
        fake.QdrantClient = FakeClient
        fake.models = NS(
            VectorParams=NS, PointStruct=NS, Filter=NS, FieldCondition=NS,
            MatchValue=NS, HasIdCondition=NS, Distance=NS(COSINE="Cosine"))
        mocked = patch.dict(sys.modules, {"qdrant_client": fake})
        mocked.start()
        self.addCleanup(mocked.stop)
        self.embedder = EmbeddingFixture()
        self.index = HybridIndex(
            self.root / "index", embedder=self.embedder,
            reranker=RerankerFixture(), client_factory=FakeClient)

    def chunks(self):
        return [
            {"evidence_id": "D1C1", "document_id": 1, "chunk_index": 1,
             "title": "Alpha", "content": "alpha lexical", "source_type": "web_text"},
            {"evidence_id": "D2C1", "document_id": 2, "chunk_index": 1,
             "title": "Beta", "content": "semantic promote", "source_type": "web_text"},
        ]

    def test_markdown_preserves_sections_and_duplicate_import_is_idempotent(self):
        first = import_file(self.store, self.task, self.file)
        second = import_file(self.store, self.task, self.file)
        self.assertEqual(first, second)
        documents = self.store.list_documents(self.task)
        self.assertEqual(len(documents), 2)
        self.assertEqual([item["metadata"]["section"] for item in documents],
                         ["First", "Second"])
        self.assertTrue(all(item["url"].startswith("file:///") for item in documents))
        self.assertEqual(documents[0]["metadata"]["file_name"], "资料.md")
        self.assertEqual(len(self.store.list_chunks(self.task)), 2)

    def test_markdown_code_fence_does_not_create_fake_chapters(self):
        self.file.write_text("# Alpha\n```python\n# not a heading\n```\nalpha\n", encoding="utf-8")
        import_file(self.store, self.task, self.file)
        self.assertEqual(len(self.store.list_documents(self.task)), 1)

    def test_changed_file_keeps_old_immutable_snapshot(self):
        old_ids = import_file(self.store, self.task, self.file)
        old_hash = self.store.list_documents(self.task)[0]["metadata"]["sha256"]
        self.file.write_text("# New\nchanged", encoding="utf-8")
        new_ids = import_file(self.store, self.task, self.file)
        self.assertFalse(set(old_ids) & set(new_ids))
        documents = self.store.list_documents(self.task)
        self.assertEqual(len(documents), 3)
        self.assertNotEqual(old_hash, documents[-1]["metadata"]["sha256"])
        self.assertEqual(self.store.list_chunks(self.task)[0]["content"], "alpha")

    def test_plain_utf8_text_and_overlap_keep_full_content(self):
        path = self.root / "plain.txt"
        text = "正文资料" * 300
        path.write_text(text, encoding="utf-8-sig")
        import_file(self.store, self.task, path)
        chunks = self.store.list_chunks(self.task)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(item["metadata"]["section"] == "正文" for item in chunks))
        for item in chunks:
            self.assertEqual(item["content"], text[item["start"]:item["end"]])
        self.assertEqual(chunks[-1]["end"], len(text))

    def test_terminal_or_unknown_tasks_and_unsupported_files_are_rejected(self):
        for status in ("completed", "partial", "cancelled", "failed"):
            self.store.update_task(self.task, status)
            with self.assertRaises(ValueError):
                import_file(self.store, self.task, self.file)
        with self.assertRaises(ValueError):
            import_file(self.store, "missing", self.file)
        self.store.update_task(self.task, "planned")
        wrong = self.root / "unsupported.bin"
        wrong.write_bytes(b"abc")
        with self.assertRaises(ValueError):
            import_file(self.store, self.task, wrong)
        self.assertEqual(self.store.list_documents(self.task), [])

    def test_incremental_index_embeds_only_missing_or_changed_chunks(self):
        chunks = self.chunks()
        self.index.search(self.task, "alpha", chunks)
        self.index.search(self.task, "alpha", chunks)
        self.assertEqual([len(batch) for batch in self.embedder.batches], [2])
        chunks[0]["content"] += " changed"
        self.index.search(self.task, "alpha", chunks)
        self.assertEqual([len(batch) for batch in self.embedder.batches], [2, 1])

    def test_rrf_fuses_both_rankings_before_independent_reranking(self):
        result = self.index.search(self.task, "alpha", self.chunks())
        self.assertEqual([item["evidence_id"] for item in result], ["D2C1", "D1C1"])
        by_id = {item["evidence_id"]: item for item in result}
        self.assertAlmostEqual(by_id["D1C1"]["rrf_score"], 1 / 61 + 1 / 62)
        self.assertAlmostEqual(by_id["D2C1"]["rrf_score"], 1 / 61)
        self.assertEqual(result[0]["rerank_score"], 10.)

    def test_task_and_allowed_chunk_filters_exclude_foreign_points(self):
        self.index.search("other", "alpha", self.chunks())
        current = self.index.search(self.task, "alpha", self.chunks())
        task, allowed = FakeClient.filters[-1]
        self.assertEqual(task, self.task)
        self.assertEqual(allowed, {
            str(uuid5(NAMESPACE_URL, self.task + "/" + item["evidence_id"]))
            for item in self.chunks()
        })
        self.assertTrue(all(item["content"] in {
            "alpha lexical", "semantic promote"} for item in current))
        self.assertEqual(len(FakeClient.stores[str(self.index.path)]
                             [self.index._collection()]), 4)

    def test_model_or_revision_changes_collection(self):
        self.index.search(self.task, "alpha", self.chunks())
        original = self.index._collection()
        for key, value in (("model_name", "another-model"), ("revision", "revision-2")):
            changed = HybridIndex(
                self.root / "index", embedder=self.embedder,
                reranker=RerankerFixture(), client_factory=FakeClient)
            setattr(changed, key, value)
            changed.search(self.task, "alpha", self.chunks())
            self.assertNotEqual(original, changed._collection())
        self.assertEqual(len(self.embedder.batches), 3)

    def test_invalid_embeddings_are_rejected_before_upsert(self):
        for values in ([[1.]], [[math.nan, 1.]], [[0., 0.]], []):
            with self.subTest(values=values), patch.object(
                self.embedder, "passage_embed", return_value=iter(values)):
                with self.assertRaises(ValueError):
                    self.index.search(self.task, "alpha", self.chunks()[:1])

    def test_invalid_reranker_scores_are_rejected(self):
        for values in ([math.nan, 0.], [1.]):
            with self.subTest(values=values), patch.object(
                self.index._reranker, "rerank", return_value=iter(values)):
                with self.assertRaises(ValueError):
                    self.index.search(self.task, "alpha", self.chunks())

    def test_empty_inputs_do_not_load_models_and_duplicate_ids_fail(self):
        with patch.object(self.index, "_load", side_effect=AssertionError("loaded")):
            self.assertEqual(self.index.search(self.task, "alpha", []), [])
        chunks = self.chunks()
        with self.assertRaises(ValueError):
            self.index.search(self.task, "alpha", [chunks[0], chunks[0]])


if __name__ == "__main__":
    unittest.main()
