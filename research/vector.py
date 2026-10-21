import hashlib
import io
import json
import math
import re
import zipfile
from collections import Counter
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

from research.rag import retrieve, split_text
from research.runtime import TaskFileLock


def _sections(raw, suffix):
    if suffix == ".pdf":
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(raw))
        if reader.is_encrypted:
            raise ValueError("不支持加密 PDF")
        return [(f"第 {number} 页", page.extract_text() or "", number)
                for number, page in enumerate(reader.pages, 1)]
    if suffix == ".docx":
        from docx import Document
        from docx.text.paragraph import Paragraph
        heading, lines, parts = "正文", [], []
        for block in Document(io.BytesIO(raw)).iter_inner_content():
            if isinstance(block, Paragraph):
                if re.match(r"Heading|标题", block.style.name) and block.text.strip():
                    if lines:
                        parts.append((heading, "\n".join(lines), None))
                    heading, lines = block.text.strip(), []
                else:
                    lines.append(block.text)
            else:
                lines.extend(" | ".join(cell.text for cell in row.cells)
                             for row in block.rows)
        return parts + [(heading, "\n".join(lines), None)]
    text = raw.decode("utf-8-sig")
    heading, lines, parts = "正文", [], []
    fence = None
    for line in text.splitlines():
        marker = re.match(r"^\s{0,3}(`{3,}|~{3,})", line) if suffix == ".md" else None
        if marker:
            value = marker.group(1)
            if fence is None:
                fence = value
            elif value[0] == fence[0] and len(value) >= len(fence):
                fence = None
            lines.append(line)
            continue
        match = re.match(r"^#{1,6}\s+(.+)", line) if suffix == ".md" and fence is None else None
        if match:
            if lines:
                parts.append((heading, "\n".join(lines), None))
            heading, lines = match.group(1).strip(), []
        else:
            lines.append(line)
    return parts + [(heading, "\n".join(lines), None)]


def import_file(store, task_id, filename):
    """调用方应持任务锁；只向尚未启动的 planning/planned 任务添加不可变快照。"""
    if store.get_task(task_id)["status"] not in {"planning", "planned"}:
        raise ValueError("本地文档只能导入尚未启动的 planning/planned 任务")
    path = Path(filename).expanduser().resolve(strict=True)
    if path.suffix.lower() not in {".txt", ".md", ".pdf", ".docx"}:
        raise ValueError("仅支持 UTF-8 TXT/Markdown、文本 PDF 和 DOCX")
    with path.open("rb") as source:
        raw = source.read(10_000_001)
    if len(raw) > 10_000_000:
        raise ValueError("单文件不能超过 10 MB")
    if path.suffix.lower() == ".docx":
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            if sum(item.file_size for item in archive.infolist()) > 30_000_000:
                raise ValueError("DOCX 解压内容超过 30 MB")
    sections = [(heading, text.strip(), page)
                for heading, text, page in _sections(raw, path.suffix.lower())
                if text.strip()]
    if not sections or sum(len(item[1]) for item in sections) > 200_000:
        raise ValueError("无可提取文本，或提取文本超过 20 万字符；扫描 PDF 需先 OCR")
    digest = hashlib.sha256(raw).hexdigest()
    documents = []
    with store.conn:
        for number, (heading, text, page) in enumerate(sections, 1):
            url = f"{path.as_uri()}?sha256={digest}#part={number}"
            metadata = {
                "file_path": str(path), "file_name": path.name,
                "section": heading, "page": page, "sha256": digest,
                "truncated": False,
            }
            cursor = store.conn.execute(
                "INSERT INTO research_documents "
                "(task_id,title,url,content,source_type,metadata,retrieved_at) "
                "VALUES (?,?,?,?,?,?,?) "
                "ON CONFLICT(task_id,url,source_type) DO NOTHING",
                (task_id, (path.name + " / " + heading)[:200], url, text,
                 "local_document", json.dumps(metadata, ensure_ascii=False),
                 datetime.now(timezone.utc).isoformat()),
            )
            document_id = store.conn.execute(
                "SELECT id FROM research_documents "
                "WHERE task_id=? AND url=? AND source_type='local_document'",
                (task_id, url),
            ).fetchone()[0]
            if cursor.rowcount == 1:
                store.conn.executemany(
                    "INSERT INTO research_chunks VALUES (?,?,?,?,?)",
                    [(document_id, chunk["index"], chunk["start"],
                      chunk["end"], chunk["content"])
                     for chunk in split_text(text, size=400, overlap=80)],
                )
            documents.append(document_id)
    return documents


class HybridIndex:
    """SQLite 为证据真源；Qdrant 是可重建的任务隔离向量索引。"""

    def __init__(self, path, *, model_name="BAAI/bge-small-zh-v1.5",
                 reranker_name="BAAI/bge-reranker-v2-m3-int8",
                 cache_dir=None, revision="v19-embedding-1",
                 embedder=None, reranker=None, client_factory=None):
        self.path = Path(path).resolve()
        self.cache_dir = str(cache_dir or self.path.parent / "models")
        self.model_name, self.reranker_name = model_name, reranker_name
        self.revision = revision
        self._embedder, self._reranker = embedder, reranker
        self.client_factory = client_factory

    @property
    def config(self):
        return {"embedding": self.model_name, "reranker": self.reranker_name,
                "revision": self.revision, "fusion": "rrf60-top24"}

    def _load(self):
        if self._embedder is None:
            from fastembed import TextEmbedding
            self._embedder = TextEmbedding(
                model_name=self.model_name, cache_dir=self.cache_dir, threads=2)
        if self._reranker is None:
            from fastembed.rerank.cross_encoder import TextCrossEncoder
            self._reranker = TextCrossEncoder(
                model_name=self.reranker_name, cache_dir=self.cache_dir, threads=2)

    @staticmethod
    def _text(chunk):
        return chunk["title"] + "\n" + chunk["content"]

    def _collection(self):
        value = json.dumps(
            [self.model_name, self._embedder.embedding_size, self.revision])
        return "research_" + hashlib.sha256(value.encode()).hexdigest()[:24]

    def _vectors(self, texts, *, query=False):
        if not texts:
            return []
        method = self._embedder.query_embed if query else self._embedder.passage_embed
        values = [[float(value) for value in vector]
                  for vector in method(texts, batch_size=32)]
        if len(values) != len(texts) or any(
            len(vector) != self._embedder.embedding_size
            or not all(math.isfinite(value) for value in vector)
            or not any(vector) for vector in values
        ):
            raise ValueError("Embedding 返回数量、维度或数值异常")
        return values

    def _dense(self, task_id, query, chunks):
        from qdrant_client import QdrantClient, models
        self.path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = self.path.parent / (self.path.name + ".db")
        factory = self.client_factory or QdrantClient
        with TaskFileLock(lock_path, "qdrant-global"):
            with closing(factory(path=str(self.path))) as client:
                name = self._collection()
                if not client.collection_exists(name):
                    client.create_collection(
                        name, vectors_config=models.VectorParams(
                            size=self._embedder.embedding_size,
                            distance=models.Distance.COSINE))
                ids = {
                    str(uuid5(NAMESPACE_URL, task_id + "/" + chunk["evidence_id"])): chunk
                    for chunk in chunks
                }
                for offset in range(0, len(ids), 64):
                    batch = list(ids)[offset:offset + 64]
                    existing = {
                        str(point.id): point.payload or {}
                        for point in client.retrieve(
                            name, ids=batch, with_payload=True, with_vectors=False)
                    }
                    pending = [
                        key for key in batch if existing.get(key, {}).get("sha256")
                        != hashlib.sha256(self._text(ids[key]).encode()).hexdigest()
                    ]
                    vectors = self._vectors([self._text(ids[key]) for key in pending])
                    if pending:
                        client.upsert(name, points=[
                            models.PointStruct(id=key, vector=vector, payload={
                                "task_id": task_id,
                                "sha256": hashlib.sha256(
                                    self._text(ids[key]).encode()).hexdigest(),
                            })
                            for key, vector in zip(pending, vectors)
                        ], wait=True)
                points = client.query_points(
                    name, query=self._vectors([query], query=True)[0], limit=24,
                    query_filter=models.Filter(must=[
                        models.FieldCondition(
                            key="task_id", match=models.MatchValue(value=task_id)),
                        models.HasIdCondition(has_id=list(ids)),
                    ]), with_payload=False, with_vectors=False,
                ).points
                return [ids[str(point.id)] for point in points if str(point.id) in ids]

    def search(self, task_id, query, chunks, top_k=6):
        if not isinstance(task_id, str) or not task_id:
            raise ValueError("task_id 无效")
        if type(top_k) is not int or not 1 <= top_k <= 12:
            raise ValueError("top_k 必须为 1 到 12")
        if not chunks or not query.strip():
            return []
        if len(chunks) > 10_000:
            raise ValueError("单任务超过 10000 分块，请使用 Qdrant 服务端")
        if len({item["evidence_id"] for item in chunks}) != len(chunks):
            raise ValueError("分块编号重复")
        self._load()
        dense = self._dense(task_id, query, chunks)
        keyword = retrieve(query, chunks, top_k=24)
        scores, available = Counter(), {item["evidence_id"]: item for item in chunks}
        for ranking in (keyword, dense):
            for rank, item in enumerate(ranking, 1):
                scores[item["evidence_id"]] += 1 / (60 + rank)
        candidates = sorted(scores, key=lambda key: (-scores[key], key))[:24]
        if not candidates:
            return []
        rerank = [float(score) for score in self._reranker.rerank(
            query, [self._text(available[key]) for key in candidates], batch_size=8)]
        if len(rerank) != len(candidates) or not all(map(math.isfinite, rerank)):
            raise ValueError("Reranker 返回数量或数值异常")
        ranked = sorted(zip(candidates, rerank),
                        key=lambda item: (-item[1], -scores[item[0]], item[0]))
        selected, per_document = [], Counter()
        for key, score in ranked:
            chunk = available[key]
            if per_document[chunk["document_id"]] < 2:
                selected.append({
                    **chunk, "rrf_score": scores[key], "rerank_score": score,
                })
                per_document[chunk["document_id"]] += 1
            if len(selected) == top_k:
                break
        return selected
