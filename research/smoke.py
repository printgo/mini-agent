"""使用真实本地模型验证 V19 检索链路；不调用研究模型或打开业务数据库。"""
import argparse
import json
import os
import sqlite3
import tempfile
from contextlib import closing
from pathlib import Path

from research.context import TokenContext
from research.documents import DocumentStore
from research.runtime import TaskFileLock
from research.vector import HybridIndex, import_file


def main(argv=None):
    parser = argparse.ArgumentParser(description="V19 真实模型与本地索引冒烟验证")
    parser.add_argument("--models", default="data/models")
    args = parser.parse_args(argv)
    models = Path(args.models).resolve()
    os.environ.setdefault("TIKTOKEN_CACHE_DIR", str(models / "tiktoken"))
    try:
        with tempfile.TemporaryDirectory(prefix="research-v19-smoke-") as folder:
            root = Path(folder)
            database = root / "smoke.db"
            with closing(sqlite3.connect(database)) as conn:
                store = DocumentStore(conn)
                question = "研究任务中断后如何恢复，怎样避免重复采集？"
                task = store.create_task(question)
                store.update_task(task, "planned")
                with TaskFileLock(database, task):
                    for name, text in {
                        "recovery.md": "# 中断恢复\n任务通过持久化 checkpoint 恢复。"
                        "已提交的采集缓存会被复用，未完成的查询才重新执行。",
                        "memory.md": "# 长期记忆\n用户偏好存入 SQLite；"
                        "只有人工显式确认的结论才存入确认记忆。",
                        "garden.md": "# 花园\n向日葵需要阳光、水分和适合生长的土壤。",
                    }.items():
                        path = root / name
                        path.write_text(text, encoding="utf-8")
                        import_file(store, task, path)
                    chunks = store.list_chunks(task)
                    index = HybridIndex(root / "qdrant", cache_dir=models)
                    evidence = index.search(task, question, chunks)
                    if not evidence:
                        raise ValueError("真实模型没有返回证据")
                    available = {item["evidence_id"] for item in chunks}
                    if not {item["evidence_id"] for item in evidence} <= available:
                        raise ValueError("检索返回了当前任务以外的证据")
                    # 再次打开真实索引，验证持久化结果可以复用。
                    repeated = index.search(task, question, chunks)
                    if [item["evidence_id"] for item in repeated] != [
                        item["evidence_id"] for item in evidence
                    ]:
                        raise ValueError("相同输入的索引重开结果不一致")
                    context = TokenContext()
                    packed = context.pack(
                        question, evidence=evidence,
                        facts=["验证引用：" + evidence[0]["evidence_id"]],
                    )
                    print(json.dumps({
                        "ok": True,
                        "scope": "真实模型/Tokenizer/Qdrant/SQLite 联通验证，不是召回率评测",
                        "documents": len(store.list_documents(task)),
                        "chunks": len(chunks),
                        "encoding": packed["encoding"],
                        "estimated_input_tokens": packed["estimated_input_tokens"],
                        "ranking": [{
                            "evidence_id": item["evidence_id"],
                            "file_name": item["metadata"]["file_name"],
                            "section": item["metadata"]["section"],
                            "rerank_score": item["rerank_score"],
                        } for item in evidence],
                    }, ensure_ascii=False, indent=2))
        return 0
    except Exception as error:
        print(json.dumps({"ok": False, "error": type(error).__name__,
                          "message": str(error)}, ensure_ascii=False))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
