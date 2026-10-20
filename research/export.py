import argparse
import hashlib
import json
import os
import sqlite3
import tempfile
import zipfile
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

from research.quality import evaluate

TERMINAL = {"completed", "partial", "cancelled", "failed"}


class ExportRejected(ValueError):
    pass


def load_snapshot(db_path, task_id):
    """所有读取使用同一个只读事务；不导入模型或创建业务表。"""
    path = Path(db_path).resolve(strict=True)
    with closing(sqlite3.connect(
        path.as_uri() + "?mode=ro", uri=True, timeout=5
    )) as conn:
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA query_only=ON")
        conn.execute("BEGIN")
        row = conn.execute(
            "SELECT id, question, status, queries, report, errors, created_at "
            "FROM research_tasks WHERE id=?", (task_id,),
        ).fetchone()
        if row is None:
            raise ValueError("研究任务不存在")
        task = dict(row)
        for key in ("queries", "errors"):
            task[key] = json.loads(task[key])

        reviews = []
        for row in conn.execute(
            "SELECT round_no, evidence_ids, draft, review, created_at "
            "FROM research_reviews WHERE task_id=? ORDER BY round_no",
            (task_id,),
        ):
            item = dict(row)
            for key in ("evidence_ids", "draft", "review"):
                item[key] = json.loads(item[key])
            reviews.append(item)

        chunks = []
        for row in conn.execute(
            "SELECT 'D'||d.id||'C'||c.chunk_index AS evidence_id, "
            "d.id AS document_id, c.chunk_index, d.title, d.url, "
            "d.source_type, d.metadata, d.retrieved_at, "
            "c.start_char AS start, c.end_char AS end, c.content "
            "FROM research_documents d JOIN research_chunks c "
            "ON c.document_id=d.id WHERE d.task_id=? "
            "ORDER BY d.id, c.chunk_index", (task_id,),
        ):
            item = dict(row)
            item["metadata"] = json.loads(item["metadata"])
            chunks.append(item)
    return task, reviews, chunks


def json_bytes(value):
    return (json.dumps(
        value, ensure_ascii=False, indent=2, allow_nan=False
    ) + "\n").encode("utf-8")


def publish_zip(output, files):
    """同目录暂存后排他发布，已有文件或目录绝不覆盖。"""
    requested = Path(output).expanduser()
    if requested.suffix.lower() != ".zip":
        raise ValueError("输出文件名必须以 .zip 结尾")
    parent = requested.parent.resolve(strict=True)
    target = parent / requested.name
    if os.path.lexists(target):
        raise FileExistsError(f"目标已存在，请使用新文件名：{target}")

    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=parent, prefix=".research-export-", suffix=".tmp", delete=False
        ) as raw:
            temporary = Path(raw.name)
            with zipfile.ZipFile(raw, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                for name, content in files.items():
                    archive.writestr(name, content)
            raw.flush()
            os.fsync(raw.fileno())
        # 不使用 os.replace；它会覆盖已有目标。
        if os.name == "nt":
            os.rename(temporary, target)
        else:
            os.link(temporary, target)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return target


def export_task(db_path, task_id, output, *, snapshot=False):
    task, reviews, chunks = load_snapshot(db_path, task_id)
    if task["status"] not in TERMINAL:
        raise ExportRejected("任务仍在规划、运行或等待审批，请先完成或取消任务")

    latest = reviews[-1] if reviews else None
    quality = evaluate(task, latest, chunks)
    if not snapshot and not quality["accepted"]:
        failed = ", ".join(
            item["code"] for item in quality["checks"] if not item["passed"]
        )
        raise ExportRejected(
            f"正式导出要求验收通过；未通过：{failed}。"
            "如需保留未验收资料，请显式使用 --snapshot"
        )

    kind = "snapshot" if snapshot else "accepted_report"
    heading = (
        "# 研究快照：不作为正式验收报告"
        if snapshot else "# 已通过规则验收的研究报告包"
    )
    report = "\n\n".join([
        heading,
        f"任务：{task_id}",
        f"任务状态：{task['status']}",
        f"规则验收：{'通过' if quality['accepted'] else '未通过'}",
        "仅检查业务数据库快照，未确认 checkpoint 已终止；"
        "规则通过不证明结论得到语义支持。",
        "---",
        task["report"] or "该任务没有保存报告。",
    ]) + "\n"
    evidence = {
        "scope": "all_task_chunks",
        "note": "包含本任务全部已保存分块，不代表所有分块均被正文引用。"
                "reviews.created_at 是该轮记录首次创建时间。",
        "task": task,
        "reviews": reviews,
        "chunks": chunks,
    }
    files = {
        "report.md": report.encode("utf-8"),
        "evidence.json": json_bytes(evidence),
        "quality.json": json_bytes(quality),
    }
    manifest = {
        "format_version": "18.0",
        "artifact_kind": kind,
        "task_id": task_id,
        "task_status": task["status"],
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "quality_accepted": quality["accepted"],
        "files": {
            name: {
                "bytes": len(content),
                "sha256": hashlib.sha256(content).hexdigest(),
            }
            for name, content in files.items()
        },
        "hash_scope": "三个数据文件的原始字节；不含 manifest 自身，不是数字签名。",
    }
    files["manifest.json"] = json_bytes(manifest)
    target = publish_zip(output, files)
    return {
        "path": str(target),
        "task_id": task_id,
        "artifact_kind": kind,
        "quality_accepted": quality["accepted"],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="V18 离线研究报告导出")
    parser.add_argument("--db", required=True, help="业务 SQLite 数据库")
    parser.add_argument("--task", required=True, help="研究任务 ID")
    parser.add_argument("--output", required=True, help="新的 ZIP 文件完整路径")
    parser.add_argument(
        "--snapshot", action="store_true", help="显式导出带状态标记的终态研究快照"
    )
    args = parser.parse_args(argv)
    try:
        result = export_task(
            args.db, args.task, args.output, snapshot=args.snapshot
        )
    except ExportRejected as error:
        print(json.dumps({"error": str(error)}, ensure_ascii=False, indent=2))
        return 1
    except (OSError, sqlite3.Error, ValueError, TypeError, KeyError,
            NotImplementedError) as error:
        print(json.dumps({
            "error": str(error),
            "hint": "目标目录须存在；请选择新 ZIP 文件名；非 Windows 文件系统须支持硬链接。",
        }, ensure_ascii=False, indent=2))
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())