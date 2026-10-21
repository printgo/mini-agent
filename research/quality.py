import argparse
import json
import re
import sqlite3
from contextlib import closing
from pathlib import Path

RULESET = "v17-quality-v1"
SOURCE_TYPES = {
    "web": {"web_text", "web_summary"},
    "paper": {"paper_abstract"},
    "github": {"github_readme"},
    "local": {"local_document"},
}
REPORT_SCOPE = (
    "本报告依据检索片段生成，正文和 README 均为限长快照。"
    "论文来源仅为摘要，GitHub 资料不代表已审计代码；"
    "引用编号合法不代表结论已经得到语义核验。"
    "模型 Critic 审查不能替代人工核验或实验复现。"
)


class AuditReadError(ValueError):
    pass


def _strings(value):
    return isinstance(value, list) and all(isinstance(x, str) for x in value)


def evaluate(task, audit, chunks):
    """验收业务数据库快照，不调用模型，不判断语义真实性。"""
    checks = []

    def check(code, ok):
        checks.append({"code": code, "passed": bool(ok)})

    def result(metrics=None):
        return {
            "ruleset": RULESET,
            "task_id": task["id"],
            "accepted": all(item["passed"] for item in checks),
            "checks": checks,
            "metrics": metrics or {},
            "limits": [
                "仅检查业务数据库快照，不判断 checkpoint 是否已终止。",
                "规则通过及 Critic pass 均不证明结论得到语义支持。",
            ],
        }

    check("task_completed", task["status"] == "completed")
    check("no_collection_errors", task["errors"] == [])
    check("latest_audit_exists", audit is not None)
    if audit is None:
        return result()

    draft, review = audit["draft"], audit["review"]
    check("round_valid", type(audit["round_no"]) is int and audit["round_no"] >= 1)
    check("critic_pass", isinstance(review, dict) and (
        review.get("decision") == "pass"
        and review.get("issues") == []
        and review.get("queries") == []
        and review.get("action") == "reviewed"
    ))

    sections = draft.get("sections") if isinstance(draft, dict) else None
    limitations = draft.get("limitations") if isinstance(draft, dict) else None
    valid = (
        isinstance(sections, list) and 1 <= len(sections) <= 5
        and _strings(limitations)
        and all(
            isinstance(section, dict)
            and all(
                isinstance(section.get(key), str) and section[key].strip()
                for key in ("heading", "text")
            )
            and _strings(section.get("evidence_ids"))
            and bool(section["evidence_ids"])
            for section in sections
        )
    )
    check("draft_schema", valid)
    if not valid:
        return result()

    available = {item["evidence_id"]: item for item in chunks}
    candidates = audit["evidence_ids"]
    candidate_valid = _strings(candidates) and bool(candidates)
    check("round_evidence_exists", candidate_valid and all(
        value in available for value in candidates
    ))
    candidate_set = set(candidates) if candidate_valid else set()
    used = {value for section in sections for value in section["evidence_ids"]}
    citations_valid = used <= candidate_set and used <= available.keys()
    check("citations_in_current_round", citations_valid)
    characters = sum(len(section["text"].strip()) for section in sections)
    check("body_within_800_chars", characters <= 800)
    check("no_manual_citations_or_urls", not any(
        re.search(r"https?://|www\.|\[(?:E\d+|D\d+C\d+)\]",
                  section["heading"] + "\n" + section["text"], re.I)
        for section in sections
    ))

    plan = task["queries"]
    plan_valid = isinstance(plan, list) and 1 <= len(plan) <= 3 and all(
        isinstance(entry, (str, dict))
        and (isinstance(entry, str) or (
            isinstance(entry.get("source"), str)
            and entry["source"] in SOURCE_TYPES
            and isinstance(entry.get("query"), str)
        ))
        and bool((entry if isinstance(entry, str) else entry["query"]).strip())
        for entry in plan
    )
    check("plan_schema", plan_valid)
    required = {
        "web" if isinstance(entry, str) else entry["source"] for entry in plan
    } if plan_valid else set()
    covered = {
        source for source, kinds in SOURCE_TYPES.items()
        if any(available[value]["source_type"] in kinds
               for value in used if value in available)
    }
    check("planned_sources_cited", plan_valid and required <= covered)

    if citations_valid:
        lines = [
            "# 研究报告", f"任务：{task['id']}", f"问题：{task['question']}",
        ]
        for section in sections:
            ids = dict.fromkeys(section["evidence_ids"])
            citations = " ".join(f"[{value}]" for value in ids)
            lines.extend([
                f"## {section['heading'].strip()}",
                f"{section['text'].strip()} {citations}",
            ])
        lines.extend(["## 局限", f"- {REPORT_SCOPE}"])
        lines.extend(f"- {item}" for item in limitations)
        if _strings(task["errors"]):
            lines.extend(f"- 检索缺口：{item}" for item in task["errors"])
        lines.append("## 参考来源")
        for value in candidates:
            if value in used:
                item = available[value]
                lines.append(
                    f"- [{item['evidence_id']}] {item['title']} — <{item['url']}>"
                )
        check("report_matches_latest_draft", task["report"] == "\n\n".join(lines))
    else:
        check("report_matches_latest_draft", False)

    return result({
        "round_no": audit["round_no"],
        "sections": len(sections),
        "body_characters": characters,
        "unique_citations": len(used),
        "required_sources": sorted(required),
        "cited_sources": sorted(covered),
    })


def audit_database(db_path, task_id):
    """mode=ro 防止创建数据库；一次读事务取得一致业务快照。"""
    try:
        path = Path(db_path).resolve(strict=True)
        with closing(sqlite3.connect(
            path.as_uri() + "?mode=ro", uri=True, timeout=5
        )) as conn:
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA query_only=ON")
            conn.execute("BEGIN")
            row = conn.execute(
                "SELECT id, question, status, queries, report, errors "
                "FROM research_tasks WHERE id=?", (task_id,),
            ).fetchone()
            if row is None:
                raise AuditReadError("研究任务不存在")
            task = dict(row)
            for key in ("queries", "errors"):
                task[key] = json.loads(task[key])
            row = conn.execute(
                "SELECT round_no, evidence_ids, draft, review "
                "FROM research_reviews WHERE task_id=? "
                "ORDER BY round_no DESC LIMIT 1", (task_id,),
            ).fetchone()
            audit = dict(row) if row else None
            if audit is not None:
                for key in ("evidence_ids", "draft", "review"):
                    audit[key] = json.loads(audit[key])
            chunks = [
                dict(row) for row in conn.execute(
                    "SELECT 'D'||d.id||'C'||c.chunk_index AS evidence_id, "
                    "d.title, d.url, d.source_type, c.content "
                    "FROM research_documents d JOIN research_chunks c "
                    "ON c.document_id=d.id WHERE d.task_id=? "
                    "ORDER BY d.id, c.chunk_index", (task_id,),
                )
            ]
        return evaluate(task, audit, chunks)
    except AuditReadError:
        raise
    except (OSError, sqlite3.Error, ValueError, TypeError) as error:
        raise AuditReadError(f"无法读取验收快照：{error}") from error


def main(argv=None):
    parser = argparse.ArgumentParser(description="V17 离线研究报告规则验收")
    parser.add_argument("--db", required=True, help="业务 SQLite 数据库路径")
    parser.add_argument("--task", required=True, help="研究任务 ID")
    args = parser.parse_args(argv)
    try:
        result = audit_database(args.db, args.task)
    except AuditReadError as error:
        print(json.dumps({"error": str(error)}, ensure_ascii=False, indent=2))
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
