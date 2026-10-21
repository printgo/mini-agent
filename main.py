import argparse
import json
import os
import sqlite3
from contextlib import ExitStack, closing
from pathlib import Path


def show(value):
    print(json.dumps(value, ensure_ascii=False, indent=2))


def drive(selected, store, task_id, events):
    while True:
        try:
            for event in events:
                for node in event:
                    print("[Workflow]", node)
        finally:
            close = getattr(events, "close", None)
            if close is not None:
                close()
        info = selected.inspect(task_id)
        if not info["pending"]:
            task = store.get_task(task_id)
            show({"task_id": task_id, "status": task["status"]})
            print(task["report"] or "\n".join(task["errors"]))
            return
        show(info["pending"][0])
        answer = input("输入 yes 批准，no 拒绝，其他输入暂停：").strip().lower()
        if answer not in {"yes", "no"}:
            print(f"任务已保存：使用 --resume {task_id} 继续")
            return
        events = selected.resume(task_id, approved=answer == "yes")


def main(argv=None):
    parser = argparse.ArgumentParser(description="V19 多 Agent 研究与长期记忆")
    parser.add_argument("--user", default="default", help="本地研究档案，不是登录鉴权")
    parser.add_argument("--db", default="data/memory.db")
    parser.add_argument("--index", default="data/qdrant")
    parser.add_argument("--models", default="data/models")
    parser.add_argument("--context-limit", type=int, default=16000)
    parser.add_argument("--web-port", type=int, default=8000)
    parser.add_argument("--file", action="append", default=[])
    parser.add_argument("--local-only", action="store_true")
    parser.add_argument("--evidence", nargs="+", default=[])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--question")
    mode.add_argument("--resume")
    mode.add_argument("--state")
    mode.add_argument("--cancel")
    mode.add_argument("--usage")
    mode.add_argument("--results")
    mode.add_argument("--limits", nargs=3, metavar=("ID", "MODEL", "QUERY"))
    mode.add_argument("--memory", nargs=4, metavar=("ACTION", "KIND", "KEY", "VALUE"))
    mode.add_argument("--memories", metavar="QUERY")
    mode.add_argument("--confirm", nargs=3, metavar=("ID", "KEY", "TEXT"))
    mode.add_argument("--warmup", action="store_true")
    mode.add_argument("--web", action="store_true", help="启动本地研究工作台")
    args = parser.parse_args(argv)
    if (args.file or args.local_only) and args.question is None:
        parser.error("--file/--local-only 只能用于 --question")
    try:
        os.environ.setdefault(
            "TIKTOKEN_CACHE_DIR", str(Path(args.models).resolve() / "tiktoken")
        )
        if args.web:
            if not 1 <= args.web_port <= 65535:
                raise ValueError("--web-port 必须为 1～65535")
            if args.user != "default":
                raise ValueError("网页第一步使用 default 本地档案")
            import uvicorn
            from api.app import create_app
            from research.web_service import WebSettings
            settings = WebSettings(
                db_path=Path(args.db).resolve(), index_path=Path(args.index).resolve(),
                model_cache=Path(args.models).resolve(), context_limit=args.context_limit,
            )
            uvicorn.run(create_app(settings), host="127.0.0.1", port=args.web_port)
            return 0
        if args.warmup:
            import tiktoken
            from research.vector import HybridIndex
            tiktoken.get_encoding("cl100k_base")
            HybridIndex(args.index, cache_dir=args.models)._load()
            print("Tokenizer、Embedding、Reranker 预热完成")
            return 0

        db_path = Path(args.db).resolve()
        db_path.parent.mkdir(parents=True, exist_ok=True)
        with ExitStack() as stack:
            conn = stack.enter_context(closing(sqlite3.connect(
                db_path, check_same_thread=False, timeout=30
            )))
            from research.knowledge import KnowledgeStore
            knowledge = KnowledgeStore(conn)
            if args.memory:
                action, kind, key, value = args.memory
                knowledge.apply(args.user, action, kind, key, value)
                show({"action": action, "kind": kind, "key": key})
                return 0
            if args.memories is not None:
                show(knowledge.retrieve(args.user, args.memories, limit=20))
                return 0
            if args.confirm:
                task_id, key, text = args.confirm
                from research.runtime import TaskFileLock
                with TaskFileLock(db_path, task_id):
                    knowledge.confirm(args.user, task_id, key, text, args.evidence)
                print("已保存人工确认结论；影响下一次研究，已有任务仍使用冻结记忆")
                return 0

            from langgraph.checkpoint.sqlite import SqliteSaver
            from research.generation import GenerationStore, GenerationWorkflowAgent
            from research.limited import LimitedWorkflowAgent
            from research.management import BudgetLedger, TaskMaintenance
            from research.recovery import RecoveryWorkflowAgent
            from research.runtime import TaskStateError
            from research.engine import TeamController, TeamWorkflowAgent
            from research.workflow import WorkflowResearchAgent

            class Legacy15(GenerationWorkflowAgent):
                VERSION = "15.0"

            store = GenerationStore(conn)
            ledger = BudgetLedger(conn)
            saver = stack.enter_context(SqliteSaver.from_conn_string(
                str(db_path.with_name("research_checkpoints.db"))
            ))
            research = TeamWorkflowAgent(
                store, saver, ledger, user_id=args.user,
                index_path=args.index, model_cache=args.models, db_path=db_path,
                context_limit=args.context_limit,
            )
            old = [
                WorkflowResearchAgent(store, saver),
                RecoveryWorkflowAgent(store, saver),
                GenerationWorkflowAgent(store, saver),
                Legacy15(store, saver),
                LimitedWorkflowAgent(store, saver, ledger),
            ]
            versions = {item.VERSION: item for item in [*old, research]}
            controller = TeamController(store, research, versions, db_path)
            maintenance = TaskMaintenance(controller, ledger)

            if args.question:
                task = research.prepare(
                    args.question, files=args.file, local_only=args.local_only
                )
                show({"task_id": task["id"], "status": task["status"]})
                if task["status"] != "planned":
                    show(task["errors"])
                    return 1
                with controller.session(task["id"]) as selected:
                    drive(selected, store, task["id"], selected.start(task["id"]))
                return 0

            task_id = next(
                value for value in (
                    args.resume, args.state, args.cancel, args.usage,
                    args.results, args.limits[0] if args.limits else None,
                ) if value is not None
            )
            owner = knowledge.owner(task_id)
            if owner is not None:
                knowledge.assert_owner(args.user, task_id)
            elif args.user != "default":
                raise PermissionError("未绑定用户的旧任务只能由 default 档案访问")
            if args.state:
                show(controller.inspect(task_id))
            elif args.cancel:
                show(controller.cancel(task_id))
            elif args.usage:
                value = maintenance.usage(task_id)
                if controller._select(task_id).VERSION == "19.0":
                    value["scope"] = "规划/写作/审查模型调用与远程查询尝试；内部重试另计"
                show(value)
            elif args.results:
                show(research.researcher.list_results(task_id))
            elif args.limits:
                show(maintenance.set_limits(
                    task_id, int(args.limits[1]), int(args.limits[2])
                ))
            else:
                with controller.session(task_id) as selected:
                    info = selected.inspect(task_id)
                    if not info["exists"]:
                        if store.get_task(task_id)["status"] != "planned":
                            raise TaskStateError("没有可恢复的计划")
                        events = selected.start(task_id)
                    elif not info["next"]:
                        raise TaskStateError("任务已经结束")
                    else:
                        events = () if info["pending"] else selected.resume(task_id)
                    drive(selected, store, task_id, events)
        return 0
    except (EOFError, KeyboardInterrupt):
        print("已停止当前命令；已提交的任务状态保留")
        return 130
    except Exception as error:
        show({"error": type(error).__name__, "message": str(error)})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
