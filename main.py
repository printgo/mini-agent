import json
import logging
import sqlite3
from contextlib import ExitStack, closing
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver

from agent.agent import ResearchAgent
from llm.client import MODEL
from memory.memory import DB_PATH
from research.generation import GenerationStore, GenerationWorkflowAgent
from research.recovery import RecoveryWorkflowAgent
from research.workflow import WorkflowResearchAgent
from research.runtime import TaskController, TaskControlError, TaskStateError
from research.management import BudgetLedger, TaskMaintenance
from research.limited import LimitedWorkflowAgent


class ManagedWorkflowAgent(GenerationWorkflowAgent):
    VERSION = "15.0"


def show_task(task):
    print(f"任务：{task['id']} | 状态：{task['status']}")
    print(task["report"] or "\n".join(task["errors"]) or "暂无报告")


def drive(research, store, task_id, events):
    while True:
        try:
            for event in events:
                for node in event:
                    if node != "__interrupt__":
                        print(f"[Workflow] {node}")
        finally:
            close = getattr(events, "close", None)
            if close is not None:
                close()

        info = research.inspect(task_id)
        if not info["pending"]:
            show_task(store.get_task(task_id))
            return

        pending = info["pending"][0]
        print("\n[Human]", "初始检索审批" if pending["kind"] == "initial"
              else "补充检索审批")
        print("任务：", task_id)
        print("问题：", pending["question"])
        for number, query in enumerate(pending["queries"], 1):
            print(f"{number}. [{query['source']}] {query['query']}")

        print(
            f"已完成 {pending['round_no']}/{pending['max_rounds']} 轮；"
            f"已使用 {pending['used_supplements']}/"
            f"{pending['max_supplements']} 项补查。"
        )
        print("每项最多 3 篇文档；正文请求和内部重试另计。")
        answer = input(
            "输入 yes 批准，no 拒绝，其他输入稍后处理："
        ).strip().lower()
        if answer not in {"yes", "no"}:
            print(f"任务已暂停；使用 /resume {task_id} 继续。")
            return
        events = research.resume(task_id, approved=answer == "yes")


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    print(f"research-agent V16.0 | Model: {MODEL}")
    print("普通输入聊天；/research 问题 创建研究；exit 退出")
    print("/tasks | /report ID | /reviews ID | /state ID | /resume ID")
    print("/retry ID | /cancel ID | /failures ID")
    print("/usage ID | /limits ID 模型上限 查询上限")
    print("/cache ID | /cache-clear ID [天数] | /cache-drop ID")
    print("/collections ID | /generations ID | /documents ID | /chunks ID")
    print("/chunk ID 片段编号 | /evidence ID")
    print("已有 V12～V15 checkpoint 按原版本恢复，新任务使用 V16。")
    print("V16 默认写作/审查调用 12 次、采集查询 8 次；规划和内部请求另计。")

    agent = ResearchAgent(session_id="session_001")
    try:
        with ExitStack() as resources:
            conn = resources.enter_context(closing(sqlite3.connect(
                DB_PATH, check_same_thread=False, timeout=30,
            )))
            store = GenerationStore(conn)
            ledger = BudgetLedger(conn)
            checkpoint_path = str(
                Path(DB_PATH).with_name("research_checkpoints.db")
            )
            saver = resources.enter_context(
                SqliteSaver.from_conn_string(checkpoint_path)
            )
            research = LimitedWorkflowAgent(store, saver, ledger)
            versions = {
                WorkflowResearchAgent.VERSION: WorkflowResearchAgent(store, saver),
                RecoveryWorkflowAgent.VERSION: RecoveryWorkflowAgent(store, saver),
                GenerationWorkflowAgent.VERSION: GenerationWorkflowAgent(store, saver),
                ManagedWorkflowAgent.VERSION: ManagedWorkflowAgent(store, saver),
                research.VERSION: research,
            }
            controller = TaskController(store, research, versions, DB_PATH)
            maintenance = TaskMaintenance(controller, ledger)

            while True:
                text = input("\nYou: ").strip()
                if text.lower() == "exit":
                    break
                if not text:
                    continue

                try:
                    command, _, argument = text.partition(" ")
                    argument = argument.strip()

                    if command == "/research":
                        task = research.prepare(argument)
                        print(f"任务：{task['id']} | 状态：{task['status']}")
                        if task["status"] != "planned":
                            show_task(task)
                            continue
                        with controller.session(task["id"]) as selected:
                            drive(
                                selected, store, task["id"],
                                selected.start(task["id"]),
                            )

                    elif command in {"/resume", "/retry"}:
                        with controller.session(argument) as selected:
                            info = selected.inspect(argument)
                            if not info["exists"]:
                                task = store.get_task(argument)
                                if task["status"] != "planned":
                                    raise TaskStateError(
                                        "任务没有 checkpoint，且不是 planned 状态"
                                    )
                                events = selected.start(argument)
                            elif not info["next"]:
                                raise TaskStateError("任务已结束，不能重复执行")
                            else:
                                events = (
                                    () if info["pending"]
                                    else selected.resume(argument)
                                )
                            drive(selected, store, argument, events)

                    elif command == "/state":
                        print(json.dumps(
                            controller.inspect(argument),
                            ensure_ascii=False, indent=2,
                        ))

                    elif command == "/cancel":
                        task = controller.cancel(argument)
                        print(f"任务：{task['id']} | 状态：{task['status']}")

                    elif command == "/failures":
                        print(json.dumps(
                            controller.list_failures(argument),
                            ensure_ascii=False, indent=2,
                        ))

                    elif command == "/usage":
                        print(json.dumps(
                            maintenance.usage(argument),
                            ensure_ascii=False, indent=2,
                        ))

                    elif command == "/limits":
                        parts = argument.split()
                        if len(parts) != 3:
                            raise TaskStateError("用法：/limits ID 模型上限 查询上限")
                        try:
                            model_limit, query_limit = map(int, parts[1:])
                        except ValueError:
                            raise TaskStateError("调用上限必须为整数") from None
                        print(json.dumps(
                            maintenance.set_limits(
                                parts[0], model_limit, query_limit
                            ),
                            ensure_ascii=False, indent=2,
                        ))

                    elif command == "/cache":
                        print(json.dumps(
                            maintenance.cache_stats(argument),
                            ensure_ascii=False, indent=2,
                        ))

                    elif command == "/cache-clear":
                        parts = argument.split()
                        if len(parts) not in {1, 2}:
                            raise TaskStateError("用法：/cache-clear ID [天数]")
                        try:
                            days = int(parts[1]) if len(parts) == 2 else 0
                        except ValueError:
                            raise TaskStateError("清理天数必须为整数") from None
                        print(json.dumps(
                            maintenance.purge_cache(parts[0], days),
                            ensure_ascii=False, indent=2,
                        ))

                    elif command == "/cache-drop":
                        print(json.dumps(
                            maintenance.invalidate_current_generation(argument),
                            ensure_ascii=False, indent=2,
                        ))

                    elif command == "/collections":
                        print(json.dumps(
                            store.list_collections(argument),
                            ensure_ascii=False, indent=2,
                        ))

                    elif command == "/generations":
                        print(json.dumps(
                            store.list_generations(argument),
                            ensure_ascii=False, indent=2,
                        ))

                    elif command == "/tasks":
                        for task in store.list_tasks():
                            print(
                                f"{task['id']} | {task['status']} | "
                                f"{task['question']}"
                            )

                    elif command == "/report":
                        show_task(store.get_task(argument))

                    elif command == "/reviews":
                        print(json.dumps(
                            store.list_reviews(argument),
                            ensure_ascii=False, indent=2,
                        ))

                    elif command == "/documents":
                        for item in store.list_documents(argument):
                            print(
                                f"D{item['id']} | {item['source_type']} | "
                                f"{item['title']}\n{item['url']}\n"
                                f"{item['metadata']}\n"
                                f"获取时间：{item['retrieved_at']}"
                            )

                    elif command == "/chunks":
                        for item in store.list_chunks(argument):
                            print(
                                f"{item['evidence_id']} | 字符范围 "
                                f"{item['start']}:{item['end']}\n"
                                f"{item['content'][:160]}\n"
                            )

                    elif command == "/chunk":
                        task_id, chunk_id = argument.split(maxsplit=1)
                        item = next((
                            item for item in store.list_chunks(task_id)
                            if item["evidence_id"] == chunk_id
                        ), None)
                        if item is None:
                            raise ValueError("该任务中不存在这个片段")
                        print(item["url"])
                        print(item["content"])

                    elif command == "/evidence":
                        for item in store.list_evidence(argument):
                            print(
                                f"{item['evidence_id']} | {item['title']}\n"
                                f"{item['url']}\n{item['source_type']}\n"
                                f"获取时间：{item['retrieved_at']}\n"
                                f"{item['content']}\n"
                            )

                    elif command.startswith("/"):
                        raise ValueError(f"未知命令：{command}")

                    else:
                        print("\nAgent:", agent.run(text))

                except TaskControlError as error:
                    print(f"[Task] {error}")
                except Exception:
                    logging.exception(
                        "本轮运行失败；研究任务请用 /state ID 查看，"
                        "用 /resume ID 恢复"
                    )

    except (EOFError, KeyboardInterrupt):
        print("\n退出程序；已保存的研究 checkpoint 可在下次恢复。")
    finally:
        agent.memory.conn.close()


if __name__ == "__main__":
    main()