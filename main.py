import json
import logging
import sqlite3
from contextlib import ExitStack, closing
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver

from agent.agent import MiniAgent
from llm.client import MODEL
from memory.memory import DB_PATH
from research.review_store import ReviewStore
from research.workflow import WorkflowResearchAgent


def show_task(task):
    print(f"任务：{task['id']} | 状态：{task['status']}")
    print(task["report"] or "\n".join(task["errors"]) or "暂无报告")


def drive(research, store, task_id, events):
    while True:
        for event in events:
            for node in event:
                if node != "__interrupt__":
                    print(f"[Workflow] {node}")

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
    print(f"MiniAgent V12.0 | Model: {MODEL}")
    print("普通输入聊天；/research 问题 创建研究；exit 退出")
    print("/tasks | /report ID | /reviews ID | /state ID | /resume ID")

    agent = MiniAgent(session_id="session_001")
    try:
        with ExitStack() as resources:
            # 图节点可能运行在线程池，不复用普通聊天的 SQLite 连接。
            conn = resources.enter_context(closing(sqlite3.connect(
                DB_PATH, check_same_thread=False, timeout=30,
            )))
            store = ReviewStore(conn)
            checkpoint_path = str(
                Path(DB_PATH).with_name("research_checkpoints.db")
            )
            saver = resources.enter_context(
                SqliteSaver.from_conn_string(checkpoint_path)
            )
            research = WorkflowResearchAgent(store, saver)

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
                        drive(
                            research, store, task["id"],
                            research.start(task["id"]),
                        )

                    elif command == "/resume":
                        info = research.inspect(argument)
                        if not info["exists"]:
                            raise ValueError("该任务没有 V12 checkpoint")
                        if not info["next"]:
                            raise ValueError("任务已结束，不能重复执行")
                        # 已暂停审批先展示原批次；异常中断则从 checkpoint 继续。
                        events = () if info["pending"] else research.resume(argument)
                        drive(research, store, argument, events)

                    elif command == "/state":
                        print(json.dumps(
                            research.inspect(argument),
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

                    else:
                        print("\nAgent:", agent.run(text))

                except Exception:
                    logging.exception(
                        "本轮运行失败；研究任务请用 /state ID 查看 checkpoint"
                    )

    except (EOFError, KeyboardInterrupt):
        print("\n退出程序；已保存的研究 checkpoint 可在下次恢复。")
    finally:
        agent.memory.conn.close()


if __name__ == "__main__":
    main()