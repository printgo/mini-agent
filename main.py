import logging

from agent.agent import MiniAgent
from llm.client import MODEL
from research.documents import DocumentStore
from research.multisource import MultiSourceResearchAgent


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    print(f"MiniAgent V10.0 | Model: {MODEL}")
    print("普通输入用于聊天；/research 问题 用于研究")
    print("/tasks 查看任务；/report ID 查看报告；/evidence ID 查看证据")

    agent = MiniAgent(session_id="session_001")
    store = DocumentStore(agent.memory.conn)
    research = MultiSourceResearchAgent(store)

    try:
        while True:
            text = input("\nYou: ").strip()
            if text.lower() == "exit":
                break
            if not text:
                continue

            try:
                if text.startswith("/research "):
                    task = research.prepare(text[len("/research "):])
                    print(f"任务：{task['id']} | 状态：{task['status']}")

                    if task["status"] != "planned":
                        print(task["errors"])
                        continue

                    for number, query in enumerate(task["queries"], 1):
                        print(
                            f"{number}. "
                            f"[{query['source']}] {query['query']}"
                        )

                    print(
                        "最多 3 项检索，每项最多 3 篇文档；"
                        "正文/README 会额外发起读取请求。"
                        "arXiv 网络错误最多重试 1 次；"
                        "计划中的每类来源均须有可用片段。"
                    )
                    approved = input(
                        "输入 yes 批准执行，其他输入取消："
                    ).strip()

                    task = research.run(
                        task["id"],
                        approved=approved.lower() == "yes",
                    )
                    print(f"任务状态：{task['status']}")
                    print(
                        task["report"]
                        or "\n".join(task["errors"])
                        or "已取消"
                    )

                elif text == "/tasks":
                    for task in store.list_tasks():
                        print(
                            f"{task['id']} | "
                            f"{task['status']} | "
                            f"{task['question']}"
                        )

                elif text.startswith("/report "):
                    task = store.get_task(text.split(maxsplit=1)[1])
                    print(f"任务状态：{task['status']}")
                    print(task["report"] or task["errors"] or "暂无报告")
                
                elif text.startswith("/documents "):
                    for item in store.list_documents(
                        text.split(maxsplit=1)[1]
                    ):
                        print(
                            f"D{item['id']} | "
                            f"{item['source_type']} | {item['title']}\n"
                            f"{item['url']}\n"
                            f"{item['metadata']}\n"
                            f"获取时间：{item['retrieved_at']}"
                        )

                elif text.startswith("/chunks "):
                    for item in store.list_chunks(
                        text.split(maxsplit=1)[1]
                    ):
                        print(
                            f"{item['evidence_id']} | 字符范围 "
                            f"{item['start']}:{item['end']}\n"
                            f"{item['content'][:160]}\n"
                        )

                elif text.startswith("/chunk "):
                    _, task_id, chunk_id = text.split(maxsplit=2)

                    item = next(
                        (
                            item
                            for item in store.list_chunks(task_id)
                            if item["evidence_id"] == chunk_id
                        ),
                        None,
                    )
                    if item is None:
                        raise ValueError("该任务中不存在这个片段")

                    print(item["url"])
                    print(item["content"])

                elif text.startswith("/evidence "):
                    task_id = text.split(maxsplit=1)[1]
                    for item in store.list_evidence(task_id):
                        print(
                            f"{item['evidence_id']} | {item['title']}\n"
                            f"{item['url']}\n"
                            f"来源类型：{item['source_type']}\n"
                            f"获取时间：{item['retrieved_at']}\n"
                            f"{item['content']}\n"
                        )

                else:
                    print("\nAgent:", agent.run(text))

            except Exception:
                logging.exception("本轮运行失败")

    except (EOFError, KeyboardInterrupt):
        print("\n退出程序")
    finally:
        agent.memory.conn.close()


if __name__ == "__main__":
    main()