from agent.agent import MiniAgent
from llm.client import MODEL
import logging


def main():

    print("==============================")
    print("      MiniAgent v8.6")
    print(f"      Model: {MODEL}")
    print("==============================")
    print("输入 exit 退出程序")

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    agent = MiniAgent(
        session_id="session_001"
    )

    while True:

        user_input = input("\nYou: ")

        if user_input.lower() == "exit":
            print("Bye!")
            break

        try:

            response = agent.run(user_input)

            print(
                "\nDeepSeek:",
                response
            )

        except Exception as e:

            print(
                "\nAgent运行失败:",
                e
            )


if __name__ == "__main__":
    main()