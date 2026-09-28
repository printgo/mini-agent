from agent.agent import MiniAgent


def main():

    print("==============================")
    print("      MiniAgent v7")
    print("      Model: DeepSeek")
    print("==============================")
    print("输入 exit 退出程序")

    agent = MiniAgent()

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