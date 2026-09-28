import json

from llm.client import chat
from tools.registry import execute_tool
from memory.memory import Memory


class MiniAgent:

    def __init__(self):

        self.memory = Memory()

        self.messages = [
            {
                "role": "system",
                "content": (
                    "你是一个AI Research Agent。"
                    "你可以自主决定是否调用工具。"
                    "对于最新信息、实时信息或外部信息，"
                    "优先使用 search_web 搜索。"
                    "当需要核实信息或深入理解网页内容时，"
                    "使用 read_webpage。"
                )
            }
        ]

        # 从SQLite恢复历史聊天
        history = self.memory.get_messages()

        self.messages.extend(history)

    def run(self, user_input: str) -> str:

        self.messages.append({
            "role": "user",
            "content": user_input
        })

        self.memory.add_message(
            "user",
            user_input
        )

        max_steps = 10

        for step in range(1, max_steps + 1):

            print(f"\n[Agent] Step {step}")

            response = chat(self.messages)

            # ========================
            # 需要调用工具
            # ========================

            if response.tool_calls:

                self.messages.append(response)

                for tool_call in response.tool_calls:

                    tool_name = tool_call.function.name

                    try:
                        arguments = json.loads(
                            tool_call.function.arguments
                        )

                    except json.JSONDecodeError as e:
                        arguments = {}

                        result = {
                            "success": False,
                            "error": f"工具参数解析失败: {str(e)}"
                        }

                    else:

                        print(
                            f"[Agent] 调用工具: {tool_name}"
                        )

                        print(
                            f"[Agent] 参数: {arguments}"
                        )

                        try:

                            result = execute_tool(
                                tool_name,
                                arguments
                            )

                        except Exception as e:

                            result = {
                                "success": False,
                                "error": str(e)
                            }

                    print(
                        f"[Tool] 返回结果: {result}"
                    )

                    # 无论成功还是失败
                    # 都必须回应这个 tool_call_id
                    self.messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": str(result)
                    })

                continue

            # ========================
            # 最终答案
            # ========================

            final_answer = response.content

            self.messages.append({
                "role": "assistant",
                "content": final_answer
            })

            self.memory.add_message(
                "assistant",
                final_answer
            )

            return final_answer

        raise RuntimeError(
            f"Agent超过最大执行步数: {max_steps}"
        )