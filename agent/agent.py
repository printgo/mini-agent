import json

from llm.client import chat
from tools.registry import execute_tool


class MiniAgent:

    def __init__(self):

        self.messages = [
            {
                "role": "system",
                "content": (
                    "你是一个AI Research Agent。"
                    "你可以自主决定是否调用工具。"
                    "对于最新信息、实时信息或外部信息，"
                    "优先使用 search_web 搜索。"
                    "搜索结果只是线索和摘要。"
                    "当需要核实信息、深入理解内容或引用具体来源时，"
                    "应使用 read_webpage 阅读重要网页的正文。"
                    "优先选择官方网站、官方文档、论文、"
                    "权威机构等高质量一手来源。"
                    "如果工具失败，可以调整策略后重试，"
                    "但不要无意义地重复相同调用。"
                )
            }
        ]

    def run(self, user_input: str) -> str:

        self.messages.append({
            "role": "user",
            "content": user_input
        })

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

            return final_answer

        raise RuntimeError(
            f"Agent超过最大执行步数: {max_steps}"
        )