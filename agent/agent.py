import json

from llm.client import chat
from tools.registry import execute_tool
from memory.memory import Memory
from memory.extractor import extract_memories


class MiniAgent:

    def __init__(
        self,
        session_id: str
    ):

        # 当前会话ID
        self.session_id = session_id

        # Memory数据库
        self.memory = Memory()

        # Agent固定规则
        self.system_prompt = (
            "你是一个AI Research Agent。"
            "你可以自主决定是否调用工具。"
            "对于最新信息、实时信息或外部信息，"
            "优先使用 search_web 搜索。"
            "当需要核实信息或深入理解网页内容时，"
            "使用 read_webpage。"
        )

        # 只恢复当前Session的聊天历史
        self.messages = self.memory.get_messages(
            self.session_id
        )

    # ==================================
    # 构建发送给LLM的Context
    # ==================================

    def build_context(self):

        # 每次调用LLM之前
        # 动态读取最新的长期记忆
        memories = self.memory.get_memories()

        memory_text = "\n".join(
            [
                f"{key}: {value}"
                for key, value
                in memories.items()
            ]
        )

        if not memory_text:
            memory_text = "暂无长期记忆"

        system_content = (
            self.system_prompt
            + "\n\n"
            + "以下是关于用户的长期记忆：\n"
            + memory_text
            + "\n\n"
            + "你可以在回答与用户背景相关的问题时，"
            + "使用这些长期记忆。"
        )

        # 临时构建Context
        context = [
            {
                "role": "system",
                "content": system_content
            },
            *self.messages
        ]

        return context

    # ==================================
    # Agent运行入口
    # ==================================

    def run(
        self,
        user_input: str
    ) -> str:

        # ==============================
        # 1. 提取长期记忆
        # ==============================

        extracted_memories = (
            extract_memories(user_input)
        )

        for item in extracted_memories:

            key = item.get("key")
            value = item.get("value")

            if key and value:

                self.memory.save_memory(
                    key,
                    value
                )

                print(
                    f"[Memory] 保存长期记忆: "
                    f"{key} = {value}"
                )

        # ==============================
        # 2. 保存用户消息到Session
        # ==============================

        self.messages.append({
            "role": "user",
            "content": user_input
        })

        self.memory.add_message(
            self.session_id,
            "user",
            user_input
        )

        # ==============================
        # 3. Agent Loop
        # ==============================

        max_steps = 10

        for step in range(
            1,
            max_steps + 1
        ):

            print(
                f"\n[Agent] Step {step}"
            )

            # 每一步都重新构建Context
            context = self.build_context()

            response = chat(
                context
            )

            # ==========================
            # 需要调用工具
            # ==========================

            if response.tool_calls:

                self.messages.append(
                    response
                )

                for tool_call in response.tool_calls:

                    tool_name = (
                        tool_call
                        .function
                        .name
                    )

                    try:

                        arguments = json.loads(
                            tool_call
                            .function
                            .arguments
                        )

                    except json.JSONDecodeError as e:

                        arguments = {}

                        result = {
                            "success": False,
                            "error": (
                                "工具参数解析失败: "
                                + str(e)
                            )
                        }

                    else:

                        print(
                            f"[Agent] 调用工具: "
                            f"{tool_name}"
                        )

                        print(
                            f"[Agent] 参数: "
                            f"{arguments}"
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
                        f"[Tool] 返回结果: "
                        f"{result}"
                    )

                    # Tool Calling协议要求：
                    # 每个tool_call_id
                    # 必须对应一个tool消息
                    self.messages.append({
                        "role": "tool",
                        "tool_call_id": (
                            tool_call.id
                        ),
                        "content": str(result)
                    })

                continue

            # ==========================
            # 最终答案
            # ==========================

            final_answer = (
                response.content
            )

            self.messages.append({
                "role": "assistant",
                "content": final_answer
            })

            self.memory.add_message(
                self.session_id,
                "assistant",
                final_answer
            )

            return final_answer

        raise RuntimeError(
            f"Agent超过最大执行步数: "
            f"{max_steps}"
        )