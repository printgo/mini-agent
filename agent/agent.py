import json

from llm.client import chat
from memory.extractor import extract_memories
from memory.memory import Memory
from memory.updater import decide_memory_action
from tools.registry import execute_tool


class MiniAgent:

    def __init__(
        self,
        session_id: str
    ):
        self.session_id = session_id

        self.memory = Memory()

        self.system_prompt = (
            "你是一个AI Research Agent。"
            "你可以自主决定是否调用工具。"
            "对于最新信息、实时信息或外部信息，"
            "优先使用 search_web 搜索。"
            "当需要核实信息或深入理解网页内容时，"
            "使用 read_webpage。"
        )

        # 这里只恢复当前 Session 的聊天历史。
        # 长期记忆不直接塞入 self.messages，
        # 而是在 build_context() 中动态注入。
        self.messages = (
            self.memory.get_messages(
                self.session_id
            )
        )

    def build_context(self) -> list[dict]:
        """
        构建当前发送给 LLM 的上下文。

        每次调用时动态读取长期记忆，
        保证刚刚新增、更新或删除的 Memory
        能立即生效。
        """

        memories = (
            self.memory.get_memories()
        )

        if memories:
            memory_text = "\n".join(
                [
                    f"{key}: {value}"
                    for key, value
                    in memories.items()
                ]
            )

            system_content = (
                f"{self.system_prompt}\n\n"
                "以下是关于用户的长期记忆。"
                "这些记忆只用于帮助你理解用户背景，"
                "回答时只使用与当前问题有关的内容。\n\n"
                f"{memory_text}"
            )

        else:
            system_content = (
                self.system_prompt
            )

        return [
            {
                "role": "system",
                "content": system_content
            },
            *self.messages
        ]

    def _process_memories(
        self,
        user_input: str
    ):
        """
        v8.4 Memory Update 流程：

        Extractor
            ↓
        Candidate Memory
            ↓
        查询旧 Memory
            ↓
        Updater
            ↓
        ADD / UPDATE / KEEP / DELETE
        """

        extracted_memories = (
            extract_memories(
                user_input
            )
        )

        print(
            "[Memory] 提取结果:",
            extracted_memories
        )

        for item in extracted_memories:

            category = item.get(
                "category"
            )

            key = item.get(
                "key"
            )

            new_value = item.get(
                "value"
            )

            if not category:
                continue

            if not key:
                continue

            if not new_value:
                continue

            old_memory = (
                self.memory.get_memory(
                    category,
                    key
                )
            )

            if old_memory:
                old_value = (
                    old_memory[
                        "value"
                    ]
                )
            else:
                old_value = None

            decision = (
                decide_memory_action(
                    user_input=(
                        user_input
                    ),
                    category=category,
                    key=key,
                    new_value=(
                        new_value
                    ),
                    old_value=(
                        old_value
                    )
                )
            )

            action = decision[
                "action"
            ]

            final_value = decision[
                "value"
            ]

            reason = decision.get(
                "reason",
                ""
            )

            print(
                f"[Memory] "
                f"{category}.{key}"
            )

            print(
                f"[Memory] old = "
                f"{old_value}"
            )

            print(
                f"[Memory] new = "
                f"{new_value}"
            )

            print(
                f"[Memory] action = "
                f"{action}"
            )

            print(
                f"[Memory] reason = "
                f"{reason}"
            )

            if action == "ADD":

                self.memory.add_memory(
                    category=category,
                    key=key,
                    value=final_value,
                    source_session_id=(
                        self.session_id
                    )
                )

            elif action == "UPDATE":

                self.memory.update_memory(
                    category=category,
                    key=key,
                    value=final_value,
                    source_session_id=(
                        self.session_id
                    )
                )

            elif action == "DELETE":

                self.memory.delete_memory(
                    category=category,
                    key=key
                )

            elif action == "KEEP":
                pass

    def run(
        self,
        user_input: str
    ) -> str:
        """
        执行一次 Agent 对话。
        """

        # ==============================
        # 1. 提取并更新长期记忆
        # ==============================

        self._process_memories(
            user_input
        )

        # ==============================
        # 2. 保存用户消息
        # ==============================

        self.memory.add_message(
            self.session_id,
            "user",
            user_input
        )

        self.messages.append(
            {
                "role": "user",
                "content": user_input
            }
        )

        # ==============================
        # 3. Agent Loop
        # ==============================

        for step in range(
            1,
            11
        ):

            print(
                f"\n[Agent] Step {step}\n"
            )

            context = (
                self.build_context()
            )

            response = chat(
                context
            )

            # --------------------------
            # 将 assistant 消息放入上下文
            # --------------------------

            assistant_message = {
                "role": "assistant",
                "content": (
                    response.content
                    or ""
                )
            }

            if response.tool_calls:

                assistant_message[
                    "tool_calls"
                ] = [
                    {
                        "id": (
                            tool_call.id
                        ),
                        "type": "function",
                        "function": {
                            "name": (
                                tool_call
                                .function
                                .name
                            ),
                            "arguments": (
                                tool_call
                                .function
                                .arguments
                            )
                        }
                    }
                    for tool_call
                    in response.tool_calls
                ]

            self.messages.append(
                assistant_message
            )

            # --------------------------
            # 没有工具调用：得到最终答案
            # --------------------------

            if not response.tool_calls:

                answer = (
                    response.content
                    or ""
                )

                self.memory.add_message(
                    self.session_id,
                    "assistant",
                    answer
                )

                return answer

            # --------------------------
            # 执行全部工具调用
            # --------------------------

            for tool_call in (
                response.tool_calls
            ):

                tool_name = (
                    tool_call
                    .function
                    .name
                )

                raw_arguments = (
                    tool_call
                    .function
                    .arguments
                )

                print(
                    f"[Tool] 调用: "
                    f"{tool_name}"
                )

                print(
                    f"[Tool] 参数: "
                    f"{raw_arguments}"
                )

                try:
                    arguments = (
                        json.loads(
                            raw_arguments
                        )
                    )

                except json.JSONDecodeError as e:

                    result = {
                        "success": False,
                        "error": (
                            "工具参数 JSON 解析失败: "
                            f"{e}"
                        )
                    }

                else:

                    try:
                        result = (
                            execute_tool(
                                tool_name,
                                arguments
                            )
                        )

                    except Exception as e:
                        result = {
                            "success": False,
                            "error": str(e)
                        }

                print(
                    f"[Tool] 结果: "
                    f"{result}"
                )

                # 每一个 tool_call 都必须紧跟
                # 一个相同 tool_call_id 的 tool 消息。
                self.messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": (
                            tool_call.id
                        ),
                        "content": str(
                            result
                        )
                    }
                )

        # ==============================
        # 4. 超过最大步数
        # ==============================

        answer = (
            "Agent 已达到最大执行步数，"
            "本轮任务停止。"
        )

        self.memory.add_message(
            self.session_id,
            "assistant",
            answer
        )

        return answer