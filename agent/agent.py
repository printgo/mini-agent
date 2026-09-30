import json

from llm.client import chat
from memory.extractor import extract_memories
from memory.memory import Memory
from memory.retriever import retrieve_memories
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

        # 只恢复当前 Session 历史。
        #
        # 长期记忆不会直接放进 self.messages，
        # 而是在 build_context() 中根据当前问题
        # 动态检索。
        self.messages = (
            self.memory.get_messages(
                self.session_id
            )
        )

    def build_context(
        self,
        user_input: str
    ) -> list[dict]:
        """
        构建当前发送给 LLM 的上下文。

        v8.5：
        不再把所有长期记忆全部注入 Prompt，
        而是根据当前 user_input 检索相关记忆。
        """

        # ==============================
        # 1. 获取全部长期记忆
        # ==============================

        all_memories = (
            self.memory.get_memories()
        )

        # ==============================
        # 2. Memory Retrieval
        # ==============================

        relevant_memories = (
            retrieve_memories(
                user_input=user_input,
                memories=all_memories
            )
        )

        print(
            "[Memory Retrieval] 命中:",
            relevant_memories
        )

        # ==============================
        # 3. 构建 System Prompt
        # ==============================

        if relevant_memories:

            memory_text = "\n".join(
                [
                    f"{key}: {value}"
                    for key, value
                    in relevant_memories.items()
                ]
            )

            system_content = (
                f"{self.system_prompt}\n\n"
                "以下是与当前问题相关的用户长期记忆。"
                "这些记忆只用于帮助你理解用户背景，"
                "不要机械复述全部记忆，"
                "只在与当前问题有关时自然使用。\n\n"
                f"{memory_text}"
            )

        else:

            system_content = (
                self.system_prompt
            )

        # ==============================
        # 4. 返回完整上下文
        # ==============================

        return [
            {
                "role": "system",
                "content": system_content
            },
            *self.messages
        ]

    def _contains_memory_change_signal(
        self,
        user_input: str
    ) -> bool:
        """
        判断当前输入是否包含明显的：

        修改
        否定
        删除

        Memory 的信号。
        """

        change_signals = [
            "不再",
            "不是",
            "不要",
            "忘记",
            "删除",
            "取消",
            "停止",
            "不做",
            "不学",
            "改成",
            "改为",
            "换成",
            "更换",
            "以后叫我",
            "现在叫我",
            "以后不要",
        ]

        return any(
            signal in user_input
            for signal in change_signals
        )

    def _process_memories(
        self,
        user_input: str
    ):
        """
        Memory Update 流程：

        Extractor
            ↓
        Candidate Memory
            ↓
        查询旧 Memory
            ↓
        相同值快速 KEEP
            ↓
        Memory Updater
            ↓
        ADD / UPDATE / KEEP / DELETE
        """

        # ==============================
        # 1. Memory Extractor
        # ==============================

        extracted_memories = (
            extract_memories(
                user_input
            )
        )

        print(
            "[Memory] 提取结果:",
            extracted_memories
        )

        # ==============================
        # 2. 逐条处理候选 Memory
        # ==============================

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

            # ==========================
            # 3. 查询旧 Memory
            # ==========================

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

            # ==========================
            # 4. v8.4.1 优化
            #
            # 新旧值完全相同的时候，
            # 一般不需要再次调用 DeepSeek。
            # ==========================

            same_value = (
                old_value is not None
                and str(old_value).strip()
                == str(new_value).strip()
            )

            has_change_signal = (
                self._contains_memory_change_signal(
                    user_input
                )
            )

            if (
                same_value
                and not has_change_signal
            ):

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
                    "[Memory] action = KEEP"
                )

                print(
                    "[Memory] reason = "
                    "新旧值一致，直接 KEEP，"
                    "跳过 Memory Updater 调用"
                )

                continue

            # ==========================
            # 5. Memory Updater
            # ==========================

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

            # ==========================
            # 6. 执行数据库操作
            # ==========================

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

            # ==========================
            # v8.5
            #
            # build_context 现在需要
            # 当前 user_input。
            #
            # Memory Retriever 会根据它
            # 选择相关长期记忆。
            # ==========================

            context = (
                self.build_context(
                    user_input
                )
            )

            response = chat(
                context
            )

            # ==========================
            # 4. Assistant Message
            # ==========================

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

            # ==========================
            # 5. 没有 Tool Call
            # ==========================

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

            # ==========================
            # 6. 执行 Tool Calls
            # ==========================

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

                # 一个 tool_call
                # 必须对应一个 tool message。
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
        # 7. 最大执行步数
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