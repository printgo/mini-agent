import json
import logging

from llm.client import chat, MAX_OUTPUT_TOKENS
from memory.extractor import extract_memories
from memory.memory import Memory
from memory.retriever import retrieve_memories
from memory.updater import decide_memory_action
from tools.registry import execute_tool, get_tool_schemas
from research.context import TokenContext
from research.knowledge import KnowledgeStore

logger = logging.getLogger(__name__)


class ResearchAgent:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.memory = Memory()
        self.knowledge = KnowledgeStore(self.memory.conn)
        self.context = TokenContext(reserve=MAX_OUTPUT_TOKENS)
        self.knowledge.seed_chat(self.memory.get_memories())

        self.system_prompt = (
            "你是一个AI Research Agent。"
            "你可以自主决定是否调用工具。"
            "对于最新信息、实时信息或外部信息，"
            "优先使用 search_web 搜索。"
            "当需要核实信息或深入理解网页内容时，"
            "使用 read_webpage。"
        )

        # 只恢复当前会话历史；长期记忆在构建上下文时检索。
        self.messages = self.memory.get_messages(self.session_id)

    def build_context(self, user_input: str) -> list[dict]:
        """检索相关长期记忆，并构建模型上下文。"""
        try:
            all_memories = self.memory.get_memories()
            relevant_memories = retrieve_memories(
                user_input=user_input,
                memories=all_memories,
            )
        except Exception:
            logger.exception("长期记忆检索失败，使用普通对话上下文")
            relevant_memories = {}

        print("[Memory Retrieval] 命中:", relevant_memories)

        system_content = self.system_prompt

        if relevant_memories:
            memory_text = "\n".join(
                f"{key}: {value}"
                for key, value in relevant_memories.items()
            )
            system_content = (
                f"{self.system_prompt}\n\n"
                "以下是与当前问题相关的用户长期记忆。"
                "这些记忆只用于帮助你理解用户背景，"
                "不要机械复述全部记忆，"
                "只在与当前问题有关时自然使用。\n\n"
                f"{memory_text}"
            )

        return self.context.chat([
            {"role": "system", "content": system_content},
            *self.messages,
        ], tools=get_tool_schemas())

    def _sync_research_memory(self, category, key, value, action):
        # 旧聊天入口是单用户；只同步到 default 研究档案。
        kind = "topic" if category == "research" else "preference"
        name = category + "." + key
        if action in {"DELETE", "KEEP"}:
            self.knowledge.apply("default", action, kind, name)
            return
        row = self.memory.conn.execute(
            "SELECT 1 FROM research_knowledge WHERE user_id='default' AND kind=? AND key=?",
            (kind, name),
        ).fetchone()
        self.knowledge.apply(
            "default", "UPDATE" if row else "ADD", kind, name, str(value)
        )

    def _contains_memory_change_signal(self, user_input: str) -> bool:
        """保留原有的记忆修改、否定、删除信号判断。"""
        change_signals = [
            "不再", "不是", "不要", "忘记", "删除",
            "取消", "停止", "不做", "不学",
            "改成", "改为", "换成", "更换",
            "以后叫我", "现在叫我", "以后不要",
        ]
        return any(signal in user_input for signal in change_signals)

    def _process_memories(self, user_input: str) -> None:
        """提取候选记忆、判断动作并执行数据库操作。"""
        extracted_memories = extract_memories(user_input)
        print("[Memory] 提取结果:", extracted_memories)

        for item in extracted_memories:
            category = item.get("category")
            key = item.get("key")
            new_value = item.get("value")

            if not category or not key or not new_value:
                continue

            old_memory = self.memory.get_memory(category, key)
            old_value = old_memory["value"] if old_memory else None

            same_value = (
                old_value is not None
                and str(old_value).strip() == str(new_value).strip()
            )
            has_change_signal = self._contains_memory_change_signal(
                user_input
            )

            if same_value and not has_change_signal:
                print(f"[Memory] {category}.{key}")
                print(f"[Memory] old = {old_value}")
                print(f"[Memory] new = {new_value}")
                print("[Memory] action = KEEP")
                print(
                    "[Memory] reason = 新旧值一致，直接 KEEP，"
                    "跳过 Memory Updater 调用"
                )
                self._sync_research_memory(category, key, new_value, "ADD")
                continue

            decision = decide_memory_action(
                user_input=user_input,
                category=category,
                key=key,
                new_value=new_value,
                old_value=old_value,
            )

            action = decision["action"]
            final_value = decision["value"]
            reason = decision.get("reason", "")

            print(f"[Memory] {category}.{key}")
            print(f"[Memory] old = {old_value}")
            print(f"[Memory] new = {new_value}")
            print(f"[Memory] action = {action}")
            print(f"[Memory] reason = {reason}")

            if action == "ADD":
                self.memory.add_memory(
                    category=category,
                    key=key,
                    value=final_value,
                    source_session_id=self.session_id,
                )
            elif action == "UPDATE":
                self.memory.update_memory(
                    category=category,
                    key=key,
                    value=final_value,
                    source_session_id=self.session_id,
                )
            elif action == "DELETE":
                self.memory.delete_memory(category=category, key=key)
            elif action == "KEEP":
                pass
            self._sync_research_memory(category, key, final_value, action)

    def run(self, user_input: str) -> str:
        """执行一次对话；长期记忆失败时继续回答。"""
        try:
            self._process_memories(user_input)
        except Exception:
            logger.exception("长期记忆更新失败，继续本轮回答")

        # 聊天消息保存仍依赖数据库，不属于长期记忆降级范围。
        self.memory.add_message(self.session_id, "user", user_input)
        self.messages.append({"role": "user", "content": user_input})

        for step in range(1, 11):
            print(f"\n[Agent] Step {step}\n")

            context = self.build_context(user_input)
            response = chat(context)

            assistant_message = {
                "role": "assistant",
                "content": response.content or "",
            }

            if response.tool_calls:
                assistant_message["tool_calls"] = [
                    {
                        "id": tool_call.id,
                        "type": "function",
                        "function": {
                            "name": tool_call.function.name,
                            "arguments": tool_call.function.arguments,
                        },
                    }
                    for tool_call in response.tool_calls
                ]

            self.messages.append(assistant_message)

            if not response.tool_calls:
                answer = response.content or ""
                self.memory.add_message(
                    self.session_id, "assistant", answer
                )
                return answer

            for tool_call in response.tool_calls:
                tool_name = tool_call.function.name
                raw_arguments = tool_call.function.arguments

                print(f"[Tool] 调用: {tool_name}")
                print(f"[Tool] 参数: {raw_arguments}")

                # JSON 解析、参数校验、执行与错误包装交给注册中心。
                result = execute_tool(tool_name, raw_arguments)
                print(f"[Tool] 结果: {result}")

                # 工具失败也必须有对应的 tool message。
                self.messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(
                        result,
                        ensure_ascii=False,
                        allow_nan=False,
                    ),
                })

        answer = "Agent 已达到最大执行步数，本轮任务停止。"
        self.messages.append({"role": "assistant", "content": answer})
        self.memory.add_message(self.session_id, "assistant", answer)
        return answer
