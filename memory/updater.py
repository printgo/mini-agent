import json
import logging

from llm.client import MEMORY_MODEL, complete

logger = logging.getLogger(__name__)

ALLOWED_ACTIONS = {"ADD", "UPDATE", "KEEP", "DELETE"}


def decide_memory_action(
    user_input: str,
    category: str,
    key: str,
    new_value: str,
    old_value: str | None,
) -> dict:
    """判断候选记忆应执行 ADD、UPDATE、KEEP 还是 DELETE。"""
    messages = [
        {
            "role": "system",
            "content": (
                "你是一个长期记忆更新决策器。"
                "你的任务不是提取记忆，"
                "而是比较已有长期记忆和新的候选记忆，"
                "决定应该执行什么操作。"
                "只允许以下四种操作："
                "ADD：数据库中没有这条记忆，"
                "并且用户明确提供了新的长期事实。"
                "UPDATE：已有记忆已经过时，"
                "用户明确提供了新的事实进行替代或纠正。"
                "KEEP：已有记忆仍然正确；"
                "或者新旧信息没有真正冲突；"
                "或者证据不足以修改旧记忆。"
                "DELETE：用户明确表示旧记忆已经不成立，"
                "或者明确要求忘记、删除这条记忆，"
                "并且没有新的值替代它。"
                "注意："
                "不要因为用户只是提到另一个技术、项目、"
                "学习内容，就自动覆盖原来的长期记忆。"
                "如果无法确定，优先 KEEP。"
                "必须只返回 JSON，不要解释。"
                "格式："
                "{"
                '"action":"KEEP",'
                '"value":"最终应该保存的值",'
                '"reason":"简短原因"'
                "}"
            ),
        },
        {
            "role": "user",
            "content": (
                f"用户原始输入：{user_input}\n\n"
                f"category：{category}\n"
                f"key：{key}\n"
                f"旧记忆：{old_value}\n"
                f"新候选记忆：{new_value}"
            ),
        },
    ]

    fallback = {
        "action": "KEEP",
        "value": old_value,
        "reason": "记忆更新失败或结果无效，保留原状态",
    }

    try:
        response = complete(
            model=MEMORY_MODEL,
            messages=messages,
        )

        content = response.choices[0].message.content
        if not isinstance(content, str) or not content.strip():
            raise ValueError("记忆决策内容为空")

        content = content.strip()

        if content.startswith("```"):
            lines = content.splitlines()[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            content = "\n".join(lines).strip()

        result = json.loads(content)
        if not isinstance(result, dict):
            raise ValueError("记忆决策必须是 JSON 对象")

        action = result.get("action")
        if not isinstance(action, str):
            raise ValueError("action 必须是字符串")

        action = action.strip().upper()
        if action not in ALLOWED_ACTIONS:
            raise ValueError("action 不在允许范围内")

        value = result.get("value")
        reason = result.get("reason", "")
        if not isinstance(reason, str):
            reason = ""

        # 保留原有的动作修正规则。
        if old_value is None:
            if action == "UPDATE":
                action = "ADD"
            elif action == "DELETE":
                action = "KEEP"
        elif action == "ADD":
            action = "UPDATE"

        if action in {"ADD", "UPDATE"}:
            if not isinstance(value, str) or not value.strip():
                raise ValueError("新增或更新的记忆必须是非空字符串")
            value = value.strip()
        elif action == "KEEP":
            value = old_value
        else:
            value = None

        return {
            "action": action,
            "value": value,
            "reason": reason,
        }

    except Exception:
        logger.exception("Memory Updater 失败，保留原状态")
        return fallback