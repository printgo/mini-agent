import json

from llm.client import client


ALLOWED_ACTIONS = {
    "ADD",
    "UPDATE",
    "KEEP",
    "DELETE"
}


def decide_memory_action(
    user_input: str,
    category: str,
    key: str,
    new_value: str,
    old_value: str | None
) -> dict:
    """
    判断候选长期记忆应该：
    ADD / UPDATE / KEEP / DELETE
    """

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
            )
        },
        {
            "role": "user",
            "content": (
                f"用户原始输入：{user_input}\n\n"
                f"category：{category}\n"
                f"key：{key}\n"
                f"旧记忆：{old_value}\n"
                f"新候选记忆：{new_value}"
            )
        }
    ]

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=messages
    )

    content = (
        response
        .choices[0]
        .message
        .content
        .strip()
    )

    # 防止模型返回 ```json ... ```
    if content.startswith("```"):
        lines = content.splitlines()

        if lines:
            lines = lines[1:]

        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]

        content = "\n".join(lines)

    try:

        result = json.loads(content)

    except json.JSONDecodeError:

        # 更新判断失败时：
        # 宁愿不修改旧记忆
        return {
            "action": "KEEP",
            "value": old_value,
            "reason": "Memory Updater 返回格式错误"
        }

    action = result.get(
        "action",
        "KEEP"
    ).upper()

    value = result.get(
        "value"
    )

    reason = result.get(
        "reason",
        ""
    )

    if action not in ALLOWED_ACTIONS:
        action = "KEEP"

    # -----------------------------
    # 防御性处理
    # -----------------------------

    # 没有旧记忆时
    if old_value is None:

        if action == "UPDATE":
            action = "ADD"

        elif action == "DELETE":
            action = "KEEP"

    # 已有相同 category + key
    if old_value is not None:

        if action == "ADD":
            action = "UPDATE"

    if action in {
        "ADD",
        "UPDATE"
    }:

        if not value:
            value = new_value

    if action == "KEEP":

        if old_value is not None:
            value = old_value
        else:
            value = new_value

    if action == "DELETE":
        value = None

    return {
        "action": action,
        "value": value,
        "reason": reason
    }