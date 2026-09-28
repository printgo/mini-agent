import json

from llm.client import client


def extract_memories(user_input: str) -> list[dict]:
    """
    从用户输入中提取值得长期保存的信息。
    """

    messages = [
        {
            "role": "system",
            "content": (
                "你是一个长期记忆提取器。"
                "你的任务是从用户消息中提取"
                "值得长期保存、未来仍然有帮助的信息。"

                "适合保存的信息包括："
                "姓名、长期职业方向、长期技术方向、"
                "稳定偏好、长期学习目标、长期项目等。"

                "不要保存："
                "普通问候、临时问题、一次性任务、"
                "短期状态、无意义内容。"

                "必须只返回 JSON 数组，不要解释。"

                "格式："
                "["
                '{"key":"name","value":"Tom"},'
                '{"key":"tech_direction","value":"Java后端开发"}'
                "]"

                "如果没有值得保存的信息，返回 []。"
            )
        },
        {
            "role": "user",
            "content": user_input
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
    )

    try:

        memories = json.loads(content)

        if not isinstance(memories, list):
            return []

        return memories

    except json.JSONDecodeError:

        return []