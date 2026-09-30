import json

from llm.client import client


ALLOWED_MEMORY_KEYS = {
    ("profile", "name"),
    ("career", "direction"),
    ("tech_stack", "primary_language"),
    ("tech_stack", "backend_framework"),
    ("tech_stack", "database"),
    ("tech_stack", "middleware"),
    ("learning", "goal"),
    ("project", "name"),
    ("research", "direction"),
    ("preference", "response_style"),
}


SYSTEM_PROMPT = """
你是一个长期记忆提取器。

你的任务是从用户输入中提取值得长期保存、并且与用户本人有关的稳定信息。

只提取用户明确陈述的事实，不要推测。

可以提取的字段只有：

1. profile.name
   用户姓名、称呼。

2. career.direction
   用户长期职业方向。

3. tech_stack.primary_language
   用户长期主要使用的编程语言。

4. tech_stack.backend_framework
   用户长期主要使用的后端框架。

5. tech_stack.database
   用户长期主要使用的数据库。

6. tech_stack.middleware
   用户长期主要使用的中间件。

7. learning.goal
   用户持续学习的长期方向或目标。

8. project.name
   用户正在持续开发的长期项目名称。

9. research.direction
   用户持续研究的研究方向。

10. preference.response_style
    用户明确提出、以后也希望持续遵守的回答风格偏好。

以下内容不要保存：

- 普通问题。
- 一次性任务。
- 临时状态。
- 当前这一次对话的操作意图。
- 用户只是询问、比较、举例或提到的技术。
- 关于其他人的信息。
- 模型自己推断出的信息。
- 对用户输入的总结。
- other、note、intent、topic、question 等元信息。

必须遵守：

Mention != Memory
Question != Memory

例如：

用户说：
“Python 的 list 怎么用？”

返回：
[]

因为用户只是询问 Python，并不能说明 Python 是他的长期技术栈。

用户说：
“我主要做 Java 后端开发。”

可以提取：
[
  {
    "category": "career",
    "key": "direction",
    "value": "Java后端开发"
  }
]

如果用户明确纠正、否定、停止或要求删除某个可能已经存在的长期事实，
仍然需要提取对应的 category、key、value，交给后续 Memory Updater 判断。

例如：

“我不再做 Java 后端了”
仍然应该识别：
{
  "category": "career",
  "key": "direction",
  "value": "Java后端开发"
}

“请忘记我正在开发 mini-agent”
仍然应该识别：
{
  "category": "project",
  "key": "name",
  "value": "mini-agent"
}

如果用户同时明确提供了新的替代事实，优先提取新的事实。

例如：

“我现在主要做 Go 后端，不再以 Java 为主。”

应该提取：
[
  {
    "category": "career",
    "key": "direction",
    "value": "Go后端开发"
  }
]

这里只负责提取候选记忆。
不要决定 ADD、UPDATE、KEEP、DELETE。

必须只返回 JSON 数组。
不要返回 Markdown。
不要返回 ```json。
不要解释。

没有值得保存的长期信息时，返回：
[]
""".strip()


def _strip_code_fence(content: str) -> str:
    """
    防止模型偶尔返回：

    ```json
    [...]
    ```
    """

    content = content.strip()

    if not content.startswith("```"):
        return content

    lines = content.splitlines()

    if lines:
        lines = lines[1:]

    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]

    return "\n".join(lines).strip()


def extract_memories(
    user_input: str
) -> list[dict]:
    """
    从用户输入中提取长期记忆候选。

    返回格式：

    [
        {
            "category": "career",
            "key": "direction",
            "value": "Java后端开发"
        }
    ]
    """

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        },
        {
            "role": "user",
            "content": user_input
        }
    ]

    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages
        )
    except Exception as e:
        print(
            "[Memory Extractor] 调用失败:",
            e
        )
        return []

    content = (
        response
        .choices[0]
        .message
        .content
        or ""
    )

    content = _strip_code_fence(
        content
    )

    try:
        result = json.loads(
            content
        )
    except json.JSONDecodeError:
        print(
            "[Memory Extractor] JSON 解析失败:",
            content
        )
        return []

    if not isinstance(
        result,
        list
    ):
        return []

    memories = []

    for item in result:

        if not isinstance(
            item,
            dict
        ):
            continue

        category = item.get(
            "category"
        )

        key = item.get(
            "key"
        )

        value = item.get(
            "value"
        )

        if not isinstance(
            category,
            str
        ):
            continue

        if not isinstance(
            key,
            str
        ):
            continue

        if value is None:
            continue

        category = (
            category
            .strip()
            .lower()
        )

        key = (
            key
            .strip()
            .lower()
        )

        value = str(
            value
        ).strip()

        if not value:
            continue

        if (
            category,
            key
        ) not in ALLOWED_MEMORY_KEYS:
            continue

        memories.append(
            {
                "category": category,
                "key": key,
                "value": value
            }
        )

    return memories