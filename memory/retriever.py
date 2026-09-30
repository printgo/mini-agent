ALL_MEMORY_TRIGGERS = [
    "你记得我吗",
    "你知道我吗",
    "你记得我什么",
    "你知道我什么",
    "关于我的信息",
    "我的信息",
    "我的情况",
    "我的画像",
    "我是谁",
]


ALWAYS_INCLUDE_KEYS = {
    "profile.name",
    "preference.response_style",
}


MEMORY_RULES = {
    "career.direction": [
        "职业",
        "工作",
        "岗位",
        "职业方向",
        "后端开发",
        "做什么工作",
        "主要做什么",
    ],

    "tech_stack.primary_language": [
        "主语言",
        "编程语言",
        "java",
        "python",
        "golang",
        "go语言",
        "技术栈",
    ],

    "tech_stack.backend_framework": [
        "spring",
        "spring boot",
        "spring cloud",
        "后端框架",
        "框架",
    ],

    "tech_stack.database": [
        "mysql",
        "redis",
        "sqlite",
        "数据库",
    ],

    "tech_stack.middleware": [
        "rabbitmq",
        "kafka",
        "消息队列",
        "中间件",
    ],

    "learning.goal": [
        "学习",
        "正在学",
        "学什么",
        "学习方向",
        "ai agent",
        "agent",
    ],

    "project.name": [
        "mini-agent",
        "mini agent",
        "项目",
        "继续开发",
        "继续下面的开发",
        "继续开发项目",
    ],

    "research.direction": [
        "研究",
        "研究方向",
        "论文",
        "推荐系统",
        "负反馈",
        "多模态",
        "对比学习",
    ],

    "preference.response_style": [
        "回答方式",
        "讲解方式",
        "回答风格",
        "完整代码",
        "代码格式",
        "偏好",
    ],
}


def retrieve_memories(
    user_input: str,
    memories: dict[str, str]
) -> dict[str, str]:
    """
    根据当前用户输入，从全部长期记忆中筛选出
    与当前问题相关的记忆。

    v8.5 第一阶段采用规则/关键词检索，
    暂时不引入 Embedding 和向量数据库。
    """

    if not memories:
        return {}

    query = (
        user_input
        .strip()
        .lower()
    )

    # ==============================
    # 1. 用户明确询问自己的全部信息
    # ==============================

    if any(
        trigger in query
        for trigger in ALL_MEMORY_TRIGGERS
    ):
        return dict(memories)

    selected_keys = set()

    # ==============================
    # 2. 默认保留少量全局 Memory
    # ==============================

    for key in ALWAYS_INCLUDE_KEYS:

        if key in memories:
            selected_keys.add(
                key
            )

    # ==============================
    # 3. 根据关键词匹配 Memory
    # ==============================

    for memory_key, keywords in (
        MEMORY_RULES.items()
    ):

        if memory_key not in memories:
            continue

        matched = any(
            keyword in query
            for keyword in keywords
        )

        if matched:
            selected_keys.add(
                memory_key
            )

    # ==============================
    # 4. mini-agent 项目关联 Memory
    # ==============================

    if (
        "mini-agent" in query
        or "mini agent" in query
        or "继续开发" in query
    ):

        related_keys = {
            "project.name",
            "learning.goal",
            "preference.response_style",
        }

        for key in related_keys:

            if key in memories:
                selected_keys.add(
                    key
                )

    # ==============================
    # 5. 返回相关 Memory
    # ==============================

    return {
        key: memories[key]
        for key in memories
        if key in selected_keys
    }