from tools.calculator import calculator, CALCULATOR_SCHEMA
from tools.time_tool import get_current_time, TIME_SCHEMA
from tools.search import search_web, SEARCH_SCHEMA


TOOL_REGISTRY = {
    "calculator": calculator,
    "get_current_time": get_current_time,
    "search_web": search_web
}


TOOL_SCHEMAS = [
    CALCULATOR_SCHEMA,
    TIME_SCHEMA,
    SEARCH_SCHEMA
]


def execute_tool(tool_name: str, arguments: dict):

    tool = TOOL_REGISTRY.get(tool_name)

    if tool is None:
        raise ValueError(
            f"未找到工具: {tool_name}"
        )

    return tool(**arguments)


def get_tool_schemas():
    return TOOL_SCHEMAS