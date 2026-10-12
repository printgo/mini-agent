import json
import logging
import math
from copy import deepcopy

from tools.calculator import calculator, CALCULATOR_SCHEMA
from tools.time_tool import get_current_time, TIME_SCHEMA
from tools.search import search_web, SEARCH_SCHEMA
from tools.web_reader import read_webpage, WEB_READER_SCHEMA

logger = logging.getLogger(__name__)

TOOL_REGISTRY = {
    "calculator": calculator,
    "get_current_time": get_current_time,
    "search_web": search_web,
    "read_webpage": read_webpage,
}

# 使用副本，避免修改原工具模块中的 Schema。
TOOL_SCHEMAS = deepcopy([
    CALCULATOR_SCHEMA,
    TIME_SCHEMA,
    SEARCH_SCHEMA,
    WEB_READER_SCHEMA,
])

for schema in TOOL_SCHEMAS:
    parameters = schema["function"]["parameters"]
    parameters["additionalProperties"] = False

    for rule in parameters.get("properties", {}).values():
        if rule.get("type") == "string":
            rule["minLength"] = 1

SCHEMA_BY_NAME = {
    schema["function"]["name"]: schema["function"]["parameters"]
    for schema in TOOL_SCHEMAS
}

SCHEMA_BY_NAME["search_web"]["properties"]["max_results"].update(
    minimum=1,
    maximum=10,
)


def failure(code: str, message: str) -> dict:
    return {
        "success": False,
        "data": None,
        "error": {"code": code, "message": message},
    }


def validate_arguments(tool_name: str, arguments: dict) -> None:
    if not isinstance(arguments, dict):
        raise ValueError("工具参数必须是 JSON 对象")

    schema = SCHEMA_BY_NAME[tool_name]
    properties = schema.get("properties", {})

    missing = set(schema.get("required", [])) - arguments.keys()
    if missing:
        raise ValueError(f"缺少参数: {', '.join(sorted(missing))}")

    unknown = arguments.keys() - properties.keys()
    if unknown:
        raise ValueError(f"未知参数: {', '.join(sorted(unknown))}")

    for name, value in arguments.items():
        rule = properties[name]
        expected = rule.get("type")

        if expected == "string":
            valid = isinstance(value, str)
        elif expected == "integer":
            valid = type(value) is int
        elif expected == "number":
            valid = (
                type(value) in (int, float)
                and math.isfinite(value)
            )
        else:
            raise ValueError(f"暂不支持参数类型: {expected}")

        if not valid:
            raise ValueError(f"{name} 必须是 {expected}")

        if expected == "string":
            if len(value.strip()) < rule.get("minLength", 0):
                raise ValueError(f"{name} 不能为空")

        if "enum" in rule and value not in rule["enum"]:
            raise ValueError(f"{name} 不在允许值范围内")

        if "minimum" in rule and value < rule["minimum"]:
            raise ValueError(f"{name} 不能小于 {rule['minimum']}")

        if "maximum" in rule and value > rule["maximum"]:
            raise ValueError(f"{name} 不能大于 {rule['maximum']}")


def execute_tool(tool_name: str, arguments: str | dict) -> dict:
    tool = TOOL_REGISTRY.get(tool_name)
    if tool is None:
        return failure("UNKNOWN_TOOL", f"未找到工具: {tool_name}")

    try:
        if isinstance(arguments, str):
            arguments = json.loads(arguments)

        validate_arguments(tool_name, arguments)
    except (ValueError, TypeError, OverflowError) as error:
        return failure("INVALID_ARGUMENTS", str(error))

    try:
        data = tool(**arguments)

        # 兼容现有搜索、网页工具返回的 JSON 字符串。
        if isinstance(data, str):
            try:
                data = json.loads(data)
            except json.JSONDecodeError:
                pass

        if isinstance(data, dict) and data.get("success") is False:
            return failure(
                "TOOL_ERROR",
                str(data.get("error", "工具执行失败")),
            )

        if isinstance(data, dict) and data.get("success") is True:
            data = {
                key: value
                for key, value in data.items()
                if key != "success"
            }

        # 防止不可序列化对象或非有限数进入消息历史。
        json.dumps(data, ensure_ascii=False, allow_nan=False)

        return {
            "success": True,
            "data": data,
            "error": None,
        }
    except Exception as error:
        logger.exception("工具执行失败: %s", tool_name)
        return failure("TOOL_ERROR", str(error))


def get_tool_schemas():
    return TOOL_SCHEMAS