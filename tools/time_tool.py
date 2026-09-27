from datetime import datetime


def get_current_time() -> str:
    return datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )


TIME_SCHEMA = {
    "type": "function",
    "function": {
        "name": "get_current_time",
        "description": "获取当前系统的日期和时间",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": []
        }
    }
}