def calculator(a: float, b: float, operation: str) -> float:
    if operation == "add":
        return a + b

    elif operation == "subtract":
        return a - b

    elif operation == "multiply":
        return a * b

    elif operation == "divide":
        if b == 0:
            raise ValueError("除数不能为0")

        return a / b

    else:
        raise ValueError(f"不支持的操作: {operation}")


CALCULATOR_SCHEMA = {
    "type": "function",
    "function": {
        "name": "calculator",
        "description": "用于执行数学计算，包括加法、减法、乘法和除法",
        "parameters": {
            "type": "object",
            "properties": {
                "a": {
                    "type": "number",
                    "description": "第一个数字"
                },
                "b": {
                    "type": "number",
                    "description": "第二个数字"
                },
                "operation": {
                    "type": "string",
                    "enum": [
                        "add",
                        "subtract",
                        "multiply",
                        "divide"
                    ],
                    "description": "数学运算类型"
                }
            },
            "required": [
                "a",
                "b",
                "operation"
            ]
        }
    }
}