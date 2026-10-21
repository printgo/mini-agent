import logging
import math
import os
from pathlib import Path
from typing import Iterator

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(Path(__file__).resolve().parents[1] / ".env")
logger = logging.getLogger(__name__)


def read_int(name: str, default: int, minimum: int) -> int:
    value = int(os.getenv(name, str(default)))
    if value < minimum:
        raise ValueError(f"{name} 必须大于等于 {minimum}")
    return value


API_KEY = os.getenv("DEEPSEEK_API_KEY", "").strip()
BASE_URL = os.getenv(
    "DEEPSEEK_BASE_URL", "https://api.deepseek.com"
).strip()
MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-flash").strip()
MEMORY_MODEL = os.getenv("DEEPSEEK_MEMORY_MODEL", MODEL).strip()

TIMEOUT = float(os.getenv("LLM_TIMEOUT_SECONDS", "30"))
MAX_RETRIES = read_int("LLM_MAX_RETRIES", 1, 0)
MAX_OUTPUT_TOKENS = read_int("LLM_MAX_OUTPUT_TOKENS", 4096, 1)

# 只查看状态、取消或管理记忆时，不创建模型客户端或要求模型凭据。
client = None


def get_client():
    global client
    if client is None:
        if not API_KEY:
            raise ValueError("请配置 DEEPSEEK_API_KEY")
        if not BASE_URL or not MODEL or not MEMORY_MODEL:
            raise ValueError("BASE_URL、MODEL 和 MEMORY_MODEL 不能为空")
        if not math.isfinite(TIMEOUT) or TIMEOUT <= 0:
            raise ValueError("LLM_TIMEOUT_SECONDS 必须是有限的正数")
        client = OpenAI(
            api_key=API_KEY,
            base_url=BASE_URL,
            timeout=TIMEOUT,
            max_retries=MAX_RETRIES,
        )
    return client


def log_usage(usage) -> None:
    if usage is not None:
        logger.info(
            "LLM usage: prompt=%s completion=%s total=%s",
            usage.prompt_tokens,
            usage.completion_tokens,
            usage.total_tokens,
        )


def request_options(messages: list[dict], model: str | None = None) -> dict:
    return {
        "model": model or MODEL,
        "messages": messages,
        "max_tokens": MAX_OUTPUT_TOKENS,
        "extra_body": {"thinking": {"type": "disabled"}},
    }


def complete(
    messages: list[dict],
    *,
    model: str | None = None,
    tools: list[dict] | None = None,
):
    options = request_options(messages, model)

    if tools:
        options["tools"] = tools

    response = get_client().chat.completions.create(**options)

    # 异常结束的响应也可能已经消耗 Token，先记录用量。
    log_usage(response.usage)

    if not response.choices:
        raise RuntimeError("模型没有返回 choices")

    reason = response.choices[0].finish_reason
    if reason not in {"stop", "tool_calls"}:
        raise RuntimeError(f"模型输出未正常完成，finish_reason={reason}")

    return response


def chat(messages: list[dict]):
    from tools.registry import get_tool_schemas

    response = complete(messages, tools=get_tool_schemas())
    return response.choices[0].message


def stream_text(messages: list[dict]) -> Iterator[str]:
    # 本版只处理纯文本流，不携带工具定义。
    options = request_options(messages)

    stream = get_client().chat.completions.create(
        **options,
        stream=True,
        stream_options={"include_usage": True},
    )

    with stream:
        for chunk in stream:
            log_usage(chunk.usage)

            if not chunk.choices:
                continue

            choice = chunk.choices[0]
            reason = choice.finish_reason

            if reason is not None and reason != "stop":
                raise RuntimeError(
                    f"流式输出未正常完成，finish_reason={reason}"
                )

            if choice.delta.content:
                yield choice.delta.content
