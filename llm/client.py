import os

from dotenv import load_dotenv
from openai import OpenAI

from tools.registry import get_tool_schemas


load_dotenv()


client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)


def chat(messages: list):

    response = client.chat.completions.create(
        model="deepseek-flash",
        messages=messages,
        tools=get_tool_schemas()
    )

    return response.choices[0].message