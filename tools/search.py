import json
import os

from dotenv import load_dotenv
from tavily import TavilyClient


load_dotenv()


def search_web(
    query: str,
    max_results: int = 5
) -> str:
    """
    使用 Tavily 搜索互联网。
    """

    api_key = os.getenv("TAVILY_API_KEY")

    if not api_key:
        return json.dumps(
            {
                "success": False,
                "error": "没有配置 TAVILY_API_KEY"
            },
            ensure_ascii=False
        )

    try:

        client = TavilyClient(
            api_key=api_key
        )

        response = client.search(
            query=query,
            max_results=max_results,
            search_depth="basic"
        )

        results = []

        for item in response.get("results", []):

            results.append({
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "content": item.get("content", "")
            })

        if not results:

            return json.dumps(
                {
                    "success": False,
                    "error": "没有搜索到相关结果"
                },
                ensure_ascii=False
            )

        return json.dumps(
            {
                "success": True,
                "query": query,
                "results": results
            },
            ensure_ascii=False
        )

    except Exception as e:

        return json.dumps(
            {
                "success": False,
                "error": f"搜索失败: {str(e)}"
            },
            ensure_ascii=False
        )


SEARCH_SCHEMA = {
    "type": "function",
    "function": {
        "name": "search_web",
        "description": (
            "搜索互联网并获取最新或外部信息。"
            "当用户询问新闻、最新技术、"
            "实时信息、当前事件、网站信息，"
            "或者模型自身知识可能过时时，应使用此工具。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "需要搜索的问题或关键词"
                },
                "max_results": {
                    "type": "integer",
                    "description": "最大搜索结果数量，默认5条"
                }
            },
            "required": [
                "query"
            ]
        }
    }
}