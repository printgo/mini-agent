import json

import requests
from bs4 import BeautifulSoup


def read_webpage(url: str) -> str:
    """
    读取网页正文内容。
    """

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 "
            "(KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }

    try:

        response = requests.get(
            url,
            headers=headers,
            timeout=15
        )

        response.raise_for_status()

        soup = BeautifulSoup(
            response.text,
            "html.parser"
        )

        # 删除不需要的内容
        for tag in soup([
            "script",
            "style",
            "nav",
            "footer",
            "header",
            "noscript"
        ]):
            tag.decompose()

        text = soup.get_text(
            separator="\n",
            strip=True
        )

        # 防止网页太长
        max_length = 10000

        if len(text) > max_length:
            text = text[:max_length]

        return json.dumps(
            {
                "success": True,
                "url": url,
                "content": text
            },
            ensure_ascii=False
        )

    except requests.Timeout:

        return json.dumps(
            {
                "success": False,
                "url": url,
                "error": "网页读取超时"
            },
            ensure_ascii=False
        )

    except requests.RequestException as e:

        return json.dumps(
            {
                "success": False,
                "url": url,
                "error": f"网页请求失败: {str(e)}"
            },
            ensure_ascii=False
        )

    except Exception as e:

        return json.dumps(
            {
                "success": False,
                "url": url,
                "error": f"网页解析失败: {str(e)}"
            },
            ensure_ascii=False
        )


WEB_READER_SCHEMA = {
    "type": "function",
    "function": {
        "name": "read_webpage",
        "description": (
            "读取指定网页URL的正文内容。"
            "当搜索结果提供了相关网页，"
            "而需要查看网页详细内容、核实信息"
            "或深入研究时使用。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "需要读取的网页URL"
                }
            },
            "required": [
                "url"
            ]
        }
    }
}