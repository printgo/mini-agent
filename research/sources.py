import base64
import json
import os
import re
import threading
import time
import xml.etree.ElementTree as ET
from urllib.parse import urlsplit

import requests
from bs4 import BeautifulSoup

from tools.registry import execute_tool

_ARXIV_LOCK = threading.Lock()
_ARXIV_LAST = 0.0

WEB_HOSTS = {
    "docs.langchain.com",
    "docs.github.com",
    "arxiv.org",
    "platform.openai.com",
    "developers.openai.com",
    "api-docs.deepseek.com",
    "react-lm.github.io",
    "huggingface.co",
}


def _request(
    url: str,
    *,
    params=None,
    headers=None,
    timeout=(5, 20),
) -> tuple[bytes, str]:
    headers = {
        "User-Agent": "research-agent/12.0",
        **(headers or {}),
    }

    with requests.get(
        url,
        params=params,
        headers=headers,
        stream=True,
        timeout=timeout,
        allow_redirects=False,
    ) as response:
        if 300 <= response.status_code < 400:
            raise RuntimeError("本版不跟随重定向")

        response.raise_for_status()
        body = bytearray()

        for part in response.iter_content(8192):
            body.extend(part)
            if len(body) > 1_000_000:
                raise ValueError("响应超过 1 MB 限制")

        return bytes(body), response.headers.get("Content-Type", "")


def _web(query: str) -> tuple[list[dict], list[str]]:
    result = execute_tool(
        "search_web",
        {"query": query, "max_results": 3},
    )
    if not result["success"]:
        raise RuntimeError(str(result["error"]))

    documents = []
    warnings = []

    for item in result["data"]["results"][:3]:
        url = item["url"]
        content = item.get("content", "")
        source_type = "web_summary"

        try:
            parsed = urlsplit(url)
            if (
                parsed.scheme != "https"
                or parsed.hostname not in WEB_HOSTS
                or parsed.port not in {None, 443}
                or parsed.username
                or parsed.password
            ):
                raise ValueError("正文读取只允许配置中的 HTTPS 站点")

            body, content_type = _request(url)
            if not any(
                kind in content_type.lower()
                for kind in (
                    "text/html",
                    "application/xhtml+xml",
                    "text/plain",
                )
            ):
                raise ValueError("响应不是 HTML 或文本")

            soup = BeautifulSoup(body, "html.parser")
            for tag in soup([
                "script", "style", "nav", "footer", "header",
            ]):
                tag.decompose()

            content = soup.get_text(separator="\n", strip=True)
            if not content:
                raise ValueError("网页正文为空")

            source_type = "web_text"

        except Exception as error:
            warnings.append(
                f"{url}: 回退搜索摘要，{error}"
            )

        documents.append({
            "title": item.get("title") or url,
            "url": url,
            "content": content,
            "source_type": source_type,
            "metadata": {"query": query},
        })

    return documents, warnings


def _paper_request(params) -> tuple[bytes, str]:
    global _ARXIV_LAST

    for attempt in range(2):
        try:
            with _ARXIV_LOCK:
                time.sleep(
                    max(0, 3 - (time.monotonic() - _ARXIV_LAST))
                )

                try:
                    return _request(
                        "https://export.arxiv.org/api/query",
                        params=params,
                        timeout=(10, 20),
                    )
                finally:
                    _ARXIV_LAST = time.monotonic()

        except requests.exceptions.SSLError:
            raise

        except (
            requests.Timeout,
            requests.ConnectionError,
            requests.exceptions.ChunkedEncodingError,
        ):
            if attempt == 1:
                raise

            print("[Paper] 临时网络错误，重试 1/1")


def _paper(query: str) -> tuple[list[dict], list[str]]:
    match = re.fullmatch(
        r"(?:arxiv:|https?://(?:www\.)?arxiv\.org/(?:abs|pdf)/)?"
        r"((?:\d{4}\.\d{4,5}|[a-z-]+(?:\.[a-z]{2})?/\d{7})"
        r"(?:v[1-9]\d*)?)(?:\.pdf)?/?",
        query.strip(),
        re.IGNORECASE,
    )
    paper_id = match.group(1) if match else None

    if paper_id:
        params = {
            "id_list": paper_id,
            "max_results": 1,
        }
    else:
        words = re.findall(r"[A-Za-z0-9_-]+", query)[:6]
        if not words:
            raise ValueError(
                "arXiv 查询需包含英文关键词或论文 ID"
            )

        params = {
            "search_query": " AND ".join(
                f'all:"{word}"' for word in words
            ),
            "start": 0,
            "max_results": 3,
        }

    body, _ = _paper_request(params)

    atom = "{http://www.w3.org/2005/Atom}"
    documents, warnings = [], []

    for entry in ET.fromstring(body).findall(atom + "entry")[:3]:
        def value(name):
            return " ".join(
                entry.findtext(atom + name, "").split()
            )

        url = value("id")
        if "/api/errors" in url:
            raise RuntimeError(value("summary"))

        parsed = urlsplit(url)
        if (
            parsed.hostname != "arxiv.org"
            or not parsed.path.startswith("/abs/")
        ):
            raise ValueError("arXiv 返回的论文地址无效")

        actual_id = parsed.path[len("/abs/"):]

        if paper_id:
            expected = paper_id.lower()
            actual = actual_id.lower()

            if (
                re.sub(r"v\d+$", "", actual)
                != re.sub(r"v\d+$", "", expected)
                or (
                    re.search(r"v\d+$", expected)
                    and actual != expected
                )
            ):
                raise ValueError(
                    "arXiv 返回的论文 ID 与指定目标不一致"
                )

        title = value("title")
        abstract = value("summary")

        if not title or not abstract:
            warnings.append(
                f"{url}: 标题或摘要为空，已跳过"
            )
            continue

        documents.append({
            "title": title,
            "url": "https://arxiv.org" + parsed.path,
            "content": (
                f"arXiv: {actual_id}\n"
                f"{title}\n{abstract}"
            ),
            "source_type": "paper_abstract",
            "metadata": {
                "authors": [
                    author.findtext(atom + "name", "")
                    for author in entry.findall(atom + "author")
                ][:10],
                "published": value("published"),
                "updated": value("updated"),
                "query": query,
            },
        })

    if not documents:
        warnings.append("arXiv 未返回有效论文摘要")

    return documents, warnings


def _github(query: str) -> tuple[list[dict], list[str]]:
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2026-03-10",
    }

    token = os.getenv("GITHUB_TOKEN", "").strip()
    if token:
        headers["Authorization"] = f"Bearer {token}"

    name_pattern = r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+"
    match = re.fullmatch(
        rf"(?:repo:)?({name_pattern})",
        query.strip(),
    )
    target = match.group(1) if match else None
    warnings = []

    if target:
        if any(
            part in {".", ".."}
            for part in target.split("/")
        ):
            raise ValueError("仓库名称无效")

        body, _ = _request(
            f"https://api.github.com/repos/{target}",
            headers=headers,
        )
        repositories = [json.loads(body)]

    else:
        body, _ = _request(
            "https://api.github.com/search/repositories",
            params={
                "q": query + " is:public",
                "per_page": 3,
                "page": 1,
            },
            headers=headers,
        )

        payload = json.loads(body)
        repositories = payload.get("items", [])[:3]

        if payload.get("incomplete_results"):
            warnings.append("GitHub 搜索结果不完整")

    documents = []

    for repo in repositories:
        name = repo.get("full_name", "")

        if (
            not re.fullmatch(name_pattern, name)
            or any(
                part in {".", ".."}
                for part in name.split("/")
            )
        ):
            raise ValueError("GitHub 返回的仓库名称无效")

        if repo.get("private"):
            raise ValueError("本版只采集公开仓库")

        if target and name.lower() != target.lower():
            raise ValueError(
                "GitHub 返回的仓库与指定目标不一致"
            )

        metadata = {
            "repository": name,
            "stars": repo.get("stargazers_count"),
            "language": repo.get("language"),
            "license": (
                repo.get("license") or {}
            ).get("spdx_id"),
            "updated_at": repo.get("updated_at"),
            "query": query,
        }

        content = (
            f"Repository: {name}\n"
            f"Description: {repo.get('description') or ''}\n"
            f"Language: {metadata['language'] or ''}\n"
            f"License: {metadata['license'] or ''}"
        )
        url = f"https://github.com/{name}"
        source_type = "github_metadata"

        try:
            raw, _ = _request(
                f"https://api.github.com/repos/{name}/readme",
                headers=headers,
            )
            readme = json.loads(raw)

            if readme.get("encoding") != "base64":
                raise ValueError("README 编码暂不支持")

            text = base64.b64decode(
                "".join(readme["content"].split()),
                validate=True,
            ).decode("utf-8", errors="replace").strip()

            if not text:
                raise ValueError("README 正文为空")

            content += "\nREADME:\n" + text
            metadata["readme_sha"] = readme.get("sha")
            url = readme.get("html_url") or url
            source_type = "github_readme"

        except Exception as error:
            warnings.append(
                f"{name}: 仅使用仓库元数据，{error}"
            )

        documents.append({
            "title": name,
            "url": url,
            "content": content,
            "source_type": source_type,
            "metadata": metadata,
        })

    return documents, warnings


def collect(
    source: str,
    query: str,
) -> tuple[list[dict], list[str]]:
    return {
        "web": _web,
        "paper": _paper,
        "github": _github,
    }[source](query)
