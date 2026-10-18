import math
import re
from collections import Counter, defaultdict


def split_text(
    text: str,
    size: int = 800,
    overlap: int = 100,
) -> list[dict]:
    if not 0 <= overlap < size:
        raise ValueError("必须满足 0 <= overlap < size")

    chunks = []
    for start in range(0, len(text), size - overlap):
        end = min(start + size, len(text))
        chunks.append({
            "index": len(chunks) + 1,
            "start": start,
            "end": end,
            "content": text[start:end],
        })

        if end == len(text):
            break

    return chunks


def tokenize(text: str) -> list[str]:
    result = re.findall(r"[a-z0-9_]+", text.lower())

    for word in re.findall(r"[\u4e00-\u9fff]+", text):
        result.extend(
            [word] if len(word) == 1
            else [
                word[i:i + 2]
                for i in range(len(word) - 1)
            ]
        )

    return result


def retrieve(
    query: str,
    chunks: list[dict],
    top_k: int = 6,
) -> list[dict]:
    """BM25 检索；每篇文档最多返回两段，不返回零分资料。"""
    query_terms = set(tokenize(query))
    if not chunks or not query_terms or top_k <= 0:
        return []

    counts = [
        Counter(tokenize(item["title"] + " " + item["content"]))
        for item in chunks
    ]
    lengths = [sum(count.values()) for count in counts]
    average = sum(lengths) / len(lengths)

    if average == 0:
        return []

    frequency = Counter(
        term for count in counts for term in count
    )
    ranked = []

    for item, count, length in zip(chunks, counts, lengths):
        score = 0.0

        for term in query_terms:
            tf = count[term]
            if tf:
                df = frequency[term]
                idf = math.log(
                    1 + (len(chunks) - df + 0.5) / (df + 0.5)
                )
                score += idf * tf * 2.5 / (
                    tf + 1.5 * (0.25 + 0.75 * length / average)
                )

        if score > 0:
            ranked.append({
                **item,
                "score": round(score, 6),
            })

    ranked.sort(key=lambda item: (
        -item["score"],
        item["document_id"],
        item["chunk_index"],
    ))

    selected = []
    per_document = defaultdict(int)

    for item in ranked:
        if per_document[item["document_id"]] >= 2:
            continue

        selected.append(item)
        per_document[item["document_id"]] += 1

        if len(selected) == top_k:
            break

    return selected