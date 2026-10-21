import json
import os
import re
from pathlib import Path

class ContextBudgetError(ValueError):
    pass

def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

def _words(text):
    return set(re.findall(r"[a-z0-9_]+|[\u4e00-\u9fff]", text.lower()))

class TokenContext:
    """明确编码的 Token 预算估算；不是 DeepSeek 服务端精确计量。"""
    def __init__(self, limit=16000, reserve=4096, encoding=None):
        if type(limit) is not int or type(reserve) is not int or not 0 < reserve < limit:
            raise ValueError("必须满足 0 < reserve < limit")
        if encoding is None:
            import tiktoken
            os.environ.setdefault(
                "TIKTOKEN_CACHE_DIR",
                str(Path(__file__).resolve().parents[1] / "data" / "models" / "tiktoken"),
            )
            encoding = tiktoken.get_encoding("cl100k_base")
        self.encoding, self.limit, self.reserve = encoding, limit, reserve

    def count(self, value):
        return len(self.encoding.encode(_json(value), disallowed_special=()))

    def cost(self, messages, tools=()):
        return self.count({"messages": messages, "tools": list(tools)}) + 8 * len(messages) + 32

    def excerpt(self, text, goal, maximum=160):
        """抽取相关句子；超长句按 Token 边界取前缀，返回原文范围。"""
        spans = [(m.start(), m.end()) for m in re.finditer(
            r".+?(?:[。！？!?\n]|(?<=\.)\s+|$)", text, re.S
        )]
        terms = _words(goal)
        ranked = sorted(spans, key=lambda span: (
            -len(terms & _words(text[slice(*span)])), span[0]
        ))
        selected, used = [], 0
        for start, end in ranked:
            part = text[start:end]
            tokens = self.encoding.encode(part, disallowed_special=())
            if len(tokens) > maximum - used:
                if selected:
                    continue
                part = self.encoding.decode(tokens[:maximum]).rstrip("\ufffd")
                while part and not text[start:end].startswith(part):
                    part = part[:-1]
                end = start + len(part)
            cost = len(self.encoding.encode(part, disallowed_special=()))
            if part and used + cost <= maximum:
                selected.append((start, end))
                used += cost
        selected.sort()
        return "\n".join(text[start:end] for start, end in selected), selected

    def _tool_summary(self, content, goal, maximum=512):
        """压缩正文时保留真实搜索/网页结果的来源，不把 JSON 片段当作来源。"""
        excerpt, _ = self.excerpt(content, goal, maximum)
        summary = {
            "extractive_summary": excerpt,
            "evidence_ids": sorted(set(re.findall(r"D\d+C\d+", content))),
        }
        try:
            value = json.loads(content)
        except (json.JSONDecodeError, TypeError):
            return summary
        if not isinstance(value, dict):
            return summary
        data = value.get("data", value)
        if not isinstance(data, dict):
            return summary
        if isinstance(data.get("results"), list):
            items = data["results"]
        elif isinstance(data.get("url"), str):
            items = [data]
        else:
            return summary
        sources = []
        for item in items:
            if not isinstance(item, dict) or not isinstance(item.get("url"), str):
                continue
            source = {"url": item["url"]}
            if isinstance(item.get("title"), str):
                source["title"] = item["title"]
            text = item.get("content", "")
            if isinstance(text, str):
                snippet, spans = self.excerpt(text, goal, max(16, maximum // max(1, len(items))))
                source.update(extractive_summary=snippet, context_ranges=spans)
            sources.append(source)
        if sources:
            # 全部来源元数据都参与预算；放不下时由 pack 明确报错。
            summary["sources"] = sources
            summary["extractive_summary"] = "按 sources 中的原文摘录及来源对应阅读。"
            if "success" in value:
                summary["success"] = value["success"]
            if value.get("error") is not None:
                summary["error"] = value["error"]
        return summary

    @staticmethod
    def _history(history):
        pending, seen = set(), set()
        for message in history:
            role = message.get("role")
            if role not in {"user", "assistant", "tool"}:
                raise ValueError("history 仅接受 user/assistant/tool")
            if role == "tool":
                ident = message.get("tool_call_id")
                if not isinstance(ident, str) or ident not in pending:
                    raise ValueError("发现孤立 tool 结果")
                pending.remove(ident)
            else:
                if pending:
                    raise ValueError("tool_call 缺少结果")
                for call in message.get("tool_calls", []):
                    ident = call.get("id")
                    if role != "assistant" or not isinstance(ident, str) or ident in seen:
                        raise ValueError("无效或重复 tool_call")
                    pending.add(ident)
                    seen.add(ident)
        if pending:
            raise ValueError("tool_call 缺少结果")
        index = next((i for i in range(len(history) - 1, -1, -1)
                      if history[i]["role"] == "user"), len(history))
        return history[:index], history[index:]

    def pack(self, goal, *, system="", payload=None, history=(), evidence=(),
             facts=(), unresolved=(), memories=(), tools=()):
        if not isinstance(goal, str) or not goal.strip():
            raise ValueError("goal 不能为空")
        if (not isinstance(facts, (list, tuple)) or not isinstance(unresolved, (list, tuple))
                or any(not isinstance(value, str) for value in (*facts, *unresolved))):
            raise ValueError("facts/unresolved 必须是字符串序列")
        older, latest = self._history(json.loads(_json(list(history))))
        for message in latest:
            if message["role"] == "tool" and isinstance(message.get("content"), str):
                content = message["content"]
                if self.count(content) > 512:
                    message["content"] = _json(self._tool_summary(content, goal))
        data = json.loads(_json(payload or {}))
        data.update(question=goal, evidence=[], memory=[], context_summary={
            "facts": list(dict.fromkeys(facts)),
            "unresolved": list(dict.fromkeys(unresolved)), "history": [],
        })

        def messages():
            return [{"role": "system", "content": system}, *latest,
                    {"role": "user", "content": _json(data)}]

        def fits():
            return self.cost(messages(), tools) + self.reserve <= self.limit

        if not fits():
            raise ContextBudgetError("目标、事实、未决问题或最新对话超出输入预算")
        evidence = list(evidence)
        current_ids = {item.get("evidence_id") for item in evidence if isinstance(item, dict)}
        required = set(re.findall(r"D\d+C\d+", " ".join(facts)))
        required.update(set(re.findall(r"D\d+C\d+", " ".join(unresolved))) & current_ids)
        dropped, seen, seen_content = [], {}, set()
        for item in evidence:
            if not isinstance(item, dict):
                raise ValueError("evidence 项必须为对象")
            ident, content = item.get("evidence_id"), item.get("content")
            if not isinstance(ident, str) or not isinstance(content, str):
                raise ValueError("evidence 必须包含 evidence_id/content")
            if ident in seen:
                if seen[ident] != item:
                    raise ValueError("同一证据编号对应不同内容")
                continue
            seen[ident] = item

        if not required <= seen.keys():
            raise ContextBudgetError("关键事实所引用的证据未能纳入预算")
        chosen = {}

        def select(ident, maximum):
            item = seen[ident]
            text, spans = self.excerpt(item["content"], goal, maximum)
            candidate = {key: item[key] for key in (
                "evidence_id", "document_id", "title", "url", "source_type",
                "chapter", "metadata", "start", "end"
            ) if key in item}
            candidate.update(content=text, context_ranges=spans)
            previous = chosen.get(ident)
            chosen[ident] = candidate
            data["evidence"] = [chosen[key] for key in seen if key in chosen]
            if fits():
                return True
            if previous is None:
                del chosen[ident]
            else:
                chosen[ident] = previous
            data["evidence"] = [chosen[key] for key in seen if key in chosen]
            return False

        # 先给所有必需引用留位置，避免可选证据或前一个长摘要挤掉后续引用。
        for ident in seen:
            if ident in required and not select(ident, 16):
                raise ContextBudgetError("关键事实所引用的证据未能纳入预算")
        for ident in seen:
            if ident in required:
                for maximum in (160, 64):
                    if select(ident, maximum):
                        break
                item = seen[ident]
                seen_content.add((item.get("source_type"), item["content"]))
        for ident, item in seen.items():
            if ident in required:
                continue
            content = item["content"]
            fingerprint = (item.get("source_type"), content)
            if fingerprint in seen_content:
                dropped.append(ident)
                continue
            accepted = False
            for maximum in (160, 64, 16):
                if select(ident, maximum):
                    accepted = True
                    seen_content.add(fingerprint)
                    break
            if not accepted:
                dropped.append(ident)
        if not required <= {item["evidence_id"] for item in data["evidence"]}:
            raise ContextBudgetError("关键事实所引用的证据未能纳入预算")
        for memory in memories:
            data["memory"].append(memory)
            if not fits():
                data["memory"].pop()
        seen_text = set()
        for message in reversed(older):
            content = message.get("content")
            if not isinstance(content, str) or content in seen_text:
                continue
            seen_text.add(content)
            text, _ = self.excerpt(content, goal, 96)
            row = {"role": message["role"], "excerpt": text,
                   "evidence_ids": sorted(set(re.findall(r"D\d+C\d+", content)))}
            if message["role"] == "tool":
                source_summary = self._tool_summary(content, goal, 96)
                if "sources" in source_summary:
                    row["sources"] = source_summary["sources"]
            data["context_summary"]["history"].insert(0, row)
            if not fits():
                data["context_summary"]["history"].pop(0)
        output = messages()
        return {"messages": output, "evidence": data["evidence"],
                "summary": data["context_summary"], "dropped_evidence_ids": dropped,
                "estimated_input_tokens": self.cost(output, tools),
                "reserved_output_tokens": self.reserve,
                "encoding": getattr(self.encoding, "name", "injected")}

    def chat(self, messages, *, tools=()):
        system = "\n\n".join(
            item["content"] for item in messages if item["role"] == "system"
        )
        history = [item for item in messages if item["role"] != "system"]
        goal = next((item["content"] for item in reversed(history)
                     if item["role"] == "user"), None)
        if not isinstance(goal, str):
            raise ValueError("聊天上下文缺少 user 消息")
        return self.pack(goal, system=system, history=history, tools=tools)["messages"]
