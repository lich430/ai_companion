from __future__ import annotations

import json
import re
from typing import List

from engine.response_generator import GLMResponseGenerator
from models.schemas import MemoryItem, Message, RelationshipState


class MemorySummarizer:
    def __init__(self, generator: GLMResponseGenerator, bible: dict | None = None):
        self.generator = generator
        self.bible = bible or {}
        self.role_name = str(self.bible.get("name", "") or "").strip() or "assistant"

    def summarize(
        self,
        user_id: str,
        recent_messages: List[Message],
        state: RelationshipState,
    ) -> List[MemoryItem]:
        dialogue = self._format_dialogue(recent_messages[-16:], self.role_name)

        persona = str(self.bible.get("persona_summary", "")).strip()
        boundaries = self.bible.get("boundaries") or []
        avoid = (self.bible.get("speech_style") or {}).get("avoid") or []
        marketing = self.bible.get("marketing") or {}
        marketing_triggers = marketing.get("triggers") or []

        system_prompt = f"""
你是“聊天记忆压缩器”。请把对话压缩成可长期存储的记忆，只输出 JSON，不要输出额外说明。
JSON 结构：
{{
  "user_profile": ["..."],
  "relationship_event": ["..."],
  "role_life": ["..."]
}}
规则：
1. 每个数组最多 3 条，短句，避免重复。
2. user_profile 只保留稳定偏好和习惯。
3. relationship_event 保留关系节点或情绪事件。
4. role_life 保留角色最近提到的生活轨迹。
5. 必须遵守角色圣经，不能生成与人设冲突的记忆，不要凭空捏造具体经历。

角色摘要：
{persona}

边界规则：
{"；".join(str(x) for x in boundaries) if boundaries else "无"}

表达禁区：
{"；".join(str(x) for x in avoid) if avoid else "无"}

营销限制：
- 只有用户明确触发以下词时，才允许保留营销相关记忆：{"、".join(str(x) for x in marketing_triggers) if marketing_triggers else "无"}
- 未触发时，不要主动写入店名/电话/价格/玩法类记忆
"""
        user_prompt = f"""
当前关系阶段：{state.stage}
当前情绪：{state.mood}

最近对话：
{dialogue}
"""

        try:
            raw = self.generator.chat(system_prompt, user_prompt, temperature=0.2)
            data = self._extract_json(raw)
            return self._to_memory_items(user_id, data)
        except Exception:
            return self._fallback(user_id, recent_messages)

    @staticmethod
    def _format_dialogue(messages: List[Message], role_name: str = "assistant") -> str:
        lines = []
        for item in messages:
            role = "用户" if item.role == "user" else role_name
            lines.append(f"{role}: {item.content}")
        return "\n".join(lines)

    @staticmethod
    def _extract_json(raw: str) -> dict:
        match = re.search(r"\{[\s\S]*\}", raw)
        if not match:
            raise ValueError("no-json")
        return json.loads(match.group(0))

    @staticmethod
    def _to_memory_items(user_id: str, data: dict) -> List[MemoryItem]:
        out: List[MemoryItem] = []
        mapping = {
            "user_profile": ("user_profile", 0.75),
            "relationship_event": ("relationship_event", 0.80),
            "role_life": ("role_life", 0.55),
        }
        for key, (category, importance) in mapping.items():
            values = data.get(key, [])
            if isinstance(values, list):
                for value in values[:3]:
                    text = str(value).strip()
                    if text:
                        out.append(
                            MemoryItem(
                                user_id=user_id,
                                category=category,
                                content=text,
                                importance=importance,
                            )
                        )
        return out

    @staticmethod
    def _fallback(user_id: str, recent_messages: List[Message]) -> List[MemoryItem]:
        out: List[MemoryItem] = []
        for item in reversed(recent_messages[-10:]):
            text = item.content.strip()
            if item.role == "user" and any(k in text for k in ["喜欢", "习惯", "平时", "总是"]):
                out.append(
                    MemoryItem(
                        user_id=user_id,
                        category="user_profile",
                        content=f"用户偏好：{text}",
                        importance=0.7,
                    )
                )
                break
        return out
