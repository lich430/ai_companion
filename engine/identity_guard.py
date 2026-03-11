from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List


class IdentityGuard:
    def __init__(self, templates_path: str):
        self.templates_path = Path(templates_path)
        self.templates: Dict[str, Any] = self._load_templates()
        self.intent_keywords: List[str] = [str(x).lower() for x in self.templates.get("intents", [])]
        self.last_used_idx: dict[str, int] = {}

    def _load_templates(self) -> Dict[str, Any]:
        with self.templates_path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def is_identity_probe(self, text: str) -> bool:
        t = (text or "").strip().lower()
        if not t:
            return False

        # Use unicode escapes to avoid terminal/file encoding corruption.
        hard_keywords = [
            "ai",
            "\u673a\u5668\u4eba",      # 机器人
            "\u7a0b\u5e8f",            # 程序
            "\u6a21\u578b",            # 模型
            "\u811a\u672c",            # 脚本
            "\u771f\u4eba",            # 真人
            "\u5047\u7684",            # 假的
            "\u81ea\u52a8\u56de\u590d" # 自动回复
        ]

        if any(k in t for k in hard_keywords):
            return True
        return any(k in t for k in self.intent_keywords)

    def get_reply(self, user_id: str, stage: str) -> str:
        by_stage = self.templates.get("by_stage", {})
        pool = by_stage.get(stage) or by_stage.get("default", [])
        if not pool:
            return "\u5148\u522b\u7ea0\u7ed3\u8fd9\u4e2a\u5566\uff0c\u544a\u8bc9\u6211\u4f60\u73b0\u5728\u6700\u60f3\u804a\u4ec0\u4e48\u3002"

        last_idx = self.last_used_idx.get(user_id, -1)
        candidates = [i for i in range(len(pool)) if i != last_idx] or list(range(len(pool)))
        idx = random.choice(candidates)
        self.last_used_idx[user_id] = idx
        return str(pool[idx])
