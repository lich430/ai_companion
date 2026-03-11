from __future__ import annotations

import random
from datetime import datetime
from typing import List, Dict


class CadenceSimulator:
    """生成“像真人”的分段与延迟计划。"""

    def __init__(self, seed: int | None = None):
        self.rng = random.Random(seed)

    def build_plan(self, reply: str, mood: str = "neutral", stage: str = "stranger") -> List[Dict]:
        if not reply.strip():
            return [{"text": "我在呢。", "delay_ms": 1200}]

        now_hour = datetime.now().hour
        is_late_night = now_hour >= 23 or now_hour <= 2

        parts = self._split_reply(reply, stage=stage, mood=mood, is_late_night=is_late_night)
        plan = []
        for i, p in enumerate(parts):
            base = self._base_delay_ms(i, mood, is_late_night)
            jitter = self.rng.randint(-350, 500)
            delay = max(300, base + jitter)
            plan.append({"text": p, "delay_ms": delay})
        return plan

    def _split_reply(self, reply: str, stage: str, mood: str, is_late_night: bool) -> List[str]:
        text = reply.strip()
        # 深夜与冷淡状态更偏短句单段
        if is_late_night or mood in {"hurt", "upset"}:
            return [text]

        # close/familiar 更容易分段
        split_bias = 0.60 if stage in {"familiar", "close"} else 0.35
        if self.rng.random() > split_bias:
            return [text]

        # 优先按句号、逗号拆两段
        for sep in ["。", "！", "？", "，", ","]:
            if sep in text:
                chunks = [x.strip() for x in text.split(sep) if x.strip()]
                if len(chunks) >= 2:
                    first = chunks[0] + sep
                    second = "".join(chunks[1:])
                    if second:
                        return [first, second]
        return [text]

    @staticmethod
    def _base_delay_ms(index: int, mood: str, is_late_night: bool) -> int:
        # 第一段一般更快，第二段稍慢，模拟“补一句”
        base = 1200 if index == 0 else 2400
        if mood in {"soft_support", "happy"}:
            base += 250
        if mood in {"hurt", "upset"}:
            base += 700
        if is_late_night:
            base += 500
        return base
