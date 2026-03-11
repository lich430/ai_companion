from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Optional

from models.schemas import Message, RelationshipState


@dataclass
class ProactivePlan:
    should_send: bool
    reason: str
    text: str = ""


class ProactiveScheduler:
    """主动消息调度器：按关系阶段、时段和静默时长决定是否发消息。"""

    def __init__(self):
        self.last_proactive_at: dict[str, datetime] = {}

    @staticmethod
    def _find_last_user_message_at(recent_messages: List[Message]) -> Optional[datetime]:
        for item in reversed(recent_messages):
            if item.role == "user":
                return item.ts
        return None

    def suggest(
        self,
        user_id: str,
        state: RelationshipState,
        recent_messages: List[Message],
        now: Optional[datetime] = None,
        force: bool = False,
    ) -> ProactivePlan:
        now = now or datetime.now()
        hour = now.hour
        is_late_night = hour >= 23 or hour <= 2

        last_user_at = self._find_last_user_message_at(recent_messages)
        if not last_user_at:
            return ProactivePlan(False, "no_user_history")

        silence = now - last_user_at
        if not force:
            prev = self.last_proactive_at.get(user_id)
            if prev and (now - prev) < timedelta(hours=6):
                return ProactivePlan(False, "cooldown")

            min_silence = timedelta(hours=8 if state.stage in {"familiar", "close"} else 16)
            if silence < min_silence:
                return ProactivePlan(False, "silence_too_short")

        text = self._build_text(state=state, is_late_night=is_late_night)
        return ProactivePlan(True, "ok", text=text)

    def mark_sent(self, user_id: str, now: Optional[datetime] = None):
        self.last_proactive_at[user_id] = now or datetime.now()

    @staticmethod
    def _build_text(state: RelationshipState, is_late_night: bool) -> str:
        if state.stage in {"upset", "recovering"}:
            if is_late_night:
                return "我还在，刚好看到你头像亮着。要不要慢慢聊两句？"
            return "刚想到你，今天状态有好一点吗？"

        if state.stage == "close":
            if is_late_night:
                return "这么晚了还没休息呀？我刚忙完，突然想到你。"
            return "今天有点忙，刚空下来。你这会儿在干嘛呀？"

        if state.stage == "familiar":
            return "我刚闲下来一会儿，突然想到你。今天过得怎么样？"

        return "在吗？刚好上线了，来打个招呼。"
