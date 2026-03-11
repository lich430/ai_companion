from __future__ import annotations

from collections import deque
from datetime import UTC, datetime

from models.schemas import Message, RelationshipState


class RelationshipStateMachine:
    def __init__(self, bible: dict):
        self.bible = bible
        self.event_windows: dict[str, deque[str]] = {}

    def apply_stage_defaults(self, state: RelationshipState) -> RelationshipState:
        defaults = self.bible["relationship_stages"].get(state.stage, {})
        state.warmth = defaults.get("warmth", state.warmth)
        state.initiative = defaults.get("initiative", state.initiative)
        state.disclosure = defaults.get("disclosure", state.disclosure)
        state.flirty = defaults.get("flirty", state.flirty)
        state.last_updated = datetime.now(UTC)
        return state

    def update_from_message(
        self,
        state: RelationshipState,
        latest_user_message: str,
        recent_messages: list[Message],
    ) -> RelationshipState:
        event = self._detect_event(latest_user_message)
        self._push_event(state.user_id, event)
        self._apply_event_transition(state, event, recent_messages)
        return self.apply_stage_defaults(state)

    def _push_event(self, user_id: str, event: str):
        window = self.event_windows.setdefault(user_id, deque(maxlen=8))
        window.append(event)

    def _count_recent(self, user_id: str, event: str, n: int = 4) -> int:
        window = list(self.event_windows.get(user_id, []))
        return sum(1 for x in window[-n:] if x == event)

    @staticmethod
    def _detect_event(text: str) -> str:
        t = (text or "").strip()
        if any(k in t for k in ["\u60f3\u4f60", "\u5728\u5417", "\u60f3\u627e\u4f60", "\u4f60\u8fd8\u5728"]):
            return "affection"
        if any(k in t for k in ["\u7d2f", "\u70e6", "\u96be\u53d7", "\u5931\u7720", "\u7126\u8651"]):
            return "distress"
        if any(k in t for k in ["\u7b97\u4e86", "\u968f\u4fbf", "\u4e0d\u804a\u4e86", "\u54e6", "\u6eda"]):
            return "cold_or_conflict"
        if any(k in t for k in ["\u5bf9\u4e0d\u8d77", "\u6211\u9519\u4e86", "\u522b\u751f\u6c14", "\u54c4\u54c4\u4f60"]):
            return "repair"
        if any(k in t for k in ["\u54c8\u54c8", "\u597d\u73a9", "\u6709\u610f\u601d", "\u4f60\u771f\u4f1a"]):
            return "engagement"
        if any(k in t for k in ["\u4f60\u662f\u4e0d\u662fAI", "\u673a\u5668\u4eba", "\u7a0b\u5e8f", "\u6a21\u578b", "\u771f\u4eba\u5417", "\u5047\u7684\u5427"]):
            return "identity_probe"
        return "neutral"

    def _apply_event_transition(self, state: RelationshipState, event: str, recent_messages: list[Message]):
        user_id = state.user_id
        conflict_policy = self.bible.get("conflict_policy") or {}
        never_upset = bool(conflict_policy.get("never_upset", False))
        conflict_mood = str(conflict_policy.get("conflict_mood", "neutral") or "neutral")

        if event == "distress":
            state.mood = "soft_support"
            state.warmth = min(1.0, state.warmth + 0.05)

        if event == "affection":
            state.mood = "happy"
            state.warmth = min(1.0, state.warmth + 0.08)

        if event == "cold_or_conflict":
            if never_upset:
                state.mood = conflict_mood
                state.initiative = max(0.10, state.initiative - 0.03)
            else:
                state.mood = "hurt"
                state.stage = "upset"

        if event == "repair":
            state.mood = "recovering"
            if state.stage in {"upset", "recovering"}:
                state.stage = "recovering"
            state.warmth = min(1.0, state.warmth + 0.05)

        if event == "identity_probe":
            state.initiative = max(0.10, state.initiative - 0.05)

        affection_recent = self._count_recent(user_id, "affection", 5)
        engage_recent = self._count_recent(user_id, "engagement", 5)
        conflict_recent = self._count_recent(user_id, "cold_or_conflict", 4)
        repair_recent = self._count_recent(user_id, "repair", 4)

        user_msg_count = sum(1 for m in recent_messages if m.role == "user")

        if conflict_recent >= 2:
            if never_upset:
                state.mood = conflict_mood
                state.initiative = max(0.10, state.initiative - 0.03)
            else:
                state.stage = "upset"
                state.mood = "hurt"
            return

        if state.stage == "upset" and repair_recent >= 1:
            state.stage = "recovering"
            return

        if state.stage == "recovering" and repair_recent >= 1 and conflict_recent == 0:
            state.stage = "familiar"
            state.mood = "neutral"
            return

        if state.stage == "stranger" and user_msg_count >= 8 and (affection_recent + engage_recent) >= 2:
            state.stage = "familiar"
            return

        if state.stage == "familiar" and user_msg_count >= 20 and affection_recent >= 2 and conflict_recent == 0:
            state.stage = "close"
