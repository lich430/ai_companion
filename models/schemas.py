from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import List, Optional


@dataclass
class Message:
    role: str
    content: str
    ts: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass
class UserProfile:
    user_id: str
    nickname: Optional[str] = None
    preferred_style: Optional[str] = None
    active_hours: Optional[str] = None
    sensitive_topics: List[str] = field(default_factory=list)
    recurring_topics: List[str] = field(default_factory=list)


@dataclass
class RelationshipState:
    user_id: str
    stage: str = "stranger"
    mood: str = "neutral"
    energy: float = 0.60
    warmth: float = 0.40
    initiative: float = 0.20
    disclosure: float = 0.10
    flirty: float = 0.05
    last_updated: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass
class MemoryItem:
    user_id: str
    category: str
    content: str
    importance: float = 0.5
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass
class ResponseContext:
    user_id: str
    latest_user_message: str
    recent_messages: List[Message]
    profile: UserProfile
    state: RelationshipState
    recalled_memories: List[MemoryItem]
    role_life_memories: List[MemoryItem]


@dataclass
class ReplyDecision:
    type: str  # "reply" | "noreply"
    text: str = ""
    recommended_delay_ms: int = 0
    reason: str = ""
