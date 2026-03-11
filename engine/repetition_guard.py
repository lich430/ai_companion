from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import List, Optional

from engine.embedding_service import EmbeddingService


@dataclass
class ReplyRecord:
    text: str
    emb: Optional[List[float]] = None


class RepetitionGuard:
    def __init__(
        self,
        embedding_service: Optional[EmbeddingService] = None,
        lexical_threshold: float = 0.72,
        semantic_threshold: float = 0.90,
        window_size: int = 20,
    ):
        self.embedding_service = embedding_service
        self.lexical_threshold = lexical_threshold
        self.semantic_threshold = semantic_threshold
        self.window_size = window_size
        self.reply_cache: dict[str, List[ReplyRecord]] = {}

    @staticmethod
    def lexical_similarity(a: str, b: str) -> float:
        return SequenceMatcher(None, a, b).ratio()

    def semantic_similarity(self, a: str, b_rec: ReplyRecord) -> float:
        if not self.embedding_service or not b_rec.emb:
            return 0.0
        a_vec = self.embedding_service.embed(a)
        return self.embedding_service.cosine(a_vec, b_rec.emb)

    def is_repetitive(self, user_id: str, new_reply: str) -> bool:
        history = self.reply_cache.get(user_id, [])
        if not history:
            return False

        for rec in history[-8:]:
            lex = self.lexical_similarity(rec.text, new_reply)
            if lex >= self.lexical_threshold:
                return True

            if len(rec.text) >= 6 and new_reply.startswith(rec.text[:6]):
                return True

            sem = self.semantic_similarity(new_reply, rec)
            if sem >= self.semantic_threshold:
                return True
        return False

    def add_reply(self, user_id: str, reply: str):
        emb = self.embedding_service.embed(reply) if self.embedding_service else None
        self.reply_cache.setdefault(user_id, []).append(ReplyRecord(text=reply, emb=emb))
        self.reply_cache[user_id] = self.reply_cache[user_id][-self.window_size :]
