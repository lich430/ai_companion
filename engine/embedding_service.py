from __future__ import annotations

import hashlib
import json
import math
from typing import List, Sequence

import requests


class EmbeddingService:
    def __init__(self, api_key: str, model: str = "embedding-3", dim: int = 128):
        self.api_key = api_key
        self.model = model
        self.dim = dim
        self.url = "https://open.bigmodel.cn/api/paas/v4/embeddings"

    def embed(self, text: str) -> List[float]:
        text = (text or "").strip()
        if not text:
            return [0.0] * self.dim
        if not self.api_key:
            return self._hash_embed(text)

        # Prefer remote embedding, fallback to local hash embedding on failure.
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
            payload = {"model": self.model, "input": text}
            resp = requests.post(self.url, headers=headers, json=payload, timeout=30)
            resp.raise_for_status()
            body = resp.json()
            emb = body.get("data", [{}])[0].get("embedding")
            if isinstance(emb, list) and emb:
                return [float(x) for x in emb]
        except Exception:
            pass

        return self._hash_embed(text)

    def _hash_embed(self, text: str) -> List[float]:
        vec = [0.0] * self.dim
        tokens = self._char_ngrams(text, n=2) + self._char_ngrams(text, n=3)
        for tok in tokens:
            idx = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16) % self.dim
            vec[idx] += 1.0
        return self._normalize(vec)

    @staticmethod
    def _char_ngrams(text: str, n: int = 2) -> List[str]:
        if len(text) <= n:
            return [text]
        return [text[i : i + n] for i in range(len(text) - n + 1)]

    @staticmethod
    def _normalize(vec: Sequence[float]) -> List[float]:
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [float(v / norm) for v in vec]

    @staticmethod
    def cosine(a: Sequence[float], b: Sequence[float]) -> float:
        if not a or not b:
            return 0.0
        m = min(len(a), len(b))
        dot = sum(float(a[i]) * float(b[i]) for i in range(m))
        na = math.sqrt(sum(float(x) * float(x) for x in a[:m])) or 1.0
        nb = math.sqrt(sum(float(x) * float(x) for x in b[:m])) or 1.0
        return dot / (na * nb)

    @staticmethod
    def dumps(vec: Sequence[float]) -> str:
        return json.dumps(list(vec), ensure_ascii=False)

    @staticmethod
    def loads(raw: str) -> List[float]:
        try:
            data = json.loads(raw)
            if isinstance(data, list):
                return [float(x) for x in data]
        except Exception:
            pass
        return []
