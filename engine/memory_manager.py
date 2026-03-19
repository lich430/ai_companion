from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from typing import List, Optional

from engine.embedding_service import EmbeddingService
from models.schemas import MemoryItem, Message, RelationshipState


class MemoryManager:
    def __init__(self, db_path: str = "memory.db", embedding_service: Optional[EmbeddingService] = None):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.embedding_service = embedding_service
        self._init_tables()

    def _init_tables(self):
        cur = self.conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                category TEXT NOT NULL,
                content TEXT NOT NULL,
                embedding TEXT,
                importance REAL DEFAULT 0.5,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS chat_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                ts TEXT NOT NULL
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS user_states (
                user_id TEXT PRIMARY KEY,
                stage TEXT NOT NULL,
                mood TEXT NOT NULL,
                energy REAL NOT NULL,
                warmth REAL NOT NULL,
                initiative REAL NOT NULL,
                disclosure REAL NOT NULL,
                flirty REAL NOT NULL,
                last_updated TEXT NOT NULL
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS user_stats (
                user_id TEXT PRIMARY KEY,
                turn_count INTEGER NOT NULL DEFAULT 0,
                updated_at TEXT NOT NULL
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS proactive_state (
                user_id TEXT PRIMARY KEY,
                last_proactive_at TEXT NOT NULL
            )
            """
        )
        cur.execute("PRAGMA table_info(memories)")
        cols = {row[1] for row in cur.fetchall()}
        if "embedding" not in cols:
            cur.execute("ALTER TABLE memories ADD COLUMN embedding TEXT")
        self.conn.commit()

    def add_memory(self, item: MemoryItem):
        emb_raw = None
        if self.embedding_service:
            emb = self.embedding_service.embed(item.content)
            emb_raw = self.embedding_service.dumps(emb)

        cur = self.conn.cursor()
        cur.execute(
            """
            INSERT INTO memories (user_id, category, content, embedding, importance)
            VALUES (?, ?, ?, ?, ?)
            """,
            (item.user_id, item.category, item.content, emb_raw, item.importance),
        )
        self.conn.commit()

    def add_chat_message(self, user_id: str, role: str, content: str, ts: Optional[datetime] = None):
        ts_raw = (ts or datetime.now(UTC)).isoformat()
        cur = self.conn.cursor()
        cur.execute(
            """
            INSERT INTO chat_messages (user_id, role, content, ts)
            VALUES (?, ?, ?, ?)
            """,
            (user_id, role, content, ts_raw),
        )
        self.conn.commit()

    def get_recent_chat_messages(self, user_id: str, limit: int = 40) -> List[Message]:
        cur = self.conn.cursor()
        rows = cur.execute(
            """
            SELECT role, content, ts
            FROM chat_messages
            WHERE user_id = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (user_id, max(1, limit)),
        ).fetchall()
        rows.reverse()

        out: List[Message] = []
        for role, content, ts in rows:
            try:
                dt = datetime.fromisoformat(str(ts).replace("Z", ""))
            except Exception:
                dt = datetime.now(UTC)
            out.append(Message(role=role, content=content, ts=dt))
        return out

    def count_chat_messages(self, user_id: str) -> int:
        cur = self.conn.cursor()
        row = cur.execute(
            "SELECT COUNT(1) FROM chat_messages WHERE user_id = ?",
            (user_id,),
        ).fetchone()
        return int(row[0]) if row and row[0] is not None else 0

    def save_user_state(self, state: RelationshipState):
        cur = self.conn.cursor()
        cur.execute(
            """
            INSERT INTO user_states (
                user_id, stage, mood, energy, warmth, initiative, disclosure, flirty, last_updated
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(user_id) DO UPDATE SET
                stage=excluded.stage,
                mood=excluded.mood,
                energy=excluded.energy,
                warmth=excluded.warmth,
                initiative=excluded.initiative,
                disclosure=excluded.disclosure,
                flirty=excluded.flirty,
                last_updated=excluded.last_updated
            """,
            (
                state.user_id,
                state.stage,
                state.mood,
                float(state.energy),
                float(state.warmth),
                float(state.initiative),
                float(state.disclosure),
                float(state.flirty),
                state.last_updated.isoformat(),
            ),
        )
        self.conn.commit()

    def get_user_state(self, user_id: str) -> Optional[RelationshipState]:
        cur = self.conn.cursor()
        row = cur.execute(
            """
            SELECT stage, mood, energy, warmth, initiative, disclosure, flirty, last_updated
            FROM user_states
            WHERE user_id = ?
            """,
            (user_id,),
        ).fetchone()
        if not row:
            return None

        stage, mood, energy, warmth, initiative, disclosure, flirty, last_updated = row
        try:
            dt = datetime.fromisoformat(str(last_updated).replace("Z", ""))
        except Exception:
            dt = datetime.now(UTC)

        return RelationshipState(
            user_id=user_id,
            stage=str(stage),
            mood=str(mood),
            energy=float(energy),
            warmth=float(warmth),
            initiative=float(initiative),
            disclosure=float(disclosure),
            flirty=float(flirty),
            last_updated=dt,
        )

    def set_turn_count(self, user_id: str, turn_count: int):
        now = datetime.now(UTC).isoformat()
        cur = self.conn.cursor()
        cur.execute(
            """
            INSERT INTO user_stats (user_id, turn_count, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(user_id) DO UPDATE SET
                turn_count=excluded.turn_count,
                updated_at=excluded.updated_at
            """,
            (user_id, int(turn_count), now),
        )
        self.conn.commit()

    def get_turn_count(self, user_id: str) -> int:
        cur = self.conn.cursor()
        row = cur.execute(
            "SELECT turn_count FROM user_stats WHERE user_id = ?",
            (user_id,),
        ).fetchone()
        return int(row[0]) if row and row[0] is not None else 0

    def save_last_proactive_at(self, user_id: str, dt: datetime):
        cur = self.conn.cursor()
        cur.execute(
            """
            INSERT INTO proactive_state (user_id, last_proactive_at)
            VALUES (?, ?)
            ON CONFLICT(user_id) DO UPDATE SET
                last_proactive_at=excluded.last_proactive_at
            """,
            (user_id, dt.isoformat()),
        )
        self.conn.commit()

    def get_last_proactive_at(self, user_id: str) -> Optional[datetime]:
        cur = self.conn.cursor()
        row = cur.execute(
            "SELECT last_proactive_at FROM proactive_state WHERE user_id = ?",
            (user_id,),
        ).fetchone()
        if not row or not row[0]:
            return None
        try:
            return datetime.fromisoformat(str(row[0]).replace("Z", ""))
        except Exception:
            return None

    def list_user_ids(self) -> List[str]:
        cur = self.conn.cursor()
        rows = cur.execute(
            """
            SELECT DISTINCT user_id FROM (
                SELECT user_id FROM chat_messages
                UNION SELECT user_id FROM user_states
                UNION SELECT user_id FROM user_stats
                UNION SELECT user_id FROM memories
            )
            ORDER BY user_id
            """
        ).fetchall()
        return [str(x[0]) for x in rows if x and x[0] is not None]

    def get_last_message_ts(self, user_id: str) -> Optional[str]:
        cur = self.conn.cursor()
        row = cur.execute(
            """
            SELECT ts FROM chat_messages
            WHERE user_id = ?
            ORDER BY id DESC
            LIMIT 1
            """,
            (user_id,),
        ).fetchone()
        return str(row[0]) if row and row[0] else None

    def get_last_message_dt(self, user_id: str) -> Optional[datetime]:
        raw = self.get_last_message_ts(user_id)
        if not raw:
            return None
        try:
            return datetime.fromisoformat(str(raw).replace("Z", ""))
        except Exception:
            return None

    def recall_memories(
        self,
        user_id: str,
        query_text: str = "",
        limit: int = 6,
        categories: Optional[List[str]] = None,
    ) -> List[MemoryItem]:
        cur = self.conn.cursor()
        if categories:
            placeholders = ",".join("?" for _ in categories)
            sql = f"""
                SELECT id, user_id, category, content, embedding, importance, created_at
                FROM memories
                WHERE user_id = ? AND category IN ({placeholders})
                ORDER BY id DESC
                LIMIT 300
            """
            rows = cur.execute(sql, [user_id, *categories]).fetchall()
        else:
            rows = cur.execute(
                """
                SELECT id, user_id, category, content, embedding, importance, created_at
                FROM memories
                WHERE user_id = ?
                ORDER BY id DESC
                LIMIT 300
                """,
                (user_id,),
            ).fetchall()

        query_vec = self.embedding_service.embed(query_text) if (self.embedding_service and query_text) else None

        scored = []
        now = datetime.now(UTC)
        for row in rows:
            _, uid, category, content, emb_raw, importance, created_at = row
            semantic_score = 0.0
            if query_vec is not None and emb_raw:
                mem_vec = self.embedding_service.loads(emb_raw)
                semantic_score = self.embedding_service.cosine(query_vec, mem_vec)

            recency_score = 0.0
            try:
                dt = datetime.fromisoformat(str(created_at).replace("Z", ""))
                days = max((now - dt).days, 0)
                recency_score = 1.0 / (1.0 + days)
            except Exception:
                pass

            final_score = semantic_score * 0.65 + float(importance) * 0.25 + recency_score * 0.10
            scored.append(
                (
                    final_score,
                    MemoryItem(
                        user_id=uid,
                        category=category,
                        content=content,
                        importance=float(importance),
                    ),
                )
            )

        scored.sort(key=lambda x: x[0], reverse=True)
        return [item for _, item in scored[:limit]]

    def extract_and_store_user_memory(self, user_id: str, latest_user_message: str):
        text = latest_user_message.strip()
        if not text:
            return

        if any(k in text for k in ["喜欢", "爱", "偏好", "想要"]):
            self.add_memory(
                MemoryItem(
                    user_id=user_id,
                    category="user_profile",
                    content=f"用户偏好线索：{text}",
                    importance=0.7,
                )
            )

        if any(k in text for k in ["累", "烦", "难受", "失眠", "压力", "崩溃"]):
            self.add_memory(
                MemoryItem(
                    user_id=user_id,
                    category="relationship_event",
                    content=f"用户近期情绪偏低：{text}",
                    importance=0.8,
                )
            )

    def add_role_life_memory(self, user_id: str, content: str, importance: float = 0.45):
        self.add_memory(
            MemoryItem(
                user_id=user_id,
                category="role_life",
                content=content,
                importance=importance,
            )
        )
