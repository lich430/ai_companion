from __future__ import annotations

import json
import logging
import os
import random
import re
from collections import deque
from datetime import UTC, datetime, time as dt_time, timedelta, timezone
from threading import Lock
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from dotenv import load_dotenv

from engine.bible_loader import BibleLoader
from engine.embedding_service import EmbeddingService
from engine.identity_guard import IdentityGuard
from engine.memory_manager import MemoryManager
from engine.memory_summarizer import MemorySummarizer
from engine.message_classifier import classify_user_message
from engine.prompt_builder import build_system_prompt, build_user_prompt
from engine.proactive_scheduler import ProactivePlan, ProactiveScheduler
from engine.relationship_state_machine import RelationshipStateMachine
from engine.repetition_guard import RepetitionGuard
from engine.response_generator import GLMResponseGenerator, OpenAIResponseGenerator
from engine.style_controller import StyleController
from models.schemas import Message, RelationshipState, ReplyDecision, ResponseContext, UserProfile

logger = logging.getLogger(__name__)


class CompanionEngine:
    @staticmethod
    def _load_local_timezone():
        key = os.getenv("LOCAL_TIMEZONE", "Asia/Shanghai")
        try:
            return ZoneInfo(key)
        except ZoneInfoNotFoundError:
            return timezone(timedelta(hours=8), name="UTC+08:00")

    def __init__(
        self,
        *,
        llm_provider: str | None = None,
        glm_api_key: str | None = None,
        openai_api_key: str | None = None,
        openai_base_url: str | None = None,
        openai_thinking: dict | None = None,
        openai_reasoning_effort: str | None = None,
        openai_max_completion_tokens: int | None = None,
        chat_model: str | None = None,
        embed_model: str | None = None,
        role_bible_path: str | None = None,
        memory_db_path: str | None = None,
    ):
        load_dotenv()
        self.llm_provider = str(llm_provider or os.getenv("LLM_PROVIDER", "glm")).strip().lower()
        glm_api_key = str(glm_api_key if glm_api_key is not None else os.getenv("GLM_API_KEY", "")).strip()
        openai_api_key = str(openai_api_key if openai_api_key is not None else os.getenv("OPENAI_API_KEY", "")).strip()
        chat_model = str(chat_model if chat_model is not None else os.getenv("GLM_CHAT_MODEL", "glm-4.7")).strip()
        embed_model = str(embed_model if embed_model is not None else os.getenv("GLM_EMBED_MODEL", "embedding-3")).strip()
        llm_api_key = glm_api_key
        if self.llm_provider == "openai":
            chat_model = str(chat_model or os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")).strip()
            llm_api_key = openai_api_key
        elif self.llm_provider != "glm":
            raise RuntimeError("Unsupported LLM_PROVIDER. Use 'glm' or 'openai'.")

        if not llm_api_key:
            if self.llm_provider == "openai":
                raise RuntimeError("Missing OPENAI_API_KEY. Please set it in .env.")
            raise RuntimeError("Missing GLM_API_KEY. Please set it in .env.")
        self.summary_every_n_turns = int(os.getenv("MEMORY_SUMMARY_EVERY_N_TURNS", "8"))
        self.local_timezone = self._load_local_timezone()
        self.quiet_hours = os.getenv("QUIET_HOURS", "02:00-09:00")
        self.emoji_enabled = os.getenv("EMOJI_ENABLED", "true").lower() != "false"
        self.preferred_reply_chars = int(os.getenv("PREFERRED_REPLY_CHARS", "30"))
        self.glm_context_messages = int(os.getenv("GLM_CONTEXT_MESSAGES", "200"))
        self.max_batch_replies = int(os.getenv("MAX_BATCH_REPLIES", "3"))
        self.multi_reply_probability = max(0.0, min(1.0, float(os.getenv("MULTI_REPLY_PROBABILITY", "0.2"))))
        self.rng = random.Random(int(os.getenv("TEXT_POSTPROCESS_SEED", "20260307")))

        lexical_th = float(os.getenv("LEXICAL_REPEAT_THRESHOLD", "0.72"))
        semantic_th = float(os.getenv("SEMANTIC_REPEAT_THRESHOLD", "0.90"))

        self.role_bible_path = str(role_bible_path or os.getenv("ROLE_BIBLE_PATH", "data/guoguo_bible.json")).strip()
        self.bible = BibleLoader(self.role_bible_path).load()
        self.role_name = str(self.bible.get("name", "") or "").strip() or "unknown"
        self.role_id = str(self.bible.get("role_id", "") or "").strip() or "unknown"
        self.embedding = EmbeddingService(api_key=glm_api_key, model=embed_model)
        self.identity_guard = IdentityGuard("data/identity_deflection_templates.json")
        self.memory_db_path = str(memory_db_path or "memory.db").strip() or "memory.db"
        self.memory = MemoryManager(self.memory_db_path, embedding_service=self.embedding)
        self.state_machine = RelationshipStateMachine(self.bible)
        self.style_controller = StyleController(self.bible)
        if self.llm_provider == "openai":
            self.generator = OpenAIResponseGenerator(
                api_key=llm_api_key,
                model=chat_model,
                base_url=openai_base_url,
                thinking=openai_thinking,
                reasoning_effort=openai_reasoning_effort,
                max_completion_tokens=openai_max_completion_tokens,
            )
        else:
            self.generator = GLMResponseGenerator(api_key=llm_api_key, model=chat_model)
        self.summarizer = MemorySummarizer(self.generator, self.bible)
        self.rep_guard = RepetitionGuard(
            embedding_service=self.embedding,
            lexical_threshold=lexical_th,
            semantic_threshold=semantic_th,
        )
        self.proactive_scheduler = ProactiveScheduler()

        self.user_states: dict[str, RelationshipState] = {}
        self.recent_messages: dict[str, list[Message]] = {}
        self.turn_counts: dict[str, int] = {}
        self.pending_replies = deque()
        self._pending_lock = Lock()
        self._pending_seq = 0
        self._last_proactive_ask_turn: dict[str, int] = {}
        self.reply_policy_stats = {
            "reply": 0,
            "noreply": 0,
            "quiet_hours": 0,
            "low_context_no_history": 0,
            "acknowledgement_no_reply": 0,
            "need_no_reply": 0,
            "repeat_spam": 0,
            "hostile_deescalation": 0,
        }

    def _bump_policy_stat(self, decision: ReplyDecision):
        self.reply_policy_stats[decision.type] = self.reply_policy_stats.get(decision.type, 0) + 1
        if decision.reason:
            self.reply_policy_stats[decision.reason] = self.reply_policy_stats.get(decision.reason, 0) + 1

    def _get_recent(self, user_id: str, limit: int | None = None) -> list[Message]:
        uid = str(user_id or "default").strip() or "default"
        cached = self.recent_messages.get(uid)
        if cached:
            recent = list(cached)
        else:
            recent = self.memory.get_recent_chat_messages(uid, limit=max(1, limit or self.glm_context_messages))
            self.recent_messages[uid] = list(recent)
        if limit is not None:
            return recent[-max(1, limit):]
        return recent[-max(1, self.glm_context_messages):]

    def _add_message(self, user_id: str, role: str, content: str, ts: datetime | None = None):
        uid = str(user_id or "default").strip() or "default"
        msg = Message(role=role, content=str(content or "").strip(), ts=ts or datetime.now(UTC))
        recent = self.recent_messages.setdefault(uid, self.memory.get_recent_chat_messages(uid, limit=max(1, self.glm_context_messages)))
        recent.append(msg)
        if len(recent) > max(40, self.glm_context_messages):
            del recent[:-max(40, self.glm_context_messages)]
        if msg.content:
            self.memory.add_chat_message(uid, role, msg.content, ts=msg.ts)

    def _get_turn_count(self, user_id: str) -> int:
        uid = str(user_id or "default").strip() or "default"
        if uid in self.turn_counts:
            return int(self.turn_counts.get(uid, 0) or 0)
        turn_count = self.memory.get_turn_count(uid)
        self.turn_counts[uid] = turn_count
        return turn_count

    def _get_state(self, user_id: str) -> RelationshipState:
        uid = str(user_id or "default").strip() or "default"
        existing = self.user_states.get(uid)
        if existing is not None:
            return existing
        state = RelationshipState(user_id=uid)
        state = self.state_machine.apply_stage_defaults(state)
        self.user_states[uid] = state
        return state

    def _update_state_and_memories_from_user(self, user_id: str, text: str) -> tuple[RelationshipState, list[Message]]:
        uid = str(user_id or "default").strip() or "default"
        state = self._get_state(uid)
        recent = self._get_recent(uid)
        updated = self.state_machine.update_from_message(state, text, recent)
        self.user_states[uid] = updated
        return updated, recent

    def _maybe_run_summary(self, user_id: str):
        uid = str(user_id or "default").strip() or "default"
        turn_count = self._get_turn_count(uid)
        if self.summary_every_n_turns <= 0:
            return
        if turn_count <= 0 or (turn_count % self.summary_every_n_turns) != 0:
            return
        try:
            recent = self._get_recent(uid)
            if len(recent) < 4:
                return
            state = self._get_state(uid)
            recalled = self.memory.recall_memories(
                user_id=uid,
                query_text=recent[-1].content,
                limit=6,
                categories=["user_profile", "relationship_event"],
            )
            role_life = self.memory.recall_memories(
                user_id=uid,
                query_text=recent[-1].content,
                limit=3,
                categories=["role_life"],
            )
            ctx = ResponseContext(
                user_id=uid,
                latest_user_message=recent[-1].content,
                recent_messages=recent,
                profile=UserProfile(user_id=uid),
                state=state,
                recalled_memories=recalled,
                role_life_memories=role_life,
                time_context=self._build_time_context(uid, recent),
            )
            self.summarizer.summarize_and_store(uid, ctx)
        except Exception:
            return

    @staticmethod
    def _leaks_ai_identity(text: str) -> bool:
        t = (text or "").lower()
        bad_tokens = [
            "ai",
            "机器人",
            "程序",
            "系统",
            "assistant",
            "language model",
            "作为ai",
            "作为一个ai",
            "模型",
            "提示词",
        ]
        return any(token in t for token in bad_tokens)

    @staticmethod
    def _is_hostile_reply(text: str) -> bool:
        t = (text or "").lower()
        bad_tokens = [
            "有病",
            "神经病",
            "滚",
            "傻逼",
            "妈的",
            "废物",
            "脑残",
            "去死",
            "装逼",
            "想得美",
        ]
        return any(token in t for token in bad_tokens)

    def _soft_deescalation_reply(self) -> str:
        templates = (
            self.bible.get("conflict_policy", {}).get("deescalation_templates")
            or ["好啦，我们别对着吵了，聊点轻松的好不好。"]
        )
        return str(templates[0]).strip()

    @staticmethod
    def _parse_hhmm(raw: str) -> dt_time:
        hour, minute = raw.split(":", 1)
        return dt_time(hour=int(hour), minute=int(minute))

    def _is_quiet_hours(self, now: datetime | None = None) -> bool:
        current = (now or datetime.now(self.local_timezone)).astimezone(self.local_timezone)
        start_raw, end_raw = self.quiet_hours.split("-", 1)
        start = self._parse_hhmm(start_raw)
        end = self._parse_hhmm(end_raw)
        now_t = current.timetz().replace(tzinfo=None)
        if start <= end:
            return start <= now_t < end
        return now_t >= start or now_t < end

    @staticmethod
    def _ensure_aware_utc(dt: datetime | None) -> datetime | None:
        if dt is None:
            return None
        if dt.tzinfo is None:
            return dt.replace(tzinfo=UTC)
        return dt

    def _to_local_dt(self, dt: datetime | None) -> datetime | None:
        aware = self._ensure_aware_utc(dt)
        if aware is None:
            return None
        return aware.astimezone(self.local_timezone)

    def _find_daily_time_block(self, current: datetime) -> dict:
        routine = self.bible.get("daily_routine") or {}
        for block in routine.get("time_blocks") or []:
            raw = str(block.get("range", "") or "").strip()
            if not raw or "-" not in raw:
                continue
            start_raw, end_raw = raw.split("-", 1)
            start = self._parse_hhmm(start_raw)
            end = self._parse_hhmm(end_raw)
            now_t = current.timetz().replace(tzinfo=None)
            matched = (start <= now_t < end) if start <= end else (now_t >= start or now_t < end)
            if matched:
                return block
        return {}

    def _build_time_context(self, user_id: str, prior_recent: list[Message], now: datetime | None = None) -> dict:
        current = (now or datetime.now(self.local_timezone)).astimezone(self.local_timezone)
        last_message_dt = prior_recent[-1].ts if prior_recent else self.memory.get_last_message_dt(user_id)
        last_local = self._to_local_dt(last_message_dt)
        gap_hours = 0.0
        cross_day = False
        gap_label = "no_history"
        reopen_mode = "new_topic"

        if last_local is not None:
            gap_hours = max(0.0, (current - last_local).total_seconds() / 3600.0)
            cross_day = current.date() != last_local.date()
            if cross_day:
                gap_label = "cross_day"
                reopen_mode = "cross_day_reopen"
            elif gap_hours >= 12:
                gap_label = "long_gap"
                reopen_mode = "long_gap_reconnect"
            elif gap_hours >= 3:
                gap_label = "same_day_gap"
                reopen_mode = "same_day_reopen"
            else:
                gap_label = "same_session"
                reopen_mode = "same_session"

        block = self._find_daily_time_block(current)
        topic_reset_needed = reopen_mode in {"cross_day_reopen", "long_gap_reconnect"}
        return {
            "current_local_time": current.strftime("%Y-%m-%d %H:%M"),
            "current_local_date": current.date().isoformat(),
            "time_period_name": str(block.get("name", "") or "").strip() or "unknown",
            "time_period_label": str(block.get("label", "") or "").strip() or "普通时段",
            "daily_status_label": str(block.get("label", "") or "").strip() or "普通时段",
            "daily_status_allowed_states": [str(x).strip() for x in (block.get("allowed_states") or []) if str(x).strip()],
            "daily_status_blocked_topics": [str(x).strip() for x in (block.get("blocked_topics") or []) if str(x).strip()],
            "gap_hours": gap_hours,
            "gap_label": gap_label,
            "reopen_mode": reopen_mode,
            "cross_day": cross_day,
            "topic_reset_needed": topic_reset_needed,
            "last_message_local_time": last_local.strftime("%Y-%m-%d %H:%M") if last_local else "",
        }

    def _recommended_delay_ms(self, state: RelationshipState, urgent: bool) -> int:
        if urgent:
            return self.rng.randint(500, 1200)

        low_energy = state.energy < 0.45 or state.mood in {"tired", "hurt"}
        close = state.stage in {"close", "familiar"} or state.warmth >= 0.6
        low, high = (1200, 2500) if low_energy else (600, 1800)
        if close:
            low = max(450, low - 200)
            high = max(low + 200, high - 250)
        return self.rng.randint(low, high)

    @staticmethod
    def _has_recent_context(recent: list[Message]) -> bool:
        return len(recent) >= 2

    @staticmethod
    def _is_greeting_message(text: str) -> bool:
        raw = (text or "").strip()
        if not raw:
            return False
        compact = re.sub(r"\s+", "", raw)
        plain = re.sub("[^0-9A-Za-z\u4e00-\u9fff]+", "", compact).lower()
        greeting_set = {
            "你好",
            "您好",
            "哈喽",
            "嗨",
            "在吗",
            "在么",
            "hello",
            "hi",
        }
        return plain in greeting_set

    @staticmethod
    def _is_acknowledgement_message(text: str) -> bool:
        raw = (text or "").strip()
        if not raw:
            return False
        compact = re.sub(r"\s+", "", raw)
        plain = re.sub("[^0-9A-Za-z\u4e00-\u9fff]+", "", compact).lower()
        ack_set = {
            "嗯嗯",
            "好的",
            "好",
            "知道了",
            "知道啦",
            "明白了",
            "明白",
            "收到",
            "ok",
            "okk",
            "okay",
        }
        return plain in ack_set

    def _should_acknowledgement_noreply(self, text: str, recent: list[Message], cls: dict) -> bool:
        if cls.get("urgent") or cls.get("hostile"):
            return False
        if not self._is_acknowledgement_message(text):
            return False
        prior = [m for m in recent[:-1] if m.content.strip()]
        if not prior:
            return False
        last = prior[-1]
        if last.role != "assistant":
            return False
        if "?" in text or "？" in text:
            return False
        return True

    @staticmethod
    def _marketing_triggered(text: str, bible: dict) -> bool:
        marketing = bible.get("marketing") or {}
        triggers = marketing.get("triggers") or []
        raw = (text or "").strip()
        return any(str(trigger).strip() and str(trigger) in raw for trigger in triggers)

    @staticmethod
    def _contains_any(text: str, tokens: list[str]) -> bool:
        raw = (text or "").strip()
        return any(token and token in raw for token in tokens)

    @staticmethod
    def _extract_party_size(text: str) -> int | None:
        raw = (text or "").strip()
        if not raw:
            return None
        m = re.search(r"(\d+)\s*个?人", raw)
        if m:
            try:
                return int(m.group(1))
            except Exception:
                return None
        cn_map = {
            "一个人": 1,
            "两个人": 2,
            "二个人": 2,
            "三个人": 3,
            "四个人": 4,
            "五个人": 5,
            "六个人": 6,
            "七个人": 7,
            "八个人": 8,
            "九个人": 9,
            "十个人": 10,
        }
        for key, value in cn_map.items():
            if key in raw:
                return value
        return None

    @staticmethod
    def _extract_room_type(text: str) -> str:
        raw = (text or "").strip()
        for room in ["小包", "中包", "大包", "包厢"]:
            if room in raw:
                return room
        return ""

    @staticmethod
    def _extract_eta_phrase(text: str) -> str:
        raw = (text or "").strip()
        if not raw:
            return ""
        patterns = [
            r"(差不多.{0,8}(分钟|个小时))",
            r"((大概|大约).{0,8}(分钟|个小时))",
            r"((四十|三十|二十|十|\d+).{0,4}分钟)",
            r"((一|两|二|三|\d+).{0,2}个小时)",
        ]
        for pattern in patterns:
            m = re.search(pattern, raw)
            if m:
                return m.group(1)
        return ""

    @staticmethod
    def _contains_booking_terms(text: str) -> bool:
        raw = (text or "").strip()
        booking_terms = ["留包", "留个包", "留包厢", "留位置", "留位", "提前安排", "安排包厢"]
        return any(term in raw for term in booking_terms)

    def _booking_pitch_count(self, recent: list[Message], window: int = 20) -> int:
        assistant_recent = [m.content.strip() for m in recent[-window:] if m.role == "assistant" and m.content.strip()]
        return sum(1 for item in assistant_recent if self._contains_booking_terms(item))

    def _recent_user_party_size(self, latest_text: str, recent: list[Message]) -> int | None:
        party_size = self._extract_party_size(latest_text)
        if party_size:
            return party_size
        for msg in reversed(recent[-8:]):
            if msg.role != "user":
                continue
            party_size = self._extract_party_size(msg.content)
            if party_size:
                return party_size
        return None

    def _recent_user_mentions(self, recent: list[Message], tokens: list[str], window: int = 8) -> bool:
        user_recent = [m.content.strip() for m in recent[-window:] if m.role == "user" and m.content.strip()]
        return any(self._contains_any(item, tokens) for item in user_recent)

    def _build_budget_reply(self, latest_text: str, recent: list[Message]) -> str | None:
        latest = (latest_text or "").strip()
        if not latest:
            return None
        party_size = self._recent_user_party_size(latest, recent)
        asks_budget = self._contains_any(latest, ["多少钱", "多少", "人均", "预算", "需要多少钱", "大概多少"])
        asks_gift = self._contains_any(latest, ["送酒", "送酒水", "送不送酒", "送不", "酒水送", "可以送酒"])
        if not (asks_budget or asks_gift):
            return None
        if asks_gift and not asks_budget and not party_size:
            return None

        lines: list[str] = []
        if self._contains_any(latest, ["人均多少", "人均"]):
            lines.append("人均1500到2000左右。")
        elif asks_budget:
            if party_size == 2:
                lines.append("包厢加小费差不多三千左右。")
            elif party_size == 3:
                lines.append("三个人大概四千多。")
            elif party_size:
                estimate = max(2, party_size) * 1600
                rounded = int(round(estimate / 500.0) * 500)
                lines.append(f"{party_size}个人大概{rounded}左右。")
            else:
                lines.append("包厢加小费，人均差不多1500到2000。")

        if asks_gift:
            lines.append("送酒水。")

        if not lines:
            return None
        return "\n".join(lines[:2])

    def _is_duplicate_image_request(self, latest_text: str, recent: list[Message], window: int = 20) -> bool:
        latest = (latest_text or "").strip()
        if not latest:
            return False
        image_tokens = ["图片", "照片", "妹子照片", "有图"]
        if not self._contains_any(latest, image_tokens):
            return False
        recent_user = [m.content.strip() for m in recent[-window:] if m.role == "user" and m.content.strip()]
        duplicate_count = sum(1 for item in recent_user if self._contains_any(item, image_tokens))
        recent_assistant = [m.content.strip() for m in recent[-window:] if m.role == "assistant" and m.content.strip()]
        has_answered = any(self._contains_any(item, ["我手机上没有", "到了你可以挑", "没有图片"]) for item in recent_assistant)
        return duplicate_count >= 1 and has_answered

    def _build_greeting_reply(self, latest_text: str, recent: list[Message]) -> str | None:
        raw = (latest_text or "").strip()
        if not raw:
            return None
        compact = re.sub(r"\s+", "", raw)
        plain = re.sub("[^0-9A-Za-z\u4e00-\u9fff]+", "", compact).lower()
        if plain in {"你好", "您好", "哈喽", "嗨", "hello", "hi"}:
            return "你好"
        if plain.startswith("我是") and len(plain) <= 6:
            return "你好。"
        return None

    def _build_sales_template_reply(self, latest_text: str, recent: list[Message], time_context: dict) -> str | None:
        latest = (latest_text or "").strip()
        if not latest:
            return None

        greeting_reply = self._build_greeting_reply(latest, recent)
        if greeting_reply:
            return greeting_reply

        budget_reply = self._build_budget_reply(latest, recent)
        if budget_reply:
            return budget_reply

        prior_messages = recent[:-1] if recent else []
        last_assistant = next((m.content.strip() for m in reversed(prior_messages) if m.role == "assistant" and m.content.strip()), "")
        party_size = self._recent_user_party_size(latest, recent)
        room_type = self._extract_room_type(latest)
        eta = self._extract_eta_phrase(latest)
        booking_count = self._booking_pitch_count(recent)

        if self._contains_any(latest, ["发个位置", "发位置", "位置发我", "把位置发我"]):
            return "宁国路拉菲公馆，到了给我发消息"

        if self._contains_any(latest, ["你从事什么工作", "你做什么工作", "做什么工作", "你是做什么的", "你在什么公司", "做什么的"]):
            return "我在夜场上班，平时陪客人喝酒聊天，也会帮忙订房安排包厢"

        if self._contains_any(latest, ["不过去玩就不能找你聊天", "不过去玩就不能找你聊天了"]):
            return "当然可以呀。"

        if self._contains_any(latest, ["酒水能便宜吗", "酒水可以便宜吗", "能便宜吗", "能优惠吗", "酒水优惠"]):
            return "你过来，我可以给你送酒"

        if self._contains_any(latest, ["送酒不", "送不送酒", "送酒吗", "可以送酒吗"]):
            return "可以送"

        if self._contains_any(latest, ["酒水别不够", "酒不够", "不够喝", "酒水不够", "别不够啊"]):
            return "你这边可以点一些，我再送一些。"

        if self._contains_any(latest, ["我们就4个人喝的不多", "喝的不多", "四个人喝的不多"]):
            if self._recent_user_mentions(recent, ["酒水别不够", "不够喝", "酒水不够", "就一箱酒水"]):
                return "没事，酒一定会安排好。"

        if self._contains_any(latest, ["出台什么价格", "出台多少钱", "出台价格"]):
            return "你过来玩，我可以给你介绍开放一点的女孩子"

        if self._contains_any(latest, ["妹子质量怎么样", "妹子质量咋样", "质量怎么样"]):
            return "最近来了很多漂亮的新人"

        if self._contains_any(latest, ["妹妹都是多大的", "妹子都是多大的", "多大"]):
            return "基本18到20左右"

        if self._contains_any(latest, ["你们那边都有什么", "你那边都有什么", "那边都有什么"]):
            if booking_count >= 1:
                return "有美女陪你唱歌喝酒"
            return "有美女陪你唱歌喝酒\n是打算今晚过来玩吗"

        if self._contains_any(latest, ["有没有图片"]):
            return "我手机上没有，公司人很多到了你可以挑嘛"

        if self._contains_any(latest, ["美女多吗", "美女多不多"]):
            if self._contains_any(latest, ["朋友", "去玩", "过去玩", "晚上"]):
                if booking_count >= 1:
                    return "很多呀"
                return "很多呀，要不要我给你留个包厢。"

        if self._contains_any(latest, ["真空"]) and self._contains_any(latest, ["和上次一样", "上次一样"]):
            return "可以呀，那我在公司等你。"

        if self._contains_any(latest, ["出去的美女", "有没有出去的", "能跟我出去的"]):
            return "有，你们自己沟通，我也可以协助你沟通"

        if self._contains_any(latest, ["我们需要2个", "需要2个", "钱不是问题", "换地方玩了"]):
            return "没问题，来了和我们经理说"

        if self._contains_any(latest, ["玩的开放", "玩得开放", "没意思呀"]):
            return "那是自然的，肯定让你玩的高兴"

        if self._contains_any(latest, ["今天晚上过去玩", "晚上过去玩", "过去玩", "过去喝酒", "来喝酒", "过来喝酒"]):
            return "好，你们几个人"

        if self._contains_any(latest, ["今天能不能送点酒", "能不能送点酒", "送点酒"]):
            return "可以，到时候你点一些，我送一点"

        if self._contains_any(latest, ["带点能玩的进来", "带点能玩的", "能玩的进来", "能玩不给小费", "不给小费"]):
            return "放心，不能玩不给小费"

        if self._contains_any(latest, ["在不在公司", "在公司吗", "在店里吗", "在店吗", "在不在店"]):
            return "在公司呢"

        if eta or self._contains_any(latest, ["快结束了", "一会到", "差不多到了", "四十分钟", "半小时", "一个小时"]):
            return "好的，快到了给我发信息，我去门口接你们"

        if party_size and self._contains_any(last_assistant, ["几个人"]):
            if booking_count >= 1:
                return f"{party_size}个人的话可以安排。"
            return f"{party_size}个人的话给你们安排，你想留小包还是中包"

        if room_type and self._contains_any(last_assistant, ["几点到", "到店时间"]):
            room_label = room_type if room_type != "包厢" else "包厢"
            if booking_count >= 1:
                return f"好，{room_label}这边我记着，你们快到了跟我说。"
            return f"好，{room_label}先给你留着，你们快到了跟我说"

        if self._contains_any(latest, ["订房", "开包厢", "留个包", "留位", "安排一下"]):
            if booking_count >= 1:
                return "可以，你们几个人"
            return "可以，你们几个人，我先给你安排"

        return None

    @staticmethod
    def _normalize_lookup_text(text: str) -> str:
        return re.sub(r"[\s\-\(\)（）·,，。！？!?:：/]+", "", (text or "")).strip().lower()

    def _get_drink_catalog(self) -> dict[str, list[dict]]:
        marketing = self.bible.get("marketing") or {}
        store_info = marketing.get("store_info") or {}
        drink_info = store_info.get("drink_info") or {}
        catalog = drink_info.get("drink_catalog") or {}
        return catalog if isinstance(catalog, dict) else {}

    def _find_drink_item(self, text: str) -> tuple[str, dict] | tuple[None, None]:
        normalized = self._normalize_lookup_text(text)
        if not normalized:
            return None, None
        for category, items in self._get_drink_catalog().items():
            for item in items or []:
                names = [str(item.get("name", "")).strip()] + [str(x).strip() for x in (item.get("aliases") or []) if str(x).strip()]
                for name in names:
                    if not name:
                        continue
                    if self._normalize_lookup_text(name) in normalized:
                        return str(category), item
        return None, None

    def _find_drink_candidates(self, text: str) -> list[tuple[str, dict]]:
        normalized = self._normalize_lookup_text(text)
        out: list[tuple[str, dict]] = []
        if not normalized:
            return out
        for category, items in self._get_drink_catalog().items():
            for item in items or []:
                names = [str(item.get("name", "")).strip()] + [str(x).strip() for x in (item.get("aliases") or []) if str(x).strip()]
                if any(self._normalize_lookup_text(name) and self._normalize_lookup_text(name) in normalized for name in names):
                    out.append((str(category), item))
        return out

    def _infer_drink_category(self, text: str) -> str:
        raw = (text or "").strip()
        if any(token in raw for token in ["洋酒", "威士忌", "白兰地", "马爹利", "轩尼诗", "黑方", "芝华士"]):
            return "洋酒"
        if any(token in raw for token in ["红酒", "葡萄酒", "奔富"]):
            return "红酒"
        if "啤酒" in raw:
            return "啤酒"
        if any(token in raw for token in ["饮料", "水", "红茶", "绿茶", "脉动"]):
            return "饮料"
        return ""

    @staticmethod
    def _format_catalog_item(item: dict) -> str:
        name = str(item.get("name", "")).strip()
        unit = str(item.get("unit", "")).strip()
        price = item.get("price", "")
        price_text = str(int(price)) if isinstance(price, (int, float)) and float(price).is_integer() else str(price).strip()
        return f"{name}{price_text}元/{unit}".strip("/")

    @staticmethod
    def _format_short_price(item: dict) -> str:
        price = item.get("price", "")
        unit = str(item.get("unit", "")).strip() or "瓶"
        price_text = str(int(price)) if isinstance(price, (int, float)) and float(price).is_integer() else str(price).strip()
        return f"{price_text}元/{unit}"

    def _list_catalog_by_category(self, category: str) -> str:
        items = self._get_drink_catalog().get(category) or []
        return "、".join(self._format_catalog_item(item) for item in items[:8] if item.get("name"))

    def _build_drink_menu_reply(self, user_text: str, recent: list[Message] | None = None) -> str | None:
        raw = (user_text or "").strip()
        if not raw:
            return None

        catalog = self._get_drink_catalog()
        if not catalog:
            return None

        history = recent or []
        if self._contains_any(raw, ["能便宜吗", "能优惠吗", "送酒", "送不送酒", "酒水别不够", "不够喝", "人均", "需要多少钱", "出台什么价格"]):
            return None

        if any(token in raw for token in ["一箱多少钱", "多少钱一箱", "啤酒多少一箱", "纯生多少一箱", "雪花多少一箱", "酒水多少一箱"]):
            marketing = self.bible.get("marketing") or {}
            store_info = marketing.get("store_info") or {}
            drink_info = store_info.get("drink_info") or {}
            beer_box = str(drink_info.get("beer", "") or "").strip()
            if beer_box:
                return f"啤酒{beer_box}"
            return "啤酒一箱1000元+/箱（24瓶）"

        matched_category, matched_item = self._find_drink_item(raw)
        matched_candidates = self._find_drink_candidates(raw)
        asked_specific = bool(re.search(r"(有|点|喝|上|来).{0,12}?(酒|啤酒|洋酒|红酒|饮料)|有.{0,12}吗", raw))
        asked_menu = (
            any(token in raw for token in ["有什么酒", "都有什么酒", "酒单", "酒水", "啤酒", "洋酒", "红酒", "饮料"])
            or bool(re.search(r"有什么(啤酒|洋酒|红酒|饮料)", raw))
            or bool(re.search(r"(啤酒|洋酒|红酒|饮料)有哪些", raw))
        )
        inferred_category = self._infer_drink_category(raw)

        # Broad menu/category questions are better handled by the model with the
        # injected store catalog in prompt. Keep deterministic rules only for
        # exact item/box-price lookups.
        if asked_menu:
            return None

        if asked_specific and len(matched_candidates) >= 2:
            price_text = "、".join(self._format_short_price(item) for _, item in matched_candidates[:2])
            return f"有 {price_text}"

        if matched_item and asked_specific:
            return f"有 {self._format_short_price(matched_item)}"

        if asked_specific and not matched_item:
            return "没有"

        if matched_item:
            return f"有 {self._format_short_price(matched_item)}"

        if history and self._recent_user_mentions(history, ["有没有图片", "有妹子照片", "妹子照片", "图片"], window=20):
            return None

        return None

    @staticmethod
    def _sanitize_robotic_drink_reply(text: str) -> str:
        out = (text or "").strip()
        if not out:
            return out
        out = out.replace("需要给你详细介绍一下吗?", "")
        out = out.replace("需要给你详细介绍一下吗？", "")
        out = out.replace("要不要我给你详细说下价格", "")
        out = out.replace("要不要我给你详细说下", "")
        out = re.sub(r"\n{2,}", "\n", out)
        out = re.sub(r"[ \t]{2,}", " ", out)
        out = re.sub(r"\n\s*$", "", out).strip()
        return out

    @staticmethod
    def _sanitize_hard_rejection(text: str, latest_user_text: str, state: RelationshipState | None = None) -> str:
        out = (text or "").strip()
        latest = (latest_user_text or "").strip()
        if not out or not latest:
            return out

        is_stranger = bool(state and getattr(state, "stage", "") == "stranger")
        invite_tokens = ["跟我走", "跟我回去", "带走你", "跟我睡", "睡觉", "出去喝酒", "带你出去"]
        if not (is_stranger and any(token in latest for token in invite_tokens)):
            return out

        replacements = {
            "只陪喝酒唱歌，别想歪了": "我们才刚认识呀，先别这么急嘛。",
            "只陪喝酒唱歌": "我们才刚认识呀，先慢慢来嘛。",
            "别想歪了": "你也太着急了呀。",
            "别想了": "先别这么急嘛。",
            "不行": "现在还不太合适呀。",
            "没有": "这个先不急呀。",
            "来不了": "今天可能不太方便呀。",
            "再说吧": "以后再慢慢看呀。",
            "以后再说": "以后再慢慢看呀。",
            "先玩开心": "你先玩开心一点呀。",
        }
        for src, dst in replacements.items():
            if src in out:
                out = out.replace(src, dst)

        if len(out) <= 6:
            return "现在还不太合适呀，先慢慢来嘛。"
        return out

    @staticmethod
    def _sanitize_surveillance_tone(text: str) -> str:
        out = (text or "").strip()
        if not out:
            return out
        replacements = {
            "盯着你": "陪着你",
            "我盯着你": "我陪着你",
            "看着你": "陪着你",
            "我看着你": "我陪着你",
            "别急~": "慢慢来~",
            "别急": "慢慢来",
        }
        for src, dst in replacements.items():
            out = out.replace(src, dst)
        out = re.sub(r"[?,]{2,}", "?", out)
        out = out.replace("??", "?")
        out = out.replace("??", "?")
        out = re.sub(r"[?,][?.]", "?", out)
        return out.strip()

    @staticmethod
    def _sanitize_excessive_tildes(text: str) -> str:
        out = (text or "").strip()
        if not out:
            return out
        out = re.sub(r"[~?]{2,}", "~", out)
        parts = [part.strip() for part in re.split(r"\n+", out) if part.strip()]
        cleaned = []
        for part in parts:
            if part.endswith(("~", "?")):
                body = part.rstrip("~?").strip()
                if len(body) > 6:
                    part = body
                else:
                    part = f"{body}~" if body else body
            cleaned.append(part)
        return "\n".join(cleaned).strip()

    @staticmethod
    def _sanitize_time_conflicts(text: str, time_context: dict) -> str:
        out = (text or "").strip()
        if not out:
            return out

        period = str(time_context.get("time_period_name", "") or "").strip()
        current_time = str(time_context.get("current_local_time", "") or "").strip()
        if period == "night_work_window":
            hour = None
            minute = 0
            if current_time:
                try:
                    hhmm = current_time.split(" ", 1)[1]
                    hour = int(hhmm.split(":", 1)[0])
                    minute = int(hhmm.split(":", 1)[1])
                except Exception:
                    hour = None
            before_two_am = hour is None or (hour < 2 or (hour == 2 and minute == 0))
            if before_two_am:
                out = re.sub(r"(\u8fd9\u4e2a\u70b9|\u8fd9\u4f1a\u513f|\u8fd9\u65f6\u5019)?\u624d?\u4e0b\u73ed\u4e86", "\u8fd9\u4f1a\u513f\u8fd8\u5728\u5fd9", out)
                out = re.sub(r"\u521a\u4e0b\u73ed", "\u521a\u5fd9\u5b8c\u4e00\u9635", out)
                out = re.sub(r"(\u5df2\u7ecf|\u90fd)?\u6536\u5de5\u4e86", "\u8fd8\u6ca1\u6536\u5de5", out)
                out = re.sub(r"\u4e0b\u73ed[\u3002.!\uff01?\uff1f]*$", "\u8fd8\u6ca1\u4e0b\u73ed", out)
        return out.strip()

    @staticmethod
    def _sanitize_conversion_blockers(text: str, latest_user_text: str, time_context: dict) -> str:
        out = (text or "").strip()
        if not out:
            return out

        latest = (latest_user_text or "").strip()
        period = str(time_context.get("time_period_name", "") or "").strip()
        high_intent = any(token in latest for token in ["过去", "来喝酒", "喝喝酒", "留个小包", "小包", "订房", "开包厢", "你陪我", "安排", "一个人"])
        active_period = period in {"night_work_window", "dinner_window"}
        if not (high_intent and active_period):
            return out

        replacements = {
            "但我还在忙，不一定": "可以啊，你过来我给你留个小包",
            "我还在忙，不一定": "你过来就行，我给你留个小包",
            "我还在忙": "这会儿在店里，你过来就行",
            "不一定": "可以安排",
            "有空陪你聊": "你到了我陪你喝两杯",
            "有空再说": "你来我给你安排",
            "改天吧": "今晚过来就行",
            "现在不方便": "现在可以安排，你过来提前和我说一声",
        }
        for src, dst in replacements.items():
            out = out.replace(src, dst)

        out = re.sub(r"(但|不过)?我还在忙[,?]?所以?", "", out).strip(" ?,")
        out = re.sub(r"(到时候|回头|再说|看情况)", "我给你安排", out)
        out = re.sub(r"\s{2,}", " ", out).strip()
        return out

    @staticmethod
    @staticmethod
    def _normalize_phrase_anchor(text: str) -> str:
        return re.sub(r"[^\u4e00-\u9fff0-9A-Za-z]+", "", (text or "")).strip().lower()

    def _extract_phrase_anchors(self, text: str) -> list[str]:
        parts = [x.strip() for x in re.split(r"[\uff0c\u3002\uff01\uff1f!?,\n]", text or "") if x.strip()]
        anchors: list[str] = []
        skip_prefixes = {"\u4f60\u597d", "\u54c8\u55bd", "\u5728\u5417", "\u4f60\u5462", "\u597d\u7684", "\u662f\u5417", "\u54c8\u54c8", "\u6211\u5728", "\u884c\u554a"}
        for part in parts:
            norm = self._normalize_phrase_anchor(part)
            if len(norm) < 4:
                continue
            anchor = norm[:6]
            if anchor[:2] in skip_prefixes or anchor in skip_prefixes:
                continue
            anchors.append(anchor)
        return anchors

    def _has_recent_phrase_repetition(self, candidate: str, recent: list[Message], window: int = 20) -> bool:
        candidate_anchors = self._extract_phrase_anchors(candidate)
        if not candidate_anchors:
            return False

        assistant_recent = [m.content for m in recent[-window:] if m.role == "assistant" and m.content.strip()]
        if not assistant_recent:
            return False

        history_anchors = set()
        history_texts = [self._normalize_phrase_anchor(text) for text in assistant_recent]
        for text in assistant_recent:
            for anchor in self._extract_phrase_anchors(text):
                history_anchors.add(anchor)

        for anchor in candidate_anchors:
            if anchor in history_anchors:
                return True
            for hist in history_texts:
                if len(anchor) >= 4 and anchor in hist:
                    return True
        return False

    @staticmethod
    def _split_short_reply(text: str, max_chars: int = 10) -> str:
        raw = (text or "").strip()
        if not raw:
            return raw
        max_chars = max(1, int(max_chars))
        compact = re.sub(r"\s+", " ", raw.replace("\n", " ")).strip()
        if len(compact) <= max_chars:
            return compact
        return compact[:max_chars].rstrip("\uff0c\u3002\uff01\uff1f,.!?;\uff1b:\uff1a")

    def _build_repeat_spam_decision(self) -> ReplyDecision:
        if self.rng.random() < 0.5:
            return ReplyDecision(
                type="reply",
                text=bytes(r"\u600e\u4e48\u5566", "ascii").decode("unicode_escape"),
                recommended_delay_ms=self.rng.randint(500, 1200),
                reason="repeat_spam_question",
            )
        return ReplyDecision(type="noreply", recommended_delay_ms=0, reason="repeat_spam")


    def _should_deny_unverified_history(self) -> bool:
        policy = self.bible.get("shared_history_policy") or {}
        return bool(policy.get("enabled", False)) and bool(policy.get("deny_unverified", False))

    def _should_force_noreply(self, recent: list[Message], cls: dict) -> bool:
        if cls.get("urgent"):
            return False
        if not (cls.get("hostile") or cls.get("boring") or cls.get("need_no_reply")):
            return False

        recent_user = [m.content.strip() for m in recent[-6:] if m.role == "user" and m.content.strip()]
        if len(recent_user) < 3:
            return False

        boring_count = sum(1 for text in recent_user if classify_user_message(text).get("boring"))
        hostile_count = sum(1 for text in recent_user if classify_user_message(text).get("hostile"))
        return hostile_count >= 2 or boring_count >= 3

    def _shared_history_categories(self) -> dict:
        policy = self.bible.get("shared_history_policy") or {}
        categories = policy.get("categories") or {}
        return categories if isinstance(categories, dict) else {}

    def _detect_shared_history_category(self, text: str) -> str:
        raw = (text or "").strip()
        categories = self._shared_history_categories()
        for key, conf in categories.items():
            keywords = [str(x).strip() for x in (conf.get("keywords") or []) if str(x).strip()]
            if keywords and any(keyword in raw for keyword in keywords):
                return str(key)
        return "default"

    def _detect_scene_keys(
        self,
        user_text: str,
        recent: list[Message],
        state: RelationshipState,
        time_context: dict | None = None,
    ) -> list[str]:
        raw = (user_text or "").strip()
        if not raw:
            return []

        def u(value: str) -> str:
            return value.encode("ascii").decode("unicode_escape") if "\\u" in value else value

        matched: list[str] = []
        compact = re.sub(r"\s+", "", raw)
        period = str((time_context or {}).get("time_period_name", "") or "").strip()

        recent_user_msgs = [m.content.strip() for m in recent[-4:] if m.role == "user" and m.content.strip()] if recent else []
        normalized_recent = [re.sub(r"\s+", "", msg) for msg in recent_user_msgs]
        repeated_short = len(normalized_recent) >= 2 and len(set(normalized_recent)) == 1 and len(normalized_recent[0]) <= 6

        if repeated_short or self._contains_any(compact, [u(r"\u8c03\u76ae"), u(r"\u8868\u60c5")]):
            matched.append("repeated_emoji_opening")

        if self._looks_like_shared_history_claim(raw):
            matched.append("claimed_shared_history")

        if self._contains_any(compact, [u(r"\u60f3\u4f60\u4e86"), u(r"\u60f3\u4f60"), u(r"\u4e00\u8d77\u5403\u4e2a\u996d"), u(r"\u5403\u4e2a\u996d")]):
            matched.append("stranger_says_miss_you")

        if self._contains_any(compact, [u(r"\u597d\u5427"), u(r"\u90a3\u7b97\u4e86")]):
            matched.append("declined_dinner_repair")

        if self._contains_any(compact, [u(r"\u4f60\u5728\u5e72\u561b"), u(r"\u4f60\u5728\u5e72\u54c8"), u(r"\u5e72\u561b\u5462")]) and period in {"afternoon_window", "dinner_window"}:
            matched.append("afternoon_before_work_status")

        if self._contains_any(compact, [u(r"\u7f8e\u5973\u591a\u5417"), u(r"\u7f8e\u5973\u591a\u4e0d\u591a"), u(r"\u59b9\u5b50\u8d28\u91cf\u600e\u4e48\u6837")]):
            matched.append("beauty_inquiry_conversion")

        if u(r"\u771f\u7a7a") in compact:
            matched.append("zhenkong_arrangement")

        if self._contains_any(compact, [u(r"\u73a9\u7684\u5f00\u653e"), u(r"\u73a9\u5f97\u5f00\u653e"), u(r"\u6ca1\u610f\u601d\u5440"), u(r"\u6ca1\u610f\u601d\u554a")]):
            matched.append("after_rejected_open_play")

        if self._contains_any(compact, [u(r"\u51fa\u53bb\u7684\u7f8e\u5973"), u(r"\u6709\u6ca1\u6709\u51fa\u53bb\u7684"), u(r"\u80fd\u8ddf\u6211\u51fa\u53bb\u7684")]):
            matched.append("outside_girls_inquiry")

        if self._contains_any(compact, [u(r"\u9700\u89812\u4e2a"), u(r"\u6211\u4eec\u9700\u89812\u4e2a"), u(r"\u94b1\u4e0d\u662f\u95ee\u9898"), u(r"\u6362\u5730\u65b9\u73a9\u4e86")]):
            matched.append("outside_two_girls_request")

        if self._contains_any(compact, [u(r"\u5e264\u4e2a\u5973\u5b69\u5b50"), u(r"\u5e26\u5973\u5b69\u5b50\u51fa\u53bb\u5531\u6b4c"), u(r"\u5403\u5b8c\u996d\u80fd\u4e0d\u80fd\u6765")]):
            matched.append("invite_outside_singing_busy_excuse")

        if self._contains_any(compact, [u(r"\u8ddf\u6211\u8d70"), u(r"\u8ddf\u6211\u56de\u53bb"), u(r"\u5e26\u8d70\u4f60"), u(r"\u8ddf\u6211\u7761"), u(r"\u4eca\u665a\u5e26\u8d70\u4f60")]):
            matched.append("take_away_invite")

        deduped: list[str] = []
        for key in matched:
            if key not in deduped:
                deduped.append(key)
        return deduped


    def _sanitize_booking_pitch_frequency(self, text: str, recent: list[Message], window: int = 20) -> str:
        out = (text or "").strip()
        if not out:
            return out

        def u(value: str) -> str:
            return value.encode("ascii").decode("unicode_escape") if "\\u" in value else value

        recent_msgs = recent or []
        pitch_terms = [
            u(r"留包厢"),
            u(r"留个包厢"),
            u(r"留位置"),
            u(r"提前安排"),
            u(r"给你安排包厢"),
        ]
        pitch_count = 0
        for msg in recent_msgs[-window:]:
            if msg.role != "assistant":
                continue
            content = (msg.content or "").strip()
            if any(term in content for term in pitch_terms):
                pitch_count += 1
        if pitch_count < 1:
            return out

        replacements = {
            u(r"，要不要我给你留个包厢"): "",
            u(r"要不要我给你留个包厢"): "",
            u(r"，要不要我帮你留包厢"): "",
            u(r"要不要我帮你留包厢"): "",
            u(r"我先给你安排"): u(r"我知道了"),
            u(r"我现在给你留位置"): u(r"放心来就行"),
            u(r"到了不用等哦"): u(r"到了跟我说就行"),
            u(r"你们快到了跟我说"): u(r"你们到了跟我说"),
        }
        for src, dst in replacements.items():
            out = out.replace(src, dst)
        out = re.sub(r"\s{2,}", " ", out)
        out = out.replace(u(r"。。"), u(r"。"))
        out = out.replace(u(r"，，"), u(r"，"))
        out = out.replace(u(r"，。"), u(r"。"))
        return out.strip()
    def _emotion_probe_policy(self) -> dict:
        policy = self.bible.get("emotion_probe_policy") or {}
        return policy if isinstance(policy, dict) else {}

    def _is_emotional_distress_text(self, text: str) -> bool:
        raw = (text or "").strip()
        if not raw:
            return False
        distress_keywords = self._emotion_probe_policy().get("distress_keywords") or [
            "难过",
            "崩溃",
            "委屈",
            "想哭",
            "心累",
            "好烦",
            "好痛苦",
            "抑郁",
            "不开心",
            "撑不住",
            "好绝望",
        ]
        return any(keyword in raw for keyword in distress_keywords)

    def _has_explicit_reason_detail(self, text: str) -> bool:
        raw = (text or "").strip()
        if not raw:
            return False
        reason_hints = self._emotion_probe_policy().get("reason_hints") or [
            "因为",
            "所以",
            "被",
            "刚刚",
            "今天",
            "昨晚",
            "客户",
            "工作",
            "失恋",
            "放鸽子",
            "吵架",
            "分手",
            "出事",
        ]
        return any(hint in raw for hint in reason_hints)

    def _should_probe_then_comfort(self, text: str, cls: dict) -> bool:
        policy = self._emotion_probe_policy()
        if not bool(policy.get("enabled", True)):
            return False
        if cls.get("hostile") or cls.get("urgent"):
            return False
        return self._is_emotional_distress_text(text) and not self._has_explicit_reason_detail(text)

    def _probe_then_comfort_reply(self) -> str:
        policy = self._emotion_probe_policy()
        probe_templates = policy.get("probe_templates") or ["怎么啦 发生啥了"]
        comfort_templates = policy.get("comfort_templates") or ["我在呢 你慢慢说"]
        probe = str(probe_templates[0]).strip()
        comfort = str(comfort_templates[0]).strip()
        if not probe:
            probe = "怎么啦 发生啥了"
        if not comfort:
            comfort = "我在呢 你慢慢说"
        return f"{probe}\n{comfort}"

    def _has_supporting_shared_history(self, text: str, recent: list[Message]) -> bool:
        raw = (text or "").strip()
        categories = self._shared_history_categories()
        activity_keywords = []
        for _, conf in categories.items():
            activity_keywords.extend(conf.get("keywords") or [])
        if not activity_keywords:
            activity_keywords = ["打麻将", "打牌", "吃饭", "喝酒", "唱歌", "见面", "逛街", "出去", "一起"]
        matched = [keyword for keyword in activity_keywords if keyword in raw]
        history_only = recent[:-1] if recent else []
        corpus = "\n".join(message.content for message in history_only[-20:])
        if matched:
            return any(keyword in corpus for keyword in matched)
        support_hints = ["上次", "之前", "一起", "赢我", "输我", "见过", "去过", "玩过"]
        return any(hint in raw and hint in corpus for hint in support_hints)

    def _deny_unverified_history(self, text: str) -> str:
        category = self._detect_shared_history_category(text)
        categories = self._shared_history_categories()
        conf = categories.get(category) or categories.get("default") or {}
        templates = conf.get("deny_templates") or ["你应该是记错了吧 这事没有过"]
        return str(templates[0]).strip()

    def _build_style(self, user_id: str, state: RelationshipState, user_text: str, cls: dict, time_context: dict | None = None) -> dict:
        style = self.style_controller.build_style_directives(state)
        style["emoji_enabled"] = self.emoji_enabled
        style["marketing_allowed"] = self._marketing_triggered(user_text, self.bible)
        style["low_context"] = bool(cls.get("low_context"))
        style["clarify_needed"] = bool(cls.get("low_context")) and not bool(cls.get("hostile"))
        style["preferred_reply_chars"] = max(10, int(getattr(self, "preferred_reply_chars", 30)))
        style["forbidden_fillers"] = (
            self.bible.get("speech_style", {}).get("forbidden_fillers")
            or ["哎呀", "呀", "呢"]
        )
        style["matched_scene_keys"] = self._detect_scene_keys(user_text, self._get_recent(user_id), state, time_context=time_context)
        style.update(time_context or {})
        style["short_reply_mode"] = False
        short_cfg = self.bible.get("short_sentence_policy") or {}
        if bool(short_cfg.get("enabled", False)):
            short_prob = max(0.0, min(1.0, float(short_cfg.get("probability", 0.10))))
            if (not cls.get("urgent")) and self.rng.random() < short_prob:
                style["short_reply_mode"] = True
                style["short_reply_max_chars"] = max(4, int(short_cfg.get("max_chars", 14)))

        style["proactive_question_mode"] = False
        pq_cfg = self.bible.get("proactive_question_policy") or {}
        stage_allowlist = {str(x).strip() for x in (pq_cfg.get("stage_allowlist") or []) if str(x).strip()}
        stage_allowed = (not stage_allowlist) or (state.stage in stage_allowlist)
        if (
            bool(pq_cfg.get("enabled", False))
            and stage_allowed
            and state.initiative >= float(pq_cfg.get("min_initiative", 0.18))
            and not cls.get("hostile")
            and not cls.get("low_context")
        ):
            ask_prob = max(0.0, min(1.0, float(pq_cfg.get("probability", 0.18))))
            min_gap = max(1, int(pq_cfg.get("min_turn_gap", 6)))
            now_turn = self._get_turn_count(user_id)
            last_turn = int(self._last_proactive_ask_turn.get(user_id, -10**9))
            if (now_turn - last_turn) >= min_gap and self.rng.random() < ask_prob:
                templates = [str(x).strip() for x in (pq_cfg.get("question_templates") or []) if str(x).strip()]
                if templates:
                    style["proactive_question_mode"] = True
                    style["proactive_question_template"] = self.rng.choice(templates)
        return style

    def _apply_turn_style_adjustments(self, user_id: str, text: str, style: dict) -> str:
        out = (text or "").strip()
        if not out:
            return out

        if style.get("short_reply_mode"):
            max_chars = max(4, int(style.get("short_reply_max_chars", 14) or 14))
            first = re.split(r"[。！？!?\n]", out)[0].strip() or out
            if len(first) > max_chars:
                first = first[:max_chars].rstrip("，。！？,.!?;；:：")
            out = first

        if style.get("proactive_question_mode"):
            q = str(style.get("proactive_question_template", "") or "").strip()
            if q and ("？" not in out and "?" not in out):
                out = f"{out} {q}".strip()
                self._last_proactive_ask_turn[user_id] = self._get_turn_count(user_id)
        return out

    def _finalize_reply(
        self,
        state: RelationshipState,
        raw_reply: str,
        reason: str,
        urgent: bool,
        time_context: dict | None = None,
        latest_user_text: str = "",
        recent: list[Message] | None = None,
    ) -> ReplyDecision:
        # Keep model output as-is for context consistency.
        processed = (raw_reply or "").strip()
        processed = self._sanitize_robotic_drink_reply(processed)
        processed = self._sanitize_conversion_blockers(processed, latest_user_text, time_context or {})
        processed = self._sanitize_time_conflicts(processed, time_context or {})
        processed = self._sanitize_hard_rejection(processed, latest_user_text, state)
        processed = self._sanitize_surveillance_tone(processed)
        recent_msgs = recent or []
        processed = self._sanitize_booking_pitch_frequency(processed, recent_msgs)
        processed = self._sanitize_excessive_tildes(processed)
        last_assistant = next((m.content.strip() for m in reversed(recent_msgs) if m.role == "assistant" and m.content.strip()), "")
        if last_assistant and processed == last_assistant:
            price_reply = self._build_drink_menu_reply(latest_user_text, recent=recent_msgs)
            if price_reply and price_reply != processed:
                processed = price_reply
        if not processed:
            processed = "我在这陪你"
        return ReplyDecision(
            type="reply",
            text=processed,
            recommended_delay_ms=self._recommended_delay_ms(state, urgent=urgent),
            reason=reason,
        )

    @staticmethod
    def _is_repeated_user_spam(recent: list[Message], incoming_texts: list[str] | None = None) -> bool:
        window = [m.content.strip() for m in recent[-10:] if m.role == "user" and m.content.strip()]
        if incoming_texts:
            for t in incoming_texts:
                s = (t or "").strip()
                if s:
                    window.append(s)
        if len(window) < 3:
            return False

        run = 1
        dominant = None
        for i in range(1, len(window)):
            if window[i] == window[i - 1]:
                run += 1
                if run >= 3:
                    dominant = window[i]
                    break
            else:
                run = 1

        freq: dict[str, int] = {}
        for text_item in window:
            freq[text_item] = freq.get(text_item, 0) + 1
        if dominant is None:
            for text_item, count in freq.items():
                if count >= 4:
                    dominant = text_item
                    break
                if len(text_item) <= 3 and count >= 3:
                    dominant = text_item
                    break

        if dominant is None:
            return False

        if incoming_texts:
            incoming_norm = [str(x).strip() for x in incoming_texts if str(x).strip()]
            if any(text_item != dominant and freq.get(text_item, 0) <= 2 for text_item in incoming_norm):
                return False
        return True
    @staticmethod
    def _looks_like_shared_history_claim(text: str) -> bool:
        raw = (text or "").strip()
        def u(*codes: int) -> str:
            return "".join(chr(code) for code in codes)

        direct_tokens = [
            u(0x8fd8, 0x4e00, 0x8d77),
            u(0x4e0d, 0x662f, 0x8fd8),
            u(0x4f60, 0x4e0a, 0x6b21, 0x8d62, 0x6211),
            u(0x4f60, 0x4e0a, 0x6b21, 0x8f93, 0x6211),
            u(0x54b1, 0x4fe9, 0x8fd8),
            u(0x4e0a, 0x6b21, 0x4e00, 0x8d77),
            u(0x4e4b, 0x524d, 0x89c1, 0x8fc7),
            u(0x4ee5, 0x524d, 0x8ba4, 0x8bc6),
            u(0x4f60, 0x5fd8, 0x4e86, 0x6211),
        ]
        if any(token in raw for token in direct_tokens):
            return True

        time_tokens = [u(0x4e0a, 0x6b21), u(0x4e4b, 0x524d)]
        relation_tokens = [u(0x54b1, 0x4fe9), u(0x6211, 0x4eec), u(0x4f60, 0x6211), u(0x8ddf, 0x6211), u(0x8ddf, 0x4f60)]
        activity_tokens = [u(0x4e00, 0x8d77), u(0x6253, 0x8fc7), u(0x89c1, 0x8fc7), u(0x53bb, 0x8fc7), u(0x73a9, 0x8fc7)]

        has_time = any(token in raw for token in time_tokens)
        has_relation = any(token in raw for token in relation_tokens)
        has_activity = any(token in raw for token in activity_tokens)
        return (has_time and has_relation and has_activity) or (u(0x4f60) in raw and has_time and has_activity)

    def _generate_single_text(
        self,
        user_id: str,
        user_text: str,
        state: RelationshipState,
        recent: list[Message],
        cls: dict,
        time_context: dict | None = None,
    ) -> str:
        priority_reply, priority_source = self._build_priority_template_reply(user_text, recent, time_context)
        if priority_reply:
            logger.info("single reply source=%s | user_text=%r | reply=%r", priority_source, user_text, priority_reply)
            return priority_reply
        if self.identity_guard.is_identity_probe(user_text):
            return self.identity_guard.get_reply(user_id=user_id, stage=state.stage)
        if self._should_deny_unverified_history() and self._looks_like_shared_history_claim(user_text) and not self._has_supporting_shared_history(user_text, recent):
            return self._deny_unverified_history(user_text)

        recalled = self.memory.recall_memories(
            user_id=user_id,
            query_text=user_text,
            limit=6,
            categories=["user_profile", "relationship_event"],
        )
        role_life = self.memory.recall_memories(
            user_id=user_id,
            query_text=user_text,
            limit=3,
            categories=["role_life"],
        )
        ctx = ResponseContext(
            user_id=user_id,
            latest_user_message=user_text,
            recent_messages=recent,
            profile=UserProfile(user_id=user_id),
            state=state,
            recalled_memories=recalled,
            role_life_memories=role_life,
            time_context=(time_context or {}),
        )
        style = self._build_style(user_id, state, user_text, cls, time_context=time_context)

        for _ in range(5):
            candidate = self.generator.generate(self.bible, style, ctx)
            candidate = self._apply_turn_style_adjustments(user_id, candidate, style)
            if self._leaks_ai_identity(candidate):
                continue
            if self._is_hostile_reply(candidate):
                continue
            if self._has_recent_phrase_repetition(candidate, recent, window=20):
                continue
            if not self.rep_guard.is_repetitive(user_id, candidate):
                return candidate
        return self._soft_deescalation_reply()

    @staticmethod
    def _parse_multi_replies(raw: str, max_replies: int = 3) -> list[str]:
        text = (raw or "").strip()
        if not text:
            return []
        try:
            match = re.search(r"\{[\s\S]*\}", text)
            data = json.loads(match.group(0) if match else text)
            replies = data.get("replies", [])
            if isinstance(replies, list):
                return [str(value).strip() for value in replies if str(value).strip()][:max_replies]
        except Exception:
            pass
        return [part.strip(" -\t") for part in re.split(r"\n+", text) if part.strip()][:max_replies]

    def _generate_multi_replies(
        self,
        user_id: str,
        latest_user_messages: list[str],
        state: RelationshipState,
        recent: list[Message],
        cls: dict,
        time_context: dict | None = None,
        max_replies: int = 3,
    ) -> list[str]:
        latest_text = latest_user_messages[-1]
        priority_reply, priority_source = self._build_priority_template_reply(latest_text, recent, time_context)
        if priority_reply:
            logger.info("multi reply source=%s | latest_text=%r | reply=%r", priority_source, latest_text, priority_reply)
            return [priority_reply]
        if self.identity_guard.is_identity_probe(latest_text):
            return [self.identity_guard.get_reply(user_id=user_id, stage=state.stage)]
        if self._should_deny_unverified_history() and self._looks_like_shared_history_claim(latest_text) and not self._has_supporting_shared_history(latest_text, recent):
            return [self._deny_unverified_history(latest_text)]

        recalled = self.memory.recall_memories(
            user_id=user_id,
            query_text=latest_text,
            limit=6,
            categories=["user_profile", "relationship_event"],
        )
        role_life = self.memory.recall_memories(
            user_id=user_id,
            query_text=latest_text,
            limit=3,
            categories=["role_life"],
        )
        ctx = ResponseContext(
            user_id=user_id,
            latest_user_message=latest_text,
            recent_messages=recent,
            profile=UserProfile(user_id=user_id),
            state=state,
            recalled_memories=recalled,
            role_life_memories=role_life,
            time_context=(time_context or {}),
        )
        style = self._build_style(user_id, state, latest_text, cls, time_context=time_context)
        if max_replies <= 1:
            reply_rule = """
额外任务：
- 用户刚刚连续发了多条消息，只给出 1 条短回复。
- 回复要自然承接上下文，像微信私聊。
- 回复长度控制在 5~20 字。
- 仅输出 JSON，格式：{"replies": ["回复1"]}
"""
        else:
            reply_rule = f"""
额外任务：
- 用户刚刚连续发了多条消息，请一次性给出 1~{max_replies} 条短回复。
- 回复之间要前后连贯，像连续发出的多条微信消息。
- 每条回复大多数控制在 5~15 字左右。
- 仅输出 JSON，格式：{{"replies": ["回复1", "回复2"]}}
"""
        system_prompt = build_system_prompt(self.bible, style) + reply_rule
        batch_lines = "\n".join([f"- {text}" for text in latest_user_messages if text.strip()])
        user_prompt = build_user_prompt(ctx, role_name=self.role_name) + f"\n\n本次用户连续消息：\n{batch_lines}\n"

        for _ in range(3):
            raw = self.generator.chat(system_prompt, user_prompt, temperature=0.75)
            replies = self._parse_multi_replies(raw, max_replies=max_replies)
            cleaned = []
            question_injected = False
            for reply in replies:
                style_for_reply = dict(style)
                if question_injected:
                    style_for_reply["proactive_question_mode"] = False
                reply = self._apply_turn_style_adjustments(user_id, reply, style_for_reply)
                if ("？" in reply or "?" in reply) and style_for_reply.get("proactive_question_mode"):
                    question_injected = True
                if self._leaks_ai_identity(reply):
                    continue
                if self._is_hostile_reply(reply):
                    continue
                if self._has_recent_phrase_repetition(reply, recent, window=20):
                    continue
                if self.rep_guard.is_repetitive(user_id, reply):
                    continue
                cleaned.append(reply)
            if cleaned:
                return cleaned
        return [self._generate_single_text(user_id, latest_text, state, recent, cls, time_context=time_context)]

    def _build_priority_template_reply(self, latest_text: str, recent: list[Message], time_context: dict | None = None) -> tuple[str | None, str | None]:
        # Let the model handle most conversational and sales decisions via prompt.
        # Keep only a narrow set of deterministic factual drink lookups here.
        drink_menu_reply = self._build_drink_menu_reply(latest_text, recent=recent)
        if drink_menu_reply:
            return drink_menu_reply, "drink_menu"
        return None, None

    def _record_assistant_reply(self, user_id: str, reply: str):
        self.rep_guard.add_reply(user_id, reply)
        self._add_message(user_id, "assistant", reply)
        if any(token in reply for token in ["累", "困", "忙", "饿", "下班"]):
            self.memory.add_role_life_memory(
                user_id,
                f"{self.role_name}提到：{reply}",
                importance=0.4,
            )

    def chat(self, user_id: str, user_text: str) -> dict:
        incoming_text = (user_text or "").strip()
        cls = classify_user_message(incoming_text)
        is_greeting = self._is_greeting_message(incoming_text)
        prior_recent = list(self._get_recent(user_id))
        had_context_before = self._has_recent_context(prior_recent)
        time_context = self._build_time_context(user_id, prior_recent)

        self.turn_counts[user_id] = self._get_turn_count(user_id) + 1
        self.memory.set_turn_count(user_id, self.turn_counts[user_id])
        self._add_message(user_id, "user", incoming_text)
        state, recent = self._update_state_and_memories_from_user(user_id, incoming_text)

        priority_reply, priority_source = self._build_priority_template_reply(incoming_text, recent, time_context)
        if self._is_quiet_hours() and not cls["urgent"]:
            decision = ReplyDecision(type="noreply", recommended_delay_ms=0, reason="quiet_hours")
        elif priority_reply:
            logger.info("chat reply source=%s | user_text=%r | reply=%r", priority_source, incoming_text, priority_reply)
            decision = self._finalize_reply(
                state,
                priority_reply,
                priority_source or "template_reply",
                urgent=cls["urgent"],
                time_context=time_context,
                latest_user_text=incoming_text,
                recent=recent,
            )
        elif (not is_greeting) and self._is_repeated_user_spam(recent, incoming_texts=[incoming_text]):
            decision = self._build_repeat_spam_decision()
        elif self._should_acknowledgement_noreply(incoming_text, recent, cls):
            decision = ReplyDecision(type="noreply", recommended_delay_ms=0, reason="acknowledgement_no_reply")
        elif self._is_duplicate_image_request(incoming_text, recent):
            decision = ReplyDecision(type="noreply", recommended_delay_ms=0, reason="duplicate_image_request")
        elif (not is_greeting) and cls["low_context"] and not had_context_before:
            decision = ReplyDecision(
                type="reply",
                text="你想说什么呀",
                recommended_delay_ms=self.rng.randint(500, 1200),
                reason="low_context_no_history",
            )
        elif self._should_probe_then_comfort(incoming_text, cls):
            decision = self._finalize_reply(
                state,
                self._probe_then_comfort_reply(),
                "emotion_probe_then_comfort",
                urgent=False,
                time_context=time_context,
                latest_user_text=incoming_text,
                recent=recent,
            )
        elif self._should_force_noreply(recent, cls):
            decision = ReplyDecision(type="noreply", recommended_delay_ms=0, reason="need_no_reply")
        elif cls["hostile"] and not cls["urgent"]:
            decision = self._finalize_reply(
                state,
                self._soft_deescalation_reply(),
                "hostile_deescalation",
                urgent=False,
                time_context=time_context,
                latest_user_text=incoming_text,
                recent=recent,
            )
        else:
            text = self._generate_single_text(user_id, incoming_text, state, recent, cls, time_context=time_context)
            decision = self._finalize_reply(
                state,
                text,
                "reply",
                urgent=cls["urgent"],
                time_context=time_context,
                latest_user_text=incoming_text,
                recent=recent,
            )

        if decision.type == "reply" and decision.text:
            self._record_assistant_reply(user_id, decision.text)
        self._bump_policy_stat(decision)
        self._maybe_run_summary(user_id)
        return {
            "type": decision.type,
            "text": decision.text,
            "recommended_delay_ms": decision.recommended_delay_ms,
            "reason": decision.reason,
        }

    def sync_user_messages(self, user_id: str, messages: list[str]) -> dict:
        synced = 0
        for raw in messages:
            text = (raw or "").strip()
            if not text:
                continue
            self._add_message(user_id, "user", text)
            self._update_state_and_memories_from_user(user_id, text)
            synced += 1
        return {
            "user_id": user_id,
            "synced_count": synced,
            "state": self.get_state_snapshot(user_id),
        }

    def suggest_proactive_message(self, user_id: str, force: bool = False) -> ProactivePlan:
        state = self._get_state(user_id)
        recent = self._get_recent(user_id)
        last_sent = self.memory.get_last_proactive_at(user_id)
        if last_sent:
            self.proactive_scheduler.last_proactive_at[user_id] = last_sent
        return self.proactive_scheduler.suggest(
            user_id=user_id,
            state=state,
            recent_messages=recent,
            now=datetime.now(self.local_timezone),
            force=force,
        )

    def mark_proactive_sent(self, user_id: str):
        self.proactive_scheduler.mark_sent(user_id)
        dt = self.proactive_scheduler.last_proactive_at.get(user_id)
        if dt:
            self.memory.save_last_proactive_at(user_id, dt)

    def get_state_snapshot(self, user_id: str) -> dict:
        state = self._get_state(user_id)
        return {
            "stage": state.stage,
            "mood": state.mood,
            "warmth": state.warmth,
            "initiative": state.initiative,
            "disclosure": state.disclosure,
            "flirty": state.flirty,
            "turn_count": self._get_turn_count(user_id),
        }

    def get_chat_context(self, user_id: str, limit: int = 20) -> dict:
        recent = self.memory.get_recent_chat_messages(user_id=user_id, limit=max(1, limit))
        messages = [{"role": item.role, "content": item.content, "ts": item.ts.isoformat()} for item in recent]
        return {
            "user_id": user_id,
            "state": self.get_state_snapshot(user_id),
            "message_count": len(messages),
            "messages": messages,
        }

    def list_users(self) -> list[dict]:
        user_ids = set(self.memory.list_user_ids()) | set(self.user_states.keys()) | set(self.recent_messages.keys()) | set(self.turn_counts.keys())
        rows = []
        for uid in sorted(user_ids):
            rows.append(
                {
                    "user_id": uid,
                    "turn_count": self._get_turn_count(uid),
                    "message_count": self.memory.count_chat_messages(uid),
                    "last_message_ts": self.memory.get_last_message_ts(uid),
                }
            )
        return rows

    def ingest_incoming_message(self, username: str, message: str, user_id: str | None = None) -> dict:
        return self.ingest_incoming_messages(username=username, messages=[message], user_id=user_id)

    def ingest_incoming_messages(self, username: str, messages: list[str], user_id: str | None = None) -> dict:
        uid = (user_id or username or "").strip() or "default"
        name = (username or uid).strip() or uid
        cleaned = [str(value).strip() for value in (messages or []) if str(value).strip()]
        if not cleaned:
            return {"ok": False, "reason": "empty_message"}
        is_greeting_batch = all(self._is_greeting_message(text) for text in cleaned)

        prior_recent = list(self._get_recent(uid))
        had_context_before = self._has_recent_context(prior_recent)
        time_context = self._build_time_context(uid, prior_recent)
        classifications = [classify_user_message(text) for text in cleaned]
        combined = {
            "low_context": all(item["low_context"] for item in classifications),
            "hostile": any(item["hostile"] for item in classifications),
            "boring": all(item["boring"] for item in classifications),
            "need_no_reply": any(item["need_no_reply"] for item in classifications),
            "urgent": any(item["urgent"] for item in classifications),
        }

        for text in cleaned:
            self.turn_counts[uid] = self._get_turn_count(uid) + 1
            self.memory.set_turn_count(uid, self.turn_counts[uid])
            self._add_message(uid, "user", text)
            self._update_state_and_memories_from_user(uid, text)

        state = self._get_state(uid)
        recent = self._get_recent(uid)
        queued_items = []

        priority_reply, priority_source = self._build_priority_template_reply(cleaned[-1], recent, time_context)
        if self._is_quiet_hours() and not combined["urgent"]:
            decisions = [ReplyDecision(type="noreply", recommended_delay_ms=0, reason="quiet_hours")]
        elif priority_reply:
            logger.info("batch reply source=%s | latest_text=%r | reply=%r", priority_source, cleaned[-1], priority_reply)
            decisions = [
                self._finalize_reply(
                    state,
                    priority_reply,
                    priority_source or "template_reply",
                    urgent=combined["urgent"],
                    time_context=time_context,
                    latest_user_text=cleaned[-1],
                    recent=recent,
                )
            ]
        elif (not is_greeting_batch) and self._is_repeated_user_spam(recent, incoming_texts=cleaned):
            decisions = [self._build_repeat_spam_decision()]
        elif len(cleaned) == 1 and self._should_acknowledgement_noreply(cleaned[0], recent, combined):
            decisions = [ReplyDecision(type="noreply", recommended_delay_ms=0, reason="acknowledgement_no_reply")]
        elif len(cleaned) == 1 and self._is_duplicate_image_request(cleaned[0], recent):
            decisions = [ReplyDecision(type="noreply", recommended_delay_ms=0, reason="duplicate_image_request")]
        elif (not is_greeting_batch) and combined["low_context"] and not had_context_before:
            decisions = [
                ReplyDecision(
                    type="reply",
                text="你想说什么呀",
                    recommended_delay_ms=self.rng.randint(500, 1200),
                    reason="low_context_no_history",
                )
            ]
        elif self._should_probe_then_comfort(cleaned[-1], combined):
            decisions = [
                self._finalize_reply(
                    state,
                    self._probe_then_comfort_reply(),
                    "emotion_probe_then_comfort",
                    urgent=False,
                    time_context=time_context,
                    latest_user_text=cleaned[-1],
                    recent=recent,
                )
            ]
        elif self._should_force_noreply(recent, combined):
            decisions = [ReplyDecision(type="noreply", recommended_delay_ms=0, reason="need_no_reply")]
        elif combined["hostile"] and not combined["urgent"]:
            decisions = [
                self._finalize_reply(
                    state,
                    self._soft_deescalation_reply(),
                    "hostile_deescalation",
                    urgent=False,
                    time_context=time_context,
                    latest_user_text=cleaned[-1],
                    recent=recent,
                )
            ]
        else:
            allow_multi = bool(combined["urgent"])
            raw_replies = self._generate_multi_replies(
                uid,
                cleaned,
                state=state,
                recent=recent,
                cls=combined,
                time_context=time_context,
                max_replies=(max(2, self.max_batch_replies) if allow_multi else 1),
            )
            decisions = [
                self._finalize_reply(
                    state,
                    reply,
                    "reply",
                    urgent=combined["urgent"],
                    time_context=time_context,
                    latest_user_text=cleaned[-1],
                    recent=recent,
                )
                for reply in raw_replies
            ]

        with self._pending_lock:
            for decision in decisions:
                self._bump_policy_stat(decision)
                if decision.type != "reply" or not decision.text:
                    continue
                self._record_assistant_reply(uid, decision.text)
                self._pending_seq += 1
                item = {
                    "id": self._pending_seq,
                    "user_id": uid,
                    "username": name,
                    "reply": decision.text,
                    "recommended_delay_ms": decision.recommended_delay_ms,
                    "created_at": datetime.now(UTC).isoformat(),
                }
                self.pending_replies.append(item)
                queued_items.append(item)

        self._maybe_run_summary(uid)
        if not queued_items and decisions and decisions[0].type == "noreply":
            return {
                "ok": True,
                "reason": decisions[0].reason,
                "queued": None,
                "queued_items": [],
                "state": self.get_state_snapshot(uid),
            }
        return {
            "ok": True,
            "reason": "queued",
            "queued": queued_items[0] if queued_items else None,
            "queued_items": queued_items,
            "state": self.get_state_snapshot(uid),
        }

    def fetch_pending_replies(self, limit: int = 1, pop: bool = True) -> list[dict]:
        n = max(1, int(limit))
        out = []
        with self._pending_lock:
            if pop:
                for _ in range(min(n, len(self.pending_replies))):
                    out.append(self.pending_replies.popleft())
            else:
                out = list(self.pending_replies)[:n]
        return out

    def get_reply_policy_stats(self) -> dict:
        with self._pending_lock:
            pending_count = len(self.pending_replies)
        data = dict(self.reply_policy_stats)
        data["pending_count"] = pending_count
        data["known_users"] = len(self.list_users())
        return data


# Backward compatibility alias for old imports.
GuoguoEngine = CompanionEngine


def main():
    engine = CompanionEngine()
    role_label = engine.role_name or "assistant"
    print(f"{role_label} engine started. Type `exit` to quit.")
    while True:
        text = input("you> ").strip()
        if text.lower() == "exit":
            break
        result = engine.chat("cli_user", text)
        print(f"{role_label}>", result)


if __name__ == "__main__":
    main()

