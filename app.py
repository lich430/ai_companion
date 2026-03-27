from __future__ import annotations

import json
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
            self.generator = OpenAIResponseGenerator(api_key=llm_api_key, model=chat_model)
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

    def _build_sales_template_reply(self, latest_text: str, recent: list[Message], time_context: dict) -> str | None:
        latest = (latest_text or "").strip()
        if not latest:
            return None

        period = str(time_context.get("time_period_name", "") or "").strip()
        if period not in {"night_work_window", "dinner_window"}:
            return None

        prior_messages = recent[:-1] if recent else []
        last_assistant = next((m.content.strip() for m in reversed(prior_messages) if m.role == "assistant" and m.content.strip()), "")
        party_size = self._extract_party_size(latest)
        room_type = self._extract_room_type(latest)
        eta = self._extract_eta_phrase(latest)

        if self._contains_any(latest, ["在不在公司", "在公司吗", "在店里吗", "在店吗", "在不在店"]):
            return "在公司呢"

        if self._contains_any(latest, ["你从事什么工作", "你做什么工作", "做什么工作", "你是做什么的", "你在什么公司", "做什么的"]):
            return "我在夜场上班 平时陪客人喝酒聊天 也会帮忙订房安排包厢"

        if self._contains_any(latest, ["今天晚上过去玩", "晚上过去玩", "过去玩", "过去喝酒", "来喝酒", "过来喝酒"]):
            return "好 你们几个人"

        if self._contains_any(latest, ["今天能不能送点酒", "能不能送点酒", "送点酒", "送酒"]):
            return "可以 到时候你点一些 我送一点"

        if self._contains_any(latest, ["带点能玩的进来", "带点能玩的", "能玩的进来", "能玩不给小费", "不给小费"]):
            return "放心 不能玩不给小费"

        if room_type and (party_size or self._contains_any(latest, ["给我留", "留个", "留一间", "开个"])):
            room_label = room_type if room_type != "包厢" else "包厢"
            if party_size:
                return f"嗯嗯 {room_label}给你们留着 你们几点到"
            return f"嗯嗯 {room_label}可以给你留 你们几点到"

        if eta or self._contains_any(latest, ["快结束了", "一会到", "差不多到了", "四十分钟", "半小时", "一个小时"]):
            return "好的 快到了给我发信息 我去门口接你们"

        if party_size and self._contains_any(last_assistant, ["几个人"]):
            return f"{party_size}个人的话给你们安排 你想留小包还是中包"

        if room_type and self._contains_any(last_assistant, ["几点到", "到店时间"]):
            room_label = room_type if room_type != "包厢" else "包厢"
            return f"好 {room_label}先给你留着 你们快到了跟我说"

        if self._contains_any(latest, ["订房", "开包厢", "留个包", "留位", "安排一下"]):
            return "可以 你们几个人 我先给你安排"

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
        prior_messages = history[:-1] if history and history[-1].role == "user" and history[-1].content.strip() == raw else history
        last_assistant = next((m.content.strip() for m in reversed(prior_messages) if m.role == "assistant" and m.content.strip()), "")
        last_user = next((m.content.strip() for m in reversed(prior_messages) if m.role == "user" and m.content.strip()), "")
        context_query = last_user if ("详细介绍" in last_assistant or "细说" in last_assistant or "发下酒单" in last_assistant) else raw

        query_for_parse = context_query if ("详细介绍" in last_assistant or "细说" in last_assistant or "发下酒单" in last_assistant) else raw
        matched_category, matched_item = self._find_drink_item(context_query)
        matched_candidates = self._find_drink_candidates(context_query)
        asked_specific = bool(re.search(r"(有(?!什么).{1,12}吗|有没有.{1,12}|能点.{1,12}吗|能不能点.{1,12})", query_for_parse))
        asked_menu = (
            any(token in query_for_parse for token in ["有什么酒", "都有什么酒", "酒单", "酒水单", "酒水", "有什么喝的", "能喝什么"])
            or bool(re.search(r"有什么(洋酒|红酒|啤酒|饮料)", query_for_parse))
            or bool(re.search(r"有(洋酒|红酒|啤酒|饮料)吗", query_for_parse))
        )
        asked_detail = (
            any(token in raw for token in ["详细", "细说", "具体", "介绍一下", "都说下", "全部", "价格", "酒单发我", "发来看看"])
            or (self._is_acknowledgement_message(raw) and ("详细介绍" in last_assistant or "细说" in last_assistant or "发下酒单" in last_assistant))
            or raw in {"需要", "你说", "发我看看", "说说看"}
        )
        inferred_category = self._infer_drink_category(context_query)
        if any(token in raw for token in ["多少钱一箱", "一箱多少钱", "一箱多钱", "啤酒多少一箱", "纯生多少一箱", "雪花多少一箱"]):
            marketing = self.bible.get("marketing") or {}
            store_info = marketing.get("store_info") or {}
            drink_info = store_info.get("drink_info") or {}
            beer_box = str(drink_info.get("beer", "") or "").strip()
            if beer_box:
                return f"啤酒一箱{beer_box}"
            return "啤酒一箱一千多"

        if asked_specific and len(matched_candidates) >= 2:
            price_text = "、".join(self._format_short_price(item) for _, item in matched_candidates[:2])
            return f"有 {price_text}"

        if matched_item and asked_specific:
            return f"有 {self._format_short_price(matched_item)}"

        if asked_specific and not matched_item:
            return "没有"

        if asked_menu and asked_detail:
            target_categories = [inferred_category] if inferred_category and inferred_category in catalog else [x for x in ["洋酒", "红酒", "啤酒"] if x in catalog]
            if inferred_category == "饮料":
                target_categories = ["饮料"]
            lines = []
            for category in target_categories:
                listing = self._list_catalog_by_category(category)
                if listing:
                    lines.append(f"{category}有{listing}")
            if "饮料" in catalog and (inferred_category == "饮料" or "饮料" in raw):
                listing = self._list_catalog_by_category("饮料")
                if listing:
                    lines.append(f"饮料有{listing}")
            return "\n".join(lines[:3]) if lines else None

        if matched_item:
            return f"有 {self._format_short_price(matched_item)}"

        if asked_menu:
            if inferred_category and inferred_category in catalog:
                return f"有 {inferred_category}这边都能点"
            return "有的 洋酒红酒啤酒这些都有"

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

    def _should_force_noreply(self, recent: list[Message], cls: dict) -> bool:
        if cls.get("urgent"):
            return False
        if not (cls.get("hostile") or cls.get("boring") or cls.get("need_no_reply")):
            return False

        recent_user = [m.content.strip() for m in recent[-6:] if m.role == "user"]
        if len(recent_user) < 3:
            return False

        boring_count = sum(1 for text in recent_user if classify_user_message(text)["boring"])
        hostile_count = sum(1 for text in recent_user if classify_user_message(text)["hostile"])
        return hostile_count >= 2 or boring_count >= 3

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

        # consecutive duplicate burst, e.g. "1 1 1" or "想你 想你 想你"
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

        # high-frequency repeats in a short window
        freq: dict[str, int] = {}
        for text in window:
            freq[text] = freq.get(text, 0) + 1
        if dominant is None:
            for text, count in freq.items():
                if count >= 4:
                    dominant = text
                    break
                if len(text) <= 3 and count >= 3:
                    dominant = text
                    break

        if dominant is None:
            return False

        # Exemption: if current incoming contains a new non-repeated content
        # different from dominant spam text, do not block.
        if incoming_texts:
            incoming_norm = [str(x).strip() for x in incoming_texts if str(x).strip()]
            if any(text != dominant and freq.get(text, 0) <= 2 for text in incoming_norm):
                return False
        return True

    def _build_repeat_spam_decision(self) -> ReplyDecision:
        # 50% reply with a single "？", 50% no reply.
        if self.rng.random() < 0.5:
            return ReplyDecision(
                type="reply",
                text="？",
                recommended_delay_ms=self.rng.randint(500, 1200),
                reason="repeat_spam_question",
            )
        return ReplyDecision(type="noreply", recommended_delay_ms=0, reason="repeat_spam")

    @staticmethod
    def _looks_like_shared_history_claim(text: str) -> bool:
        raw = (text or "").strip()
        claim_patterns = [
            "还一起",
            "不是还",
            "你上次赢我",
            "你上次输我",
            "咱俩还",
        ]
        if any(pattern in raw for pattern in claim_patterns):
            return True
        regex_patterns = [
            r"(上次|之前).{0,8}(咱俩|我们|你我|跟我|跟你).{0,8}(一起|打过|见过|去过|玩过)",
            r"(你|咱俩|我们).{0,8}(上次|之前).{0,8}(一起|打过|见过|去过|玩过)",
        ]
        return any(re.search(pattern, raw) for pattern in regex_patterns)

    def _shared_history_categories(self) -> dict:
        policy = (self.bible.get("shared_history_policy") or {})
        return policy.get("categories") or {}

    def _detect_shared_history_category(self, text: str) -> str:
        raw = (text or "").strip()
        categories = self._shared_history_categories()
        for category, conf in categories.items():
            if category == "default":
                continue
            keywords = conf.get("keywords") or []
            if any(keyword in raw for keyword in keywords):
                return category
        return "default"

    def _should_deny_unverified_history(self) -> bool:
        policy = self.bible.get("shared_history_policy") or {}
        return bool(policy.get("enabled", False)) and bool(policy.get("deny_unverified", False))

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

    def _get_state(self, user_id: str) -> RelationshipState:
        if user_id not in self.user_states:
            state = self.memory.get_user_state(user_id)
            if state is None:
                state = RelationshipState(user_id=user_id)
                state = self.state_machine.apply_stage_defaults(state)
                self.memory.save_user_state(state)
            self.user_states[user_id] = self.state_machine.apply_stage_defaults(state)
        return self.user_states[user_id]

    def _add_message(self, user_id: str, role: str, content: str):
        msg = Message(role=role, content=content)
        self.memory.add_chat_message(user_id=user_id, role=role, content=content, ts=msg.ts)
        self.recent_messages.setdefault(user_id, []).append(msg)
        self.recent_messages[user_id] = self.recent_messages[user_id][-self.glm_context_messages :]

    def _get_recent(self, user_id: str) -> list[Message]:
        if user_id not in self.recent_messages:
            self.recent_messages[user_id] = self.memory.get_recent_chat_messages(
                user_id=user_id,
                limit=self.glm_context_messages,
            )
        return self.recent_messages.setdefault(user_id, [])

    def _get_turn_count(self, user_id: str) -> int:
        if user_id not in self.turn_counts:
            self.turn_counts[user_id] = self.memory.get_turn_count(user_id)
        return self.turn_counts[user_id]

    def _maybe_run_summary(self, user_id: str):
        if not self.summary_every_n_turns:
            return
        if self.turn_counts.get(user_id, 0) % self.summary_every_n_turns != 0:
            return
        state = self._get_state(user_id)
        recent = self._get_recent(user_id)
        items = self.summarizer.summarize(user_id, recent, state)
        for item in items:
            self.memory.add_memory(item)

    def _update_state_and_memories_from_user(self, user_id: str, user_text: str) -> tuple[RelationshipState, list[Message]]:
        state = self._get_state(user_id)
        recent = self._get_recent(user_id)
        state = self.state_machine.update_from_message(state, user_text, recent)
        state.last_updated = datetime.now(UTC)
        self.user_states[user_id] = state
        self.memory.save_user_state(state)
        self.memory.extract_and_store_user_memory(user_id, user_text)
        return state, recent

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
        processed = self._sanitize_surveillance_tone(processed)
        processed = self._sanitize_excessive_tildes(processed)
        recent_msgs = recent or []
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
        # Remove overly permissive tail phrase.
        out = re.sub(r"(，|,)?\s*只要不过分(就行|就好|就可以)?[。.!！?？]*$", "", out).strip()
        return out

    @staticmethod
    def _sanitize_excessive_tildes(text: str) -> str:
        out = (text or "").strip()
        if not out:
            return out
        out = re.sub(r"[~～]{2,}", "~", out)
        parts = [part.strip() for part in re.split(r"\n+", out) if part.strip()]
        cleaned = []
        for part in parts:
            if part.endswith(("~", "～")):
                body = part.rstrip("~～").strip()
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

        # During the night work window, avoid saying the role has already clocked off.
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
                out = re.sub(r"(这个点|这会儿|这时候)?才?下班了", "这会儿还在忙", out)
                out = re.sub(r"刚下班", "刚忙完一阵", out)
                out = re.sub(r"(已经|都)?收工了", "还没收工", out)
                out = re.sub(r"下班[。.!！?？]*$", "还没下班", out)
        return out.strip()

    @staticmethod
    def _sanitize_conversion_blockers(text: str, latest_user_text: str, time_context: dict) -> str:
        out = (text or "").strip()
        if not out:
            return out

        latest = (latest_user_text or "").strip()
        period = str(time_context.get("time_period_name", "") or "").strip()
        high_intent = any(
            token in latest
            for token in ["过去", "来喝酒", "喝喝酒", "留个小包", "小包", "订房", "开包厢", "你陪我", "安排", "一个人"]
        )
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

        out = re.sub(r"(但|不过)?我还在忙[,，]?所以?", "", out).strip(" ，,")
        out = re.sub(r"(到时候|回头|再说|看情况)", "我给你安排", out)
        out = re.sub(r"\s{2,}", " ", out).strip()
        return out

    @staticmethod
    def _sanitize_forbidden_invite(text: str) -> str:
        out = (text or "").strip()
        if not out:
            return out

        replacements = {
            "来店里": "来找我",
            "到店": "来找我",
            "来消费": "来聊聊",
            "来玩": "来聊聊",
            "来坐坐": "来聊聊",
            "过来店里": "来找我",
            "来我店里": "来找我",
        }
        for src, dst in replacements.items():
            out = out.replace(src, dst)

        # Strip remaining explicit store-pull wording if still present.
        out = re.sub(r"(快)?来找我让我哄哄你(吧|嘛|呀|啊)?", "我在这陪你", out)
        out = re.sub(r"(快)?来.*?(店里|到店|消费|玩).*", "我在这陪你", out)
        out = re.sub(r"\s{2,}", " ", out).strip()
        return out

    @staticmethod
    def _normalize_phrase_anchor(text: str) -> str:
        return re.sub(r"[^\u4e00-\u9fff0-9A-Za-z]+", "", (text or "")).strip().lower()

    def _extract_phrase_anchors(self, text: str) -> list[str]:
        parts = [x.strip() for x in re.split(r"[，。！？!?,\n]", text or "") if x.strip()]
        anchors: list[str] = []
        skip_prefixes = {
            "你好",
            "哈喽",
            "在吗",
            "你呢",
            "好的",
            "是吗",
            "哈哈",
            "我在",
            "行啊",
        }
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
        # Keep a single short line by default to avoid over-splitting into many messages.
        compact = re.sub(r"\s+", " ", raw.replace("\n", " ")).strip()
        if len(compact) <= max_chars:
            return compact
        return compact[:max_chars].rstrip("，。！？,.!?;；:：")

    def _generate_single_text(
        self,
        user_id: str,
        user_text: str,
        state: RelationshipState,
        recent: list[Message],
        cls: dict,
        time_context: dict | None = None,
    ) -> str:
        drink_menu_reply = self._build_drink_menu_reply(user_text, recent=recent)
        if drink_menu_reply:
            return drink_menu_reply
        template_reply = self._build_sales_template_reply(user_text, recent, time_context or {})
        if template_reply:
            return template_reply
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
        drink_menu_reply = self._build_drink_menu_reply(latest_text, recent=recent)
        if drink_menu_reply:
            return [drink_menu_reply]
        template_reply = self._build_sales_template_reply(latest_text, recent, time_context or {})
        if template_reply:
            return [template_reply]
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

        if self._is_quiet_hours() and not cls["urgent"]:
            decision = ReplyDecision(type="noreply", recommended_delay_ms=0, reason="quiet_hours")
        elif (not is_greeting) and self._is_repeated_user_spam(recent, incoming_texts=[incoming_text]):
            decision = self._build_repeat_spam_decision()
        elif self._should_acknowledgement_noreply(incoming_text, recent, cls):
            decision = ReplyDecision(type="noreply", recommended_delay_ms=0, reason="acknowledgement_no_reply")
        elif (not is_greeting) and cls["low_context"] and not had_context_before:
            decision = ReplyDecision(
                type="reply",
                text="？",
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

        if self._is_quiet_hours() and not combined["urgent"]:
            decisions = [ReplyDecision(type="noreply", recommended_delay_ms=0, reason="quiet_hours")]
        elif (not is_greeting_batch) and self._is_repeated_user_spam(recent, incoming_texts=cleaned):
            decisions = [self._build_repeat_spam_decision()]
        elif len(cleaned) == 1 and self._should_acknowledgement_noreply(cleaned[0], recent, combined):
            decisions = [ReplyDecision(type="noreply", recommended_delay_ms=0, reason="acknowledgement_no_reply")]
        elif (not is_greeting_batch) and combined["low_context"] and not had_context_before:
            decisions = [
                ReplyDecision(
                    type="reply",
                    text="？",
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
            allow_multi = combined["urgent"] or (self.rng.random() < self.multi_reply_probability)
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

