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

    def __init__(self):
        load_dotenv()
        self.llm_provider = os.getenv("LLM_PROVIDER", "glm").strip().lower()
        glm_api_key = os.getenv("GLM_API_KEY", "")
        openai_api_key = os.getenv("OPENAI_API_KEY", "")
        chat_model = os.getenv("GLM_CHAT_MODEL", "glm-4.7")
        embed_model = os.getenv("GLM_EMBED_MODEL", "embedding-3")
        llm_api_key = glm_api_key
        if self.llm_provider == "openai":
            chat_model = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
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

        self.role_bible_path = os.getenv("ROLE_BIBLE_PATH", "data/guoguo_bible.json")
        self.bible = BibleLoader(self.role_bible_path).load()
        self.role_name = str(self.bible.get("name", "") or "").strip() or "unknown"
        self.role_id = str(self.bible.get("role_id", "") or "").strip() or "unknown"
        self.embedding = EmbeddingService(api_key=glm_api_key, model=embed_model)
        self.identity_guard = IdentityGuard("data/identity_deflection_templates.json")
        self.memory = MemoryManager("memory.db", embedding_service=self.embedding)
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
    def _marketing_triggered(text: str, bible: dict) -> bool:
        marketing = bible.get("marketing") or {}
        triggers = marketing.get("triggers") or []
        raw = (text or "").strip()
        return any(str(trigger).strip() and str(trigger) in raw for trigger in triggers)

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

    def _build_style(self, user_id: str, state: RelationshipState, user_text: str, cls: dict) -> dict:
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

    def _finalize_reply(self, state: RelationshipState, raw_reply: str, reason: str, urgent: bool) -> ReplyDecision:
        # Keep model output as-is for context consistency.
        processed = (raw_reply or "").strip()
        processed = self._sanitize_surveillance_tone(processed)
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

    def _generate_single_text(self, user_id: str, user_text: str, state: RelationshipState, recent: list[Message], cls: dict) -> str:
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
        )
        style = self._build_style(user_id, state, user_text, cls)

        for _ in range(5):
            candidate = self.generator.generate(self.bible, style, ctx)
            candidate = self._apply_turn_style_adjustments(user_id, candidate, style)
            if self._leaks_ai_identity(candidate):
                continue
            if self._is_hostile_reply(candidate):
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
        max_replies: int = 3,
    ) -> list[str]:
        latest_text = latest_user_messages[-1]
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
        )
        style = self._build_style(user_id, state, latest_text, cls)
        system_prompt = build_system_prompt(self.bible, style) + f"""
额外任务：
- 用户刚刚连续发了多条消息，请一次性给出 1~3 条短回复。
- 回复之间要前后连贯，像连续发出的多条微信消息。
- 每条回复大多数控制在 5~15 字左右。
- 仅输出 JSON，格式：{{"replies": ["回复1", "回复2"]}}
"""
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
                if self.rep_guard.is_repetitive(user_id, reply):
                    continue
                cleaned.append(reply)
            if cleaned:
                return cleaned
        return [self._generate_single_text(user_id, latest_text, state, recent, cls)]

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
        had_context_before = self._has_recent_context(self._get_recent(user_id))

        self.turn_counts[user_id] = self._get_turn_count(user_id) + 1
        self.memory.set_turn_count(user_id, self.turn_counts[user_id])
        self._add_message(user_id, "user", incoming_text)
        state, recent = self._update_state_and_memories_from_user(user_id, incoming_text)

        if self._is_quiet_hours() and not cls["urgent"]:
            decision = ReplyDecision(type="noreply", recommended_delay_ms=0, reason="quiet_hours")
        elif (not is_greeting) and self._is_repeated_user_spam(recent, incoming_texts=[incoming_text]):
            decision = self._build_repeat_spam_decision()
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
            )
        elif self._should_force_noreply(recent, cls):
            decision = ReplyDecision(type="noreply", recommended_delay_ms=0, reason="need_no_reply")
        elif cls["hostile"] and not cls["urgent"]:
            decision = self._finalize_reply(state, self._soft_deescalation_reply(), "hostile_deescalation", urgent=False)
        else:
            text = self._generate_single_text(user_id, incoming_text, state, recent, cls)
            decision = self._finalize_reply(state, text, "reply", urgent=cls["urgent"])

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

        had_context_before = self._has_recent_context(self._get_recent(uid))
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
                )
            ]
        elif self._should_force_noreply(recent, combined):
            decisions = [ReplyDecision(type="noreply", recommended_delay_ms=0, reason="need_no_reply")]
        elif combined["hostile"] and not combined["urgent"]:
            decisions = [self._finalize_reply(state, self._soft_deescalation_reply(), "hostile_deescalation", urgent=False)]
        else:
            allow_multi = combined["urgent"] or (self.rng.random() < self.multi_reply_probability)
            raw_replies = self._generate_multi_replies(
                uid,
                cleaned,
                state=state,
                recent=recent,
                cls=combined,
                max_replies=(max(2, self.max_batch_replies) if allow_multi else 1),
            )
            decisions = [self._finalize_reply(state, reply, "reply", urgent=combined["urgent"]) for reply in raw_replies]

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

