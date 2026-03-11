from datetime import UTC, datetime
from threading import Lock


from app import GuoguoEngine
from models.schemas import Message, RelationshipState


class DummyMemory:
    def __init__(self):
        self.turns = {}
        self.states = {}
        self.messages = {}

    def get_user_state(self, user_id):
        return self.states.get(user_id)

    def save_user_state(self, state):
        self.states[state.user_id] = state

    def add_chat_message(self, user_id, role, content, ts=None):
        self.messages.setdefault(user_id, []).append(Message(role=role, content=content, ts=ts or datetime.now(UTC)))

    def get_recent_chat_messages(self, user_id, limit=40):
        return self.messages.get(user_id, [])[-limit:]

    def get_turn_count(self, user_id):
        return self.turns.get(user_id, 0)

    def set_turn_count(self, user_id, turn_count):
        self.turns[user_id] = turn_count

    def extract_and_store_user_memory(self, user_id, user_text):
        return None

    def recall_memories(self, **kwargs):
        return []

    def add_role_life_memory(self, user_id, content, importance=0.4):
        return None

    def add_memory(self, item):
        return None

    def get_last_proactive_at(self, user_id):
        return None

    def save_last_proactive_at(self, user_id, dt):
        return None

    def count_chat_messages(self, user_id):
        return len(self.messages.get(user_id, []))

    def get_last_message_ts(self, user_id):
        return None

    def list_user_ids(self):
        return list(self.messages.keys())


class DummyStateMachine:
    def apply_stage_defaults(self, state):
        return state

    def update_from_message(self, state, latest_user_message, recent_messages):
        return state


class DummyRepGuard:
    def is_repetitive(self, user_id, reply):
        return False

    def add_reply(self, user_id, reply):
        return None


class FixedRng:
    def __init__(self):
        self.seed_values = [0.99, 0.99]

    def random(self):
        if self.seed_values:
            return self.seed_values.pop(0)
        return 0.99

    def choice(self, seq):
        return seq[0]

    def randint(self, a, b):
        return a


def make_engine():
    engine = GuoguoEngine.__new__(GuoguoEngine)
    engine.bible = {
        "marketing": {"triggers": []},
        "conflict_policy": {"deescalation_templates": ["好啦，我们先缓一缓。"]},
        "emotion_probe_policy": {
            "enabled": True,
            "distress_keywords": ["难过", "委屈"],
            "reason_hints": ["因为", "客户", "放鸽子"],
            "probe_templates": ["怎么啦 发生啥了"],
            "comfort_templates": ["我在呢 你慢慢说"],
        },
        "shared_history_policy": {
            "enabled": True,
            "deny_unverified": True,
            "categories": {
                "mahjong": {
                    "keywords": ["麻将", "打麻将", "打过麻将", "打牌"],
                    "deny_templates": ["你记错了吧 我没跟你一起打过麻将"],
                },
                "meal": {
                    "keywords": ["吃饭", "吃过饭", "一起吃", "请吃"],
                    "deny_templates": ["你是不是记混了 我们没有一起吃过饭"],
                },
                "meet": {
                    "keywords": ["见面", "见过", "出去", "逛街"],
                    "deny_templates": ["这个你记错啦 我们之前没有一起出去过"],
                },
                "drink": {
                    "keywords": ["喝酒", "喝过", "一起喝"],
                    "deny_templates": ["这个没有过 我们之前没一起喝过酒"],
                },
                "default": {"keywords": [], "deny_templates": ["你应该是记错了吧 这事没有过"]},
            },
        },
    }
    engine.summary_every_n_turns = 999999
    engine.local_timezone = UTC
    engine.quiet_hours = "02:00-09:00"
    engine.emoji_enabled = False
    engine.rng = FixedRng()
    engine.memory = DummyMemory()
    engine.state_machine = DummyStateMachine()
    engine.style_controller = type(
        "DummyStyle",
        (),
        {
            "build_style_directives": staticmethod(
                lambda state: {
                    "stage": state.stage,
                    "mood": state.mood,
                    "reply_length": "short",
                    "warmth_level": state.warmth,
                    "initiative_level": state.initiative,
                    "disclosure_level": state.disclosure,
                    "allow_counter_question": True,
                    "allow_life_detail": False,
                    "emoji_enabled": False,
                    "marketing_allowed": False,
                    "low_context": False,
                    "clarify_needed": False,
                }
            )
        },
    )()
    engine.identity_guard = type(
        "DummyIdentityGuard",
        (),
        {
            "is_identity_probe": staticmethod(lambda text: False),
            "get_reply": staticmethod(lambda **kwargs: "先不聊这个啦"),
        },
    )()
    engine.generator = None
    engine.summarizer = type("DummySummarizer", (), {"summarize": staticmethod(lambda user_id, recent, state: [])})()
    engine.rep_guard = DummyRepGuard()
    engine.proactive_scheduler = type(
        "DummyProactive",
        (),
        {
            "last_proactive_at": {},
            "suggest": staticmethod(lambda **kwargs: None),
            "mark_sent": staticmethod(lambda user_id: None),
        },
    )()
    engine.user_states = {}
    engine.recent_messages = {}
    engine.turn_counts = {}
    engine.pending_replies = []
    engine._pending_lock = Lock()
    engine._pending_seq = 0
    engine.reply_policy_stats = {
        "reply": 0,
        "noreply": 0,
        "quiet_hours": 0,
        "low_context_no_history": 0,
        "need_no_reply": 0,
        "repeat_spam": 0,
        "hostile_deescalation": 0,
    }
    engine.max_reply_chars = 20
    engine.glm_context_messages = 200
    engine.max_batch_replies = 3
    engine._generate_single_text = lambda user_id, user_text, state, recent, cls: "先说说怎么了"
    engine._is_quiet_hours = lambda now=None: False
    return engine


def test_quiet_hours_default_noreply():
    engine = make_engine()
    engine._is_quiet_hours = lambda now=None: True
    result = engine.chat("u1", "你好")
    assert result["type"] == "noreply"
    assert result["reason"] == "quiet_hours"


def test_quiet_hours_urgent_exception():
    engine = make_engine()
    engine._is_quiet_hours = lambda now=None: True
    result = engine.chat("u1", "救命 我很难受")
    assert result["type"] == "reply"
    assert result["text"]


def test_low_context_without_history_returns_question_mark():
    engine = make_engine()
    result = engine.chat("u2", "啊")
    assert result["type"] == "reply"
    assert result["text"] == "？"


def test_boring_or_hostile_can_result_in_noreply():
    engine = make_engine()
    engine.chat("u3", "无聊")
    engine.chat("u3", "无聊")
    result = engine.chat("u3", "无聊")
    assert result["type"] == "noreply"


def test_repeated_message_burst_results_noreply():
    engine = make_engine()
    engine.chat("u_repeat", "1")
    engine.chat("u_repeat", "1")
    result = engine.chat("u_repeat", "1")
    assert result["type"] == "noreply"
    assert result["reason"] == "repeat_spam"


def test_ingest_repeated_batch_results_noreply():
    engine = make_engine()
    out = engine.ingest_incoming_messages(
        username="u_batch",
        user_id="u_batch",
        messages=["想你", "想你", "想你", "想你"],
    )
    assert out["ok"] is True
    assert out["queued"] is None
    assert out["reason"] == "repeat_spam"


def test_repeated_message_burst_can_reply_question_mark():
    class LowRng:
        def random(self):
            return 0.1

        def randint(self, a, b):
            return a

        def choice(self, seq):
            return seq[0]

    engine = make_engine()
    engine.rng = LowRng()
    engine.chat("u_repeat_q", "1")
    engine.chat("u_repeat_q", "1")
    result = engine.chat("u_repeat_q", "1")
    assert result["type"] == "reply"
    assert result["text"] == "？"
    assert result["reason"] == "repeat_spam_question"


def test_new_non_repeated_message_is_not_blocked_by_old_repeat_spam():
    engine = make_engine()
    engine.chat("u_repeat_new", "1")
    engine.chat("u_repeat_new", "1")
    engine.chat("u_repeat_new", "1")
    result = engine.chat("u_repeat_new", "问你吃饭了没")
    assert result["reason"] not in {"repeat_spam", "repeat_spam_question"}


def test_ingest_single_message_can_queue_multiple_replies():
    engine = make_engine()
    engine.recent_messages["u_multi"] = [
        Message(role="user", content="你好"),
        Message(role="assistant", content="在呢"),
    ]
    engine._generate_multi_replies = lambda *args, **kwargs: ["第一句", "第二句"]
    out = engine.ingest_incoming_messages(
        username="u_multi",
        user_id="u_multi",
        messages=["今天怎么样"],
    )
    assert out["ok"] is True
    assert out["reason"] == "queued"
    assert len(out["queued_items"]) == 2


def test_emotion_without_reason_probe_then_comfort():
    engine = make_engine()
    result = engine.chat("u3x", "我好难过啊")
    assert result["type"] == "reply"
    assert "怎么啦" in result["text"]
    assert "慢慢说" in result["text"]
    assert result["reason"] == "emotion_probe_then_comfort"


def test_emotion_probe_templates_come_from_bible_policy():
    engine = make_engine()
    engine.bible["emotion_probe_policy"]["probe_templates"] = ["你咋啦"]
    engine.bible["emotion_probe_policy"]["comfort_templates"] = ["我在听 你说"]
    result = engine.chat("u3y", "我很难过")
    assert result["type"] == "reply"
    assert "你咋啦" in result["text"]
    assert "我在听 你说" in result["text"]


def test_unverified_shared_history_is_denied():
    engine = make_engine()
    engine._generate_single_text = GuoguoEngine._generate_single_text.__get__(engine, GuoguoEngine)
    result = engine.chat("u4", "你上次还跟我一起打过麻将")
    assert result["type"] == "reply"
    assert "没跟你一起打过麻将" in result["text"]


def test_unverified_shared_history_meal_category():
    engine = make_engine()
    engine._generate_single_text = GuoguoEngine._generate_single_text.__get__(engine, GuoguoEngine)
    result = engine.chat("u5", "你上次还说要跟我一起吃饭")
    assert result["type"] == "reply"
    assert "没有一起吃过饭" in result["text"]


def test_forbidden_store_invite_is_sanitized():
    engine = make_engine()
    state = RelationshipState(user_id="u6")
    decision = engine._finalize_reply(
        state=state,
        raw_reply="你别难过 快来店里让我哄哄你 来消费也行",
        reason="reply",
        urgent=False,
    )
    assert "来店里" not in decision.text
    assert "来消费" not in decision.text


def test_reply_is_split_to_max_20_chars():
    engine = make_engine()
    out = engine._split_short_reply("这是一段超过二十个字符的回复文本用于测试分段逻辑", max_chars=20)
    parts = [x for x in out.split("\n") if x]
    assert len(parts) >= 2
    assert all(len(p) <= 20 for p in parts)


def test_reply_split_prefers_space_semantics():
    engine = make_engine()
    out = engine._split_short_reply("你好 你在干嘛呢 今天忙不忙", max_chars=7)
    parts = [x for x in out.split("\n") if x]
    assert parts == ["你好", "你在干嘛呢", "今天忙不忙"]
