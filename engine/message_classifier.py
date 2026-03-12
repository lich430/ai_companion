from __future__ import annotations

import re


HOSTILE_KEYWORDS = {
    "傻逼",
    "煞笔",
    "滚",
    "有病",
    "神经病",
    "废物",
    "脑残",
    "去死",
    "装逼",
    "妈的",
    "操",
    "mlgb",
    "草",
}

BORING_EXACT = {
    "哦",
    "嗯",
    "呵呵",
    "在不在",
    "在吗",
    "。。",
    "...",
    "无聊",
}

URGENT_KEYWORDS = {
    "急",
    "难受",
    "出事",
    "救命",
    "快",
    "严重",
    "崩溃",
    "受不了",
}

LOW_CONTEXT_EXEMPT = {"你好", "在吗", "在么", "哈喽"}


def classify_user_message(text: str) -> dict:
    raw = (text or "").strip()
    lowered = raw.lower()
    compact = re.sub(r"\s+", "", raw)
    # Normalize punctuation/symbols so greetings like "你好。/你好~" still match exemptions.
    compact_plain = re.sub("[^0-9A-Za-z\u4e00-\u9fff]+", "", compact)
    only_marks = bool(compact) and all(ch in "？?!.。…~～，,、🙂😳😅😂🥺🙃😮‍💨" for ch in compact)
    alpha_num = re.sub(r"[\W_]+", "", lowered)

    low_context = (
        len(compact) <= 1
        or only_marks
        or (len(compact_plain) <= 3 and compact_plain not in LOW_CONTEXT_EXEMPT)
        or (len(alpha_num) <= 3 and any(ch.isascii() and ch.isalpha() for ch in alpha_num))
    )
    hostile = any(k in lowered for k in HOSTILE_KEYWORDS)
    boring = (compact in BORING_EXACT and compact not in LOW_CONTEXT_EXEMPT) or (
        len(compact_plain) <= 2
        and compact_plain not in LOW_CONTEXT_EXEMPT
        and not hostile
        and not low_context
    )
    urgent = any(k in raw for k in URGENT_KEYWORDS)
    need_no_reply = hostile or boring

    return {
        "low_context": low_context,
        "hostile": hostile,
        "boring": boring,
        "need_no_reply": need_no_reply,
        "urgent": urgent,
    }
