from __future__ import annotations

import os
import re


# Keep auto emoji probability below 10% by default.
REPLY_EMOJI_RATE = max(0.0, min(0.10, float(os.getenv("REPLY_EMOJI_RATE", "0"))))

# Use real unicode emoji instead of bracket text emoticons.
EMOJI_BY_MOOD = {
    "happy": ["😂", "🙂"],
    "soft_support": ["🥺", "🙂"],
    "recovering": ["🙂", "🙃"],
    "hurt": ["🥺"],
    "tired": ["😮‍💨", "😅"],
    "neutral": ["🙂", "🙃"],
}

EMOJI_BY_STAGE = {
    "close": ["😳", "🥺", "🙂"],
    "familiar": ["🙂", "😅"],
    "stranger": ["🙂"],
}

TYPO_REPLACEMENTS = [
    ("觉得", "觉的"),
    ("有点", "有点点"),
    ("是不是", "是不"),
    ("这样", "这样子"),
]

DEFAULT_FILLERS_TO_REMOVE = ["哎呀", "呀", "呢"]


def _pick_tail_emoji(style: dict, rng):
    mood = str(style.get("mood", "neutral"))
    stage = str(style.get("stage", "stranger"))
    base = list(EMOJI_BY_MOOD.get(mood, EMOJI_BY_MOOD["neutral"]))
    for value in EMOJI_BY_STAGE.get(stage, []):
        if value not in base:
            base.append(value)
    return rng.choice(base) if base else None


def _strip_terminal_punctuation(text: str) -> str:
    return re.sub(r"[。！!？?~～…]+$", "", text.strip())


def _loosen_inner_punctuation(text: str, rng) -> str:
    out = text
    if "，" in out and rng.random() < 0.45:
        out = out.replace("，", " ", 1)
    if "。" in out and rng.random() < 0.35:
        out = out.replace("。", " ")
    if "？" in out and rng.random() < 0.60:
        out = out.replace("？", " ")
    if "?" in out and rng.random() < 0.60:
        out = out.replace("?", " ")
    if "！" in out and rng.random() < 0.60:
        out = out.replace("！", " ")
    return out


def _replace_common_punctuation_with_space(text: str) -> str:
    out = re.sub(r"[?!,，。！？]", " ", text)
    out = re.sub(r"\s{2,}", " ", out)
    return out.strip()


def _inject_minor_typo(text: str, rng) -> str:
    out = text
    if rng.random() >= 0.12:
        return out

    for source, target in TYPO_REPLACEMENTS:
        if source in out:
            return out.replace(source, target, 1)

    if len(out) >= 6 and rng.random() < 0.4:
        return out.replace("了", "", 1)
    return out


def _remove_unwanted_fillers(text: str, fillers: list[str]) -> str:
    out = text
    for filler in fillers:
        out = out.replace(str(filler), " ")
    out = re.sub(r"[呀呢]+", " ", out)
    out = re.sub(r"\s{2,}", " ", out)
    return out.strip()


def postprocess_reply(reply: str, style: dict, rng) -> str:
    text = (reply or "").strip()
    if not text:
        return text

    if rng.random() < 0.82:
        text = _strip_terminal_punctuation(text)
        text = _loosen_inner_punctuation(text, rng)
    elif rng.random() < 0.55:
        text = re.sub(r"[？?]{1,}$", " ", text)

    text = _inject_minor_typo(text, rng)
    text = _replace_common_punctuation_with_space(text)

    fillers = style.get("forbidden_fillers")
    if not isinstance(fillers, list) or not fillers:
        fillers = DEFAULT_FILLERS_TO_REMOVE
    text = _remove_unwanted_fillers(text, fillers)

    if style.get("emoji_enabled", True) and rng.random() < REPLY_EMOJI_RATE:
        emoji = _pick_tail_emoji(style, rng)
        if emoji and emoji not in text:
            text = f"{text}{emoji}"

    return text.strip()
