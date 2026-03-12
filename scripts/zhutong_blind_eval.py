from __future__ import annotations

import json
import os
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app import CompanionEngine
from engine.message_classifier import classify_user_message


Z_NAME = "绔圭瓛"
U_NAME = "灏忔潹鐧捐揣鍒嗕韩"
STICKER_PREFIX = "锛堣〃鎯呭寘锛?
STICKER_SUFFIX = "锛?


def is_timestamp(line: str) -> bool:
    return bool(re.search(r"\d{1,2}:\d{2}", line)) and (
        "2025" in line or "2026" in line or "骞? in line
    )


def parse_dialog(path: Path) -> list[dict]:
    lines = path.read_text(encoding="utf-8").splitlines()
    messages: list[dict] = []
    pending_speaker = None
    active_speaker = None
    buf: list[str] = []

    def flush():
        nonlocal buf
        if active_speaker and buf:
            text = "\n".join(x for x in buf if x.strip()).strip()
            if text:
                messages.append({"speaker": active_speaker, "text": text})
        buf = []

    for raw in lines:
        s = raw.strip()
        if s in {Z_NAME, U_NAME}:
            flush()
            pending_speaker = s
            active_speaker = None
            continue
        if pending_speaker and is_timestamp(s):
            active_speaker = pending_speaker
            buf = []
            continue
        if not s:
            flush()
            continue
        if active_speaker:
            buf.append(s)
    flush()

    return [m for m in messages if m["speaker"] in {Z_NAME, U_NAME}]


def build_pairs(dialog: list[dict]) -> list[dict]:
    pairs: list[dict] = []
    for i, item in enumerate(dialog):
        if item["speaker"] != U_NAME:
            continue
        replies: list[str] = []
        j = i + 1
        while j < len(dialog) and dialog[j]["speaker"] == Z_NAME:
            replies.append(dialog[j]["text"])
            j += 1
        if replies:
            pairs.append({"user": item["text"], "real": "\n".join(replies)})
    return pairs


def trigram_jaccard(a: str, b: str) -> float:
    def grams(s: str) -> set[str]:
        s = (s or "").strip()
        if not s:
            return set()
        if len(s) < 3:
            return {s}
        return {s[i : i + 3] for i in range(len(s) - 2)}

    g1 = grams(a)
    g2 = grams(b)
    den = len(g1 | g2) or 1
    return len(g1 & g2) / den


def run(sample_size: int = 40, seed: int = 20260311):
    random.seed(seed)
    base = Path(".")
    src = base / "zhutong"
    blind = base / "eval" / "zhutong_blind_eval.jsonl"
    answer = base / "eval" / "zhutong_blind_answer.jsonl"
    report = base / "eval" / "zhutong_blind_report.md"
    details = base / "eval" / "zhutong_blind_details.json"

    os.environ["ROLE_BIBLE_PATH"] = "data/zhutong_bible.json"

    dialog = parse_dialog(src)
    pairs = build_pairs(dialog)
    candidates = [
        p
        for p in pairs
        if 2 <= len(p["user"]) <= 120 and 2 <= len(p["real"]) <= 220
    ]
    random.shuffle(candidates)
    cases = candidates[:sample_size]

    engine = CompanionEngine()
    records = []
    for idx, p in enumerate(cases, 1):
        uid = f"zhutong_eval_user_{idx}"
        try:
            user_text = str(p["user"] or "").strip()
            # Force generation mode for evaluation to avoid noreply policy skew.
            state = engine._get_state(uid)
            recent = engine._get_recent(uid)
            cls = classify_user_message(user_text)
            model_reply = engine._generate_single_text(uid, user_text, state, recent, cls).strip()
        except Exception as e:
            model_reply = f"[MODEL_ERROR] {e}"

        sim = round(trigram_jaccard(p["real"], model_reply), 4)
        if random.random() < 0.5:
            reply_a, reply_b = p["real"], model_reply
            which_real = "A"
        else:
            reply_a, reply_b = model_reply, p["real"]
            which_real = "B"

        records.append(
            {
                "id": idx,
                "user_msg": p["user"],
                "reply_a": reply_a,
                "reply_b": reply_b,
                "which_real": which_real,
                "real_reply": p["real"],
                "model_reply": model_reply,
                "trigram_jaccard": sim,
            }
        )

    with blind.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(
                json.dumps(
                    {
                        "id": r["id"],
                        "user_msg": r["user_msg"],
                        "reply_a": r["reply_a"],
                        "reply_b": r["reply_b"],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    with answer.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(
                json.dumps(
                    {
                        "id": r["id"],
                        "which_real": r["which_real"],
                        "trigram_jaccard": r["trigram_jaccard"],
                        "real_reply": r["real_reply"],
                        "model_reply": r["model_reply"],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    avg_sim = sum(r["trigram_jaccard"] for r in records) / (len(records) or 1)
    low_cases = sorted(records, key=lambda x: x["trigram_jaccard"])[:10]
    details.write_text(
        json.dumps(
            {
                "sample_size": len(records),
                "avg_trigram_jaccard": round(avg_sim, 4),
                "lowest_cases": low_cases,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    report.write_text(
        "\n".join(
            [
                "# Zhutong Blind Eval Report",
                "",
                f"- Sample size: {len(records)}",
                f"- Avg trigram Jaccard: {avg_sim:.4f}",
                "- Note: this metric is lexical overlap only.",
                "",
                "Files:",
                "- eval/zhutong_blind_eval.jsonl (A/B shuffled for review)",
                "- eval/zhutong_blind_answer.jsonl (answer key)",
                "- eval/zhutong_blind_details.json (low-score cases for tuning)",
            ]
        ),
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "sample_size": len(records),
                "avg_trigram_jaccard": round(avg_sim, 4),
                "pairs_total": len(pairs),
                "dialog_total": len(dialog),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    sample = int(os.getenv("ZHUTONG_BLIND_SAMPLE", "40"))
    seed = int(os.getenv("ZHUTONG_BLIND_SEED", "20260311"))
    run(sample_size=max(1, sample), seed=seed)

