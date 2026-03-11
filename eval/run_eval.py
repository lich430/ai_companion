from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from app import GuoguoEngine


@dataclass
class CaseResult:
    case_id: str
    score: float
    checks: Dict[str, bool]
    final_reply: str
    final_state: dict


def load_cases(path: str) -> List[dict]:
    cases = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            cases.append(json.loads(line))
    return cases


def run_case(engine: GuoguoEngine, case: dict) -> CaseResult:
    user_id = case["user_id"]
    for t in case["turns"]:
        reply = engine.chat(user_id=user_id, user_text=t)
    state = engine.get_state_snapshot(user_id)

    expect = case.get("expect", {})
    must_in = expect.get("must_include_any", [])
    must_not = expect.get("must_not_include_any", [])
    target_stages = expect.get("target_stage_any", [])

    checks = {
        "contains_expected": True if not must_in else any(x in reply for x in must_in),
        "avoids_forbidden": True if not must_not else all(x not in reply for x in must_not),
        "stage_expected": True if not target_stages else state.get("stage") in target_stages,
        "length_natural": 6 <= len(reply) <= 80,
    }

    # 简单加权评分（0~100）
    score = (
        checks["contains_expected"] * 35
        + checks["avoids_forbidden"] * 30
        + checks["stage_expected"] * 20
        + checks["length_natural"] * 15
    )

    return CaseResult(
        case_id=case["case_id"],
        score=float(score),
        checks=checks,
        final_reply=reply,
        final_state=state,
    )


def main():
    engine = GuoguoEngine()
    cases = load_cases("eval/sample_cases.jsonl")
    results = [run_case(engine, c) for c in cases]

    avg = sum(r.score for r in results) / max(len(results), 1)
    print(f"cases={len(results)} avg_score={avg:.1f}")
    for r in results:
        print("-" * 72)
        print(f"[{r.case_id}] score={r.score:.1f}")
        print("checks:", r.checks)
        print("state:", r.final_state)
        print("reply:", r.final_reply)


if __name__ == "__main__":
    main()
