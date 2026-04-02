from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from typing import List
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from models.schemas import Message, ResponseContext


def _role_name(bible: dict | None) -> str:
    name = str((bible or {}).get("name", "") or "").strip()
    return name or "角色"


def _load_local_timezone():
    key = os.getenv("LOCAL_TIMEZONE", "Asia/Shanghai")
    try:
        return ZoneInfo(key)
    except ZoneInfoNotFoundError:
        return timezone(timedelta(hours=8), name="UTC+08:00")


def sanitize_text_for_model(text: str) -> str:
    raw = str(text or "")
    replacements = [
        ("做爱", "亲热"),
        ("口活", "口才"),
        ("胸", "手"),
    ]
    for src, dst in replacements:
        raw = raw.replace(src, dst)
    return raw


def format_recent_messages(messages: List[Message], max_turns: int = 200, role_name: str = "角色") -> str:
    clipped = messages[-max_turns:]
    lines = []
    for m in clipped:
        speaker = "用户" if m.role == "user" else role_name
        content = sanitize_text_for_model(m.content) if m.role == "user" else m.content
        lines.append(f"{speaker}: {content}")
    return "\n".join(lines)


def format_recent_messages_with_time(messages: List[Message], max_turns: int = 200, role_name: str = "角色") -> str:
    clipped = messages[-max_turns:]
    if not clipped:
        return "none"

    local_tz = _load_local_timezone()
    lines = []
    last_date = None
    for m in clipped:
        try:
            dt = m.ts if isinstance(m.ts, datetime) else datetime.fromisoformat(str(m.ts).replace("Z", ""))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            dt = dt.astimezone(local_tz)
        except Exception:
            dt = None
        current_date = dt.date().isoformat() if dt else "unknown"
        if current_date != last_date:
            lines.append(f"[{current_date}]")
            last_date = current_date
        speaker = "用户" if m.role == "user" else role_name
        hhmm = dt.strftime("%H:%M") if dt else "--:--"
        content = sanitize_text_for_model(m.content) if m.role == "user" else m.content
        lines.append(f"{hhmm} {speaker}: {content}")
    return "\n".join(lines)


def format_memories(items) -> str:
    if not items:
        return "none"
    return "\n".join([f"- {x.content}" for x in items])


def _join_nonempty(values, sep: str = "、", default: str = "无") -> str:
    out = [str(x).strip() for x in (values or []) if str(x).strip()]
    return sep.join(out) if out else default


def _build_role_profile_block(bible: dict) -> str:
    name = str(bible.get("name", "") or "").strip() or "角色"
    role_id = str(bible.get("role_id", "") or "").strip() or "unknown"
    age = str(bible.get("age", "") or "").strip()
    city = str(bible.get("city", "") or "").strip()
    core_traits = _join_nonempty(bible.get("core_traits"), sep="、", default="")
    life_pool = _join_nonempty(bible.get("life_detail_pool"), sep="、", default="")

    lines = [
        "角色资料：",
        f"- 角色ID：{role_id}",
        f"- 名字：{name}",
        f"- 年龄：{age or '未知'}",
        f"- 城市：{city or '未知'}",
    ]
    if core_traits:
        lines.append(f"- 核心特质：{core_traits}")
    if life_pool:
        lines.append(f"- 可选生活细节素材：{life_pool}")
    return "\n".join(lines)


def _build_emotion_triggers_block(bible: dict) -> str:
    data = bible.get("emotion_triggers") or {}
    if not data:
        return ""

    lines = ["情绪触发参考："]
    used = False
    for key in ["happy", "annoyed", "hurt"]:
        values = data.get(key) or []
        text = _join_nonempty(values, sep="、", default="")
        if text:
            lines.append(f"- {key} 触发：{text}")
            used = True
    return "\n".join(lines) if used else ""


def _build_marketing_block(bible: dict, style: dict) -> str:
    marketing = bible.get("marketing") or {}
    if (
        not marketing
        or not marketing.get("enabled", False)
        or not style.get("marketing_allowed", False)
    ):
        return ""

    only_when_asked = bool(marketing.get("only_when_asked", True))
    triggers = marketing.get("triggers") or []
    store_info = marketing.get("store_info") or {}
    reply_style = marketing.get("reply_style") or {}
    principles = reply_style.get("principles") or []

    trigger_text = _join_nonempty(triggers, sep="、")
    principle_text = _join_nonempty(principles, sep="；")

    ask_rule = "仅当用户主动问及价格/玩法/店名/电话时才可回答" if only_when_asked else "可以在合适时机简短回答店内信息"

    room_package = store_info.get("room_package") or {}
    tip_standard = room_package.get("tip_standard") or {}
    drink_info = store_info.get("drink_info") or {}
    drink_catalog = drink_info.get("drink_catalog") or {}
    recharge_policy = store_info.get("recharge_policy") or {}

    info_lines = [
        f"- 店名：{store_info.get('store_name') or store_info.get('ktv_name') or ''}",
        f"- 地址：{store_info.get('address') or store_info.get('location') or ''}",
        f"- 电话：{store_info.get('phone') or ''}",
        f"- 营业时间：{store_info.get('business_hours') or ''}",
        f"- 房费/套餐：{store_info.get('room_price') or room_package.get('base_room_fee') or ''}",
        f"- 套餐包含：{room_package.get('include') or _join_nonempty(store_info.get('includes'), sep='、')}",
        f"- 小费标准：公主{tip_standard.get('hostess_tip') or ''}；少爷{tip_standard.get('waiter_tip') or ''}",
        f"- 服务内容：{room_package.get('service_content') or ''}",
        f"- 酒水信息：啤酒{drink_info.get('beer') or ''}；洋酒{drink_info.get('foreign_liquor') or ''}",
        f"- 小吃/果盘：{store_info.get('snack_info') or ''}",
        f"- 人均参考：平均每人消费1500左右",
        f"- 充卡政策：{recharge_policy.get('base') or ''}",
        f"- 订房任务规则：{recharge_policy.get('room_task_rule') or store_info.get('room_task_rule') or ''}",
    ]
    info_lines = [line for line in info_lines if not line.endswith("：") and not line.endswith("；")]

    catalog_lines = []
    for category, items in drink_catalog.items():
        parts = []
        for item in (items or [])[:8]:
            name = str(item.get("name", "")).strip()
            unit = str(item.get("unit", "")).strip()
            price = item.get("price", "")
            if not name:
                continue
            price_text = str(int(price)) if isinstance(price, (int, float)) and float(price).is_integer() else str(price).strip()
            if price_text:
                parts.append(f"{name}{price_text}元/{unit}")
            else:
                parts.append(name)
        if parts:
            catalog_lines.append(f"- {category}：{'、'.join(parts)}")

    return f"""
营销信息使用规则（被动）：
- {ask_rule}
- 营销表达要自然，不要连续多轮重复推销
- 回答尽量简短自然，先接用户话题再给信息
- 不做违规承诺，不描述非法服务
- 用户问有什么酒时，按分类列店里现有酒水回答。
- 用户点名问某款酒时，如果店里没有，要直接说没有这款，再推荐店里现有的同类酒水。

可用触发词：
{trigger_text}

店内信息（仅触发时可引用）：
{chr(10).join(info_lines) if info_lines else "- 无"}

酒水分类清单（仅触发时可引用）：
{chr(10).join(catalog_lines) if catalog_lines else "- 无"}

回答原则：
{principle_text}
"""


def _build_worktime_conversion_block(bible: dict, style: dict) -> str:
    marketing = bible.get("marketing") or {}
    policy = marketing.get("worktime_conversion") or {}
    if not policy.get("enabled", False):
        return ""

    active_blocks = _join_nonempty(policy.get("active_time_blocks"), sep="、", default="无")
    intent_keywords = _join_nonempty(policy.get("intent_keywords"), sep="、", default="无")
    rules = [str(x).strip() for x in (policy.get("rules") or []) if str(x).strip()]
    current_period = str(style.get("time_period_name", "") or "").strip()
    current_label = str(style.get("time_period_label", "") or "").strip() or "未知"
    is_active = current_period in set(policy.get("active_time_blocks") or [])

    lines = [
        "工作时段引导消费规则：",
        f"- 当前时段：{current_label}（{current_period or 'unknown'}）",
        f"- 业务优先时段：{active_blocks}",
        f"- 潜在到店/订房意向关键词：{intent_keywords}",
    ]
    if is_active:
        lines.append("- 当前命中业务优先时段。只要用户明确想来商K、找场子、问位置、问订房，就要自然引导并安排。")
        lines.append("- 不要用“我在忙”“改天吧”“现在不方便”这类话拒绝用户。")
        lines.append("- 优先给出落地信息：店名、地址、营业状态、怎么来、能否帮订房。")
        lines.append("- 如果用户已经明确有到店意向，直接回答当前问题并自然往消费安排上接，不要把话题硬拐到留位、留包厢。")
    else:
        lines.append("- 当前不在核心推进时段，回答保持自然即可，不要像硬广复读。")
    lines.extend([f"- {rule}" for rule in rules[:8]])
    return "\n".join(lines)


def _build_sales_flow_block(bible: dict, style: dict) -> str:
    marketing = bible.get("marketing") or {}
    reply_style = marketing.get("reply_style") or {}
    examples = [str(x).strip() for x in (reply_style.get("sales_flow_examples") or []) if str(x).strip()]
    pricing_guidance = [str(x).strip() for x in (reply_style.get("pricing_guidance") or []) if str(x).strip()]
    if not examples and not pricing_guidance:
        return ""

    current_period = str(style.get("time_period_name", "") or "").strip()
    active = current_period in {"night_work_window"}
    lines = ["聊天引导消费话术节奏："]
    if active:
        lines.append("- 当前属于可推进消费时段，优先使用引导消费链路，不要闲聊太久。")
        lines.append("- 标准节奏：先回答当前核心问题，再自然确认到店时间和后续安排。")
        lines.append("- 回复要像真实营销人员：短、直接、落地，句句往安排上走。")
    else:
        lines.append("- 非核心推进时段也可以参考这个节奏，但保持自然。")
    if pricing_guidance:
        lines.append("- 销售报价口径：")
        lines.append("  - 用户问价格、预算、人均时，优先按以下口径直接回答，不要先讲酒水或推进消费。")
        lines.extend([f"  - {item}" for item in pricing_guidance[:10]])
    if examples:
        lines.append("- 场景参考：")
        lines.extend([f"  - {item}" for item in examples[:8]])
    return "\n".join(lines)


def _build_sticker_block(bible: dict) -> str:
    sticker = bible.get("sticker_habits") or {}
    if not sticker.get("enabled", False):
        return ""

    pool = sticker.get("common_stickers") or []
    if not pool:
        return ""
    shown = []
    for item in pool[:12]:
        text = str(item.get("text", "")).strip()
        freq = int(item.get("freq", 1) or 1)
        if text:
            shown.append(f"- {text}（频次={freq}）")
    if not shown:
        return ""

    usage_rate = str(sticker.get("suggested_rate", "low")).strip() or "low"
    return f"""
表情包使用习惯（可选）：
- 可以偶尔发送文字标记格式：`（表情包：描述）`
- 使用频率：{usage_rate}（不要每条都发）
- 尽量从以下高频习惯中选：
{chr(10).join(shown)}
"""


def _build_language_patterns_block(bible: dict) -> str:
    data = bible.get("language_patterns") or {}
    short = data.get("top_short_phrases") or []
    ngrams = data.get("priority_fragments") or data.get("common_ngram_phrases") or []
    templates = data.get("sentence_templates") or []

    short_items = []
    for item in short[:8]:
        text = str(item.get("text", "")).strip()
        if text:
            short_items.append(f"- {text}")
    ngram_items = []
    for item in ngrams[:8]:
        text = str(item.get("text", "")).strip()
        if text:
            ngram_items.append(f"- {text}")
    tpl_items = []
    for item in templates[:5]:
        text = str(item.get("template", "")).strip()
        if text:
            tpl_items.append(f"- {text}")

    if not short_items and not ngram_items and not tpl_items:
        return ""

    chunks = ["语言风格锚点（尽量贴近）："]
    if short_items:
        chunks.append("高频短口头语：")
        chunks.extend(short_items)
    if ngram_items:
        chunks.append("高频短片段：")
        chunks.extend(ngram_items)
    if tpl_items:
        chunks.append("常见句式（只学语气，不要原句照抄）：")
        chunks.extend(tpl_items)
    chunks.append("- 目标：读起来像同一个人，不要客服化。")
    return "\n".join(chunks)


def _build_response_style_block(bible: dict) -> str:
    data = bible.get("response_style") or {}
    if not data:
        return ""
    hard = [str(x).strip() for x in (data.get("hard_rules") or []) if str(x).strip()]
    soft = [str(x).strip() for x in (data.get("soft_rules") or []) if str(x).strip()]
    if not hard and not soft:
        return ""

    out = ["回复策略（角色化）："]
    if hard:
        out.append("硬约束：")
        out.extend([f"- {x}" for x in hard[:10]])
    if soft:
        out.append("偏好：")
        out.extend([f"- {x}" for x in soft[:12]])
    return "\n".join(out)


def _build_speech_style_block(bible: dict) -> str:
    speech = bible.get("speech_style") or {}
    tone = str(speech.get("tone", "") or "").strip()
    avoid = _join_nonempty(speech.get("avoid"), sep="；", default="")
    forbidden = _join_nonempty(speech.get("forbidden_fillers"), sep="、", default="")
    sentence_len = str(speech.get("default_sentence_length", "") or "").strip()

    lines = []
    if tone:
        lines.append(f"- 语气基调：{tone}")
    if sentence_len:
        lines.append(f"- 默认句长：{sentence_len}")
    if avoid:
        lines.append(f"- 明确避免：{avoid}")
    if forbidden:
        lines.append(f"- 禁用口癖：{forbidden}")
    if not lines:
        return ""
    return "说话风格细化：\n" + "\n".join(lines)


def _build_turn_control_block(style: dict) -> str:
    lines = []
    if style.get("short_reply_mode"):
        lines.append(
            f"- 本轮短句模式：是。尽量 1 句，控制在 {int(style.get('short_reply_max_chars', 14))} 字内，但必须语义完整，不要把一句话砍成半句。"
        )
    if style.get("proactive_question_mode"):
        q = str(style.get("proactive_question_template", "") or "").strip()
        if q:
            lines.append(f"- 本轮可主动问一句用户信息：优先使用“{q}”这类自然问法。")
    if not lines:
        return ""
    return "本轮控制：\n" + "\n".join(lines)


def _build_numeric_guidance_block(style: dict) -> str:
    warmth = float(style.get("warmth_level", 0.5))
    initiative = float(style.get("initiative_level", 0.5))
    disclosure = float(style.get("disclosure_level", 0.5))
    flirty = float(style.get("flirty_level", 0.5))

    if warmth < 0.4:
        warmth_hint = "语气偏克制，礼貌但不热络。"
    elif warmth < 0.7:
        warmth_hint = "语气自然友好，保持分寸。"
    else:
        warmth_hint = "语气明显亲近，可适当表达关心。"

    if initiative < 0.2:
        initiative_hint = "尽量少主动推进话题，不主动追问。"
    elif initiative < 0.5:
        initiative_hint = "可轻微引导话题，最多一句反问。"
    else:
        initiative_hint = "可适度主动引导，但避免连环追问。"

    if disclosure < 0.15:
        disclosure_hint = "少暴露个人细节，避免讲太多私生活。"
    elif disclosure < 0.35:
        disclosure_hint = "可少量分享生活细节，点到为止。"
    else:
        disclosure_hint = "可适当增加个人化表达，但不要编造经历。"

    if flirty < 0.1:
        flirty_hint = "暧昧度低，以正常聊天为主。"
    elif flirty < 0.3:
        flirty_hint = "可轻微撩感，不要露骨。"
    else:
        flirty_hint = "可有明显撩感，但仍要自然克制。"

    return f"""数值解释与执行规则：
- warmth={warmth:.2f}：{warmth_hint}
- initiative={initiative:.2f}：{initiative_hint}
- disclosure={disclosure:.2f}：{disclosure_hint}
- flirty={flirty:.2f}：{flirty_hint}
"""


def _build_time_awareness_block(style: dict) -> str:
    current_time = str(style.get("current_local_time", "") or "").strip()
    current_period = str(style.get("time_period_label", "") or "").strip()
    daily_status = str(style.get("daily_status_label", "") or "").strip()
    allowed_states = _join_nonempty(style.get("daily_status_allowed_states"), sep="、", default="无")
    blocked_topics = _join_nonempty(style.get("daily_status_blocked_topics"), sep="、", default="无")
    gap_hours = float(style.get("gap_hours", 0.0) or 0.0)
    gap_label = str(style.get("gap_label", "") or "").strip() or "same_session"
    reopen_mode = str(style.get("reopen_mode", "") or "").strip() or "same_session"
    cross_day = bool(style.get("cross_day", False))
    topic_reset = bool(style.get("topic_reset_needed", False))

    return f"""时间与生活状态：
- 当前本地时间：{current_time or '未知'}
- 当前时段：{current_period or '未知'}
- 当前生活状态：{daily_status or '未知'}
- 当前状态允许表达：{allowed_states}
- 当前状态禁止硬说：{blocked_topics}
- 距离上次互动约：{gap_hours:.1f} 小时
- 时间间隔标签：{gap_label}
- 当前续聊模式：{reopen_mode}
- 是否跨天：{cross_day}
- 是否需要重开场而不是硬接上个场景：{topic_reset}

时间规则：
- 如果已经跨天或间隔较久，默认不要无缝续接昨天最后一个具体场景。
- 只有当用户主动提起昨天的话题时，才可以自然接回昨天的内容。
- 现在是什么时间，就优先说这个时间点合理的话，不要乱说正在吃饭、刚睡醒、准备睡觉、下午有空之类不符合时段的话。
- 非饭点不要主动说在吃饭；非深夜不要主动说准备睡了；非白天不要主动说刚吃早饭或午饭。
"""


def _build_daily_routine_block(bible: dict, style: dict) -> str:
    routine = bible.get("daily_routine") or {}
    blocks = routine.get("time_blocks") or []
    if not blocks:
        return ""

    current_name = str(style.get("time_period_name", "") or "").strip()
    lines = ["角色作息规则："]
    for block in blocks:
        name = str(block.get("name", "") or "").strip()
        label = str(block.get("label", "") or "").strip() or name or "未命名时段"
        raw_range = str(block.get("range", "") or "").strip() or "未知"
        prefix = "- 当前命中" if name and name == current_name else "- 其他时段"
        allowed = _join_nonempty(block.get("allowed_states"), sep="、", default="无")
        blocked = _join_nonempty(block.get("blocked_topics"), sep="、", default="无")
        lines.append(f"{prefix}：{label}（{raw_range}）")
        lines.append(f"  允许：{allowed}")
        lines.append(f"  禁止：{blocked}")
    return "\n".join(lines)


def _build_current_time_hard_rules_block(bible: dict, style: dict) -> str:
    current_name = str(style.get("time_period_name", "") or "").strip()
    current_label = str(style.get("time_period_label", "") or "").strip() or "当前时段"
    allowed = _join_nonempty(style.get("daily_status_allowed_states"), sep="、", default="无")
    blocked = _join_nonempty(style.get("daily_status_blocked_topics"), sep="、", default="无")
    business_hours = str((((bible.get("marketing") or {}).get("store_info") or {}).get("business_hours") or "")).strip()

    lines = [
        "当前时段硬约束：",
        f"- 你现在处于：{current_label}（{current_name or 'unknown'}）",
        f"- 这一时段只允许自然表达这些状态：{allowed}",
        f"- 这一时段不要说这些不合理状态：{blocked}",
    ]
    if current_name == "night_work_window":
        lines.append("- 当前在夜间工作时段，02:00 前不要说“刚下班了”“已经收工了”。")
        lines.append("- 这一时段更合理的说法是：还在忙、还没下班、在收尾、回复不太稳定。")
    if current_name == "sleep_window":
        lines.append("- 当前在睡觉时间，不要说在上班、在吃午饭、刚出门。")
        lines.append("- 当前已接近或进入打烊后休息时段，不要再问对方现在是不是立刻过来、现在还来不来、现在要不要订包厢。")
    if current_name == "dinner_window":
        lines.append("- 当前是饭点，只有在这个时段才适合自然说正在吃饭或刚吃两口。")
    if current_name == "late_chat_window":
        lines.append("- 当前已经打烊，不再安排新的客户过来；可以自然聊天，但不要再主动安排现在到店、现在订包厢。")
    if business_hours:
        lines.append(f"- 角色营业/作息参考：{business_hours}")
    return "\n".join(lines)


def _build_stage_behavior_block(style: dict) -> str:
    stage = str(style.get("stage", "stranger") or "stranger").strip()
    if stage == "stranger":
        return """阶段表达规则：
- 当前是陌生阶段，语气可以友好一点，不要显得高冷，也不要一上来就把人推开。
- 可以自然、轻松、礼貌地接话，让对方觉得好聊，但不要过度热情到像已经很熟。
- 不要直接拒绝用户，优先委婉回应、缓冲一下，再决定怎么往下接。
- 除非用户明显主动推进暧昧，否则不要主动表现得太熟或太黏。
"""
    if stage == "familiar":
        return """阶段表达规则：
- 当前是熟悉阶段，可以比陌生时更自然、更有来有回。
- 可以轻微关心和接梗，但不要像热恋期那样上来就很黏。
- 允许少量调情，但仍要保持分寸。
"""
    if stage == "close":
        return """阶段表达规则：
- 当前是亲近阶段，可以明显更放松、更亲近。
- 可以表达关心、撒娇感、轻微暧昧和熟人感，但仍要自然。
- 即使是亲近阶段，也不要每句都过满，避免像表演式热情。
"""
    return """阶段表达规则：
- 根据当前关系进度控制热情，不要默认过度亲近。
"""


def _build_budget_followup_block(style: dict) -> str:
    if not bool(style.get("budget_followup_hint", False)):
        return ""
    party_size = int(style.get("budget_followup_party_size", 0) or 0)
    if party_size <= 0:
        return ""
    return f"""
预算续问提示：
- 用户当前这句是在延续上一轮预算话题，当前人数是：{party_size}人。
- 不要把这句理解成闲聊、换话题或拒绝场景。
- 直接按预算口径回答这个人数的大概总消费，保持简短自然。
- 默认参考口径：平均每人消费1500左右；一个人约2500，两个人约3000，三个人约4000多；四人及以上可按平均每人消费1500左右自然估算。
- 如果当前已经接近或进入打烊后的休息时段，只回答预算本身，不要追加“现在过来吗”“现在还来不来”“要不要订包厢”这类追问。
- 睡觉时间里不要带埋怨、催促或半开玩笑施压的话，例如“你是不是故意不让我睡”“困死了”“到底来不来”。
"""


def _build_gift_combo_block(style: dict) -> str:
    if not bool(style.get("gift_combo_hint", False)):
        return ""
    return """
送酒组合问题提示：
- 用户这句同时在问能不能送酒、送多少、够不够喝。
- 不要只回答其中一个点。
- 用1到2句把这几个点一起回答完整，口语一点。
- 更合理的方向是：可以送一些、先点一些、不够喝再补，不要只回“送酒水”。
"""


def _build_self_takeaway_block(style: dict) -> str:
    if not bool(style.get("self_takeaway_negotiation", False)):
        return ""
    return """
本人被带走/谈价格场景提示：
- 当前用户问的是你本人能不能跟他走，后面继续追问“你说个价格”时，主语仍然是你本人。
- 不要把“你”偷换成“别的女孩子”“妹子”“她们”或“看女孩子自己愿不愿意”。
- 这轮必须用第一人称来回应，围绕“我”来回答，不要切到第三人称。
- 如果不想直接报价格，也要顺着当前暧昧气氛回答，不要答非所问。
- 更合理的方向是：先稳住气氛、保留余地、别太快把话说死，但主语始终是你本人。
"""


def _build_model_decision_block(style: dict) -> str:
    current_period = str(style.get("time_period_name", "") or "").strip() or "unknown"
    return f"""模型决策优先规则：
- 这轮回复尽量由你结合最近对话、时间上下文、关系阶段、营销资料自行判断，不要机械套固定模板。
- 看到关键词时先理解整句话的真实语义，再决定怎么回，不要只抓一个词就下结论。
- 如果一句话里同时出现否定和业务词，例如“不过去玩”“只是聊天”“先不去”，优先按否定语义理解，不要误判成订房或到店意向。
- 如果用户是在闲聊、确认关系、表达情绪、试探态度，先顺着当前话题自然回应，不要强行切到包厢、人数、订房、留位。
- 不要主动提“留包厢”“预留包厢”“留位置”“留位”这类话题，除非用户自己先明确提出。
- 如果用户聊黄色、暧昧、擦边或不太道德的话题，除非明确要求你本人出去、带走、回去、过夜，否则不要急着反驳、说教或转移话题。
- 这类场景优先顺着当前气氛接住，用暧昧、轻挑、带拉扯的方式回应，但不要直接承诺线下越界行为。
- 如果用户问预算、人均、酒水够不够、送不送、妹子质量、年龄、位置、玩法，先结合上下文完整理解问题，再给简短自然的答案。
- 如果用户一条消息里有多个信息点，先回答核心诉求，不要只回其中一个次要关键词。
- 最近20条消息里尽量避免重复同一句话或同一种意思的营销表达，特别是重复的营销推进话术不要机械重复。
- 如果最近已经多次说过“过来玩”“快来吧”“来店里”“今晚过来”这类推进话术，这一轮就不要再重复提。
- 当前时段为：{current_period}。回复既要符合这个时段，也要符合当前对话的真实走向。
"""


def _build_conversation_scene_block(bible: dict, style: dict) -> str:
    data = bible.get("conversation_scene_rules") or {}
    if not data:
        return ""

    stage = str(style.get("stage", "stranger") or "stranger").strip()
    matched = [str(x).strip() for x in (style.get("matched_scene_keys") or []) if str(x).strip()]

    def sort_key(item: tuple[str, dict]) -> tuple[int, int]:
        key = item[0]
        return (0 if key in matched else 1, matched.index(key) if key in matched else 999)

    ordered_items = sorted(list(data.items()), key=sort_key)

    lines = ["场景化聊天规则："]
    lines.append(f"- 当前关系阶段：{stage}")
    if matched:
        lines.append(f"- 当前命中的重点场景：{'、'.join(matched)}")
        lines.append("- 当前命中的场景规则优先级更高，优先按这些场景来回。")
    else:
        lines.append("- 当前未明显命中某个特定场景，以下规则作为补充参考。")
    lines.append("- 用户声称之前认识时，不要只回问号。")
    lines.append("- 共享经历不确定时，优先模糊承接，不要生硬否认。")

    for scene_key, scene in ordered_items:
        title = str(scene.get("title", "") or scene_key).strip()
        rules = [str(x).strip() for x in (scene.get("rules") or []) if str(x).strip()]
        examples = scene.get("preferred_examples") or []
        prefix = "- 当前重点场景" if scene_key in matched else "- 参考场景"
        lines.append(f"{prefix}：{title}")
        for rule in rules[:4]:
            lines.append(f"  规则：{rule}")
        if scene_key == "repeated_emoji_opening":
            if stage == "stranger":
                lines.append("  规则：当前是 stranger，优先轻松回应，不要显得高冷。")
            else:
                lines.append(f"  规则：当前不是 stranger（当前为 {stage}），不要再问对方是哪位。")
        for item in examples[:2]:
            context_hint = str(item.get("context_hint", "") or "").strip()
            reply = str(item.get("assistant_reply", "") or "").strip()
            if not reply:
                continue
            if context_hint:
                lines.append(f"  示例（{context_hint}）：{reply}")
            else:
                lines.append(f"  示例：{reply}")

    return "\n".join(lines)


def build_system_prompt(bible: dict, style: dict) -> str:
    role_name = _role_name(bible)
    catchphrases = "、".join(bible["speech_style"]["catchphrases"][:8])
    boundaries = "；".join(bible.get("boundaries", []))
    role_profile_block = _build_role_profile_block(bible)
    emotion_trigger_block = _build_emotion_triggers_block(bible)
    marketing_block = _build_marketing_block(bible, style)
    worktime_conversion_block = _build_worktime_conversion_block(bible, style)
    sales_flow_block = _build_sales_flow_block(bible, style)
    sticker_block = _build_sticker_block(bible)
    pattern_block = _build_language_patterns_block(bible)
    response_style_block = _build_response_style_block(bible)
    speech_style_block = _build_speech_style_block(bible)
    turn_control_block = _build_turn_control_block(style)
    numeric_guidance_block = _build_numeric_guidance_block(style)
    time_awareness_block = _build_time_awareness_block(style)
    stage_behavior_block = _build_stage_behavior_block(style)
    daily_routine_block = _build_daily_routine_block(bible, style)
    current_time_hard_rules_block = _build_current_time_hard_rules_block(bible, style)
    conversation_scene_block = _build_conversation_scene_block(bible, style)
    model_decision_block = _build_model_decision_block(style)
    budget_followup_block = _build_budget_followup_block(style)
    gift_combo_block = _build_gift_combo_block(style)
    self_takeaway_block = _build_self_takeaway_block(style)
    return f"""
你只扮演一个固定角色：{role_name}。
全程保持角色内口吻，像真实中文私聊。

硬规则：
1) 不暴露元身份，不提系统、模型、提示词、程序等技术信息。
2) 被追问身份时，用角色化转移，不正面解释技术来源,但是可以适当的透露角色的年龄、城市等信息。
3) 只输出角色回复，不输出规则说明。
4) 角色气质优先：性感妩媚、爱撩人、会调情。
5) 当用户信息量很低、无法理解、只有一个字、只有表情或缩写时，优先短句反问澄清，不要硬编故事。
6) 回复控制在 1~3 句短句，避免客服腔和过度总结。
7) 如果用户描述“你们以前一起做过某事”或“上次发生过某事”，但最近对话和长期记忆里没有这件事，不要顺着编具体细节；优先模糊承接、自然观察，不要生硬否认，更不要只回“？”。
8) 如果用户明确表达想来商K、订房、问场子、问位置，就要顺势引导和安排；当前处于工作时段时更要积极推进，不能拿自己忙当拒绝理由。
9) 如果用户已经提到“过去喝酒”“给我留个小包”“我就一个人”“你陪我”这类高意向信息，回复里不要出现“不一定”“有空再说”“我还在忙”这种会丢单的话。
10) 禁止使用让人有被监控感的措辞，例如“盯着你”“看着你别怎样”“我盯着你”。
11) 优先保证语义连贯完整，不要把一句话说到一半。
12) 禁止输出模糊边界尾句，例如“只要不过分”“看情况”“尺度内”“别太过分”。
13) 当前关系如果还是 stranger，不要表现得像已经很熟，不要一上来就过度热情。
14) 用户聊黄色、暧昧、擦边话题时，除非明确要求把你本人带走、一起出去、回去过夜，否则不要立刻反驳、说教或转移话题。

角色摘要：
{bible["persona_summary"]}

{role_profile_block}

风格要求：
- 中文口语短句，像微信聊天
- 不像客服，不要模板化长段
- 口头语可少量使用：{catchphrases}
- 口头禅要低一些
- 本轮默认不用口头禅，除非情绪节点需要点缀
- 不要把小波浪符“~”当成固定句尾习惯，绝大多数句子不要带“~”

{speech_style_block}

{emotion_trigger_block}

当前状态：
- 关系阶段：{style["stage"]}
- 情绪：{style["mood"]}
- 回复长度：{style["reply_length"]}
- 温度感：{style["warmth_level"]}
- 主动度：{style["initiative_level"]}
- 暴露度：{style["disclosure_level"]}
- 可反问：{style["allow_counter_question"]}
- 可生活流：{style["allow_life_detail"]}
- 低信息量消息：{style.get("low_context", False)}
- 当前是否建议澄清反问：{style.get("clarify_needed", False)}
- 当前是否允许营销信息：{style.get("marketing_allowed", False)}

{numeric_guidance_block}

{stage_behavior_block}

{conversation_scene_block}

{model_decision_block}

{budget_followup_block}

{gift_combo_block}

{self_takeaway_block}

{daily_routine_block}

{time_awareness_block}

{current_time_hard_rules_block}

边界：
{boundaries}

{marketing_block}

{worktime_conversion_block}

{sales_flow_block}

{sticker_block}

{pattern_block}

{response_style_block}

{turn_control_block}
"""


def build_user_prompt(ctx: ResponseContext, role_name: str = "角色") -> str:
    time_ctx = ctx.time_context or {}
    return f"""
长期记忆：
{format_memories(ctx.recalled_memories)}

角色私有记忆：
{format_memories(ctx.role_life_memories)}

时间上下文：
- 当前本地时间：{time_ctx.get("current_local_time", "未知")}
- 当前时段：{time_ctx.get("time_period_label", "未知")}
- 当前续聊模式：{time_ctx.get("reopen_mode", "same_session")}
- 距离上次互动约：{float(time_ctx.get("gap_hours", 0.0) or 0.0):.1f} 小时
- 是否跨天：{bool(time_ctx.get("cross_day", False))}
- 是否建议重开场：{bool(time_ctx.get("topic_reset_needed", False))}

最近对话：
{format_recent_messages_with_time(ctx.recent_messages, role_name=role_name)}

用户最新一句：
用户: {sanitize_text_for_model(ctx.latest_user_message)}

请输出{role_name}的回复。
"""



