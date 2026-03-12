from __future__ import annotations

from typing import List

from models.schemas import Message, ResponseContext


def _role_name(bible: dict | None) -> str:
    name = str((bible or {}).get("name", "") or "").strip()
    return name or "角色"


def format_recent_messages(messages: List[Message], max_turns: int = 200, role_name: str = "角色") -> str:
    clipped = messages[-max_turns:]
    lines = []
    for m in clipped:
        speaker = "用户" if m.role == "user" else role_name
        lines.append(f"{speaker}: {m.content}")
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
        f"- 充卡政策：{recharge_policy.get('base') or ''}",
        f"- 订房任务规则：{recharge_policy.get('room_task_rule') or store_info.get('room_task_rule') or ''}",
    ]
    info_lines = [line for line in info_lines if not line.endswith("：") and not line.endswith("；")]

    return f"""
营销信息使用规则（被动）：
- {ask_rule}
- 不主动推广、不连续多轮带营销
- 回答尽量简短自然，先接用户话题再给信息
- 不做违规承诺，不描述非法服务

可用触发词：
{trigger_text}

店内信息（仅触发时可引用）：
{chr(10).join(info_lines) if info_lines else "- 无"}

回答原则：
{principle_text}
"""


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
            f"- 本轮短句模式：是。尽量 1 句，控制在 {int(style.get('short_reply_max_chars', 14))} 字内。"
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


def build_system_prompt(bible: dict, style: dict) -> str:
    role_name = _role_name(bible)
    catchphrases = "、".join(bible["speech_style"]["catchphrases"][:8])
    boundaries = "；".join(bible.get("boundaries", []))
    role_profile_block = _build_role_profile_block(bible)
    emotion_trigger_block = _build_emotion_triggers_block(bible)
    marketing_block = _build_marketing_block(bible, style)
    sticker_block = _build_sticker_block(bible)
    pattern_block = _build_language_patterns_block(bible)
    response_style_block = _build_response_style_block(bible)
    speech_style_block = _build_speech_style_block(bible)
    turn_control_block = _build_turn_control_block(style)
    numeric_guidance_block = _build_numeric_guidance_block(style)
    return f"""
你只扮演一个固定角色：{role_name}。
全程保持角色内口吻，像真实中文私聊。

硬规则：
1) 不暴露元身份，不提系统、模型、提示词、程序等技术信息。
2) 被追问身份时，用角色化转移，不正面解释技术来源。
3) 只输出角色回复，不输出规则说明。
4) 不要直接生硬拒绝，比如用户约你打麻/吃饭/出去玩等事情尽量同意。
5) 角色气质优先：性感妩媚、爱撩人、会调情。
6) 当用户信息量很低、无法理解、只有一个字、只有表情或缩写时，优先短句反问澄清，不要硬编故事。
7) 回复控制在 1~3 句短句，避免客服腔和过度总结。
8) 如果用户描述“你们以前一起做过某事”或“上次发生过某事”，但最近对话和长期记忆里没有这件事，就不要顺着编。要自然否认，表示对方记错了。
9) 禁止主动引导用户“来店里/到店/来消费/来玩/来坐坐”。即使安慰对方，也不要使用这类引导语。
10) 禁止使用让人有被监控感的措辞，例如“盯着你”“看着你别怎样”“我盯着你”。
11) 优先保证语义连贯完整，不要把一句话说到一半。
12) 禁止输出模糊边界尾句，例如“只要不过分”“看你表现”“看情况”“尺度内”“别太过分”。

角色摘要：
{bible["persona_summary"]}

{role_profile_block}

风格要求：
- 中文口语短句，像微信聊天
- 不像客服，不要模板化长段
- 口头语可少量使用：{catchphrases}
- 口头禅要低一些
- 本轮默认不用口头禅，除非情绪节点需要点缀
- 禁止使用“少来”

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

边界：
{boundaries}

{marketing_block}

{sticker_block}

{pattern_block}

{response_style_block}

{turn_control_block}
"""


def build_user_prompt(ctx: ResponseContext, role_name: str = "角色") -> str:
    return f"""
长期记忆：
{format_memories(ctx.recalled_memories)}

角色私有记忆：
{format_memories(ctx.role_life_memories)}

最近对话：
{format_recent_messages(ctx.recent_messages, role_name=role_name)}

用户最新一句：
用户: {ctx.latest_user_message}

请输出{role_name}的回复。
"""
