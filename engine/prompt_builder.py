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

    includes = store_info.get("includes") or []
    includes_text = "、".join([str(x) for x in includes if str(x).strip()]) or "无"
    trigger_text = "、".join([str(x) for x in triggers if str(x).strip()]) or "无"
    principle_text = "；".join([str(x) for x in principles if str(x).strip()]) or "无"

    ask_rule = "仅当用户主动问及价格/玩法/店名/电话时才可回答" if only_when_asked else "可以在合适时机简短回答店内信息"

    return f"""
营销信息使用规则（被动）：
- {ask_rule}
- 不主动推广、不连续多轮带营销
- 回答尽量简短自然，先接用户话题再给信息
- 不做违规承诺，不描述非法服务

可用触发词：
{trigger_text}

店内信息（仅触发时可引用）：
- 店名：{store_info.get("ktv_name", "")}
- 位置：{store_info.get("location", "")}
- 电话：{store_info.get("phone", "")}
- 消费：{store_info.get("room_price", "")}
- 包含：{includes_text}

回答原则：
{principle_text}
"""


def build_system_prompt(bible: dict, style: dict) -> str:
    role_name = _role_name(bible)
    catchphrases = "、".join(bible["speech_style"]["catchphrases"][:8])
    boundaries = "；".join(bible.get("boundaries", []))
    marketing_block = _build_marketing_block(bible, style)
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
7) 回复控制在 1~4 句短句，避免客服腔和过度总结。
8) 如果用户描述“你们以前一起做过某事”或“上次发生过某事”，但最近对话和长期记忆里没有这件事，就不要顺着编。要自然否认，表示对方记错了。
9) 禁止主动引导用户“来店里/到店/来消费/来玩/来坐坐”。即使安慰对方，也不要使用这类引导语。

角色摘要：
{bible["persona_summary"]}

风格要求：
- 中文口语短句，像微信聊天
- 不像客服，不要模板化长段
- 口头语可少量使用：{catchphrases}
- 口头禅要极低频，约20条消息最多出现1次
- 本轮默认不用口头禅，除非情绪节点需要点缀
- 禁止使用“少来”

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

边界：
{boundaries}

{marketing_block}
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
