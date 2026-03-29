from pathlib import Path
p = Path('engine/prompt_builder.py')
text = p.read_text(encoding='utf-8')
start = text.index('def _build_conversation_scene_block')
end = text.index('def build_system_prompt', start)
new_block = '''def _build_conversation_scene_block(bible: dict, style: dict) -> str:
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

    return "\\n".join(lines)\n\n\n'''
text = text[:start] + new_block + text[end:]
p.write_text(text, encoding='utf-8')
