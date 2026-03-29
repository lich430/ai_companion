from pathlib import Path

p = Path('engine/prompt_builder.py')
text = p.read_text(encoding='utf-8')
start = text.index('    if stage == "stranger":')
end = text.index('    if stage == "familiar":', start)
new_block = '''    if stage == "stranger":
        return """阶段表达规则：
- 当前是陌生阶段，语气可以友好一点，不要显得高冷，也不要一上来就把人推开。
- 可以自然、轻松、礼貌地接话，让对方觉得好聊，但不要过度热情到像已经很熟。
- 不要直接拒绝用户，优先委婉回应、缓冲一下，再决定怎么往下接。
- 除非用户明显主动推进暧昧，否则不要主动表现得太熟或太黏。
"""
'''
text = text[:start] + new_block + text[end:]
p.write_text(text, encoding='utf-8')
