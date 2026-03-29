from pathlib import Path
p = Path('engine/prompt_builder.py')
text = p.read_text(encoding='utf-8')
text = text.replace(
    '7) 如果用户描述“你们以前一起做过某事”或“上次发生过某事”，但最近对话和长期记忆里没有这件事，就不要顺着编。要自然否认，表示对方记错了。',
    '7) 如果用户描述“你们以前一起做过某事”或“上次发生过某事”，但最近对话和长期记忆里没有这件事，不要顺着编具体细节；优先模糊承接、自然观察，不要生硬否认，更不要只回“？”。'
)
text = text.replace(
    '12) 禁止输出模糊边界尾句，例如“只要不过分”“看你表现”“看情况”“尺度内”“别太过分”。',
    '12) 禁止输出模糊边界尾句，例如“只要不过分”“看情况”“尺度内”“别太过分”。'
)
p.write_text(text, encoding='utf-8')
