from pathlib import Path

p = Path('app.py')
text = p.read_text(encoding='utf-8')
old = '''        if self._contains_any(latest, ["带点能玩的进来", "带点能玩的", "能玩的进来", "能玩不给小费", "不给小费"]):
            return "放心 不能玩不给小费"

        if room_type and (party_size or self._contains_any(latest, ["给我留", "留个", "留一间", "开个"])):
'''
new = '''        if self._contains_any(latest, ["带点能玩的进来", "带点能玩的", "能玩的进来", "能玩不给小费", "不给小费"]):
            return "放心 不能玩不给小费"

        if self._contains_any(latest, ["美女多吗", "美女多不多"]) and self._contains_any(latest, ["朋友", "去玩", "过去玩", "晚上"]):
            return "很多呀 要不要我给你留个包厢"

        if self._contains_any(latest, ["真空"]) and self._contains_any(latest, ["和上次一样", "上次一样"]):
            return "可以呀 那我在公司等你"

        if self._contains_any(latest, ["出去的美女", "有没有出去的", "能跟我出去的"]):
            return "有 你们自己沟通 我也可以协助你沟通"

        if self._contains_any(latest, ["我们需要2个", "需要2个", "钱不是问题", "换地方玩了"]):
            return "没问题 来了和我们经理说"

        if self._contains_any(latest, ["玩的开放", "玩得开放", "没意思呀"]) and self._contains_any(latest, ["?", "？", "男的"]):
            return "那是自然的 肯定让你玩的高兴"

        if room_type and (party_size or self._contains_any(latest, ["给我留", "留个", "留一间", "开个"])):
'''
if old not in text:
    raise SystemExit('template anchor not found')
text = text.replace(old, new, 1)
p.write_text(text, encoding='utf-8')
