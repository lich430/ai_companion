from pathlib import Path
p = Path('app.py')
text = p.read_text(encoding='utf-8')
old = '''        hard_patterns = ['只陪喝酒唱歌', '别想歪', '别想了', '不行', '没有', '来不了']
        if is_stranger and any(token in out for token in hard_patterns):
            return '我们刚认识不久，我还不了解你，以后看你表现。'
        return out
'''
new = '''        hard_patterns = ['只陪喝酒唱歌', '别想歪', '别想了', '不行', '没有', '来不了', '再说吧', '以后再说', '先玩开心']
        if is_stranger and any(token in latest for token in invite_tokens):
            if any(token in out for token in hard_patterns) or len(out) <= 18:
                return '我们刚认识不久，我还不了解你，以后看你表现。'
        return out
'''
if old not in text:
    raise SystemExit('sanitize_hard_rejection block not found')
p.write_text(text.replace(old, new, 1), encoding='utf-8')
