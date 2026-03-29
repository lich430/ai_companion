from pathlib import Path
p = Path('app.py')
text = p.read_text(encoding='utf-8')
anchor = '''    @staticmethod
    def _sanitize_surveillance_tone(text: str) -> str:
'''
insert = '''    @staticmethod
    def _sanitize_hard_rejection(text: str, latest_user_text: str, state: RelationshipState | None = None) -> str:
        out = (text or '').strip()
        latest = (latest_user_text or '').strip()
        if not out or not latest:
            return out

        is_stranger = bool(state and getattr(state, 'stage', '') == 'stranger')
        invite_tokens = ['跟我走', '跟我回去', '带走你', '跟我睡', '睡觉', '出去喝酒', '带你出去']
        if not any(token in latest for token in invite_tokens):
            return out

        hard_patterns = ['只陪喝酒唱歌', '别想歪', '别想了', '不行', '没有', '来不了']
        if is_stranger and any(token in out for token in hard_patterns):
            return '我们刚认识不久，我还不了解你，以后看你表现。'
        return out

'''
if anchor not in text:
    raise SystemExit('anchor not found')
text = text.replace(anchor, insert + anchor, 1)
text = text.replace(
    "        processed = self._sanitize_time_conflicts(processed, time_context or {})\n        processed = self._sanitize_surveillance_tone(processed)\n",
    "        processed = self._sanitize_time_conflicts(processed, time_context or {})\n        processed = self._sanitize_hard_rejection(processed, latest_user_text, state)\n        processed = self._sanitize_surveillance_tone(processed)\n",
    1,
)
p.write_text(text, encoding='utf-8')
