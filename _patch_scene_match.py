from pathlib import Path
p = Path('app.py')
text = p.read_text(encoding='utf-8')
anchor = '''    def _build_style(self, user_id: str, state: RelationshipState, user_text: str, cls: dict, time_context: dict | None = None) -> dict:
'''
insert = '''    def _detect_scene_keys(self, user_text: str, recent: list[Message], state: RelationshipState, time_context: dict | None = None) -> list[str]:
        raw = (user_text or '').strip()
        if not raw:
            return []
        period = str((time_context or {}).get('time_period_name', '') or '').strip()
        hits: list[str] = []
        if self._contains_any(raw, ['调皮', '😜', '😝', '😏']) and len(re.findall(r'(调皮|😜|😝|😏)', raw)) >= 1:
            hits.append('repeated_emoji_opening')
        if self._contains_any(raw, ['上次一起喝酒', '你这么快就忘记了', '你忘了我吗', '之前见过']):
            hits.append('claimed_shared_history')
        if self._contains_any(raw, ['想你了', '一起吃个饭']) and state.stage == 'stranger':
            hits.append('stranger_says_miss_you')
        if self._contains_any(raw, ['好吧', '那算了']):
            hits.append('declined_dinner_repair')
        if period == 'afternoon_window' and self._contains_any(raw, ['你在干嘛', '你在干嘛呢']):
            hits.append('afternoon_before_work_status')
        if self._contains_any(raw, ['美女多吗', '美女多不多']):
            hits.append('beauty_inquiry_conversion')
        if self._contains_any(raw, ['真空']):
            hits.append('zhenkong_arrangement')
        if self._contains_any(raw, ['玩的开放', '玩得开放', '没意思呀']):
            hits.append('after_rejected_open_play')
        if self._contains_any(raw, ['出去的美女', '有没有出去的', '能跟我出去的']):
            hits.append('outside_girls_inquiry')
        if self._contains_any(raw, ['需要2个', '钱不是问题', '换地方玩了']):
            hits.append('outside_two_girls_request')
        if self._contains_any(raw, ['你带4个女孩子过来', '一起去唱歌']):
            hits.append('invite_outside_singing_busy_excuse')
        if self._contains_any(raw, ['跟我走', '跟我回去', '带走你', '跟我睡']):
            hits.append('take_away_invite')
        # keep order, dedupe
        seen = set()
        ordered = []
        for key in hits:
            if key not in seen:
                seen.add(key)
                ordered.append(key)
        return ordered

'''
if anchor not in text:
    raise SystemExit('build_style anchor not found')
text = text.replace(anchor, insert + anchor, 1)
text = text.replace('        style["forbidden_fillers"] = (\n            self.bible.get("speech_style", {}).get("forbidden_fillers")\n            or ["哎呀", "呀", "呢"]\n        )\n', '        style["forbidden_fillers"] = (\n            self.bible.get("speech_style", {}).get("forbidden_fillers")\n            or ["哎呀", "呀", "呢"]\n        )\n        style["matched_scene_keys"] = self._detect_scene_keys(user_text, self._get_recent(user_id), state, time_context=time_context)\n', 1)
p.write_text(text, encoding='utf-8')
