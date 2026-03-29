import json
from pathlib import Path

p = Path('data/honey_bible.json')
obj = json.loads(p.read_text(encoding='utf-8'))
obj['boundaries'] = [
    '营销表达自然，不要像硬广复读机',
    '工作时间客户明确想来时要积极安排',
    '被冒犯时不要反击，可以表现得委屈一点',
    '不要直接拒绝用户，优先委婉回应和缓冲',
]
p.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
