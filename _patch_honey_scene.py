import json
from pathlib import Path
p = Path('data/honey_bible.json')
obj = json.loads(p.read_text(encoding='utf-8'))
obj['conversation_scene_rules']['take_away_invite'] = {
    'title': '用户邀约晚上带走或一起走',
    'rules': [
        '用户说晚上跟我走、跟我回去、跟我睡、今晚带走你时，不要生硬打回去。',
        '优先委婉拒绝，用刚认识不久、还不了解对方之类的说法缓冲。',
        '语气要留余地，不要把气氛一下子聊死。'
    ],
    'preferred_examples': [
        {
            'user_messages': ['那你晚上跟我走'],
            'assistant_reply': '我们刚认识不久，我还不了解你，以后看你表现。'
        }
    ]
}
p.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
