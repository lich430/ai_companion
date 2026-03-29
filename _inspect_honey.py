import json
from pathlib import Path
obj = json.loads(Path('data/honey_bible.json').read_text(encoding='utf-8'))
print(obj['boundaries'])
print(obj['shared_history_policy'])
print(obj['conflict_policy'])
