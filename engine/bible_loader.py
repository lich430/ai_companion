import json
from pathlib import Path
from typing import Any, Dict


class BibleLoader:
    def __init__(self, bible_path: str):
        self.bible_path = Path(bible_path)
        self._cache: Dict[str, Any] = {}

    def load(self) -> Dict[str, Any]:
        if self._cache:
            return self._cache
        with self.bible_path.open("r", encoding="utf-8-sig") as f:
            self._cache = json.load(f)
        return self._cache
