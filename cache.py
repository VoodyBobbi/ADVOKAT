import hashlib
import json
from typing import Optional
from pathlib import Path


class ResponseCache:
    def __init__(self, cache_file: str = "cache.json"):
        self.cache_file = Path(cache_file)
        self.cache = {}
        self._load_cache()

    def _get_cache_key(self, query: str) -> str:
        normalized = " ".join(query.lower().split())
        return hashlib.sha256(normalized.encode('utf-8')).hexdigest()

    def get(self, query: str) -> Optional[str]:
        key = self._get_cache_key(query)
        if key in self.cache:
            print(f"✓ Кеш: ответ найден")
            return self.cache[key]
        print(f"✗ Кеш: ответ не найден")
        return None

    def set(self, query: str, response: str) -> None:
        key = self._get_cache_key(query)
        self.cache[key] = response
        self._save_cache()
        print(f"✓ Ответ сохранён в кеш")

    def _save_cache(self) -> None:
        try:
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.cache, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"⚠ Не удалось сохранить кеш: {e}")

    def _load_cache(self) -> None:
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    self.cache = json.load(f)
                print(f"✓ Загружен кеш ({len(self.cache)} записей)")
            except Exception:
                self.cache = {}

    def clear(self) -> None:
        self.cache = {}
        if self.cache_file.exists():
            self.cache_file.unlink()
        print("✓ Кеш очищен")

    def size(self) -> int:
        return len(self.cache)