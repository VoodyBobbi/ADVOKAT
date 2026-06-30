"""
Кеш ответов на SQLite. Thread-safe, с TTL (время жизни записи).
ВАЖНО: кеш работает только для вопросов БЕЗ контекста диалога —
смотри логику в main.py (answer_question), где кеш пропускается
для пользователей с активной историей переписки.
"""
import hashlib
import sqlite3
import time
import threading
from pathlib import Path
from typing import Optional

from config import CACHE_DB_PATH, CACHE_TTL_SECONDS


class ResponseCache:
    def __init__(self, db_path: str = None, ttl_seconds: int = None):
        self.db_path = Path(db_path or CACHE_DB_PATH)
        self.ttl = ttl_seconds if ttl_seconds is not None else CACHE_TTL_SECONDS
        self._lock = threading.Lock()
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cache (
                    key TEXT PRIMARY KEY,
                    response TEXT NOT NULL,
                    created_at REAL NOT NULL
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_created ON cache(created_at)")
            conn.commit()

    def _get_key(self, query: str) -> str:
        normalized = " ".join(query.lower().split())
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    def get(self, query: str) -> Optional[str]:
        key = self._get_key(query)
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                row = conn.execute(
                    "SELECT response, created_at FROM cache WHERE key = ?", (key,)
                ).fetchone()
        if row:
            response, created_at = row
            if time.time() - created_at < self.ttl:
                return response
            self._delete(key)
        return None

    def set(self, query: str, response: str) -> None:
        key = self._get_key(query)
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO cache (key, response, created_at) VALUES (?, ?, ?)",
                    (key, response, time.time())
                )
                conn.commit()

    def _delete(self, key: str) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM cache WHERE key = ?", (key,))
            conn.commit()

    def clear(self) -> None:
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("DELETE FROM cache")
                conn.commit()

    def cleanup_expired(self) -> int:
        cutoff = time.time() - self.ttl
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("DELETE FROM cache WHERE created_at < ?", (cutoff,))
                conn.commit()
                return cursor.rowcount

    def size(self) -> int:
        with sqlite3.connect(self.db_path) as conn:
            return conn.execute("SELECT COUNT(*) FROM cache").fetchone()[0]
