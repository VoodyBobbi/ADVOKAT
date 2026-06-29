import sqlite3
import csv
from datetime import datetime
from typing import Optional
from pathlib import Path


class DatabaseLogger:
    def __init__(self, db_path: str = "logs.db"):
        self.db_path = Path(db_path)
        self._init_database()

    def _init_database(self) -> None:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                user_id TEXT,
                username TEXT,
                source TEXT NOT NULL,
                query TEXT NOT NULL,
                response TEXT NOT NULL,
                from_cache INTEGER DEFAULT 0,
                response_time_ms INTEGER,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON logs(timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_user_id ON logs(user_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_source ON logs(source)")
        conn.commit()
        conn.close()

    def log_interaction(self, query: str, response: str, source: str = "console",
                        user_id: Optional[str] = None, username: Optional[str] = None,
                        from_cache: bool = False, response_time_ms: Optional[int] = None) -> None:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        timestamp = datetime.now().isoformat()
        cursor.execute("""
            INSERT INTO logs (timestamp, user_id, username, source, query, response,
                             from_cache, response_time_ms)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (timestamp, user_id, username, source, query, response,
              1 if from_cache else 0, response_time_ms))
        conn.commit()
        conn.close()

    def get_stats(self) -> dict:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM logs")
        total = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM logs WHERE from_cache = 1")
        cached = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(DISTINCT user_id) FROM logs WHERE user_id IS NOT NULL")
        unique_users = cursor.fetchone()[0]
        cursor.execute("SELECT AVG(response_time_ms) FROM logs WHERE response_time_ms IS NOT NULL")
        avg_time = cursor.fetchone()[0] or 0
        conn.close()
        return {
            "total_requests": total,
            "cached_requests": cached,
            "unique_users": unique_users,
            "avg_response_time_ms": int(avg_time),
        }

    def export_to_csv(self, output_path: str = "logs_export.csv") -> str:
        """Экспортирует все записи из logs.db в CSV-файл."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, timestamp, user_id, username, source,
                   query, response, from_cache, response_time_ms
            FROM logs
            ORDER BY timestamp DESC
        """)
        rows = cursor.fetchall()
        conn.close()

        headers = [
            "id", "timestamp", "user_id", "username", "source",
            "query", "response", "from_cache", "response_time_ms"
        ]

        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            writer.writerows(rows)

        return output_path

    def get_history(self, user_id: str, limit: int = 20) -> list:
        """Возвращает последние N записей диалога конкретного пользователя."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT query, response, timestamp
            FROM logs
            WHERE user_id = ?
            ORDER BY timestamp DESC
            LIMIT ?
        """, (user_id, limit))
        rows = cursor.fetchall()
        conn.close()
        return rows
