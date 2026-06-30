"""
Логирование всех взаимодействий с ботом в SQLite.
"""
import sqlite3
import csv
from datetime import datetime
from typing import Optional
from pathlib import Path

from config import DB_PATH


class DatabaseLogger:
    def __init__(self, db_path: str = None):
        self.db_path = Path(db_path or DB_PATH)
        self._init_database()

    def _init_database(self) -> None:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        # Добавляем новую колонку is_valid
        try:
            cursor.execute("ALTER TABLE logs ADD COLUMN is_valid INTEGER")
        except sqlite3.OperationalError:
            pass  # Колонка уже существует

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
                is_valid INTEGER,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_user_id ON logs(user_id)")
        conn.commit()
        conn.close()

    def log_interaction(self, query: str, response: str, source: str = "console",
                        user_id: Optional[str] = None, username: Optional[str] = None,
                        from_cache: bool = False, response_time_ms: Optional[int] = None,
                        is_valid: Optional[bool] = None) -> None:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        timestamp = datetime.now().isoformat()
        cursor.execute("""
            INSERT INTO logs (timestamp, user_id, username, source, query, response,
                             from_cache, response_time_ms, is_valid)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (timestamp, user_id, username, source, query, response,
              1 if from_cache else 0, response_time_ms,
              1 if is_valid else 0 if is_valid is not None else None))
        conn.commit()
        conn.close()

    def get_stats(self) -> dict:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        stats = {
            "total_requests": cursor.execute("SELECT COUNT(*) FROM logs").fetchone()[0],
            "cached_requests": cursor.execute("SELECT COUNT(*) FROM logs WHERE from_cache = 1").fetchone()[0],
            "unique_users": cursor.execute("SELECT COUNT(DISTINCT user_id) FROM logs WHERE user_id IS NOT NULL").fetchone()[0],
            "avg_response_time_ms": int(cursor.execute("SELECT AVG(response_time_ms) FROM logs WHERE response_time_ms IS NOT NULL").fetchone()[0] or 0),
            "invalid_responses": cursor.execute("SELECT COUNT(*) FROM logs WHERE is_valid = 0").fetchone()[0]
        }
        conn.close()
        return stats
    
    # ... (остальные методы без изменений)
    def export_to_csv(self, output_path: str = "logs_export.csv") -> str:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM logs ORDER BY timestamp DESC")
        rows = cursor.fetchall()
        headers = [description[0] for description in cursor.description]
        conn.close()

        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            writer.writerows(rows)
        return output_path

    def get_history(self, user_id: str, limit: int = 20) -> list:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT query, response, timestamp FROM logs WHERE user_id = ? ORDER BY timestamp DESC LIMIT ?", (user_id, limit))
        rows = cursor.fetchall()
        conn.close()
        return rows
