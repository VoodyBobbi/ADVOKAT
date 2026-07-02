"""
Просмотр логов в терминале PyCharm.
Запуск: python show_logs.py

Показывает последние 20 диалогов из logs.db
"""
import sqlite3
from pathlib import Path
from datetime import date, timedelta
import sys

DB_PATH = "logs.db"
LOGS_DIR = Path("logs")


def show_recent(limit: int = 20):
    if not Path(DB_PATH).exists():
        print("⚠ Файл logs.db не найден. Запусти бота хотя бы один раз.")
        return

    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute("""
        SELECT timestamp, username, query, response, response_time_ms, is_valid
        FROM logs ORDER BY timestamp DESC LIMIT ?
    """, (limit,)).fetchall()
    conn.close()

    if not rows:
        print("Логов пока нет.")
        return

    print(f"\n{'='*70}")
    print(f"ПОСЛЕДНИЕ {len(rows)} ДИАЛОГОВ")
    print(f"{'='*70}\n")

    for ts, username, query, response, ms, valid in rows:
        mark = "✓" if valid else "✗" if valid == 0 else "?"
        print(f"[{ts[:16]}] {username or '?'} [{mark}] {ms or '?'} мс")
        print(f"  ❓ {query[:100]}")
        print(f"  💬 {response[:150]}{'...' if len(response) > 150 else ''}")
        print()


def show_stats():
    if not Path(DB_PATH).exists():
        print("⚠ logs.db не найден.")
        return

    conn = sqlite3.connect(DB_PATH)
    total = conn.execute("SELECT COUNT(*) FROM logs").fetchone()[0]
    cached = conn.execute("SELECT COUNT(*) FROM logs WHERE from_cache=1").fetchone()[0]
    users = conn.execute("SELECT COUNT(DISTINCT user_id) FROM logs").fetchone()[0]
    avg_ms = conn.execute("SELECT AVG(response_time_ms) FROM logs WHERE response_time_ms IS NOT NULL").fetchone()[0] or 0
    invalid = conn.execute("SELECT COUNT(*) FROM logs WHERE is_valid=0").fetchone()[0]
    conn.close()

    print(f"\n{'='*40}")
    print(f"СТАТИСТИКА")
    print(f"{'='*40}")
    print(f"Всего запросов:       {total}")
    print(f"Из кеша:              {cached}")
    print(f"Уникальных юзеров:    {users}")
    print(f"Среднее время ответа: {int(avg_ms)} мс")
    print(f"Невалидных ответов:   {invalid}")


def export_today():
    if not Path(DB_PATH).exists():
        print("⚠ logs.db не найден.")
        return

    from db_logger import DatabaseLogger
    logger = DatabaseLogger()
    path = logger.export_daily_csv(for_date=date.today())
    if path:
        print(f"✓ Экспортировано в {path}")
    else:
        print("Сегодня записей нет.")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "logs"

    if cmd == "stats":
        show_stats()
    elif cmd == "export":
        export_today()
    else:
        limit = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 20
        show_recent(limit)
