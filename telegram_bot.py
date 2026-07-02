"""
Telegram-интерфейс. Никаких команд — только разговор.
Технические команды: python show_logs.py [stats|export|N]
"""
import asyncio
import logging
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, date, timedelta

from telegram import Update
from telegram.ext import Application, MessageHandler, filters, ContextTypes
from main import initialize_system
from config import TELEGRAM_BOT_TOKEN, OLLAMA_URL

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Антифлуд
MIN_INTERVAL = 3
_last_seen: dict[str, float] = {}

# Состояния пользователей в текущей сессии бота
# "new"      → первый раз видим, нет истории
# "awaiting" → вернулся, спросили "продолжим или нова?"
# "active"   → уже в диалоге
_user_state: dict[str, str] = {}


# ──────────────────────────────────────────────
# Ollama авто-запуск
# ──────────────────────────────────────────────

def _is_ollama_running() -> bool:
    try:
        urllib.request.urlopen(OLLAMA_URL, timeout=2)
        return True
    except Exception:
        return False


def _ensure_ollama(max_wait: float = 20.0):
    if _is_ollama_running():
        logger.info("Ollama уже запущен")
        return
    logger.info("Запускаю Ollama автоматически...")
    kwargs = {}
    if sys.platform == "win32":
        kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
    try:
        subprocess.Popen(["ollama", "serve"],
                         stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL, **kwargs)
    except FileNotFoundError:
        raise RuntimeError("Ollama не найден. Проверь установку.")
    waited = 0.0
    while waited < max_wait:
        if _is_ollama_running():
            logger.info(f"Ollama поднялся за {waited:.1f} сек")
            return
        time.sleep(0.5)
        waited += 0.5
    raise RuntimeError(f"Ollama не ответил за {max_wait:.0f} сек")


# ──────────────────────────────────────────────
# Ежедневный экспорт в 00:00
# ──────────────────────────────────────────────

async def _daily_export(app: Application):
    while True:
        now = datetime.now()
        next_midnight = datetime.combine(
            date.today() + timedelta(days=1), datetime.min.time()
        )
        await asyncio.sleep((next_midnight - now).total_seconds())
        service = app.bot_data.get("service")
        if service:
            yesterday = date.today() - timedelta(days=1)
            service.logger.export_daily_csv(for_date=yesterday)


# ──────────────────────────────────────────────
# Обработка сообщений
# ──────────────────────────────────────────────

def _is_flooding(user_id: str) -> bool:
    now = time.time()
    if user_id in _last_seen and now - _last_seen[user_id] < MIN_INTERVAL:
        return True
    _last_seen[user_id] = now
    return False


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    user_id = str(user.id)
    query = update.message.text.strip()

    if not query:
        return
    if _is_flooding(user_id):
        await update.message.reply_text("Подождите секунду.")
        return

    service = context.bot_data["service"]
    state = _user_state.get(user_id)

    # ── Возвращающийся пользователь ждёт выбора ──
    if state == "awaiting":
        low = query.lower()
        if any(w in low for w in ["продолж", "да", "прошл", "тот", "его", "её", "конечно"]):
            _user_state[user_id] = "active"
            await update.message.reply_text(
                "Хорошо, продолжаем. Что изменилось или что хотите уточнить?"
            )
        else:
            # Любой другой ответ = новая ситуация
            service.rag.clear_memory(user_id)
            _user_state[user_id] = "active"
            await update.message.reply_text("Понял. Опишите новую ситуацию.")
        return

    # ── Первое сообщение в этой сессии ──
    if state is None:
        has_history = bool(service.logger.get_history(user_id, limit=1))
        if has_history:
            _user_state[user_id] = "awaiting"
            await update.message.reply_text(
                "С возвращением. Продолжим обсуждение прошлой ситуации "
                "или расскажете о новом случае?"
            )
            return
        else:
            _user_state[user_id] = "active"
            await update.message.reply_text("Опишите вашу ситуацию.")
            return

    # ── Обычный диалог ──
    async def keep_typing():
        while True:
            await context.bot.send_chat_action(
                chat_id=update.effective_chat.id, action="typing"
            )
            await asyncio.sleep(4)

    typing_task = asyncio.create_task(keep_typing())
    try:
        t0 = time.perf_counter()
        answer = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: service.answer_query(
                query=query, user_id=user_id,
                username=user.username or user.first_name or "?",
                source="telegram"
            )
        )
        elapsed = time.perf_counter() - t0
        logger.info(f"Ответ для {user_id} за {elapsed:.2f}с")
    finally:
        typing_task.cancel()
        try:
            await typing_task
        except asyncio.CancelledError:
            pass

    if len(answer) > 4096:
        for i in range(0, len(answer), 4096):
            await update.message.reply_text(answer[i:i + 4096])
    else:
        await update.message.reply_text(answer)


# ──────────────────────────────────────────────
# Запуск
# ──────────────────────────────────────────────

def run_bot():
    if not TELEGRAM_BOT_TOKEN:
        raise ValueError("TELEGRAM_BOT_TOKEN не задан в .env")

    _ensure_ollama()
    query_service = initialize_system()

    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    app.bot_data["service"] = query_service

    # Только текстовые сообщения — никаких команд
    app.add_handler(MessageHandler(filters.TEXT, handle_message))

    async def post_init(application: Application):
        application.create_task(_daily_export(application))

    app.post_init = post_init

    logger.info("Бот запущен")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    run_bot()
