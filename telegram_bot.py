import asyncio
import subprocess
import sys
import time
import logging
import urllib.request
from telegram import Update
from telegram.ext import (
    Application, CommandHandler, MessageHandler,
    filters, ContextTypes
)
from main import initialize_system, answer_question
from config import TELEGRAM_BOT_TOKEN

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

MIN_INTERVAL = 3
_last_seen: dict[str, float] = {}


def _is_flooding(user_id: str) -> bool:
    now = time.time()
    if user_id in _last_seen and now - _last_seen[user_id] < MIN_INTERVAL:
        return True
    _last_seen[user_id] = now
    return False


async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Привет. Я даю оценку содеянному по закону — коротко и по-человечески, "
        "без юридической волокиты.\n\n"
        "Опиши ситуацию своими словами — задам пару уточняющих вопросов и дам квалификацию.\n\n"
        "Команды:\n"
        "/start — начало\n"
        "/clear — забыть текущий разговор и начать заново\n"
        "/stats — статистика использования бота"
    )


async def cmd_clear(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = str(update.effective_user.id)
    rag = context.bot_data["rag"]
    rag.clear_memory(user_id)
    await update.message.reply_text("Разговор очищен. Начнём заново — расскажи, что случилось.")


async def cmd_stats(update: Update, context: ContextTypes.DEFAULT_TYPE):
    log_db = context.bot_data["logger"]
    stats = log_db.get_stats()
    text = (
        f"Всего запросов: {stats['total_requests']}\n"
        f"Из кеша: {stats['cached_requests']}\n"
        f"Уникальных пользователей: {stats['unique_users']}\n"
        f"Среднее время ответа: {stats['avg_response_time_ms']} мс"
    )
    await update.message.reply_text(text)


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    # user.id уникален для каждого человека в Telegram — поэтому переписка
    # одного человека никогда не путается с перепиской другого.
    user_id = str(user.id)
    username = user.username or user.first_name or "Unknown"
    query = update.message.text.strip()

    if not query:
        return

    if _is_flooding(user_id):
        await update.message.reply_text("Подожди секунду перед следующим вопросом.")
        return

    rag = context.bot_data["rag"]
    cache = context.bot_data["cache"]
    log_db = context.bot_data["logger"]

    # Индикатор "печатает..." живёт 5 сек, а Ollama думает дольше —
    # обновляем индикатор каждые 4 секунды пока считается ответ.
    async def keep_typing():
        while True:
            await context.bot.send_chat_action(
                chat_id=update.effective_chat.id,
                action="typing"
            )
            await asyncio.sleep(4)

    typing_task = asyncio.create_task(keep_typing())

    try:
        answer = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: answer_question(
                query=query,
                rag=rag,
                cache=cache,
                logger=log_db,
                source="telegram",
                user_id=user_id,
                username=username
            )
        )
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


def _is_ollama_running(url: str = "http://localhost:11434", timeout: float = 2.0) -> bool:
    try:
        urllib.request.urlopen(url, timeout=timeout)
        return True
    except Exception:
        return False


def _ensure_ollama_running(max_wait: float = 20.0):
    """
    Если Ollama уже отвечает (например, запущен через десктоп-приложение
    с автозапуском) — ничего не делает, просто идём дальше.
    Если нет — поднимает `ollama serve` в фоне без видимого окна консоли
    и ждёт, реально опрашивая сервер каждые полсекунды, а не по таймеру.
    Процесс остаётся жить после закрытия бота — при следующем запуске
    Ollama уже будет наготове.
    """
    if _is_ollama_running():
        logger.info("Ollama уже запущен")
        return

    logger.info("Ollama не отвечает, пробую запустить автоматически...")

    popen_kwargs = {}
    if sys.platform == "win32":
        popen_kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW

    try:
        subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            **popen_kwargs
        )
    except FileNotFoundError:
        raise RuntimeError(
            "Команда 'ollama' не найдена в системе. Проверь, что Ollama "
            "установлен (попробуй вручную в терминале: ollama list)."
        )
    except Exception as e:
        raise RuntimeError(f"Не удалось запустить Ollama автоматически: {e}")

    waited = 0.0
    step = 0.5
    while waited < max_wait:
        if _is_ollama_running():
            logger.info(f"Ollama поднялся за {waited:.1f} сек")
            return
        time.sleep(step)
        waited += step

    raise RuntimeError(
        f"Ollama не ответил за {max_wait:.0f} сек после автозапуска. "
        "Попробуй вручную: открой терминал, введи 'ollama serve' и посмотри, "
        "что он пишет — там будет настоящая причина."
    )


def run_bot():
    if not TELEGRAM_BOT_TOKEN:
        raise ValueError("TELEGRAM_BOT_TOKEN не задан в .env")

    _ensure_ollama_running()

    _, rag, cache, log_db = initialize_system()

    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    app.bot_data["rag"] = rag
    app.bot_data["cache"] = cache
    app.bot_data["logger"] = log_db

    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("clear", cmd_clear))
    app.add_handler(CommandHandler("stats", cmd_stats))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    logger.info("Бот запущен")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    run_bot()
