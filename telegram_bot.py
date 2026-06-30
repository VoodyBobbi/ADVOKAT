"""
Интерфейс для Telegram.

Отвечает только за:
- Приём сообщений от пользователя.
- Передачу запроса в сервисный слой (QueryService).
- Отправку готового ответа пользователю.
- Обработку команд (/start, /clear, /stats).

Этот модуль не содержит никакой бизнес-логики.
"""
import asyncio
import os
import time
import logging
import urllib.request
from telegram import Update
from telegram.ext import (
    Application, CommandHandler, MessageHandler,
    filters, ContextTypes
)
from main import initialize_system
from config import TELEGRAM_BOT_TOKEN, OLLAMA_URL

# Исправление для Windows: принудительно указываем правильный адрес Ollama
if OLLAMA_URL:
    try:
        host, port = OLLAMA_URL.replace("http://", "").split(":")
        os.environ["OLLAMA_HOST"] = host
        os.environ["OLLAMA_PORT"] = port
    except ValueError:
        print(f"Неверный формат OLLAMA_URL: {OLLAMA_URL}. Ожидается http://host:port")


logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

MIN_INTERVAL = 3
_last_seen: dict[str, float] = {}


def _is_flooding(user_id: str) -> bool:
    """Простая защита от флуда."""
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
    service = context.bot_data["service"]
    service.clear_user_memory(user_id)
    await update.message.reply_text("Разговор очищен. Начнём заново — расскажи, что случилось.")


async def cmd_stats(update: Update, context: ContextTypes.DEFAULT_TYPE):
    service = context.bot_data["service"]
    stats = service.get_statistics()
    text = (
        f"Всего запросов: {stats['total_requests']}\n"
        f"Из кеша: {stats['cached_requests']}\n"
        f"Уникальных пользователей: {stats['unique_users']}\n"
        f"Среднее время ответа: {stats['avg_response_time_ms']} мс"
    )
    await update.message.reply_text(text)


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    user_id = str(user.id)
    username = user.username or user.first_name or "Unknown"
    query = update.message.text.strip()

    if not query:
        return

    if _is_flooding(user_id):
        await update.message.reply_text("Подожди секунду перед следующим вопросом.")
        return

    service = context.bot_data["service"]

    # Индикатор "печатает..."
    async def keep_typing():
        while True:
            await context.bot.send_chat_action(
                chat_id=update.effective_chat.id,
                action="typing"
            )
            await asyncio.sleep(4)

    typing_task = asyncio.create_task(keep_typing())

    try:
        # Вызываем единый метод сервисного слоя
        answer = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: service.answer_query(
                query=query,
                user_id=user_id,
                username=username,
                source="telegram"
            )
        )
    finally:
        typing_task.cancel()
        try:
            await typing_task
        except asyncio.CancelledError:
            pass

    # Отправка ответа
    if len(answer) > 4096:
        for i in range(0, len(answer), 4096):
            await update.message.reply_text(answer[i:i + 4096])
    else:
        await update.message.reply_text(answer)


def run_bot():
    if not TELEGRAM_BOT_TOKEN:
        raise ValueError("TELEGRAM_BOT_TOKEN не задан в .env")

    # Проверяем, что Ollama запущен
    try:
        urllib.request.urlopen(OLLAMA_URL, timeout=3)
    except Exception:
        raise RuntimeError(f"Ollama не отвечает по адресу {OLLAMA_URL}! Запусти приложение Ollama и попробуй снова.")

    # Инициализируем всю систему и получаем один объект-сервис
    query_service = initialize_system()

    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    # Передаём в контекст только один объект
    app.bot_data["service"] = query_service

    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("clear", cmd_clear))
    app.add_handler(CommandHandler("stats", cmd_stats))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    logger.info("Бот запущен")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    run_bot()
