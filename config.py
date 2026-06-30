"""
Централизованная конфигурация проекта.
Все настраиваемые значения (пути, URL, имена БД) берутся из .env.
Если переменная не задана в .env — используется значение по умолчанию.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# Telegram
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")

# Модель Ollama
MODEL_NAME = os.getenv("MODEL_NAME", "gemma3:4b")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_CHAT_URL = f"{OLLAMA_URL}/api/chat"
OLLAMA_EMBED_URL = OLLAMA_URL  # chromadb сам достраивает нужный путь
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
# Сколько чанков отправлять в Ollama за один запрос при индексации.
# Маленькое значение — важно на слабой оперативке: большой пакет (например,
# все 934 чанка УК РФ разом) может уронить внутренний процесс Ollama,
# который реально считает эмбеддинги.
EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "16"))

# Пути к данным
DB_PATH = os.getenv("DB_PATH", "logs.db")
CACHE_DB_PATH = os.getenv("CACHE_DB_PATH", "cache.db")
CHROMA_PATH = os.getenv("CHROMA_PATH", "./chroma_db")
DOCS_PATH = os.getenv("DOCS_PATH", "docs")

# Кеш
CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", str(7 * 24 * 3600)))  # 7 дней по умолчанию

# Память диалога
MAX_MEMORY_MESSAGES = int(os.getenv("MAX_MEMORY_MESSAGES", "20"))
MEMORY_RESTORE_LIMIT = int(os.getenv("MEMORY_RESTORE_LIMIT", "10"))

# Поиск
SEARCH_TOP_K = int(os.getenv("SEARCH_TOP_K", "5"))
SEARCH_DISTANCE_THRESHOLD = float(os.getenv("SEARCH_DISTANCE_THRESHOLD", "1.5"))
