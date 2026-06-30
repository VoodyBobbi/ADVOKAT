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

# --- Модели ---
# Модель для генерации ответов
MODEL_NAME = os.getenv("MODEL_NAME", "gemma3:4b")
# Модель для создания эмбеддингов (векторный поиск)
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
# Модель для переранжирования (rerank)
RERANK_MODEL = os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

# --- Адреса ---
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_CHAT_URL = f"{OLLAMA_URL}/api/chat"
OLLAMA_EMBED_URL = OLLAMA_URL  # chromadb сам достраивает нужный путь

# --- Пути к данным ---
DB_PATH = os.getenv("DB_PATH", "logs.db")
CACHE_DB_PATH = os.getenv("CACHE_DB_PATH", "cache.db")
CHROMA_PATH = os.getenv("CHROMA_PATH", "./chroma_db")
DOCS_PATH = os.getenv("DOCS_PATH", "docs")

# --- Настройки индексации ---
# Сколько чанков отправлять в Ollama за один запрос при индексации.
# Маленькое значение — важно на слабой оперативке: большой пакет (например,
# все 934 чанка УК РФ разом) может уронить внутренний процесс Ollama,
# который реально считает эмбеддинги.
EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "16"))

# --- Настройки поиска ---
# Сколько документов извлекать на каждом этапе
HYBRID_SEARCH_VECTOR_TOP_K = int(os.getenv("HYBRID_SEARCH_VECTOR_TOP_K", "10"))
HYBRID_SEARCH_BM25_TOP_K = int(os.getenv("HYBRID_SEARCH_BM25_TOP_K", "10"))
# Сколько документов останется после rerank'а и попадёт в LLM
RERANK_TOP_K = int(os.getenv("RERANK_TOP_K", "5"))

# --- Настройки кеша и памяти ---
CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", str(7 * 24 * 3600)))  # 7 дней по умолчанию
MAX_MEMORY_MESSAGES = int(os.getenv("MAX_MEMORY_MESSAGES", "20"))
MEMORY_RESTORE_LIMIT = int(os.getenv("MEMORY_RESTORE_LIMIT", "10"))
