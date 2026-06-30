"""
Точка входа для инициализации системы.
"""
from document_index_manager import DocumentIndexManager
from rag import RAGAssistant
from cache import ResponseCache
from db_logger import DatabaseLogger
from service import QueryService
from validator import Validator


def initialize_system() -> QueryService:
    """Создаёт и возвращает полностью готовый к работе сервисный слой."""
    print("=" * 70)
    print("🚀 АГЕНТ-АДВОКАТ — ЗАПУСК")
    print("=" * 70)

    # 1. Инициализация компонентов
    index_manager = DocumentIndexManager()
    rag = RAGAssistant(index_manager)
    cache = ResponseCache()
    logger = DatabaseLogger()
    validator = Validator()

    # 2. Проверка и индексация документов
    index_manager.add_documents()

    # 3. Создание и возврат единого сервисного слоя
    query_service = QueryService(
        rag=rag,
        cache=cache,
        logger=logger,
        validator=validator
    )

    print("✅ Система готова")
    return query_service


if __name__ == "__main__":
    print("Это модуль для инициализации. Запускай через telegram_bot.py")
