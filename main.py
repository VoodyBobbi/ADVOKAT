import time
from embeddings import EmbeddingStore
from rag import RAGAssistant
from cache import ResponseCache
from db_logger import DatabaseLogger


def initialize_system():
    print("=" * 70)
    print("🚀 АГЕНТ-АДВОКАТ — ЗАПУСК")
    print("=" * 70)

    cache = ResponseCache()
    embedding_store = EmbeddingStore()

    # Вызываем add_documents() при КАЖДОМ запуске — внутри неё уже есть
    # проверка по каждому файлу отдельно, поэтому уже проиндексированные
    # документы пропускаются быстро (без обращений к Ollama), а новые
    # файлы из docs/ подхватываются сами, без какой-либо ручной настройки.
    print(f"✓ В базе сейчас {embedding_store.collection.count()} чанков")
    embedding_store.add_documents()

    rag = RAGAssistant(embedding_store)
    logger = DatabaseLogger()

    print("✅ Система готова")
    return embedding_store, rag, cache, logger


def answer_question(query: str, rag: RAGAssistant, cache: ResponseCache,
                    logger: DatabaseLogger, source: str = "console",
                    user_id: str = "console", username: str = "User"):
    start_time = time.time()

    # Кеш используем только для вопросов БЕЗ активной истории диалога.
    # Диалоговые сообщения зависят от контекста — кешировать их нельзя.
    has_history = bool(rag.get_memory(user_id))
    cached_answer = cache.get(query) if not has_history else None
    from_cache = cached_answer is not None

    if cached_answer:
        answer = cached_answer
    else:
        answer, _ = rag.generate_response(query, user_id=user_id)
        if not has_history:
            cache.set(query, answer)

    response_time_ms = int((time.time() - start_time) * 1000)

    logger.log_interaction(
        query=query,
        response=answer,
        source=source,
        user_id=user_id,
        username=username,
        from_cache=from_cache,
        response_time_ms=response_time_ms
    )

    return answer


if __name__ == "__main__":
    print("Запускай через telegram_bot.py")
