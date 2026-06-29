import os
import time
from dotenv import load_dotenv
from embeddings import EmbeddingStore
from rag import RAGAssistant
from cache import ResponseCache
from db_logger import DatabaseLogger


def initialize_system():
    print("=" * 70)
    print("🚀 АГЕНТ-АДВОКАТ — ЗАПУСК")
    print("=" * 70)

    load_dotenv()

    cache = ResponseCache()
    embedding_store = EmbeddingStore()

    if embedding_store.collection.count() == 0:
        print("📚 Загружаем УК РФ...")
        embedding_store.add_documents()
    else:
        print(f"✓ База УК РФ готова ({embedding_store.collection.count()} чанков)")

    rag = RAGAssistant(embedding_store)
    logger = DatabaseLogger(db_path="logs.db")

    print("✅ Система готова. Логи пишутся в logs.db")
    return embedding_store, rag, cache, logger


def answer_question(query: str, rag: RAGAssistant, cache: ResponseCache,
                    logger: DatabaseLogger, source: str = "console",
                    user_id: str = "console", username: str = "User"):
    start_time = time.time()

    cached_answer = cache.get(query)
    from_cache = cached_answer is not None

    if cached_answer:
        answer = cached_answer
    else:
        answer, _ = rag.generate_response(query, user_id=user_id)
        cache.set(query, answer)

    response_time_ms = int((time.time() - start_time) * 1000)

    # Полноценное логирование
    logger.log_interaction(
        query=query,
        response=answer,
        source=source,
        user_id=user_id,
        username=username,
        from_cache=from_cache,
        response_time_ms=response_time_ms
    )

    print("\n" + "=" * 70)
    print(answer)
    print("=" * 70)
    return answer


def interactive_mode(rag, cache, logger):
    print("\nАгент-Адвокат готов.")
    print("Команды: exit, stats, logs, clear_cache\n")

    while True:
        q = input("\n👤 Ты: ").strip()

        if q.lower() in ['exit', 'quit', 'выход']:
            break
        if q.lower() == 'stats':
            print(logger.get_stats())
            continue
        if q.lower() == 'logs':
            filename = f"logs_{int(time.time())}.csv"
            logger.export_to_csv(output_path=filename)
            print(f"Логи сохранены в {filename}")
            continue
        if q.lower() == 'clear_cache':
            cache.clear()
            continue

        answer_question(q, rag, cache, logger)


if __name__ == "__main__":
    _, rag, cache, logger = initialize_system()
    interactive_mode(rag, cache, logger)