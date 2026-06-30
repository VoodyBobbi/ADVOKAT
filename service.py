"""
Сервисный слой — «мозг» приложения.
"""
import time
from rag import RAGAssistant
from cache import ResponseCache
from db_logger import DatabaseLogger
from validator import Validator, SAFE_FALLBACK_ANSWER


class QueryService:
    def __init__(self, rag: RAGAssistant, cache: ResponseCache, logger: DatabaseLogger, validator: Validator):
        self.rag = rag
        self.cache = cache
        self.logger = logger
        self.validator = validator
        print("✓ Сервисный слой готов")

    def answer_query(self, query: str, user_id: str, username: str, source: str) -> str:
        """Основной метод для обработки запроса пользователя."""
        start_time = time.time()

        # 1. Управление кешем и памятью
        has_history = bool(self.rag.get_memory(user_id))
        cached_answer = self.cache.get(query) if not has_history else None
        from_cache = cached_answer is not None

        if from_cache:
            answer = cached_answer
            is_valid = True  # Считаем, что кешированные ответы уже валидны
        else:
            # 2. Генерация ответа
            raw_answer, search_results = self.rag.generate_response(query, user_id=user_id)

            # 3. Валидация ответа
            context_for_validation = "\n\n---\n\n".join([doc["text"] for doc in search_results])
            is_valid, _ = self.validator.validate(raw_answer, context_for_validation)

            if is_valid:
                answer = raw_answer
                if not has_history:
                    self.cache.set(query, answer)
            else:
                # Если ответ не прошел валидацию, возвращаем безопасный ответ
                answer = SAFE_FALLBACK_ANSWER
                # Не кешируем невалидные или запасные ответы

        response_time_ms = int((time.time() - start_time) * 1000)

        # 4. Логирование
        self.logger.log_interaction(
            query=query,
            response=answer,
            source=source,
            user_id=user_id,
            username=username,
            from_cache=from_cache,
            response_time_ms=response_time_ms,
            is_valid=is_valid  # Добавляем флаг валидности в лог
        )

        return answer

    def clear_user_memory(self, user_id: str):
        self.rag.clear_memory(user_id)

    def get_statistics(self) -> dict:
        return self.logger.get_stats()
