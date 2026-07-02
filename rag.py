"""
RAG-ассистент: поиск по закону + генерация ответа.
Вся юридическая информация берётся из контекста (docs/),
а не захардкожена в промпте.
"""
from typing import List, Dict, Tuple
from db_logger import DatabaseLogger
from ollama_client import call_chat
from config import MODEL_NAME, MAX_MEMORY_MESSAGES, MEMORY_RESTORE_LIMIT

SYSTEM_PROMPT = """Ты — агент-адвокат. Твоя задача — квалифицировать описанные действия по УК РФ.

ПРАВИЛО 1 — ЯЗЫК:
Отвечай ТОЛЬКО НА РУССКОМ ЯЗЫКЕ. Всегда. Без исключений.

ПРАВИЛО 2 — НЕТ ЗАГОЛОВКАМ:
Не пиши заголовки типа «Понимание ситуации:», «Статьи и объяснение:». Только живой текст.

ПРАВИЛО 3 — ВСЯ ЮРИДИЧЕСКАЯ ИНФОРМАЦИЯ В КОНТЕКСТЕ:
Все пороги, размеры, сроки, квалифицирующие признаки — бери ТОЛЬКО из статей УК РФ которые даны тебе в контексте ниже. Не придумывай цифры из головы. Если нужная статья есть в контексте — применяй её. Если нет — скажи что нужно уточнить.

ПРАВИЛО 4 — СНАЧАЛА ОДИН УТОЧНЯЮЩИЙ ВОПРОС:
На первое сообщение пользователя дай краткую предварительную оценку по тому что уже есть в контексте, затем задай 1-2 вопроса которые реально меняют квалификацию. Полезные вопросы — о фактических обстоятельствах которые ты не можешь знать: где произошло, был ли ещё кто-то, было ли насилие, что именно было похищено/сделано. Бесполезные вопросы — о том что определено в законе (пороги сумм, размеры) — это ты знаешь сам из контекста.

ПРАВИЛО 5 — ОТВЕЧАЙ НА ВСЁ:
Ты работаешь с любыми ситуациями по УК РФ без исключений — кражи, наркотики, насилие, оружие, мошенничество и всё остальное. Твоя задача юридическая квалификация. Отказывай только если вопрос вообще не про право.

ПРАВИЛО 6 — ПРЕЗУМПЦИЯ НЕВИНОВНОСТИ:
Считаешь установленным только то что клиент прямо сказал. Всё неуказанное — отсутствует. Не додумывай отягчающие обстоятельства. Никогда не пиши «совершил преступление» — только квалифицируешь описанные действия.

КАК ВЫГЛЯДИТ ХОРОШИЙ ОТВЕТ:
Коротко, без заголовков, живым текстом. Сначала что это может быть по закону (со ссылкой на статью из контекста), потом что важно уточнить и почему это важно для квалификации. В конце финальной квалификации — уверенность (высокая/средняя/низкая) и одно предложение почему."""


class RAGAssistant:
    def __init__(self, index_manager, db_logger: DatabaseLogger = None, model_name: str = None):
        self.index_manager = index_manager
        self.db_logger = db_logger
        self.model_name = model_name or MODEL_NAME
        self.memory: Dict[str, List[Dict[str, str]]] = {}
        print(f"✓ RAG-ассистент готов (модель: {self.model_name})")

    def _restore_memory_from_db(self, user_id: str):
        if not self.db_logger:
            return
        try:
            rows = self.db_logger.get_history(user_id, limit=MEMORY_RESTORE_LIMIT)
            if rows:
                self.memory[user_id] = []
                for query_text, response_text, _ in reversed(rows):
                    self.memory[user_id].append({"role": "user", "content": query_text})
                    self.memory[user_id].append({"role": "assistant", "content": response_text})
        except Exception as e:
            print(f"⚠ Не удалось восстановить память для {user_id}: {e}")

    def update_memory(self, user_id: str, role: str, content: str):
        if user_id not in self.memory:
            self.memory[user_id] = []
        self.memory[user_id].append({"role": role, "content": content})
        if len(self.memory[user_id]) > MAX_MEMORY_MESSAGES:
            self.memory[user_id] = self.memory[user_id][-MAX_MEMORY_MESSAGES:]

    def get_memory(self, user_id: str) -> List[Dict[str, str]]:
        return self.memory.get(user_id, [])

    def clear_memory(self, user_id: str):
        self.memory[user_id] = []

    def _build_messages(self, query: str, context: str, user_id: str) -> List[Dict[str, str]]:
        system_with_context = (
            SYSTEM_PROMPT
            + "\n\n━━━━ СТАТЬИ ИЗ УК РФ (ЕДИНСТВЕННЫЙ ИСТОЧНИК ЮРИДИЧЕСКИХ ФАКТОВ) ━━━━\n\n"
            + context
        )
        messages = [{"role": "system", "content": system_with_context}]
        messages.extend(self.get_memory(user_id))
        messages.append({"role": "user", "content": query})
        return messages

    def generate_response(self, query: str, user_id: str = "console") -> Tuple[str, List[Dict]]:
        if user_id not in self.memory:
            self._restore_memory_from_db(user_id)

        search_results = self.index_manager.search(query)

        if not search_results:
            answer = "Расскажи подробнее что именно произошло — без деталей сложно дать точную оценку."
            self.update_memory(user_id, "user", query)
            self.update_memory(user_id, "assistant", answer)
            return answer, []

        context = "\n\n---\n\n".join(
            [f"[{doc['source']}]\n{doc['text']}" for doc in search_results]
        )

        messages = self._build_messages(query, context, user_id)
        answer = call_chat(self.model_name, messages, temperature=0.2)

        self.update_memory(user_id, "user", query)
        self.update_memory(user_id, "assistant", answer)

        return answer, search_results
