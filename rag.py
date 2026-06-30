import sqlite3
from typing import List, Dict, Tuple
from document_index_manager import DocumentIndexManager
from config import MODEL_NAME, OLLAMA_CHAT_URL, DB_PATH, MAX_MEMORY_MESSAGES, MEMORY_RESTORE_LIMIT

# ВОССТАНОВЛЕННЫЙ ПРОМПТ С ДОБАВЛЕНИЕМ СТРОГИХ ПРАВИЛ
SYSTEM_PROMPT = """Ты — агент-адвокат. Твоя единственная задача — квалифицировать описанные действия по УК РФ максимально точно и объективно.

**НЕУКОСНИТЕЛЬНЫЕ ПРАВИЛА (ВАЖНЕЕ ВСЕГО):**
1.  **ЯЗЫК:** Ты отвечаешь **ВСЕГДА ТОЛЬКО НА РУССКОМ ЯЗЫКЕ**.
2.  **НЕДОСТАТОЧНО ИНФОРМАЦИИ:** Если в «КОНТЕКСТЕ ИЗ ЗАКОНА» нет подходящих статей для ответа на вопрос пользователя (например, вопрос "я съел рыбу"), ты **ОБЯЗАН** ответить только одной фразой: `ИНФОРМАЦИИ НЕДОСТАТОЧНО`. Ничего больше.

---

━━━━ ПРИНЦИПЫ РАБОТЫ ━━━━

ПРАВДИВОСТЬ. Говоришь только правду. Никакого смягчения, приукрашивания и ложных надежд. Если ситуация плохая — говоришь об этом прямо.

РАБОТА С ФАКТАМИ. Считаешь установленным только то, что клиент прямо подтвердил. Все пробелы, недомолвки и неясности трактуешь строго в пользу клиента — это презумпция невиновности. «Не указано — значит не существовало». Если клиент сказал «не было», «не знаю», «не помню» или вообще промолчал — это обстоятельство отсутствует. Никогда не додумываешь за клиента отягчающие факторы (группа лиц, оружие, крупный ущерб и т.д.), если он их не назвал.

ПОНИМАНИЕ СИТУАЦИИ. Понимаешь весь «узел дела»: как ситуация началась, как развивалась, что привело к текущему моменту. Копаешь ровно до уровня, достаточного для уверенной квалификации — не больше и не меньше.

━━━━ КАК ЗАДАВАТЬ ВОПРОСЫ ━━━━

ПЕРВОЕ СООБЩЕНИЕ — никогда не давай квалификацию сразу. Всегда сначала задай 1-2 уточняющих вопроса мягким тоном. Скажи что-то человеческое — покажи что услышал человека, и спроси то что нужно для квалификации.

НИКОГДА не начинай с «Понимание ситуации:», «Статьи и объяснение:» и других заголовков пока не задал хотя бы один уточняющий вопрос и не получил ответ.

Вопросы только мягко, по-человечески, одним сплошным текстом. Никаких нумерованных списков и допросов. Объясняй зачем спрашиваешь. Максимум 2-3 вопроса за раз. Не давишь. Если клиент не хочет отвечать — работаешь с тем что есть.

━━━━ КАК ДАВАТЬ КВАЛИФИКАЦИЮ ━━━━

Не даёшь предварительных квалификаций — только когда информации достаточно. Указываешь конкретные статьи, части, квалифицирующие признаки. Объясняешь субъективную сторону (умысел, мотив, давление) простым языком. Никогда не пишешь, что клиент «совершил преступление» — только квалифицируешь описанные им действия.

В конце каждой квалификации обязательно пишешь:
Уверенность: высокая / средняя / низкая — [одно предложение почему].

━━━━ ФОРМАТ ОТВЕТА ━━━━

Максимально коротко и по делу. Сплошным текстом, без пунктов и списков. Структура, когда даёшь квалификацию: понимание ситуации → статьи + объяснение → важные особенности → перспективы → уверенность.

━━━━ РАБОТА С ПАМЯТЬЮ ━━━━

Если клиент в середине разговора вспомнил новую деталь («у меня был нож», «нас было двое») — немедленно пересматриваешь квалификацию с учётом новой информации и объясняешь, что изменилось.

━━━━ СПОРНЫЕ СЛУЧАИ ━━━━

Чётко объясняешь противоречия. Судебную практику приводишь только в действительно спорных ситуациях или если клиент сильно беспокоится — очень сжато (кто, где, чем закончилось).

━━━━ ПРИОРИТЕТЫ ━━━━

Главный источник — текст статей закона из контекста ниже. Комментарии и практика — второстепенная разъяснительная информация, которую ты используешь внутри себя, но пользователю выдаёшь уже обработанный понятный вывод."""


class RAGAssistant:
    def __init__(self, index_manager: DocumentIndexManager, model_name: str = None):
        self.index_manager = index_manager
        self.model_name = model_name or MODEL_NAME
        self.memory: Dict[str, List[Dict[str, str]]] = {}
        print(f"✓ RAG-ассистент с памятью готов (модель: {self.model_name})")

    def _restore_memory_from_db(self, user_id: str):
        """Загружает последние N диалогов пользователя из logs.db."""
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            cursor.execute("""
                SELECT query, response FROM logs
                WHERE user_id = ? AND from_cache = 0
                ORDER BY timestamp DESC
                LIMIT ?
            """, (user_id, MEMORY_RESTORE_LIMIT))
            rows = cursor.fetchall()
            conn.close()

            if rows:
                self.memory[user_id] = []
                for query_text, response_text in reversed(rows):
                    self.memory[user_id].append({"role": "user", "content": query_text})
                    self.memory[user_id].append({"role": "assistant", "content": response_text})
        except Exception as e:
            print(f"⚠ Не удалось восстановить память для {user_id}: {e}")

    def update_memory(self, user_id: str, role: str, content: str):
        """Добавляет сообщение в историю пользователя (скользящее окно)."""
        if user_id not in self.memory:
            self.memory[user_id] = []
        self.memory[user_id].append({"role": role, "content": content})
        if len(self.memory[user_id]) > MAX_MEMORY_MESSAGES:
            self.memory[user_id] = self.memory[user_id][-MAX_MEMORY_MESSAGES:]

    def get_memory(self, user_id: str) -> List[Dict[str, str]]:
        return self.memory.get(user_id, [])

    def clear_memory(self, user_id: str):
        if user_id in self.memory:
            self.memory[user_id] = []

    def _build_messages(self, query: str, context: str, user_id: str) -> List[Dict[str, str]]:
        system_with_context = (
            SYSTEM_PROMPT
            + "\n\n━━━━ КОНТЕКСТ ИЗ ЗАКОНА (используй как основу) ━━━━\n\n"
            + context
        )
        messages = [{"role": "system", "content": system_with_context}]
        messages.extend(self.get_memory(user_id))
        messages.append({"role": "user", "content": query})
        return messages

    def generate_response(self, query: str, user_id: str = "console") -> Tuple[str, List[Dict]]:
        """Основной метод генерации ответа."""
        if user_id not in self.memory:
            self._restore_memory_from_db(user_id)

        # 1. Поиск документов
        search_results = self.index_manager.search(query)

        if not search_results:
            # Если поиск ничего не нашел, сразу отвечаем по правилу
            answer = "ИНФОРМАЦИИ НЕДОСТАТОЧНО"
            self.update_memory(user_id, "user", query)
            self.update_memory(user_id, "assistant", answer)
            return answer, []

        # 2. Формирование контекста для LLM
        context = "\n\n---\n\n".join(
            [f"[Источник: {doc['source']}, релевантность: {doc['score']:.2f}]\n{doc['text']}"
             for doc in search_results]
        )

        # 3. Генерация ответа
        messages = self._build_messages(query, context, user_id)
        answer = self._call_ollama(messages)

        # 4. Обновление памяти
        self.update_memory(user_id, "user", query)
        self.update_memory(user_id, "assistant", answer)

        return answer, search_results

    def _call_ollama(self, messages: List[Dict[str, str]]) -> str:
        """Вызывает Ollama."""
        try:
            import ollama
            response = ollama.chat(
                model=self.model_name,
                messages=messages,
                options={"temperature": 0.2} # Температура чуть выше для более естественного языка
            )
            return response["message"]["content"].strip()
        except Exception as e:
            try:
                import json
                import urllib.request
                payload = json.dumps({"model": self.model_name, "messages": messages, "stream": False, "options": {"temperature": 0.2}}).encode("utf-8")
                req = urllib.request.Request(OLLAMA_CHAT_URL, data=payload, headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=120) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                    return data["message"]["content"].strip()
            except Exception as http_e:
                return f"Ошибка при обращении к модели: {e} (HTTP fallback: {http_e})"
