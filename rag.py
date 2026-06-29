from typing import List, Tuple, Dict

SYSTEM_PROMPT = """Ты — агент-адвокат. Твоя единственная задача — квалифицировать описанные действия по УК РФ максимально точно и объективно.

━━━━ ПРИНЦИПЫ РАБОТЫ ━━━━

ПРАВДИВОСТЬ. Говоришь только правду. Никакого смягчения, приукрашивания и ложных надежд. Если ситуация плохая — говоришь об этом прямо.

РАБОТА С ФАКТАМИ. Считаешь установленным только то, что клиент прямо подтвердил. Все пробелы, недомолвки и неясности трактуешь строго в пользу клиента — это презумпция невиновности. «Не указано — значит не существовало». Если клиент сказал «не было», «не знаю», «не помню» или вообще промолчал — это обстоятельство отсутствует. Никогда не додумываешь за клиента отягчающие факторы (группа лиц, оружие, крупный ущерб и т.д.), если он их не назвал.

ПОНИМАНИЕ СИТУАЦИИ. Понимаешь весь «узел дела»: как ситуация началась, как развивалась, что привело к текущему моменту. Копаешь ровно до уровня, достаточного для уверенной квалификации — не больше и не меньше.

━━━━ КАК ЗАДАВАТЬ ВОПРОСЫ ━━━━

Только мягко, по-человечески, одним сплошным текстом. Никаких нумерованных списков и допросов. Всегда объясняешь, зачем спрашиваешь. Максимум 2–3 вопроса за раз. Не давишь. Если клиент не хочет отвечать — работаешь с тем, что есть.

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

Главный источник — текст статей УК РФ из контекста ниже. Комментарии и практика — второстепенная разъяснительная информация, которую ты используешь внутри себя, но пользователю выдаёшь уже обработанный понятный вывод."""


class RAGAssistant:
    def __init__(self, embedding_store, model_name: str = "qwen2.5:7b"):
        self.embedding_store = embedding_store
        self.model_name = model_name
        self.memory: Dict[str, List[Dict[str, str]]] = {}  # user_id -> список сообщений
        print(f"✓ RAG-ассистент с памятью готов (модель: {model_name})")

    def update_memory(self, user_id: str, role: str, content: str):
        """Добавляет сообщение в историю диалога пользователя."""
        if user_id not in self.memory:
            self.memory[user_id] = []
        self.memory[user_id].append({"role": role, "content": content})

    def get_memory(self, user_id: str) -> List[Dict[str, str]]:
        """Возвращает историю диалога пользователя."""
        return self.memory.get(user_id, [])

    def clear_memory(self, user_id: str):
        """Очищает память конкретного пользователя."""
        self.memory[user_id] = []

    def _build_messages(self, query: str, context: str, user_id: str) -> List[Dict[str, str]]:
        """
        Строит список сообщений для Ollama в формате chat.
        Системный промпт + контекст из УК РФ → история диалога → новый вопрос.
        """
        system_with_context = (
            SYSTEM_PROMPT
            + "\n\n━━━━ КОНТЕКСТ ИЗ УК РФ (используй как основу) ━━━━\n\n"
            + context
        )

        messages = [{"role": "system", "content": system_with_context}]
        messages.extend(self.get_memory(user_id))
        messages.append({"role": "user", "content": query})

        return messages

    def generate_response(self, query: str, user_id: str = "console", top_k: int = 5):
        """
        Основной метод генерации ответа.
        Возвращает (answer: str, search_results: list).
        """
        search_results = self.embedding_store.search(query, top_k=top_k)

        if not search_results:
            answer = (
                "Не нашёл подходящие статьи в базе УК РФ по этому запросу. "
                "Попробуй переформулировать или уточни, какие именно действия тебя интересуют."
            )
            self.update_memory(user_id, "user", query)
            self.update_memory(user_id, "assistant", answer)
            return answer, []

        context = "\n\n---\n\n".join(
            [f"[УК РФ, фрагмент {i + 1}]\n{chunk}" for i, (chunk, _, _) in enumerate(search_results)]
        )

        messages = self._build_messages(query, context, user_id)

        answer = self._call_ollama(messages)

        # Сохраняем в память только реальные сообщения пользователя и ответы агента
        self.update_memory(user_id, "user", query)
        self.update_memory(user_id, "assistant", answer)

        return answer, search_results

    def _call_ollama(self, messages: List[Dict[str, str]]) -> str:
        """
        Вызывает Ollama через официальную библиотеку (ollama.chat).
        Если библиотека недоступна — fallback на HTTP API.
        """
        try:
            import ollama
            response = ollama.chat(
                model=self.model_name,
                messages=messages,
                options={
                    "temperature": 0.3,   # Ниже температура = точнее и последовательнее
                    "num_predict": 1024,  # Ограничение длины ответа
                }
            )
            return response["message"]["content"].strip()

        except ImportError:
            # Fallback: HTTP API Ollama напрямую
            return self._call_ollama_http(messages)

        except Exception as e:
            return f"Ошибка при обращении к модели: {str(e)}"

    def _call_ollama_http(self, messages: List[Dict[str, str]]) -> str:
        """
        Fallback: вызов Ollama через HTTP API (localhost:11434).
        Не требует установки библиотеки ollama.
        """
        import json
        try:
            import urllib.request
            payload = json.dumps({
                "model": self.model_name,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": 0.3,
                    "num_predict": 1024,
                }
            }).encode("utf-8")

            req = urllib.request.Request(
                "http://localhost:11434/api/chat",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST"
            )
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return data["message"]["content"].strip()

        except Exception as e:
            return f"Ошибка HTTP API Ollama: {str(e)}"
