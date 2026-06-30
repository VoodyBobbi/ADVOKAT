"""
Validation Layer — Слой проверки качества ответа.

Отвечает за то, чтобы ответ LLM был:
1.  Фактологически корректным (не выдумывал статьи).
2.  Обоснованным (основан только на предоставленном контексте).
3.  Честным (признавал недостаточность данных).
"""
import re
import json
import urllib.request
from typing import List, Dict, Tuple

from config import MODEL_NAME, OLLAMA_CHAT_URL

# Системный промпт для LLM-валидатора. Он должен быть очень строгим.
VALIDATOR_PROMPT = """Ты — беспристрастный и строгий ИИ-валидатор. Твоя задача — проверить ответ, сгенерированный другим ИИ-ассистентом, на соответствие предоставленным документам.

ПРАВИЛА ПРОВЕРКИ:
1.  **ПОЛНАЯ ОБОСНОВАННОСТЬ**: Ответ считается ВАЛИДНЫМ, только если КАЖДОЕ утверждение в нём напрямую подтверждается текстом из предоставленных «Исходных документов».
2.  **НИКАКИХ ДОМЫСЛОВ**: Если ассистент сделал вывод, который логически следует, но не написан в документах прямым текстом, ответ НЕВАЛИДНЫЙ.
3.  **ПРОВЕРКА СТАТЕЙ**: Все номера статей (например, "ст. 158 УК РФ"), упомянутые в ответе, должны присутствовать в «Исходных документах».
4.  **ОДНО СЛОВО**: Твой ответ — это ВСЕГДА только одно слово: `VALID` или `INVALID`.

Вот данные для проверки:
---
ИСХОДНЫЕ ДОКУМЕНТЫ:
{context}
---
ОТВЕТ АССИСТЕНТА:
{response}
---

ОСНОВАН ЛИ ОТВЕТ АССИСТЕНТА **ИСКЛЮЧИТЕЛЬНО** НА ИНФОРМАЦИИ ИЗ ИСХОДНЫХ ДОКУМЕНТОВ?
Ответь одним словом: `VALID` или `INVALID`.
"""

SAFE_FALLBACK_ANSWER = (
    "На основе предоставленной информации я не могу дать точную правовую квалификацию. "
    "Ситуация требует дополнительного анализа или в ней могут быть нюансы, не отраженные в базе знаний. "
    "Пожалуйста, попробуйте переформулировать запрос или предоставьте больше деталей."
)


class Validator:
    def __init__(self, model_name: str = None):
        self.model_name = model_name or MODEL_NAME
        print("✓ Валидатор готов")

    def _extract_articles(self, text: str) -> List[str]:
        """Извлекает все упоминания статей (например, 'ст. 158', 'статья 159.1')."""
        # Находит "ст. X", "статья X", "ст.X" и т.д. с разными пробелами и регистрами
        return re.findall(r'ст(?:атья)?\.?\s*(\d+(?:\.\d+)?)', text, re.IGNORECASE)

    def _check_articles_in_context(self, response_articles: List[str], context: str) -> bool:
        """Проверяет, что все упомянутые в ответе статьи есть в контексте."""
        if not response_articles:
            return True  # Если в ответе нет статей, проверка пройдена

        context_articles = self._extract_articles(context)
        if not context_articles:
            return False  # В ответе есть статьи, а в контексте нет — провал

        # Все статьи из ответа должны быть в контексте
        return all(article in context_articles for article in response_articles)

    def _is_faithful_to_context(self, response: str, context: str) -> bool:
        """Проверяет ответ на обоснованность с помощью LLM-as-Judge."""
        prompt = VALIDATOR_PROMPT.format(context=context, response=response)
        messages = [{"role": "system", "content": prompt}]

        try:
            llm_verdict = self._call_ollama(messages).strip().upper()
            return "VALID" in llm_verdict
        except Exception as e:
            print(f"⚠ Ошибка при вызове LLM-валидатора: {e}")
            return False  # В случае ошибки считаем ответ невалидным

    def validate(self, response: str, context: str) -> Tuple[bool, str]:
        """
        Основной метод валидации. Возвращает (is_valid, error_message).
        """
        # 1. Проверка на "галлюцинации" статей
        response_articles = self._extract_articles(response)
        if not self._check_articles_in_context(response_articles, context):
            msg = f"Ответ ссылается на статьи, которых нет в контексте: {response_articles}"
            print(f"INVALID: {msg}")
            return False, msg

        # 2. Проверка на обоснованность с помощью LLM
        if not self._is_faithful_to_context(response, context):
            msg = "Ответ содержит информацию, не подтвержденную контекстом."
            print(f"INVALID: {msg}")
            return False, msg

        print("✓ Ответ прошел валидацию")
        return True, ""

    def _call_ollama(self, messages: List[Dict[str, str]]) -> str:
        """Приватный метод для вызова Ollama."""
        payload = json.dumps({
            "model": self.model_name,
            "messages": messages,
            "stream": False,
            "options": {"temperature": 0.0}  # Нулевая температура для строгости
        }).encode("utf-8")
        req = urllib.request.Request(
            OLLAMA_CHAT_URL,
            data=payload,
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data["message"]["content"]
