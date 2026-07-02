"""
Валидатор: только проверка на галлюцинации статей.
LLM-judge убран — он ломал все ответы, возвращая INVALID.
"""
import re
from typing import List, Tuple

# Дружеский запасной ответ — используется только если ответ не прошёл проверку
SAFE_FALLBACK_ANSWER = (
    "Расскажи подробнее — что именно случилось и при каких обстоятельствах? "
    "Это поможет дать точную оценку."
)


class Validator:
    def __init__(self, model_name: str = None):
        print("✓ Валидатор готов")

    def _extract_articles(self, text: str) -> List[str]:
        return re.findall(r'ст(?:атья)?\.?\s*(\d+(?:\.\d+)?)', text, re.IGNORECASE)

    def validate(self, response: str, context: str) -> Tuple[bool, str]:
        """
        Проверяет что все статьи в ответе есть в контексте УК РФ.
        Защита от галлюцинаций — модель не должна называть статьи которых нет в найденных чанках.
        """
        if not context.strip():
            return True, ""

        response_articles = self._extract_articles(response)
        if not response_articles:
            return True, ""

        context_articles = self._extract_articles(context)
        if not context_articles:
            return False, "В ответе есть статьи, которых нет в контексте"

        fake = [a for a in response_articles if a not in context_articles]
        if fake:
            return False, f"Возможные галлюцинации: ст. {', '.join(fake)}"

        return True, ""
