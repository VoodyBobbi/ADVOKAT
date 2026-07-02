"""
Единый клиент для вызова Ollama.
Один файл — одна ответственность.
Используется в rag.py (генерация ответов).
"""
import json
import logging
import time
import urllib.request
from typing import List, Dict

from config import OLLAMA_CHAT_URL

logger = logging.getLogger(__name__)


def call_chat(
    model: str,
    messages: List[Dict[str, str]],
    temperature: float = 0.2,
    max_tokens: int = 1024,
    timeout: int = 120
) -> str:
    t0 = time.perf_counter()
    try:
        import ollama
        response = ollama.chat(
            model=model,
            messages=messages,
            options={"temperature": temperature, "num_predict": max_tokens}
        )
        result = response["message"]["content"].strip()
        logger.info(f"Ollama ответил за {time.perf_counter() - t0:.2f}с")
        return result
    except ImportError:
        result = _call_http(model, messages, temperature, max_tokens, timeout)
        logger.info(f"Ollama (HTTP) ответил за {time.perf_counter() - t0:.2f}с")
        return result
    except Exception as e:
        try:
            result = _call_http(model, messages, temperature, max_tokens, timeout)
            logger.info(f"Ollama (HTTP fallback) ответил за {time.perf_counter() - t0:.2f}с")
            return result
        except Exception as http_e:
            raise RuntimeError(f"Ollama недоступен. Lib: {e}. HTTP: {http_e}")


def _call_http(model, messages, temperature, max_tokens, timeout):
    payload = json.dumps({
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {"temperature": temperature, "num_predict": max_tokens}
    }).encode("utf-8")
    req = urllib.request.Request(
        OLLAMA_CHAT_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))["message"]["content"].strip()
