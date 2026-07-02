"""
Менеджер индексов документов.
Гибридный поиск: ChromaDB (вектор) + BM25 (ключевые слова).
Профилирование: замер времени каждого компонента.
Валидация: проверка документов перед индексацией.
"""
import logging
import time
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from rank_bm25 import BM25Okapi
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from docx import Document
import numpy as np

from config import (
    CHROMA_PATH, DOCS_PATH, OLLAMA_EMBED_URL, EMBEDDING_MODEL,
    EMBED_BATCH_SIZE, HYBRID_SEARCH_VECTOR_TOP_K,
    HYBRID_SEARCH_BM25_TOP_K, RERANK_TOP_K
)

logger = logging.getLogger(__name__)


class DocumentIndexManager:
    def __init__(self, collection_name: str = "legal_docs"):
        logger.info("Инициализация менеджера индексов...")
        self.client = chromadb.PersistentClient(
            path=CHROMA_PATH,
            settings=Settings(anonymized_telemetry=False)
        )
        self.embedding_fn = embedding_functions.OllamaEmbeddingFunction(
            model_name=EMBEDDING_MODEL,
            url=OLLAMA_EMBED_URL
        )
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "Нормативно-правовые акты"},
            embedding_function=self.embedding_fn
        )
        self.bm25_index = None
        self.all_docs: Dict[str, Dict] = {}
        self._build_bm25()
        logger.info(f"Менеджер готов (чанков: {self.collection.count()}, BM25: {len(self.all_docs)})")

    def _build_bm25(self):
        if self.collection.count() == 0:
            return
        t0 = time.perf_counter()
        data = self.collection.get(include=["documents", "metadatas"])
        for id_, doc, meta in zip(data["ids"], data["documents"], data["metadatas"]):
            self.all_docs[id_] = {"text": doc, "meta": meta}
        corpus = [doc["text"].split() for doc in self.all_docs.values()]
        self.bm25_index = BM25Okapi(corpus)
        logger.info(f"BM25 построен за {time.perf_counter() - t0:.2f}с ({len(self.all_docs)} документов)")

    # ──────────────────────────────────────────────
    # Поиск с профилированием
    # ──────────────────────────────────────────────

    def search(self, query: str) -> List[Dict]:
        if self.collection.count() == 0:
            return []

        t0 = time.perf_counter()
        vector_scores = self._vector_search(query)
        t_vec = time.perf_counter() - t0

        t0 = time.perf_counter()
        bm25_scores = self._bm25_search(query)
        t_bm25 = time.perf_counter() - t0

        logger.info(f"Поиск: Chroma={t_vec:.3f}с, BM25={t_bm25:.3f}с")

        all_ids = set(vector_scores) | set(bm25_scores)
        combined = {
            id_: 0.6 * vector_scores.get(id_, 0.0) + 0.4 * bm25_scores.get(id_, 0.0)
            for id_ in all_ids
        }
        top_ids = sorted(combined, key=lambda x: combined[x], reverse=True)[:RERANK_TOP_K]

        return [
            {
                "text": self.all_docs[id_]["text"],
                "source": self.all_docs[id_]["meta"].get("source", "Документ"),
                "score": combined[id_]
            }
            for id_ in top_ids if id_ in self.all_docs
        ]

    def _vector_search(self, query: str) -> Dict[str, float]:
        results = self.collection.query(
            query_texts=[query],
            n_results=HYBRID_SEARCH_VECTOR_TOP_K
        )
        if not results["ids"]:
            return {}
        return {
            id_: 1.0 / (1.0 + dist)
            for id_, dist in zip(results["ids"][0], results["distances"][0])
        }

    def _bm25_search(self, query: str) -> Dict[str, float]:
        if not self.bm25_index:
            return {}
        scores = self.bm25_index.get_scores(query.split())
        doc_ids = list(self.all_docs.keys())
        top_n = np.argsort(scores)[::-1][:HYBRID_SEARCH_BM25_TOP_K]
        max_score = max(scores[top_n]) if len(top_n) > 0 else 1.0
        return {
            doc_ids[i]: float(scores[i]) / max_score
            for i in top_n if scores[i] > 0 and max_score > 0
        }

    # ──────────────────────────────────────────────
    # Индексация с валидацией
    # ──────────────────────────────────────────────

    def add_documents(self):
        """Проверяет папку docs/ при каждом запуске. Индексирует только новые файлы."""
        docs_path = Path(DOCS_PATH)
        if not docs_path.exists():
            logger.warning(f"Папка {DOCS_PATH}/ не найдена")
            return

        docx_files = list(docs_path.glob("*.docx"))
        if not docx_files:
            logger.warning(f"В папке {DOCS_PATH}/ нет .docx файлов")
            return

        existing_ids = set(self.collection.get(include=[])["ids"])
        new_files_found = False

        for file_path in docx_files:
            prefix = f"{file_path.stem}_chunk_"
            if any(id_.startswith(prefix) for id_ in existing_ids):
                logger.info(f"{file_path.stem} — уже в базе, пропускаем")
                continue

            # Валидация перед индексацией
            ok, reason = self._validate_docx(file_path)
            if not ok:
                logger.error(f"Пропускаем {file_path.name}: {reason}")
                continue

            new_files_found = True
            self._index_file(file_path)

        if new_files_found:
            self._build_bm25()
        else:
            logger.info("Новых документов нет — индексация не нужна")

    def _validate_docx(self, file_path: Path) -> Tuple[bool, str]:
        """Проверяет файл перед индексацией."""
        if not file_path.exists():
            return False, "файл не существует"
        if file_path.stat().st_size == 0:
            return False, "файл пустой (0 байт)"
        try:
            text = self._load_docx(str(file_path))
            if len(text.strip()) < 100:
                return False, f"слишком мало текста ({len(text.strip())} символов)"
            return True, ""
        except Exception as e:
            return False, f"не удалось открыть: {e}"

    def _index_file(self, file_path: Path):
        source_name = file_path.stem
        logger.info(f"Индексируем: {file_path.name}...")
        t0 = time.perf_counter()

        text = self._load_docx(str(file_path))
        chunks = self._create_chunks(text)
        logger.info(f"  {len(chunks)} чанков, пачками по {EMBED_BATCH_SIZE}...")

        for batch_start in range(0, len(chunks), EMBED_BATCH_SIZE):
            batch = chunks[batch_start:batch_start + EMBED_BATCH_SIZE]
            ids = [f"{source_name}_chunk_{batch_start + i}" for i in range(len(batch))]
            metas = [{"source": source_name} for _ in batch]
            try:
                self.collection.add(documents=batch, metadatas=metas, ids=ids)
                logger.info(f"  ...{min(batch_start + EMBED_BATCH_SIZE, len(chunks))}/{len(chunks)}")
            except Exception as e:
                logger.error(f"Ошибка при добавлении чанков {batch_start}-{batch_start + len(batch)}: {e}")
                return

        elapsed = time.perf_counter() - t0
        logger.info(f"✓ {source_name} добавлен за {elapsed:.1f}с")

    def _load_docx(self, file_path: str) -> str:
        doc = Document(file_path)
        return "\n".join([p.text for p in doc.paragraphs if p.text.strip()])

    def _create_chunks(self, text: str, chunk_size: int = 900, overlap: int = 100) -> List[str]:
        chunks, start = [], 0
        while start < len(text):
            chunk = text[start:start + chunk_size].strip()
            if chunk:
                chunks.append(chunk)
            start += chunk_size - overlap
        return chunks
