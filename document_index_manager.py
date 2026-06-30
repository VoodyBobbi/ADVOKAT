"""
Менеджер индексов документов.

Отвечает за:
1.  Загрузку и чанкинг документов.
2.  Создание двух типов индексов:
    - Векторный (ChromaDB) для семантического поиска.
    - Лексический (BM25) для поиска по ключевым словам.
3.  Выполнение гибридного поиска и переранжирования (rerank).
"""
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from typing import List, Tuple, Dict
from pathlib import Path
from docx import Document
import numpy as np

from config import (
    CHROMA_PATH, DOCS_PATH, OLLAMA_EMBED_URL, EMBEDDING_MODEL,
    EMBED_BATCH_SIZE, RERANK_MODEL, HYBRID_SEARCH_VECTOR_TOP_K,
    HYBRID_SEARCH_BM25_TOP_K, RERANK_TOP_K
)


class DocumentIndexManager:
    def __init__(self, collection_name: str = "legal_docs"):
        print("Инициализация менеджера индексов...")
        # --- Векторный индекс (ChromaDB) ---
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

        # --- Лексический индекс (BM25) и Reranker ---
        self.bm25_index = None
        self.reranker = None
        self.all_docs: Dict[str, Dict] = {}  # {id: {"text": ..., "meta": ...}}

        self._load_all_docs_and_build_bm25()
        self._load_reranker()
        print(f"✓ Менеджер индексов готов (векторных: {self.collection.count()}, лексических: {len(self.all_docs)})")

    def _load_reranker(self):
        """Загружает модель для переранжирования."""
        print(f"Загрузка rerank-модели: {RERANK_MODEL}...")
        try:
            self.reranker = CrossEncoder(RERANK_MODEL)
            print("✓ Rerank-модель готова")
        except Exception as e:
            print(f"⚠ Не удалось загрузить rerank-модель: {e}")
            print("  Поиск будет работать без переранжирования.")

    def _load_all_docs_and_build_bm25(self):
        """Загружает все документы из ChromaDB в память и строит BM25 индекс."""
        if self.collection.count() == 0:
            return

        print("Загрузка документов из ChromaDB для BM25 и rerank...")
        # get() без where возвращает ВСЕ документы
        docs_data = self.collection.get(include=["documents", "metadatas"])
        if not docs_data["ids"]:
            return

        for id_, doc, meta in zip(docs_data["ids"], docs_data["documents"], docs_data["metadatas"]):
            self.all_docs[id_] = {"text": doc, "meta": meta}

        # Создаём BM25 индекс
        tokenized_corpus = [doc["text"].split() for doc in self.all_docs.values()]
        self.bm25_index = BM25Okapi(tokenized_corpus)
        print(f"✓ BM25-индекс построен на {len(self.all_docs)} документах")

    def _vector_search(self, query: str) -> List[Tuple[str, float]]:
        """Поиск по векторам."""
        results = self.collection.query(
            query_texts=[query],
            n_results=HYBRID_SEARCH_VECTOR_TOP_K
        )
        # Возвращает [(id, score), ...]
        return list(zip(results["ids"][0], results["distances"][0])) if results["ids"] else []

    def _bm25_search(self, query: str) -> List[Tuple[str, float]]:
        """Поиск по ключевым словам."""
        if not self.bm25_index:
            return []
        tokenized_query = query.split()
        doc_scores = self.bm25_index.get_scores(tokenized_query)
        
        # Получаем top N результатов
        top_n_indices = np.argsort(doc_scores)[::-1][:HYBRID_SEARCH_BM25_TOP_K]
        
        # Сопоставляем индексы с id документов
        doc_ids = list(self.all_docs.keys())
        return [(doc_ids[i], doc_scores[i]) for i in top_n_indices if doc_scores[i] > 0]

    def _rerank(self, query: str, doc_ids: List[str]) -> List[Tuple[str, float]]:
        """Переранжирование с помощью Cross-Encoder."""
        if not self.reranker or not doc_ids:
            # Если reranker не загружен, просто возвращаем уникальные id
            return [(id_, 0.0) for id_ in doc_ids]

        pairs = [(query, self.all_docs[id_]["text"]) for id_ in doc_ids]
        scores = self.reranker.predict(pairs)
        
        scored_docs = list(zip(doc_ids, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        return scored_docs

    def search(self, query: str) -> List[Dict]:
        """Выполняет гибридный поиск + rerank."""
        if self.collection.count() == 0:
            return []

        # 1. Hybrid Search
        vector_results = self._vector_search(query)
        bm25_results = self._bm25_search(query)

        # 2. Объединение результатов
        # Используем dict для автоматического удаления дубликатов
        combined_ids = {id_ for id_, _ in vector_results}
        combined_ids.update(id_ for id_, _ in bm25_results)
        
        if not combined_ids:
            return []

        # 3. Rerank
        reranked_results = self._rerank(query, list(combined_ids))

        # 4. Формирование финального списка документов
        final_docs = []
        for id_, score in reranked_results[:RERANK_TOP_K]:
            doc_info = self.all_docs[id_]
            final_docs.append({
                "text": doc_info["text"],
                "source": doc_info["meta"].get("source", "Неизвестно"),
                "score": float(score)
            })
        return final_docs

    def add_documents(self):
        """Загружает .docx файлы, индексирует и обновляет BM25 индекс."""
        docs_path = Path(DOCS_PATH)
        if not docs_path.exists():
            print(f"⚠ Папка {DOCS_PATH}/ не найдена")
            return

        docx_files = list(docs_path.glob("*.docx"))
        if not docx_files:
            print(f"⚠ В папке {DOCS_PATH}/ нет .docx файлов")
            return

        existing_ids = set(self.collection.get(include=[])["ids"])
        new_files_found = False
        for file_path in docx_files:
            prefix = f"{file_path.stem}_chunk_"
            if any(id_.startswith(prefix) for id_ in existing_ids):
                continue

            new_files_found = True
            source_name = file_path.stem
            print(f"📚 Индексируем новый файл: {file_path.name}...")
            text = self._load_docx(str(file_path))
            chunks = self._create_chunks(text)
            print(f"   Создано {len(chunks)} чанков, отправляю в Ollama пачками по {EMBED_BATCH_SIZE}...")

            for batch_start in range(0, len(chunks), EMBED_BATCH_SIZE):
                batch_chunks = chunks[batch_start:batch_start + EMBED_BATCH_SIZE]
                batch_ids = [f"{source_name}_chunk_{batch_start + i}" for i in range(len(batch_chunks))]
                batch_metas = [{"source": source_name} for _ in range(len(batch_chunks))]
                
                try:
                    self.collection.add(documents=batch_chunks, metadatas=batch_metas, ids=batch_ids)
                except Exception as e:
                    print(f"⚠ Ошибка при добавлении чанков в ChromaDB: {e}")
                    return

        if new_files_found:
            print("Новые документы добавлены, перестраиваю BM25-индекс...")
            self._load_all_docs_and_build_bm25()
        else:
            print("✓ Новых документов нет, индексация не нужна.")

    def _load_docx(self, file_path: str) -> str:
        doc = Document(file_path)
        return "\n".join([para.text for para in doc.paragraphs if para.text.strip()])

    def _create_chunks(self, text: str, chunk_size: int = 900, overlap: int = 100) -> List[str]:
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            start = end - overlap
        return chunks
