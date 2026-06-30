import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from typing import List, Tuple
from pathlib import Path
from docx import Document

from config import CHROMA_PATH, DOCS_PATH, OLLAMA_EMBED_URL, EMBEDDING_MODEL, SEARCH_DISTANCE_THRESHOLD, EMBED_BATCH_SIZE


class EmbeddingStore:
    def __init__(self, collection_name: str = "legal_docs",
                 persist_directory: str = None):
        print("Инициализация ChromaDB...")
        self.persist_directory = persist_directory or CHROMA_PATH
        self.client = chromadb.PersistentClient(
            path=self.persist_directory,
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
        print(f"✓ Коллекция готова. Документов: {self.collection.count()}")

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

    def load_docx(self, file_path: str) -> str:
        doc = Document(file_path)
        return "\n".join([para.text for para in doc.paragraphs if para.text.strip()])

    def search(self, query: str, top_k: int = 5,
               distance_threshold: float = None) -> List[Tuple[str, str, float]]:
        """
        Ищет релевантные фрагменты в базе.
        distance_threshold: чем ниже — тем строже фильтр.
        ChromaDB возвращает L2-расстояние: 0.0 = идеально совпадение.
        """
        threshold = distance_threshold if distance_threshold is not None else SEARCH_DISTANCE_THRESHOLD

        if self.collection.count() == 0:
            return []

        results = self.collection.query(
            query_texts=[query],
            n_results=top_k
        )

        formatted = []
        if results["documents"]:
            docs = results["documents"][0]
            metas = results["metadatas"][0]
            distances = results["distances"][0]

            for doc, meta, dist in zip(docs, metas, distances):
                if dist <= threshold:
                    source = meta.get("source", "Документ")
                    formatted.append((doc, source, dist))

        return formatted

    def add_documents(self):
        """
        Загружает .docx файлы из папки docs/, которых ЕЩЁ НЕТ в базе.
        Вызывается при КАЖДОМ запуске бота — но реально отправляет в Ollama
        только новые файлы, для уже проиндексированных делает быструю
        локальную проверку по id (без обращения к модели).

        Чанки одного файла отправляются в Ollama небольшими пачками
        (EMBED_BATCH_SIZE), а не всем файлом разом — это сильно снижает
        нагрузку на оперативную память при индексации.
        """
        docs_path = Path(DOCS_PATH)
        if not docs_path.exists():
            print(f"⚠ Папка {DOCS_PATH}/ не найдена")
            return

        docx_files = list(docs_path.glob("*.docx"))
        if not docx_files:
            print(f"⚠ В папке {DOCS_PATH}/ нет .docx файлов")
            return

        existing_ids = set(self.collection.get()["ids"])

        new_files = []
        for file_path in docx_files:
            prefix = f"{file_path.stem}_chunk_"
            if any(id_.startswith(prefix) for id_ in existing_ids):
                print(f"✓ {file_path.stem} — уже в базе, пропускаем")
            else:
                new_files.append(file_path)

        if not new_files:
            print("✓ Новых файлов нет, индексация не нужна")
            return

        for file_path in new_files:
            source_name = file_path.stem
            print(f"📚 Загружаем: {file_path.name}...")
            text = self.load_docx(str(file_path))
            chunks = self._create_chunks(text)
            print(f"   Создано {len(chunks)} чанков, отправляю в Ollama пачками по {EMBED_BATCH_SIZE}...")

            total_added = 0
            for batch_start in range(0, len(chunks), EMBED_BATCH_SIZE):
                batch_chunks = chunks[batch_start:batch_start + EMBED_BATCH_SIZE]
                batch_ids = [
                    f"{source_name}_chunk_{batch_start + i}"
                    for i in range(len(batch_chunks))
                ]
                batch_metas = [
                    {"source": source_name, "chunk": batch_start + i}
                    for i in range(len(batch_chunks))
                ]
                try:
                    self.collection.add(
                        documents=batch_chunks,
                        metadatas=batch_metas,
                        ids=batch_ids
                    )
                    total_added += len(batch_chunks)
                    print(f"   ...{total_added}/{len(chunks)}")
                except Exception as e:
                    print(f"⚠ Ollama не ответил на пачке {batch_start}-{batch_start + len(batch_chunks)}: {e}")
                    print(
                        "   Проверь: 1) Ollama запущен, 2) команда `ollama list` "
                        f"показывает модель {EMBEDDING_MODEL}, 3) хватает ли оперативной памяти "
                        "(попробуй закрыть другие модели/программы и перезапустить Ollama)."
                    )
                    print(f"   Уже сохранено {total_added}/{len(chunks)} чанков из {source_name} — "
                          "при следующем запуске продолжится с этого места не получится, "
                          "файл переиндексируется заново при следующем запуске.")
                    return

            print(f"✓ {source_name} добавлен полностью ({total_added} чанков)")
