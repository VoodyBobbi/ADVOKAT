import chromadb
from chromadb.config import Settings
from typing import List, Tuple
import os
from pathlib import Path
from docx import Document


class EmbeddingStore:
    def __init__(self, collection_name: str = "uk_rf", persist_directory: str = "./chroma_db"):
        print("Инициализация ChromaDB...")
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "УК РФ"}
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

    def add_documents(self):
        docs_path = Path("docs")
        uk_file = docs_path / "УК РФ.docx"

        if uk_file.exists() and self.collection.count() == 0:
            print("Загружаем УК РФ...")
            text = self.load_docx(str(uk_file))
            chunks = self._create_chunks(text)
            print(f"Создано {len(chunks)} чанков")

            ids = [f"uk_chunk_{i}" for i in range(len(chunks))]
            metadatas = [{"source": "УК РФ", "chunk": i} for i in range(len(chunks))]

            self.collection.add(
                documents=chunks,
                metadatas=metadatas,
                ids=ids
            )
            print(f"✓ УК РФ добавлен ({len(chunks)} чанков)")
        else:
            print(f"✓ В базе уже {self.collection.count()} чанков")

    def search(self, query: str, top_k: int = 5) -> List[Tuple[str, str, float]]:
        if self.collection.count() == 0:
            return []
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k
        )
        formatted = []
        if results['documents']:
            for i in range(len(results['documents'][0])):
                chunk = results['documents'][0][i]
                source = results['metadatas'][0][i]['source']
                formatted.append((chunk, source, 0.0))
        return formatted