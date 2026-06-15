import chromadb
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
from typing import List


class VectorStore:
    def __init__(self) -> None:
        self._client = chromadb.EphemeralClient()
        self._ef = DefaultEmbeddingFunction()
        self._collection = self._client.get_or_create_collection(
            name="documents",
            embedding_function=self._ef,
        )

    def add_documents(self, chunks: List[str], source: str) -> None:
        ids = [f"{source}_{i}" for i in range(len(chunks))]
        self._collection.add(
            documents=chunks,
            ids=ids,
            metadatas=[{"source": source}] * len(chunks),
        )

    def query(self, query_text: str, n_results: int = 3) -> List[str]:
        results = self._collection.query(
            query_texts=[query_text],
            n_results=n_results,
        )
        return results["documents"][0] if results["documents"] else []
