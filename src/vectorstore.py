import chromadb
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
from typing import List


class VectorStore:
    """In-memory vector database for semantic search over document chunks."""

    def __init__(self) -> None:
        # EphemeralClient keeps everything in RAM only — no data is written to
        # disk, so each Streamlit session starts with a clean, isolated store.
        self._client = chromadb.EphemeralClient()
        self._ef = DefaultEmbeddingFunction()
        self._collection = self._client.get_or_create_collection(
            name="documents",
            embedding_function=self._ef,
        )

    def add_documents(self, chunks: List[str], source: str) -> None:
        """Embed and store text chunks, tagged with their source file name."""
        ids = [f"{source}_{i}" for i in range(len(chunks))]
        self._collection.add(
            documents=chunks,
            ids=ids,
            metadatas=[{"source": source}] * len(chunks),
        )

    def query(self, query_text: str, n_results: int = 3) -> List[str]:
        """Return the n most semantically similar chunks to the query."""
        results = self._collection.query(
            query_texts=[query_text],
            n_results=n_results,
        )
        return results["documents"][0] if results["documents"] else []
