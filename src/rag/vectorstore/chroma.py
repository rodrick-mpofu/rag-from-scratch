from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeAlias

import chromadb
from chromadb.utils import embedding_functions

MetadataValue: TypeAlias = str | int | float | bool


@dataclass(slots=True, frozen=True)
class VectorSearchResult:
    """Single result returned from the vector database."""

    id: str
    document: str
    metadata: dict[str, Any]
    distance: float | None


class ChromaVectorStore:
    """ChromaDB-backed vector store."""

    def __init__(
        self,
        persist_directory: str | Path = "data/chroma",
        collection_name: str = "documents",
        embedding_model: str = "all-MiniLM-L6-v2",
    ) -> None:
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.embedding_function = (
            embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=embedding_model
            )
        )

        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory)
        )

        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_function,
        )

    def upsert_documents(
        self,
        ids: Sequence[str],
        documents: Sequence[str],
        metadatas: Sequence[Mapping[str, MetadataValue]],
        batch_size: int = 100,
    ) -> None:
        """Insert or update documents in ChromaDB."""
        if not (
            len(ids)
            == len(documents)
            == len(metadatas)
        ):
            raise ValueError(
                "ids, documents, and metadatas must have the same length"
            )

        if batch_size <= 0:
            raise ValueError("batch_size must be greater than 0")

        for start in range(0, len(documents), batch_size):
            end = start + batch_size

            self.collection.upsert(
                ids=list(ids[start:end]),
                documents=list(documents[start:end]),
                metadatas=[
                    dict(metadata)
                    for metadata in metadatas[start:end]
                ],
            )

    def search(
        self,
        query: str,
        k: int = 3,
    ) -> list[VectorSearchResult]:
        """Perform semantic similarity search."""
        query = query.strip()

        if not query:
            raise ValueError("query cannot be empty")

        if k <= 0:
            raise ValueError("k must be greater than 0")

        results = self.collection.query(
            query_texts=[query],
            n_results=k,
            include=[
                "documents",
                "metadatas",
                "distances",
            ],
        )

        ids = (results.get("ids") or [[]])[0]
        documents = (results.get("documents") or [[]])[0]
        metadatas = (results.get("metadatas") or [[]])[0]
        distances = (results.get("distances") or [[]])[0]

        search_results: list[VectorSearchResult] = []

        for index, document in enumerate(documents):
            if document is None:
                continue

            metadata = (
                metadatas[index]
                if index < len(metadatas)
                else None
            )

            distance = (
                distances[index]
                if index < len(distances)
                else None
            )

            result_id = (
                ids[index]
                if index < len(ids)
                else f"result-{index}"
            )

            search_results.append(
                VectorSearchResult(
                    id=result_id,
                    document=document,
                    metadata=dict(metadata or {}),
                    distance=distance,
                )
            )

        return search_results

    def count(self) -> int:
        """Return number of indexed chunks."""
        return self.collection.count()