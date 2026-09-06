from dataclasses import dataclass
from typing import Any, Protocol

from rag.vectorstore.chroma import ChromaVectorStore


@dataclass(slots=True, frozen=True)
class RetrievedChunk:
    """A document chunk returned by a retrieval system."""

    id: str
    text: str
    source: str | None
    chunk_index: int | None
    distance: float | None
    metadata: dict[str, Any]


class RetrieverProtocol(Protocol):
    """Common interface that all retrievers must implement."""

    def retrieve(
        self,
        query: str,
        k: int = 3,
    ) -> list[RetrievedChunk]:
        """Retrieve the top-k chunks for a query."""
        ...


class DenseRetriever:
    """Retrieve document chunks using dense vector similarity search."""

    def __init__(
        self,
        vector_store: ChromaVectorStore,
    ) -> None:
        self.vector_store = vector_store

    def retrieve(
        self,
        query: str,
        k: int = 3,
    ) -> list[RetrievedChunk]:
        """Retrieve the top-k most semantically similar chunks."""

        query = query.strip()

        if not query:
            raise ValueError("query cannot be empty")

        if k <= 0:
            raise ValueError("k must be greater than 0")

        results = self.vector_store.search(
            query=query,
            k=k,
        )

        chunks: list[RetrievedChunk] = []

        for result in results:
            source = result.metadata.get("source")
            chunk_index = result.metadata.get("chunk")

            chunks.append(
                RetrievedChunk(
                    id=result.id,
                    text=result.document,
                    source=(
                        str(source)
                        if source is not None
                        else None
                    ),
                    chunk_index=(
                        int(chunk_index)
                        if isinstance(chunk_index, int)
                        else None
                    ),
                    distance=result.distance,
                    metadata=result.metadata,
                )
            )

        return chunks


def build_context(
    chunks: list[RetrievedChunk],
) -> str:
    """Format retrieved chunks into context for the LLM."""

    context_parts: list[str] = []

    for index, chunk in enumerate(chunks, start=1):
        source = chunk.source or "unknown"

        chunk_label = (
            str(chunk.chunk_index)
            if chunk.chunk_index is not None
            else "unknown"
        )

        context_parts.append(
            f"[Document {index}]\n"
            f"Source: {source}\n"
            f"Chunk: {chunk_label}\n"
            f"{chunk.text}"
        )

    return "\n\n".join(context_parts)