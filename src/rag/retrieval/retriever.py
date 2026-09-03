from dataclasses import dataclass
from typing import Any

from rag.vectorstore.chroma import ChromaVectorStore


@dataclass(slots=True, frozen=True)
class RetrievedChunk:
    """A document chunk returned by the retrieval system."""

    id: str
    text: str
    source: str | None
    chunk_index: int | None
    distance: float | None
    metadata: dict[str, Any]


class Retriever:
    """Retrieve relevant document chunks for a query."""

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
        """Retrieve the top-k most relevant chunks."""
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

    @staticmethod
    def build_context(
        chunks: list[RetrievedChunk],
    ) -> str:
        """Format retrieved chunks into LLM context."""
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