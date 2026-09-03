from dataclasses import dataclass

from rag.generation.llm import OllamaLLM
from rag.generation.prompts import build_rag_messages
from rag.retrieval.retriever import Retriever


@dataclass(slots=True)
class Source:
    """Source metadata returned alongside an answer."""

    id: str
    document: str | None
    chunk_index: int | None
    distance: float | None


@dataclass(slots=True)
class RAGResult:
    """Result produced by the RAG pipeline."""

    query: str
    answer: str
    sources: list[Source]


class RAGService:
    """Orchestrates retrieval and generation."""

    def __init__(
        self,
        retriever: Retriever,
        llm: OllamaLLM,
        default_top_k: int = 3,
    ) -> None:
        if default_top_k <= 0:
            raise ValueError(
                "default_top_k must be greater than 0"
            )

        self.retriever = retriever
        self.llm = llm
        self.default_top_k = default_top_k

    def query(
        self,
        query: str,
        top_k: int | None = None,
    ) -> RAGResult:
        """Run the complete RAG pipeline."""
        query = query.strip()

        if not query:
            raise ValueError("query cannot be empty")

        k = (
            top_k
            if top_k is not None
            else self.default_top_k
        )

        if k <= 0:
            raise ValueError("top_k must be greater than 0")

        # 1. Retrieval
        chunks = self.retriever.retrieve(
            query=query,
            k=k,
        )

        if not chunks:
            return RAGResult(
                query=query,
                answer=(
                    "I cannot answer this based on the "
                    "provided information."
                ),
                sources=[],
            )

        # 2. Context construction
        context = self.retriever.build_context(chunks)

        # 3. Prompt construction
        messages = build_rag_messages(
            query=query,
            context=context,
        )

        # 4. Generation
        answer = self.llm.generate(messages)

        # 5. Source attribution
        sources = [
            Source(
                id=chunk.id,
                document=chunk.source,
                chunk_index=chunk.chunk_index,
                distance=chunk.distance,
            )
            for chunk in chunks
        ]

        return RAGResult(
            query=query,
            answer=answer,
            sources=sources,
        )