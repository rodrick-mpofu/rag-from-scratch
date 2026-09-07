import re

from rank_bm25 import BM25Okapi

from rag.retrieval.retriever import RetrievedChunk
from rag.vectorstore.chroma import ChromaVectorStore


def tokenize(text: str) -> list[str]:
    return re.findall(r"\b\w+\b", text.lower())


class BM25Retriever:
    def __init__(
        self,
        vector_store: ChromaVectorStore,
    ) -> None:
        records = vector_store.get_all()

        if not records:
            raise ValueError(
                "Cannot initialize BM25 with an empty corpus"
            )

        self.records = records

        tokenized_corpus = [
            tokenize(record.document)
            for record in records
        ]

        self.index = BM25Okapi(tokenized_corpus)

    def retrieve(
        self,
        query: str,
        k: int = 3,
    ) -> list[RetrievedChunk]:
        if k <= 0:
            raise ValueError("k must be greater than 0")

        query_tokens = tokenize(query)

        scores = self.index.get_scores(query_tokens)

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True,
        )[:k]

        chunks = []

        for index in ranked_indices:
            record = self.records[index]
            metadata = record.metadata

            source = metadata.get("source")
            chunk_index = metadata.get("chunk")

            bm25_score = float(scores[index])

            chunks.append(
                RetrievedChunk(
                    id=record.id,
                    text=record.document,
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
                    distance=None,
                    metadata={
                        **metadata,
                        "bm25_score": bm25_score,
                    },
                )
            )

        return chunks