from rag.generation.llm import OllamaLLM
from rag.retrieval.bm25 import BM25Retriever
from rag.retrieval.retriever import DenseRetriever
from rag.services.rag_service import RAGService
from rag.vectorstore.chroma import ChromaVectorStore


def main() -> None:
    vector_store = ChromaVectorStore(
        persist_directory="data/chroma",
        collection_name="documents",
    )

    dense_retriever = DenseRetriever(
        vector_store=vector_store,
    )

    bm25_retriever = BM25Retriever(
        vector_store=vector_store,
    )

    llm = OllamaLLM(
        model="qwen3:4b",
    )

    dense_rag_service = RAGService(
        retriever=dense_retriever,
        llm=llm,
        default_top_k=3,
    )

    bm25_rag_service = RAGService(
        retriever=bm25_retriever,
        llm=llm,
        default_top_k=3,
    )

    dense_result = dense_rag_service.query(
        "What is a natural parameter?"
    )

    bm25_result = bm25_rag_service.query(
        "What is a natural parameter?"
    )

    print("\n Dense Answer:")
    print(dense_result.answer)

    print("\nSources:")

    for source in dense_result.sources:
        print(
            f"- {source.document} "
            f"(chunk {source.chunk_index}, "
            f"distance={source.distance})"
        )

    print("\n BM25 Answer:")
    print(bm25_result.answer)

    print("\nSources:")

    for source in bm25_result.sources:
        print(
            f"- {source.document} "
            f"(chunk {source.chunk_index}, "
            f"distance={source.distance})"
        )


if __name__ == "__main__":
    main()