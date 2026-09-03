from rag.generation.llm import OllamaLLM
from rag.retrieval.retriever import Retriever
from rag.services.rag_service import RAGService
from rag.vectorstore.chroma import ChromaVectorStore


def main() -> None:
    vector_store = ChromaVectorStore(
        persist_directory="data/chroma",
        collection_name="documents",
    )

    retriever = Retriever(
        vector_store=vector_store,
    )

    llm = OllamaLLM(
        model="qwen3:4b",
    )

    rag_service = RAGService(
        retriever=retriever,
        llm=llm,
        default_top_k=3,
    )

    result = rag_service.query(
        "What is a natural parameter?"
    )

    print("\nAnswer:")
    print(result.answer)

    print("\nSources:")

    for source in result.sources:
        print(
            f"- {source.document} "
            f"(chunk {source.chunk_index}, "
            f"distance={source.distance})"
        )


if __name__ == "__main__":
    main()