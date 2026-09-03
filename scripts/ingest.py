from pathlib import Path

from rag.ingestion.chunking import TextChunker
from rag.ingestion.loaders import read_document
from rag.vectorstore.chroma import ChromaVectorStore


def main() -> None:
    file_path = Path("data/raw/cs229-notes1.pdf")

    text = read_document(file_path)

    chunker = TextChunker(
        chunk_size=500,
        chunk_overlap=50,
    )

    chunks = chunker.split(text)

    vector_store = ChromaVectorStore(
        persist_directory="data/chroma",
        collection_name="documents",
    )

    ids = [
        f"{file_path.name}-chunk-{index}"
        for index in range(len(chunks))
    ]

    metadatas = [
        {
            "source": file_path.name,
            "chunk": index,
        }
        for index in range(len(chunks))
    ]

    vector_store.upsert_documents(
        ids=ids,
        documents=chunks,
        metadatas=metadatas,
    )

    print(
        f"Indexed {len(chunks)} chunks "
        f"from {file_path.name}"
    )

    print(
        f"Collection now contains "
        f"{vector_store.count()} chunks"
    )


if __name__ == "__main__":
    main()