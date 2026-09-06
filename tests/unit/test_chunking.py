from rag.ingestion.chunking import TextChunker


def test_chunker_returns_chunks():
    chunker = TextChunker(
        chunk_size=50,
        chunk_overlap=10,
    )

    text = (
        "This is the first sentence. "
        "This is the second sentence. "
        "This is the third sentence."
    )

    chunks = chunker.split(text)

    assert len(chunks) > 0
    assert all(isinstance(chunk, str) for chunk in chunks)


def test_empty_text():
    chunker = TextChunker()

    assert chunker.split("") == []


def test_invalid_overlap():
    try:
        TextChunker(
            chunk_size=100,
            chunk_overlap=100,
        )
    except ValueError:
        return

    raise AssertionError("Expected ValueError")