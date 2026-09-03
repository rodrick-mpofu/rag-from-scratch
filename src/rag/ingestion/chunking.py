import re
from dataclasses import dataclass


def normalize_text(text: str) -> str:
    """Normalize whitespace while preserving textual content."""
    return re.sub(r"\s+", " ", text).strip()


@dataclass(slots=True)
class TextChunker:
    """Simple sentence-aware text chunker."""

    chunk_size: int = 500
    chunk_overlap: int = 50

    def __post_init__(self) -> None:
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be greater than 0")

        if self.chunk_overlap < 0:
            raise ValueError("chunk_overlap cannot be negative")

        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                "chunk_overlap must be smaller than chunk_size"
            )

    def split(self, text: str) -> list[str]:
        """Split text into approximately chunk_size character chunks."""
        text = normalize_text(text)

        if not text:
            return []

        sentences = re.split(r"(?<=[.!?])\s+", text)

        chunks: list[str] = []
        current_chunk: list[str] = []
        current_size = 0

        for sentence in sentences:
            sentence = sentence.strip()

            if not sentence:
                continue

            sentence_size = len(sentence)

            if (
                current_chunk
                and current_size + sentence_size > self.chunk_size
            ):
                completed_chunk = " ".join(current_chunk).strip()
                chunks.append(completed_chunk)

                overlap_text = self._create_overlap(completed_chunk)

                current_chunk = [overlap_text] if overlap_text else []
                current_size = len(overlap_text)

            current_chunk.append(sentence)
            current_size += sentence_size + 1

        if current_chunk:
            final_chunk = " ".join(current_chunk).strip()

            if final_chunk:
                chunks.append(final_chunk)

        return chunks

    def _create_overlap(self, text: str) -> str:
        """Take approximately chunk_overlap characters from previous chunk."""
        if self.chunk_overlap == 0:
            return ""

        if len(text) <= self.chunk_overlap:
            return text

        overlap = text[-self.chunk_overlap :]

        # Avoid starting in the middle of a word.
        first_space = overlap.find(" ")

        if first_space != -1:
            overlap = overlap[first_space + 1 :]

        return overlap.strip()


def split_text(
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> list[str]:
    """Convenience wrapper around TextChunker."""
    chunker = TextChunker(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    return chunker.split(text)