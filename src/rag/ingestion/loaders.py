from pathlib import Path

import docx
from pypdf import PdfReader

SUPPORTED_EXTENSIONS = {".txt", ".pdf", ".docx"}


def _validate_file(file_path: str | Path) -> Path:
    """Validate that the requested file exists."""
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File does not exist: {path}")

    if not path.is_file():
        raise ValueError(f"Path is not a file: {path}")

    return path


def read_text_file(file_path: str | Path) -> str:
    """Read plain text from a .txt file."""
    path = _validate_file(file_path)

    return path.read_text(encoding="utf-8")


def read_pdf_file(file_path: str | Path) -> str:
    """Extract text from a PDF file."""
    path = _validate_file(file_path)

    reader = PdfReader(path)

    pages: list[str] = []

    for page in reader.pages:
        text = page.extract_text()

        if text:
            pages.append(text)

    return "\n".join(pages)


def read_docx_file(file_path: str | Path) -> str:
    """Extract text from a DOCX file."""
    path = _validate_file(file_path)

    document = docx.Document(path)

    return "\n".join(
        paragraph.text
        for paragraph in document.paragraphs
        if paragraph.text.strip()
    )


def read_document(file_path: str | Path) -> str:
    """Read a supported document based on its file extension."""
    path = _validate_file(file_path)

    extension = path.suffix.lower()

    readers = {
        ".txt": read_text_file,
        ".pdf": read_pdf_file,
        ".docx": read_docx_file,
    }

    reader = readers.get(extension)

    if reader is None:
        raise ValueError(
            f"Unsupported file format: {extension}. "
            f"Supported formats: {sorted(SUPPORTED_EXTENSIONS)}"
        )

    return reader(path)