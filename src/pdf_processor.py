from pypdf import PdfReader
from typing import List

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


def extract_and_chunk_pdf(file_path: str) -> List[str]:
    """Extract text from a PDF and split it into overlapping chunks."""
    reader = PdfReader(file_path)

    full_text = ""
    for page in reader.pages:
        text = page.extract_text()
        if text:
            full_text += text + "\n"

    return _chunk_text(full_text)


def _chunk_text(text: str) -> List[str]:
    chunks: List[str] = []
    start = 0

    while start < len(text):
        end = start + CHUNK_SIZE
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk)
        # Overlap keeps context across chunk boundaries so an answer
        # spanning two chunks isn't split at the wrong place.
        start += CHUNK_SIZE - CHUNK_OVERLAP

    return chunks
