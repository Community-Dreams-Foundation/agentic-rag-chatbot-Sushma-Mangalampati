"""Document ingestion: parse and chunk documents for RAG."""

import re
from pathlib import Path
from typing import Iterator

from pypdf import PdfReader


def parse_file(file_path: Path) -> str:
    """Parse a file and return its text content."""
    suffix = file_path.suffix.lower()
    if suffix == ".txt" or suffix == ".md":
        return file_path.read_text(encoding="utf-8", errors="replace")
    if suffix == ".pdf":
        reader = PdfReader(str(file_path))
        return "\n\n".join(
            page.extract_text() or "" for page in reader.pages
        )
    raise ValueError(f"Unsupported file type: {suffix}")


def chunk_text(
    text: str,
    chunk_size: int = 500,
    overlap: int = 50,
) -> Iterator[tuple[str, int, str | None]]:
    """Split text into overlapping chunks. Yields (chunk, chunk_index, section)."""
    # Try to split at paragraph/section boundaries first
    sections = re.split(r"\n\s*\n", text.strip())
    current_chunk = []
    current_len = 0
    chunk_idx = 0
    section_name = None

    for i, section in enumerate(sections):
        section = section.strip()
        if not section:
            continue
        # Extract potential section header (first line if it looks like a heading)
        lines = section.split("\n")
        if lines and (lines[0].startswith("#") or lines[0].endswith(":")):
            section_name = lines[0][:80].strip()

        words = section.split()
        for word in words:
            current_chunk.append(word)
            current_len += len(word) + 1
            if current_len >= chunk_size:
                chunk_text = " ".join(current_chunk)
                yield chunk_text, chunk_idx, section_name
                chunk_idx += 1
                # Overlap: keep last overlap words
                overlap_words = current_chunk[-overlap:] if len(current_chunk) > overlap else current_chunk
                current_chunk = overlap_words
                current_len = sum(len(w) + 1 for w in overlap_words)

    if current_chunk:
        yield " ".join(current_chunk), chunk_idx, section_name


def ingest_document(
    file_path: Path,
    chunk_size: int = 500,
    overlap: int = 50,
) -> list[dict]:
    """Ingest a document: parse and chunk. Returns list of chunk dicts."""
    text = parse_file(file_path)
    source_name = file_path.name
    chunks = []

    for text_chunk, chunk_idx, section in chunk_text(text, chunk_size, overlap):
        locator = f"chunk {chunk_idx}"
        if section:
            locator = f"{section} ({locator})"

        chunks.append({
            "text": text_chunk,
            "metadata": {
                "source": source_name,
                "chunk_id": chunk_idx,
                "locator": locator,
            },
        })

    return chunks


def ingest_directory(dir_path: Path, chunk_size: int = 500, overlap: int = 50) -> list[dict]:
    """Ingest all supported files from a directory."""
    all_chunks = []
    supported = {".txt", ".md", ".pdf"}

    for path in sorted(dir_path.iterdir()):
        if path.suffix.lower() in supported:
            chunks = ingest_document(path, chunk_size, overlap)
            all_chunks.extend(chunks)

    return all_chunks
