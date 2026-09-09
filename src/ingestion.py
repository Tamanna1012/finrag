"""
Level 1: turn a PDF annual report into a list of overlapping text chunks.

Pipeline: PDF -> raw text (per page) -> cleaned text -> fixed-size chunks.
"""

import os
import re
from dataclasses import dataclass, field

import pdfplumber


@dataclass
class Chunk:
    text: str
    source_file: str
    page_number: int
    chunk_id: int
    metadata: dict = field(default_factory=dict)


def extract_text_by_page(pdf_path: str) -> list[str]:
    """Return a list of raw text strings, one per PDF page."""
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            pages.append(text)
    return pages


def clean_text(text: str) -> str:
    """Collapse whitespace/line breaks that PDF extraction leaves messy."""
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def chunk_text(text: str, chunk_size: int = 400, overlap: int = 50) -> list[str]:
    """
    Split text into ~chunk_size-word pieces with a small overlap so a fact
    that falls near a chunk boundary still appears in full in one chunk.
    """
    words = text.split()
    if not words:
        return []

    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunks.append(" ".join(words[start:end]))
        if end >= len(words):
            break
        start = end - overlap
    return chunks


def build_chunks_from_pdf(
    pdf_path: str,
    company: str = "",
    year: str = "",
    chunk_size: int = 400,
    overlap: int = 50,
) -> list[Chunk]:
    """Full ingestion for one PDF: extract -> clean -> chunk -> attach metadata."""
    source_file = os.path.basename(pdf_path)
    pages = extract_text_by_page(pdf_path)

    all_chunks: list[Chunk] = []
    chunk_id = 0
    for page_number, raw_page_text in enumerate(pages, start=1):
        cleaned = clean_text(raw_page_text)
        if not cleaned:
            continue
        for piece in chunk_text(cleaned, chunk_size=chunk_size, overlap=overlap):
            all_chunks.append(
                Chunk(
                    text=piece,
                    source_file=source_file,
                    page_number=page_number,
                    chunk_id=chunk_id,
                    metadata={"company": company, "year": year},
                )
            )
            chunk_id += 1
    return all_chunks


def build_chunks_from_directory(
    directory: str, chunk_size: int = 400, overlap: int = 50
) -> list[Chunk]:
    """
    Ingest every PDF in a directory.
    Filename convention expected: <Company>_<Year>.pdf (e.g. TCS_2024.pdf).
    Falls back to blank metadata if the filename doesn't match.
    """
    chunks: list[Chunk] = []
    for filename in sorted(os.listdir(directory)):
        if not filename.lower().endswith(".pdf"):
            continue
        name_without_ext = os.path.splitext(filename)[0]
        parts = name_without_ext.split("_")
        company, year = (parts[0], parts[1]) if len(parts) >= 2 else (name_without_ext, "")
        pdf_path = os.path.join(directory, filename)
        chunks.extend(
            build_chunks_from_pdf(pdf_path, company=company, year=year,
                                   chunk_size=chunk_size, overlap=overlap)
        )
    return chunks


if __name__ == "__main__":
    import sys

    directory = sys.argv[1] if len(sys.argv) > 1 else "data/raw_pdfs"
    result = build_chunks_from_directory(directory)
    print(f"Built {len(result)} chunks from PDFs in '{directory}'")
    if result:
        print("Example chunk:\n", result[0])
