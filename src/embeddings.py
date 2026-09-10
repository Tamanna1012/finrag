"""
Level 1: turn chunks into numeric vectors so we can search them by meaning
instead of exact text match.

We use TF-IDF (Term Frequency - Inverse Document Frequency) instead of a
neural embedding model. TF-IDF turns each chunk into a vector where common
words (like "the", "company") get a low weight and distinctive words (like
"revenue", "2024") get a high weight, so chunks sharing distinctive words
score as similar. It's not as smart as a neural embedding model (it matches
words, not deeper meaning), but it needs no download, no GPU, and no
internet connection — good for learning the retrieval part of RAG first.
Swapping in a neural embedding model later (e.g. sentence-transformers) is
a drop-in upgrade to this one file.
"""

import os
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer

from ingestion import Chunk


def build_vectorizer_and_matrix(chunks: list[Chunk]):
    """Fit a TF-IDF vectorizer on all chunks and return (vectorizer, matrix)."""
    texts = [chunk.text for chunk in chunks]
    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform(texts)
    return vectorizer, matrix


def save_index(vectorizer, matrix, chunks: list[Chunk], out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "index.pkl"), "wb") as f:
        pickle.dump({"vectorizer": vectorizer, "matrix": matrix, "chunks": chunks}, f)


def load_index(out_dir: str):
    with open(os.path.join(out_dir, "index.pkl"), "rb") as f:
        data = pickle.load(f)
    return data["vectorizer"], data["matrix"], data["chunks"]


def build_and_save(chunks: list[Chunk], out_dir: str = "data/processed") -> None:
    vectorizer, matrix = build_vectorizer_and_matrix(chunks)
    save_index(vectorizer, matrix, chunks, out_dir)
    print(f"Saved TF-IDF index with {len(chunks)} chunks to '{out_dir}'")


if __name__ == "__main__":
    from ingestion import build_chunks_from_directory

    chunks = build_chunks_from_directory("data/raw_pdfs")
    build_and_save(chunks)
