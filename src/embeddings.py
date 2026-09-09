"""
Level 1: turn chunks into embeddings and store them in a FAISS index.

An embedding is a list of numbers that represents the *meaning* of a piece
of text. FAISS lets us store many embeddings and quickly find the ones
closest to a new query embedding.
"""

import json
import os
import pickle

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from ingestion import Chunk

MODEL_NAME = "all-MiniLM-L6-v2"  # small, fast, good enough for this project


def get_embedding_model() -> SentenceTransformer:
    return SentenceTransformer(MODEL_NAME)


def embed_chunks(chunks: list[Chunk], model: SentenceTransformer) -> np.ndarray:
    texts = [chunk.text for chunk in chunks]
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    return embeddings.astype("float32")


def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatL2:
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index


def save_index(index: faiss.IndexFlatL2, chunks: list[Chunk], out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    faiss.write_index(index, os.path.join(out_dir, "index.faiss"))
    with open(os.path.join(out_dir, "chunks.pkl"), "wb") as f:
        pickle.dump(chunks, f)


def load_index(out_dir: str) -> tuple[faiss.IndexFlatL2, list[Chunk]]:
    index = faiss.read_index(os.path.join(out_dir, "index.faiss"))
    with open(os.path.join(out_dir, "chunks.pkl"), "rb") as f:
        chunks = pickle.load(f)
    return index, chunks


def build_and_save(chunks: list[Chunk], out_dir: str = "data/processed") -> None:
    model = get_embedding_model()
    embeddings = embed_chunks(chunks, model)
    index = build_faiss_index(embeddings)
    save_index(index, chunks, out_dir)
    print(f"Saved FAISS index with {len(chunks)} chunks to '{out_dir}'")


if __name__ == "__main__":
    from ingestion import build_chunks_from_directory

    chunks = build_chunks_from_directory("data/raw_pdfs")
    build_and_save(chunks)
