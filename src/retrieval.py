"""
Level 1/2: given a question, find the most relevant chunks.

This is the "R" in RAG. We embed the query with the same model used for the
chunks, then ask FAISS for the nearest neighbours.
"""

from ingestion import Chunk
from embeddings import get_embedding_model, load_index


class Retriever:
    def __init__(self, index_dir: str = "data/processed"):
        self.index, self.chunks = load_index(index_dir)
        self.model = get_embedding_model()

    def search(self, query: str, top_k: int = 5,
               company: str | None = None, year: str | None = None) -> list[Chunk]:
        """
        Return the top_k chunks most similar to the query.
        Optional company/year filters narrow the search to Level 2's
        "financial metadata aware" retrieval.
        """
        query_embedding = self.model.encode([query], convert_to_numpy=True).astype("float32")

        # Over-fetch, then filter by metadata, since FAISS itself has no filter step.
        fetch_k = top_k * 5 if (company or year) else top_k
        distances, indices = self.index.search(query_embedding, fetch_k)

        results = []
        for idx in indices[0]:
            if idx == -1:
                continue
            chunk = self.chunks[idx]
            if company and chunk.metadata.get("company") != company:
                continue
            if year and chunk.metadata.get("year") != year:
                continue
            results.append(chunk)
            if len(results) >= top_k:
                break
        return results


if __name__ == "__main__":
    retriever = Retriever()
    hits = retriever.search("What was the total revenue?", top_k=3)
    for h in hits:
        print(f"[{h.source_file} p.{h.page_number}] {h.text[:150]}...")
