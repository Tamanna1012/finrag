"""
Level 1/2: given a question, find the most relevant chunks.

This is the "R" in RAG. We turn the query into the same kind of vector used
for the chunks (TF-IDF), then rank every chunk by cosine similarity — a
measure of how much two vectors point in the same direction, which for
TF-IDF vectors means "how much distinctive vocabulary do these two texts
share."
"""

from sklearn.metrics.pairwise import cosine_similarity

from ingestion import Chunk
from embeddings import load_index


class Retriever:
    def __init__(self, index_dir: str = "data/processed"):
        self.vectorizer, self.matrix, self.chunks = load_index(index_dir)

    def search(self, query: str, top_k: int = 5,
               company: str | None = None, year: str | None = None) -> list[Chunk]:
        """
        Return the top_k chunks most similar to the query.
        Optional company/year filters narrow the search to Level 2's
        "financial metadata aware" retrieval.
        """
        query_vector = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vector, self.matrix)[0]
        ranked_indices = scores.argsort()[::-1]  # highest similarity first

        results = []
        for idx in ranked_indices:
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
