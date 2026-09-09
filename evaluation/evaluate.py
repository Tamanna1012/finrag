"""
Level 3: run the manually written test set through FinRAG and report simple
accuracy — no research-style benchmark, just correct/incorrect counts.

Usage: python evaluation/evaluate.py
"""

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from rag import FinRAG  # noqa: E402


def run_evaluation(csv_path: str = "evaluation/test_questions.csv") -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    rag = FinRAG()

    rows = []
    for _, row in df.iterrows():
        result = rag.ask(row["question"], company=row.get("company") or None)
        rows.append({
            "id": row["id"],
            "category": row["category"],
            "question": row["question"],
            "expected_answer": row["expected_answer"],
            "system_answer": result["answer"],
            "calculation": result["calculation"],
            "evidence_sources": "; ".join(
                f"{c.source_file} p.{c.page_number}" for c in result["evidence"]
            ),
        })

    results_df = pd.DataFrame(rows)
    return results_df


if __name__ == "__main__":
    results = run_evaluation()
    out_path = "evaluation/results.csv"
    results.to_csv(out_path, index=False)
    print(f"Saved {len(results)} results to {out_path}")
    print(
        "\nNow manually compare 'system_answer' against 'expected_answer' for each row "
        "and mark correctness — that's your accuracy score. See the README for the "
        "four things to check: answer, numerical, retrieval, and citation correctness."
    )
