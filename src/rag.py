"""
Level 2/3: ties retrieval + numerical calculation + the LLM together.

This is the main entry point: ask() takes a plain-English question and
returns an answer, the evidence chunks, and (when relevant) the numbers
that were calculated.
"""

import os
import re

from anthropic import Anthropic

from calculations import extract_primary_number, growth_percent, difference, ratio
from ingestion import Chunk
from retrieval import Retriever

SYSTEM_PROMPT = """You are a financial analyst assistant. You are given a user
question, one or more excerpts from a company's annual report, and (if
applicable) a number that was already calculated in Python. Write a short,
clear answer using ONLY the given information. Do not invent numbers. If a
calculated result is provided, state it plainly."""

# A question is treated as "two-value" if it mentions two distinct years.
_YEAR_PATTERN = re.compile(r"\b(19|20)\d{2}\b")


def _detect_years(question: str) -> list[str]:
    return sorted(set(_YEAR_PATTERN.findall(question)))


def _detect_operation(question: str) -> str:
    q = question.lower()
    if "growth" in q or "increase" in q or "% change" in q or "percentage change" in q:
        return "growth_percent"
    if "difference" in q or "how much more" in q or "how much less" in q:
        return "difference"
    if "ratio" in q or "times" in q:
        return "ratio"
    return "growth_percent"  # reasonable default for "compare two years" questions


class FinRAG:
    def __init__(self, index_dir: str = "data/processed", top_k: int = 3):
        self.retriever = Retriever(index_dir)
        self.top_k = top_k
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        self.llm = Anthropic(api_key=api_key) if api_key else None

    def _call_llm(self, prompt: str) -> str:
        if self.llm is None:
            return (
                "[LLM not configured — set ANTHROPIC_API_KEY to get a generated "
                "explanation. Raw evidence and calculation are shown above.]"
            )
        response = self.llm.messages.create(
            model="claude-sonnet-5",
            max_tokens=300,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text

    def ask(self, question: str, company: str | None = None) -> dict:
        years = _detect_years(question)

        if len(years) >= 2:
            return self._ask_two_value(question, years[0], years[1], company)
        return self._ask_simple(question, company)

    def _ask_simple(self, question: str, company: str | None) -> dict:
        chunks = self.retriever.search(question, top_k=self.top_k, company=company)
        evidence_text = "\n\n".join(f"[{c.source_file} p.{c.page_number}] {c.text}" for c in chunks)
        prompt = f"Question: {question}\n\nRetrieved excerpts:\n{evidence_text}\n\nAnswer:"
        answer = self._call_llm(prompt)
        return {"question": question, "answer": answer, "evidence": chunks, "calculation": None}

    def _ask_two_value(self, question: str, year_a: str, year_b: str,
                        company: str | None) -> dict:
        chunks_a = self.retriever.search(f"{question} {year_a}", top_k=self.top_k,
                                          company=company, year=year_a)
        chunks_b = self.retriever.search(f"{question} {year_b}", top_k=self.top_k,
                                          company=company, year=year_b)

        value_a = extract_primary_number(chunks_a[0]) if chunks_a else None
        value_b = extract_primary_number(chunks_b[0]) if chunks_b else None

        calculation = None
        if value_a is not None and value_b is not None:
            operation = _detect_operation(question)
            if operation == "growth_percent":
                result = growth_percent(value_a, value_b)
                calculation = f"Growth from {year_a} ({value_a}) to {year_b} ({value_b}) = {result:.2f}%"
            elif operation == "difference":
                result = difference(value_b, value_a)
                calculation = f"Difference {year_b} - {year_a} = {result}"
            elif operation == "ratio":
                result = ratio(value_b, value_a)
                calculation = f"Ratio {year_b} / {year_a} = {result:.2f}"

        all_chunks = chunks_a + chunks_b
        evidence_text = "\n\n".join(
            f"[{c.source_file} p.{c.page_number}] {c.text}" for c in all_chunks
        )
        calc_line = f"\n\nPython-calculated result: {calculation}" if calculation else ""
        prompt = f"Question: {question}\n\nRetrieved excerpts:\n{evidence_text}{calc_line}\n\nAnswer:"
        answer = self._call_llm(prompt)

        return {
            "question": question,
            "answer": answer,
            "evidence": all_chunks,
            "calculation": calculation,
        }


if __name__ == "__main__":
    rag = FinRAG()
    result = rag.ask("What was the revenue growth from 2023 to 2024?")
    print("Answer:", result["answer"])
    print("Calculation:", result["calculation"])
