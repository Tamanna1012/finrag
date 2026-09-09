# FinRAG — Beginner-Friendly Financial Filing Q&A

A Retrieval-Augmented Generation (RAG) system that answers financial questions
from company annual reports, including comparison questions that need two
facts and a calculation (e.g. revenue growth between two years).

This project is intentionally scoped to be **understandable and buildable by
a beginner** — not a research system. See "What this project is (and isn't)"
below.

## What is RAG, in one paragraph

Large language models (LLMs) don't know the contents of a specific company's
annual report — it wasn't in their training data — and they can hallucinate
(confidently invent) numbers they don't actually know. RAG fixes this by
first **retrieving** the relevant passages from the report (using embedding
similarity search) and then giving those passages to the LLM so it answers
using real, grounded text instead of guessing.

## What this project is (and isn't)

**Is:**
- A working RAG pipeline: PDF → text → chunks → embeddings → vector search → LLM answer.
- Two-step retrieval for questions that need two facts (e.g. "revenue in 2022 vs 2024"), with the actual arithmetic done in Python, not left to the LLM.
- Evidence/citations shown for every answer.

**Isn't:**
- True dynamic multi-hop agent reasoning (the system doesn't decide retrieval steps on the fly with an agent loop — it detects "this question mentions two years" and runs retrieval twice). We call this **two-step retrieval**, not multi-hop, to keep the claim honest.
- Autonomous/multi-agent systems, fine-tuning, custom model training, or complex cloud infrastructure.

## Architecture

```
Annual Report (PDF)
   ↓ extract text (pdfplumber)
   ↓ split into overlapping chunks
   ↓ embed chunks (Sentence Transformers)
   ↓ store in a vector index (FAISS)
User asks a question
   ↓ embed the question, retrieve top-k similar chunks
   ↓ if the question mentions two years → retrieve twice (once per year)
   ↓ extract the numeric figures from the retrieved chunks
   ↓ calculate growth % / difference / ratio in Python
   ↓ send question + evidence + calculation to the LLM
   ↓ LLM writes the final natural-language answer
   ↓ show the answer + calculation + source evidence
```

## Tech stack

| Tool | Purpose |
|---|---|
| Python | Core language |
| pdfplumber | Extract text from annual report PDFs |
| Sentence Transformers (`all-MiniLM-L6-v2`) | Turn text into embeddings |
| FAISS | Store embeddings, do fast similarity search |
| Anthropic API (Claude) | Generate the final natural-language answer |
| Pandas | Handle the evaluation test set |
| Streamlit | Simple web UI |

## Project levels

- **Level 1 — Basic RAG**: `ingestion.py` + `embeddings.py` + `retrieval.py` on any document.
- **Level 2 — Financial RAG**: adds company/year metadata filtering and source citations (already built into `retrieval.py`).
- **Level 3 — Simple FinRAG**: adds two-step retrieval + numerical calculation (`calculations.py`, `rag.py`) and the UI (`app.py`).

## Folder structure

```
finrag/
├── data/
│   ├── raw_pdfs/       # put your annual report PDFs here, named Company_Year.pdf
│   └── processed/      # generated FAISS index + chunk store
├── notebooks/          # experiments
├── src/
│   ├── ingestion.py     # PDF -> chunks
│   ├── embeddings.py    # chunks -> FAISS index
│   ├── retrieval.py     # question -> relevant chunks
│   ├── calculations.py  # extract numbers, compute growth/diff/ratio
│   ├── rag.py            # ties retrieval + calculation + LLM together
│   └── app.py            # Streamlit UI
├── evaluation/
│   ├── test_questions.csv  # manually written test set (fill in with real Q&A)
│   └── evaluate.py         # runs the test set through FinRAG
├── requirements.txt
└── README.md
```

## Getting started

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Get some annual reports.** Download 1–2 PDFs from
   [SEC EDGAR](https://www.sec.gov/edgar) (US companies) or a company's
   investor relations page. Name them `Company_Year.pdf`
   (e.g. `TCS_2024.pdf`) and put them in `data/raw_pdfs/`.

3. **Set your LLM API key** (optional but needed for generated answers):
   ```bash
   export ANTHROPIC_API_KEY=your_key_here
   ```

4. **Build the vector index**
   ```bash
   python src/embeddings.py
   ```

5. **Try retrieval + Q&A from the command line**
   ```bash
   python src/rag.py
   ```

6. **Run the UI**
   ```bash
   streamlit run src/app.py
   ```

7. **Evaluate.** Fill in real questions/answers in
   `evaluation/test_questions.csv` (based on the PDFs you actually
   downloaded), then run:
   ```bash
   python evaluation/evaluate.py
   ```
   Manually compare `system_answer` to `expected_answer` in the generated
   `evaluation/results.csv` and score four things: answer correctness,
   numerical correctness, retrieval correctness, and citation correctness.

## Example

Question: *"What was the revenue growth from 2023 to 2024?"*

1. The system detects two years in the question (2023, 2024).
2. It retrieves the chunk mentioning 2023 revenue, and separately the chunk mentioning 2024 revenue.
3. It extracts both numeric values.
4. Python computes `((2024_value - 2023_value) / 2023_value) * 100`.
5. The LLM turns the result into a sentence: "Revenue increased by Z%."
6. Both source excerpts are shown as evidence underneath.

## Learning roadmap

If you're new to RAG, work through this order:

1. Python basics (functions, files, lists/dicts)
2. Embeddings — what they are, try `sentence-transformers` on a few sentences
3. Vector search basics — cosine/L2 similarity, FAISS
4. What RAG is, end to end
5. Build Level 1 on any text file first (not even financial)
6. Add a real financial PDF (Level 2)
7. Add numerical extraction + calculation (Level 3)
8. Build the Streamlit UI
9. Build the evaluation set and score the system

## Baseline vs FinRAG

- **Baseline (plain RAG)**: retrieve once, let the LLM answer directly — including doing any math itself, which LLMs are unreliable at.
- **FinRAG**: two-step retrieval for comparison questions + Python-calculated arithmetic + evidence shown.

Run both against the same `test_questions.csv` set and report the real
counts you observe — don't assume or fabricate an improvement number.
