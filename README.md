# FinRAG — Beginner-Friendly Financial Filing Q&A

A Retrieval-Augmented Generation (RAG) system that answers financial questions
from company annual reports, including comparison questions that need two
facts and a calculation (e.g. revenue growth between two years).

This project is intentionally scoped to be **understandable and buildable by
a beginner** — not a research system. See "What this project is (and isn't)"
below. It ships with two small sample annual reports so you can run the
whole thing in under 2 minutes before touching any real data.

## What is RAG, in one paragraph

Large language models (LLMs) don't know the contents of a specific company's
annual report — it wasn't in their training data — and they can hallucinate
(confidently invent) numbers they don't actually know. RAG fixes this by
first **retrieving** the relevant passages from the report (by comparing
numeric text vectors) and then giving those passages to the LLM so it
answers using real, grounded text instead of guessing.

## What this project is (and isn't)

**Is:**
- A working RAG pipeline: PDF → text → chunks → vectors → similarity search → LLM answer.
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
   ↓ turn chunks into TF-IDF vectors (scikit-learn)
   ↓ store the vectors + chunks (one local file, no server/database needed)
User asks a question
   ↓ turn the question into a vector the same way, rank chunks by similarity
   ↓ if the question mentions two years → retrieve twice (once per year)
   ↓ extract the numeric figures from the retrieved chunks
   ↓ calculate growth % / difference / ratio in Python
   ↓ send question + evidence + calculation to the LLM
   ↓ LLM writes the final natural-language answer
   ↓ show the answer + calculation + source evidence
```

**Why TF-IDF instead of a neural embedding model?** TF-IDF (Term Frequency -
Inverse Document Frequency) turns each chunk into a vector where distinctive
words (like "revenue", "2024") get a high weight and common words get a low
weight, so chunks sharing distinctive vocabulary score as similar. It's a
simpler idea than a neural embedding model, needs **no download, no GPU, and
no internet connection** to run, and is exactly what many people learn
before moving to neural embeddings. Swapping in a real embedding model
(e.g. `sentence-transformers`) later is a self-contained change to
`src/embeddings.py` and `src/retrieval.py` only — a natural "next step" once
you're comfortable with everything else.

## Tech stack

| Tool | Purpose |
|---|---|
| Python | Core language |
| pdfplumber | Extract text from annual report PDFs |
| scikit-learn (TF-IDF + cosine similarity) | Turn text into vectors and rank chunks by similarity |
| Anthropic API (Claude) | Generate the final natural-language answer (optional — works without it too) |
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
│   ├── raw_pdfs/       # DemoCorp_2023.pdf / DemoCorp_2024.pdf sample reports are here already
│   └── processed/      # generated index (created by embeddings.py, not committed to git)
├── notebooks/          # experiments
├── src/
│   ├── ingestion.py     # PDF -> chunks
│   ├── embeddings.py    # chunks -> TF-IDF index
│   ├── retrieval.py     # question -> relevant chunks
│   ├── calculations.py  # extract numbers, compute growth/diff/ratio
│   ├── rag.py            # ties retrieval + calculation + LLM together
│   └── app.py            # Streamlit UI
├── evaluation/
│   ├── test_questions.csv  # sample test set (about DemoCorp) — add more once you use real data
│   └── evaluate.py         # runs the test set through FinRAG
├── requirements.txt
└── README.md
```

## Quickstart (using the bundled sample data)

No API key, no PDFs to find, no internet connection needed for this part —
you'll have a working RAG pipeline in about a minute.

```bash
git clone https://github.com/Tamanna1012/finrag
cd finrag
python3 -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python src/embeddings.py     # builds the index from the sample DemoCorp PDFs
python src/rag.py            # asks "What was the revenue growth from 2023 to 2024?" and prints the answer
```

You should see something like:
```
Answer: [LLM not configured — set ANTHROPIC_API_KEY to get a generated explanation. Raw evidence and calculation are shown above.]
Calculation: Growth from 2023 (4500000000.0) to 2024 (5400000000.0) = 20.00%
```
That confirms the whole pipeline (ingestion → retrieval → two-step
calculation) is working correctly — the numbers say revenue grew from
₹450 crore to ₹540 crore, a 20% increase, which is exactly right for the
sample data. The bracketed message just means no LLM is turning that into a
sentence yet (see next step).

**Optional — get a written-out answer instead of just the raw calculation:**
```bash
export ANTHROPIC_API_KEY=your_key_here   # get one at console.anthropic.com
python src/rag.py
```

**Try the UI:**
```bash
streamlit run src/app.py
```

**Run the evaluation set:**
```bash
python evaluation/evaluate.py
```
Then open `evaluation/results.csv` and compare `system_answer` to
`expected_answer` yourself — that manual comparison *is* the evaluation
(see "Evaluation" below).

## Using your own real annual reports

1. Download 1–2 PDFs from [SEC EDGAR](https://www.sec.gov/edgar) (US
   companies) or a company's investor relations page.
2. Name them `Company_Year.pdf` (e.g. `TCS_2024.pdf`) and put them in
   `data/raw_pdfs/` (alongside or instead of the DemoCorp samples).
3. Re-run `python src/embeddings.py` to rebuild the index.
4. Write real questions and answers into `evaluation/test_questions.csv`
   (read the PDF yourself to know the correct answer first).

## Example

Question: *"What was the revenue growth from 2023 to 2024?"*

1. The system detects two years in the question (2023, 2024).
2. It retrieves the chunk mentioning 2023 revenue, and separately the chunk mentioning 2024 revenue.
3. It extracts both numeric values (specifically looking for the number near the word "revenue", so it doesn't confuse revenue with profit).
4. Python computes `((2024_value - 2023_value) / 2023_value) * 100`.
5. The LLM turns the result into a sentence: "Revenue increased by Z%."
6. Both source excerpts are shown as evidence underneath.

## Learning roadmap

If you're new to RAG, work through this order:

1. Python basics (functions, files, lists/dicts)
2. What a vector representation of text is — read about TF-IDF, then look at `src/embeddings.py`
3. Vector search basics — cosine similarity (used in `src/retrieval.py`)
4. What RAG is, end to end
5. Run Level 1 on the sample data (already done if you followed Quickstart above)
6. Add a real financial PDF (Level 2)
7. Read `src/calculations.py` and `src/rag.py` to understand numerical extraction (Level 3)
8. Build the Streamlit UI
9. Build the evaluation set and score the system
10. (Later, optional) swap TF-IDF for a neural embedding model like `sentence-transformers`

## Evaluation

`evaluation/test_questions.csv` has a small manually-written test set split
into simple factual and two-value numerical questions. Run
`python evaluation/evaluate.py`, then manually check each row in the
generated `evaluation/results.csv` for four things:
- **Answer correctness** — does the answer match what the report actually says?
- **Numerical correctness** — is the calculated number right?
- **Retrieval correctness** — did it fetch the chunk that actually contains the answer?
- **Citation correctness** — is the shown evidence the right source?

## Baseline vs FinRAG

- **Baseline (plain RAG)**: retrieve once, let the LLM answer directly — including doing any math itself, which LLMs are unreliable at.
- **FinRAG**: two-step retrieval for comparison questions + Python-calculated arithmetic + evidence shown.

Run both against the same `test_questions.csv` set and report the real
counts you observe — don't assume or fabricate an improvement number.
