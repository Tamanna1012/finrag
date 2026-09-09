"""
Level 3: a simple Streamlit UI on top of FinRAG.

Run with: streamlit run src/app.py
"""

import streamlit as st

from rag import FinRAG

st.set_page_config(page_title="FinRAG", page_icon="📊")
st.title("📊 FinRAG — Financial Filing Q&A")
st.caption(
    "Ask questions about the ingested annual reports. "
    "Comparison questions (e.g. mentioning two years) trigger two-step "
    "retrieval and a Python-calculated result."
)


@st.cache_resource
def load_rag():
    return FinRAG()


try:
    rag = load_rag()
except Exception as e:
    st.error(
        "Could not load the FAISS index. Run `python src/embeddings.py` first "
        f"to build it from your PDFs in data/raw_pdfs/.\n\nDetails: {e}"
    )
    st.stop()

question = st.text_input(
    "Your question",
    placeholder="e.g. What was the revenue growth from 2023 to 2024?",
)
company = st.text_input("Company filter (optional)", placeholder="e.g. TCS")

if st.button("Ask") and question:
    with st.spinner("Retrieving and reasoning..."):
        result = rag.ask(question, company=company or None)

    st.subheader("Answer")
    st.write(result["answer"])

    if result["calculation"]:
        st.subheader("Calculation")
        st.code(result["calculation"])

    st.subheader("Evidence")
    for chunk in result["evidence"]:
        with st.expander(f"{chunk.source_file} — page {chunk.page_number}"):
            st.write(chunk.text)
