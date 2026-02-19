"""Streamlit UI for the Agentic RAG Chatbot."""

import os
import sys
from pathlib import Path

# Ensure project root is on path so "src" package can be found
_project_root = Path(__file__).resolve().parent.parent

# Load .env from project root before any code reads env vars
try:
    from dotenv import load_dotenv
    load_dotenv(_project_root / ".env")
except ImportError:
    pass
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import streamlit as st

from src.ingestion import ingest_document, ingest_directory
from src.indexing import clear_collection, index_chunks
from src.memory import load_memory_for_context, process_memory
from src.rag_chain import answer_with_citations

# Page config
st.set_page_config(page_title="Agentic RAG Chatbot", page_icon="🤖", layout="centered")

st.title("🤖 Agentic RAG Chatbot")
st.caption("Upload documents, ask questions with citations, and persistent memory")

# Sidebar: file upload and indexing
with st.sidebar:
    st.header("📁 Documents")
    st.markdown("Upload files or use sample docs to build the RAG index.")

    sample_docs = Path("sample_docs")
    if sample_docs.exists():
        if st.button("📂 Index sample_docs/", use_container_width=True):
            chunks = ingest_directory(sample_docs)
            if chunks:
                clear_collection()
                n = index_chunks(chunks)
                st.success(f"Indexed {n} chunks from sample_docs/")
            else:
                st.warning("No supported files in sample_docs/")

    uploaded_files = st.file_uploader(
        "Or upload files",
        type=["txt", "md", "pdf"],
        accept_multiple_files=True,
    )
    if uploaded_files:
        if st.button("Index uploaded files", use_container_width=True):
            all_chunks = []
            for f in uploaded_files:
                path = Path(f.name)
                path.write_bytes(f.getvalue())
                try:
                    chunks = ingest_document(path)
                    all_chunks.extend(chunks)
                finally:
                    path.unlink(missing_ok=True)
            if all_chunks:
                clear_collection()
                n = index_chunks(all_chunks)
                st.success(f"Indexed {n} chunks")
            else:
                st.warning("Could not parse uploaded files.")

    st.divider()
    st.markdown("**Memory files**")
    st.markdown("- USER_MEMORY.md")
    st.markdown("- COMPANY_MEMORY.md")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("citations"):
            with st.expander("📎 Citations"):
                for c in msg["citations"]:
                    st.markdown(f"- **{c['source']}** ({c['locator']})")
                    st.caption(c["snippet"][:150] + "..." if len(c["snippet"]) > 150 else c["snippet"])

# Chat input
if prompt := st.chat_input("Ask a question about your documents..."):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer, citations = answer_with_citations(prompt)
            st.markdown(answer)
            if citations:
                with st.expander("📎 Citations"):
                    for c in citations:
                        st.markdown(f"- **{c['source']}** ({c['locator']})")
                        st.caption(c["snippet"][:150] + "..." if len(c["snippet"]) > 150 else c["snippet"])

            # Memory: extract and write selective facts
            memory_writes = process_memory(prompt, answer)
            if memory_writes:
                st.caption(f"Remembered: {len(memory_writes)} fact(s)")

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "citations": citations,
    })
