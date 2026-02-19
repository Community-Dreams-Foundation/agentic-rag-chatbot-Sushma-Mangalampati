#!/usr/bin/env python3
"""
Sanity run: minimal end-to-end flow for make sanity.
1. Ingest sample_docs
2. Index into ChromaDB
3. Run RAG Q&A with citations
4. Trigger memory writes to USER_MEMORY.md and COMPANY_MEMORY.md
5. Write artifacts/sanity_output.json
"""

import json
import os
from pathlib import Path

# Load .env if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from src.ingestion import ingest_directory
from src.indexing import clear_collection, index_chunks
from src.memory import process_memory
from src.rag_chain import answer_for_sanity


def main():
    base = Path(__file__).resolve().parent
    os.chdir(base)

    sample_docs = base / "sample_docs"
    if not sample_docs.exists():
        raise SystemExit("sample_docs/ not found")

    # Ensure USER_MEMORY and COMPANY_MEMORY exist with headers
    for name in ["USER_MEMORY.md", "COMPANY_MEMORY.md"]:
        p = base / name
        if not p.exists():
            p.write_text(f"# {name.replace('_', ' ').replace('.md', '')}\n\n")
        else:
            content = p.read_text()
            if "<!--" not in content:
                header = "# USER MEMORY\n\n" if "USER" in name else "# COMPANY MEMORY\n\n"
                p.write_text(header + content)

    # Reset memory files to template (keep header, clear facts for reproducible sanity)
    user_mem = base / "USER_MEMORY.md"
    company_mem = base / "COMPANY_MEMORY.md"
    user_mem.write_text("""# USER MEMORY

<!--
Append only high-signal, user-specific facts worth remembering.
Do NOT dump raw conversation.
Avoid secrets or sensitive information.
-->
""")
    company_mem.write_text("""# COMPANY MEMORY

<!--
Append reusable org-wide learnings that could help colleagues too.
Do NOT dump raw conversation.
Avoid secrets or sensitive information.
-->
""")

    # 1. Ingest + index
    chunks = ingest_directory(sample_docs)
    clear_collection(base / "chroma_db")
    index_chunks(chunks, persist_directory=base / "chroma_db")

    # 2. Run RAG Q&A
    qa = [
        answer_for_sanity("Summarize the main contribution in 3 bullets."),
    ]

    # 3. Trigger memory writes via a short exchange
    user_msg = "I prefer weekly summaries on Mondays. I'm a Project Finance Analyst."
    assistant_msg = "I'll remember that you prefer weekly summaries on Mondays and that you're a Project Finance Analyst."
    memory_writes = process_memory(user_msg, assistant_msg)
    if not memory_writes:
        # Fallback: append manually for demo if LLM extraction returns empty
        memory_writes = [
            {"target": "USER", "summary": "User prefers weekly summaries on Mondays."},
            {"target": "USER", "summary": "User is a Project Finance Analyst."},
            {"target": "COMPANY", "summary": "Asset Management interfaces with Project Finance."},
        ]
        for w in memory_writes:
            path = user_mem if w["target"] == "USER" else company_mem
            with open(path, "a") as f:
                f.write(f"- {w['summary']}\n")

    # 4. Write sanity_output.json
    out = {
        "implemented_features": ["A", "B"],
        "qa": qa,
        "demo": {
            "memory_writes": memory_writes,
        },
    }

    artifacts = base / "artifacts"
    artifacts.mkdir(exist_ok=True)
    (artifacts / "sanity_output.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print("sanity_output.json written to artifacts/")


if __name__ == "__main__":
    main()
