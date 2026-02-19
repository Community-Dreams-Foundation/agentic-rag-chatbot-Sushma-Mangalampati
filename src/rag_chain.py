"""RAG chain: retrieve + LLM with citations."""

import json
import os

from src.llm_client import get_client, get_model
from src.retrieval import retrieve


CITATION_PROMPT = """You are a helpful assistant that answers questions based ONLY on the provided context.
If the answer cannot be found in the context, say "I couldn't find relevant information in the uploaded documents."
Do NOT make up information or cite sources that don't exist.

Context (retrieved passages):
{context}

For each fact you state, cite the source using this exact format: [Source: filename, Locator: locator]
Example: [Source: report.pdf, Locator: page 2 / section 1]

Question: {question}

Answer (with inline citations):"""


def _format_context(chunks: list[dict]) -> str:
    """Format retrieved chunks for the prompt."""
    parts = []
    for i, c in enumerate(chunks, 1):
        parts.append(
            f"[{i}] (Source: {c['source']}, Locator: {c['locator']})\n{c['text']}"
        )
    return "\n\n".join(parts)


def answer_with_citations(
    question: str,
    top_k: int = 5,
    api_key: str | None = None,
) -> tuple[str, list[dict]]:
    """
    Retrieve chunks, call LLM, return (answer, citations).
    citations: list of {source, locator, snippet}
    """
    chunks = retrieve(question, top_k=top_k)

    if not chunks:
        return (
            "I couldn't find relevant information in the uploaded documents. "
            "Please upload documents first or try a different question.",
            [],
        )

    context = _format_context(chunks)
    prompt = CITATION_PROMPT.format(context=context, question=question)

    client = get_client(api_key=api_key)
    if client is None:
        citations = []
        seen = set()
        for c in chunks:
            key = (c["source"], c["locator"])
            if key not in seen:
                seen.add(key)
                citations.append({"source": c["source"], "locator": c["locator"], "snippet": c["snippet"]})
        return (
            "No LLM configured. Set USE_OLLAMA=1 for local Ollama, or OPENAI_API_KEY for OpenAI. "
            f"Top result: {chunks[0]['snippet'][:100]}...",
            citations,
        )

    # Build citations from chunks (used on success and on API failure fallback)
    citations = []
    seen = set()
    for c in chunks:
        key = (c["source"], c["locator"])
        if key not in seen:
            seen.add(key)
            citations.append({
                "source": c["source"],
                "locator": c["locator"],
                "snippet": c["snippet"],
            })

    try:
        response = client.chat.completions.create(
            model=get_model(),
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        answer = response.choices[0].message.content or ""
    except Exception as e:
        err_msg = str(e)
        if "429" in err_msg or "quota" in err_msg.lower() or "insufficient_quota" in err_msg:
            fallback = (
                "Relevant passages retrieved (LLM unavailable - quota exceeded). "
                "Please check https://platform.openai.com/account/billing. "
                f"Top result: {chunks[0]['snippet'][:150]}..."
            )
            return (fallback, citations)
        return (f"OpenAI API error: {err_msg}", citations)

    return answer.strip(), citations


def answer_for_sanity(
    question: str,
    top_k: int = 5,
    api_key: str | None = None,
) -> dict:
    """Return a qa item for sanity_output.json: {question, answer, citations}."""
    answer, citations = answer_with_citations(question, top_k, api_key)
    return {
        "question": question,
        "answer": answer,
        "citations": citations,
    }
