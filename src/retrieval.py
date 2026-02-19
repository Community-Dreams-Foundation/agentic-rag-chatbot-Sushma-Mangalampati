"""Retrieval: query ChromaDB and return relevant chunks with metadata."""

from pathlib import Path

import chromadb
from chromadb.config import Settings

from src.indexing import DEFAULT_CHROMA_PATH, DEFAULT_COLLECTION_NAME, get_embedding_model


def retrieve(
    query: str,
    top_k: int = 5,
    persist_directory: str | Path = DEFAULT_CHROMA_PATH,
    collection_name: str = DEFAULT_COLLECTION_NAME,
) -> list[dict]:
    """
    Retrieve top-k chunks for a query. Returns list of dicts with:
    - text, source, locator, snippet (first ~200 chars of text)
    """
    try:
        client = chromadb.PersistentClient(
            path=str(persist_directory),
            settings=Settings(anonymized_telemetry=False),
        )
        collection = client.get_collection(collection_name)
    except Exception:
        return []

    model = get_embedding_model()
    query_embedding = model.encode([query], show_progress_bar=False).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=min(top_k, 10),
        include=["documents", "metadatas"],
    )

    if not results["documents"] or not results["documents"][0]:
        return []

    chunks = []
    docs = results["documents"][0]
    metadatas = results["metadatas"][0] or []

    for i, doc in enumerate(docs):
        meta = metadatas[i] if i < len(metadatas) else {}
        source = meta.get("source", "unknown")
        locator = meta.get("locator", "unknown")
        snippet = (doc[:200] + "..." if len(doc) > 200 else doc) if doc else ""

        chunks.append({
            "text": doc,
            "source": source,
            "locator": locator,
            "snippet": snippet,
        })

    return chunks
