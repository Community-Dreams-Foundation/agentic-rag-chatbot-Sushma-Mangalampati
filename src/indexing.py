"""Indexing: embed chunks and store in ChromaDB."""

from pathlib import Path

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer


# Default persistence path
DEFAULT_CHROMA_PATH = Path("chroma_db")
DEFAULT_COLLECTION_NAME = "rag_chunks"


def get_embedding_model() -> SentenceTransformer:
    """Load the sentence-transformers embedding model (cached after first load)."""
    return SentenceTransformer("all-MiniLM-L6-v2")


def get_or_create_collection(
    persist_directory: str | Path = DEFAULT_CHROMA_PATH,
    collection_name: str = DEFAULT_COLLECTION_NAME,
) -> chromadb.Collection:
    """Get or create a ChromaDB collection with persistence."""
    client = chromadb.PersistentClient(
        path=str(persist_directory),
        settings=Settings(anonymized_telemetry=False),
    )
    return client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )


def index_chunks(
    chunks: list[dict],
    persist_directory: str | Path = DEFAULT_CHROMA_PATH,
    collection_name: str = DEFAULT_COLLECTION_NAME,
) -> int:
    """
    Embed chunks and add to ChromaDB. Returns number of chunks indexed.
    """
    if not chunks:
        return 0

    model = get_embedding_model()
    collection = get_or_create_collection(persist_directory, collection_name)

    texts = [c["text"] for c in chunks]
    metadatas = [c["metadata"] for c in chunks]
    ids = [f"{m['source']}_{m['chunk_id']}" for m in metadatas]

    embeddings = model.encode(texts, show_progress_bar=False)

    collection.add(
        ids=ids,
        embeddings=embeddings.tolist(),
        documents=texts,
        metadatas=metadatas,
    )

    return len(chunks)


def clear_collection(
    persist_directory: str | Path = DEFAULT_CHROMA_PATH,
    collection_name: str = DEFAULT_COLLECTION_NAME,
) -> None:
    """Delete the collection (for re-indexing)."""
    client = chromadb.PersistentClient(
        path=str(persist_directory),
        settings=Settings(anonymized_telemetry=False),
    )
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass
