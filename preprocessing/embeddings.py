"""
Embedding and vector store setup using Google Gemini embeddings and ChromaDB.
"""

import json
import shutil
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    GOOGLE_API_KEY,
    EMBEDDING_MODEL,
    CHROMA_PERSIST_DIR,
    COLLECTION_NAME,
)


def get_embeddings() -> GoogleGenerativeAIEmbeddings:
    """Initialize Gemini embeddings for documents."""
    return GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=GOOGLE_API_KEY,
        task_type="RETRIEVAL_DOCUMENT"
    )


def get_query_embeddings() -> GoogleGenerativeAIEmbeddings:
    """Initialize Gemini embeddings for queries."""
    return GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=GOOGLE_API_KEY,
        task_type="RETRIEVAL_QUERY"
    )


def load_processed_vendors(path: str) -> list[dict]:
    """Load preprocessed vendor data."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def create_documents(vendors: list[dict]) -> tuple[list[Document], list[str]]:
    """Convert processed vendors to LangChain Documents along with their ids."""
    documents: list[Document] = []
    ids: list[str] = []
    for vendor in vendors:
        doc_id = str(vendor["id"])
        doc = Document(
            page_content=vendor["text"],
            metadata={**vendor["metadata"], "doc_id": doc_id}
        )
        documents.append(doc)
        ids.append(doc_id)
    return documents, ids


def create_vector_store(documents: list[Document], embeddings: GoogleGenerativeAIEmbeddings) -> Chroma:
    """Create and persist ChromaDB vector store."""
    print(f"Creating vector store with {len(documents)} documents...")

    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_PERSIST_DIR
    )

    print(f"Vector store created and persisted to {CHROMA_PERSIST_DIR}")
    return vector_store


def load_vector_store(embeddings: GoogleGenerativeAIEmbeddings) -> Chroma:
    """Load existing ChromaDB vector store."""
    return Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_PERSIST_DIR,
        embedding_function=embeddings
    )


def index_vendors(processed_path: str):
    """Main function to index all vendors into ChromaDB."""
    return index_vendors_with_dedup(processed_path, dedup=True, reset=False)


def index_vendors_with_dedup(processed_path: str, dedup: bool = True, reset: bool = False):
    """
    Index vendors into ChromaDB with optional deduplication and reset.

    Args:
        processed_path: Path to preprocessed vendors JSON.
        dedup: If True, skip documents whose ids already exist in the store.
        reset: If True, delete the existing persisted store before indexing.
    """
    # Load processed data
    vendors = load_processed_vendors(processed_path)
    print(f"Loaded {len(vendors)} processed vendors")

    if not vendors:
        print("No vendors to index; exiting.")
        return None

    # Create documents
    documents, ids = create_documents(vendors)

    # Initialize embeddings
    embeddings = get_embeddings()

    persist_path = Path(CHROMA_PERSIST_DIR)
    if reset and persist_path.exists():
        print(f"Reset requested: removing existing index at {persist_path}")
        shutil.rmtree(persist_path, ignore_errors=True)

    # If no existing store, build fresh
    if not persist_path.exists():
        return create_vector_store(documents, embeddings)

    # Otherwise load existing and optionally deduplicate
    vector_store = load_vector_store(embeddings)

    existing_ids: set[str] = set()
    try:
        existing = vector_store.get(include=[])
        existing_ids.update(existing.get("ids", []))
    except Exception as exc:  # pragma: no cover - defensive
        print(f"Warning: could not read existing ids ({exc}); proceeding without dedup")
        dedup = False

    docs_to_add = []
    ids_to_add = []
    skipped = 0

    for doc, doc_id in zip(documents, ids):
        if dedup and doc_id in existing_ids:
            skipped += 1
            continue
        docs_to_add.append(doc)
        ids_to_add.append(doc_id)

    if not docs_to_add:
        print(f"No new documents to add (skipped {skipped} duplicates).")
        return vector_store

    print(f"Adding {len(docs_to_add)} new documents to existing store (skipped {skipped} duplicates).")
    vector_store.add_documents(documents=docs_to_add, ids=ids_to_add)
    print(f"Vector store updated and persisted to {CHROMA_PERSIST_DIR}")
    return vector_store
