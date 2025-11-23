"""
Retrieve Node - Fetches candidate vendors from vector store.
"""

import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

from config import (
    GOOGLE_API_KEY,
    EMBEDDING_MODEL,
    CHROMA_PERSIST_DIR,
    COLLECTION_NAME,
    TOP_K_RETRIEVAL,
)
from graph.state import GraphState, VendorCandidate


def get_vector_store():
    """Load the ChromaDB vector store."""
    # Check if vector store exists
    if not os.path.exists(CHROMA_PERSIST_DIR):
        raise FileNotFoundError(
            f"Vector store not found at '{CHROMA_PERSIST_DIR}'. "
            "Please run 'python run_preprocessing.py' first to create the index."
        )

    embeddings = GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=GOOGLE_API_KEY,
        task_type="RETRIEVAL_QUERY"
    )

    return Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_PERSIST_DIR,
        embedding_function=embeddings
    )


def distance_to_similarity(distance: float) -> float:
    """
    Convert ChromaDB distance to similarity score.
    ChromaDB returns L2 distance by default (lower = more similar).
    We convert to similarity where higher = more similar (0.0 to 1.0 range).
    """
    # Using 1 / (1 + distance) to map [0, inf) -> (0, 1]
    return 1.0 / (1.0 + distance)


def retrieve_node(state: GraphState) -> GraphState:
    """
    Retrieve candidate vendors from vector store using optimized query.

    Input: extracted_info (uses optimized_query)
    Output: candidates
    """
    print("\n[Retrieve Node] Searching for candidates...")

    extracted_info = state.get("extracted_info")
    if not extracted_info:
        return {
            **state,
            "candidates": [],
            "error": "No extracted info available for retrieval",
        }

    # Use optimized query from extraction
    query = extracted_info.get("optimized_query", state["original_query"])
    print(f"[Retrieve Node] Query: {query}")

    # Search vector store with error handling
    try:
        vector_store = get_vector_store()
        results = vector_store.similarity_search_with_score(query, k=TOP_K_RETRIEVAL)
    except FileNotFoundError as e:
        print(f"[Retrieve Node] ERROR: {e}")
        return {
            **state,
            "candidates": [],
            "error": str(e),
        }
    except Exception as e:
        print(f"[Retrieve Node] ERROR: Vector store query failed: {e}")
        return {
            **state,
            "candidates": [],
            "error": f"Vector store query failed: {str(e)}",
        }

    # Convert to VendorCandidate format with stable IDs
    candidates: list[VendorCandidate] = []
    for idx, (doc, distance) in enumerate(results):
        meta = doc.metadata

        # Prefer persisted doc_id; fallback to positional index
        candidate_id = str(meta.get("doc_id", idx))

        # Convert distance to similarity (higher = better)
        similarity = distance_to_similarity(distance)

        candidate: VendorCandidate = {
            "candidate_id": candidate_id,  # Stable ID for lookup
            "company_name": meta.get("company_name", "Unknown"),
            "trading_name": meta.get("trading_name"),
            "services": meta.get("services"),
            "products": meta.get("products"),
            "industry": meta.get("industry"),
            "about": meta.get("about"),
            "city": meta.get("city"),
            "address": meta.get("address"),
            "phone": meta.get("phone"),
            "email": meta.get("email"),
            "website": meta.get("website"),
            "employees": meta.get("employees"),
            "certifications": meta.get("certifications"),
            "similarity_score": round(similarity, 4),  # Now correctly: higher = better
        }
        candidates.append(candidate)

    print(f"[Retrieve Node] Found {len(candidates)} candidates")

    # Preview top 3 (sorted by similarity, highest first)
    for i, c in enumerate(candidates[:3]):
        print(f"  [{i+1}] {c['company_name']} (similarity: {c['similarity_score']:.4f})")

    return {
        **state,
        "candidates": candidates,
    }
