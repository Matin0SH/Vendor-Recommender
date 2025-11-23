"""
Main script for preprocessing vendor data.

This script:
1. Loads raw vendor data from all_results.json
2. Preprocesses and combines text fields
3. Creates embeddings using Gemini
4. Indexes into ChromaDB vector store

Run this BEFORE using the recommender system.

Usage:
    python run_preprocessing.py
"""

import argparse
from preprocessing.preprocess import preprocess_vendors, save_processed
from preprocessing.embeddings import index_vendors_with_dedup, get_query_embeddings, load_vector_store
from config import RAW_DATA_PATH, PROCESSED_DATA_PATH


def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess vendor data and build/update vector index.")
    parser.add_argument(
        "--reset-index",
        action="store_true",
        help="Delete any existing Chroma index before indexing (start clean)."
    )
    parser.add_argument(
        "--no-dedup",
        action="store_true",
        help="Disable deduplication (allow re-adding existing ids)."
    )
    return parser.parse_args()


def main():
    """Run the full preprocessing pipeline."""
    args = parse_args()

    print("=" * 60)
    print("VENDOR DATA PREPROCESSING")
    print("=" * 60)

    # Step 1: Preprocess vendors
    print("\n[Step 1] Preprocessing vendor data...")
    processed = preprocess_vendors(RAW_DATA_PATH)
    save_processed(processed, PROCESSED_DATA_PATH)

    print(f"  Processed {len(processed)} vendors")

    # Step 2: Create embeddings and index
    print("\n[Step 2] Creating embeddings and indexing...")
    vector_store = index_vendors_with_dedup(
        PROCESSED_DATA_PATH,
        dedup=not args.no_dedup,
        reset=args.reset_index,
    )

    # Step 3: Verify with test search
    print("\n[Step 3] Verifying index with test search...")
    query_embeddings = get_query_embeddings()
    vs = load_vector_store(query_embeddings)

    test_query = "fire protection sprinkler systems"
    results = vs.similarity_search_with_score(test_query, k=3)

    print(f"\n  Test query: '{test_query}'")
    print("  " + "-" * 40)
    for i, (doc, score) in enumerate(results, 1):
        print(f"  [{i}] {doc.metadata.get('company_name')} (score: {score:.4f})")

    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETE")
    print("=" * 60)
    print("\nYou can now run the recommender:")
    print("  python run_recommender.py")


if __name__ == "__main__":
    main()
