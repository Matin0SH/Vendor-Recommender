"""
Vendor Recommender - Main Entry Point

A LangGraph-based recommendation system that:
1. Extracts job requirements from natural language queries
2. Retrieves candidate vendors via semantic search
3. Reranks with LLM Chain-of-Thought reasoning

Usage:
    python run_recommender.py                    # Interactive mode
    python run_recommender.py "your query here"  # Single query mode
"""

import sys
from graph.workflow import run_recommendation, print_results


def interactive_mode():
    """Run interactive recommendation session."""
    print("\n" + "=" * 70)
    print("VENDOR RECOMMENDER SYSTEM")
    print("Powered by LangGraph + Gemini")
    print("=" * 70)
    print("\nDescribe the job you need done, and I'll find the best vendors.")
    print("Type 'quit' or 'exit' to stop.\n")

    while True:
        print("\nEnter your job request:")
        query = input("> ").strip()

        if query.lower() in ["quit", "exit", "q"]:
            print("\nGoodbye!")
            break

        if not query:
            print("Please enter a job description.")
            continue

        print("\n" + "-" * 70)
        print(f"Processing: {query}")
        print("-" * 70)

        # Run recommendation pipeline
        result = run_recommendation(query)

        # Print results
        print_results(result)


def single_query_mode(query: str):
    """Run a single recommendation query."""
    print("\n" + "=" * 70)
    print("VENDOR RECOMMENDER")
    print("=" * 70)
    print(f"\nQuery: {query}")

    # Run recommendation pipeline
    result = run_recommendation(query)

    # Print results
    print_results(result)


def main():
    """Main entry point."""
    if len(sys.argv) > 1:
        # Single query from command line
        query = " ".join(sys.argv[1:])
        single_query_mode(query)
    else:
        # Interactive mode
        interactive_mode()


if __name__ == "__main__":
    main()
