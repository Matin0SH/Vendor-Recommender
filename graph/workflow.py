"""
LangGraph workflow definition for vendor recommendation.
"""

from langgraph.graph import StateGraph, END

from graph.state import GraphState
from graph.nodes.extract import extract_node
from graph.nodes.retrieve import retrieve_node
from graph.nodes.rerank import rerank_node


def create_graph() -> StateGraph:
    """
    Create the vendor recommendation graph.

    Flow:
        START -> extract -> retrieve -> rerank -> END

    Architecture:
        +-------------+
        |   START     |
        +------+------+
               |
               v
        +-------------+
        |   Extract   |  Parse user query, extract job info
        +------+------+
               |
               v
        +-------------+
        |  Retrieve   |  Vector search with optimized query
        +------+------+
               |
               v
        +-------------+
        |   Rerank    |  LLM CoT reasoning with ORIGINAL query
        +------+------+
               |
               v
        +-------------+
        |    END      |  Return ranked vendors with reasoning
        +-------------+
    """
    # Create graph with state schema
    workflow = StateGraph(GraphState)

    # Add nodes
    workflow.add_node("extract", extract_node)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("rerank", rerank_node)

    # Define edges (linear flow)
    workflow.set_entry_point("extract")
    workflow.add_edge("extract", "retrieve")
    workflow.add_edge("retrieve", "rerank")
    workflow.add_edge("rerank", END)

    # Compile
    graph = workflow.compile()

    return graph


def run_recommendation(query: str) -> dict:
    """
    Run the full recommendation pipeline.

    Args:
        query: User's natural language job request

    Returns:
        Final state with ranked_vendors and reasoning
    """
    graph = create_graph()

    # Initialize state
    initial_state: GraphState = {
        "original_query": query,
        "extracted_info": None,
        "candidates": None,
        "ranked_vendors": None,
        "error": None,
    }

    # Run graph
    final_state = graph.invoke(initial_state)

    return final_state


def print_results(state: dict):
    """Pretty print the recommendation results."""
    print("\n" + "=" * 70)
    print("VENDOR RECOMMENDATIONS")
    print("=" * 70)

    if state.get("error"):
        print(f"\nWarning: {state['error']}")

    ranked = state.get("ranked_vendors", [])

    if not ranked:
        print("\nNo vendors found matching your request.")
        return

    print(f"\nFound {len(ranked)} relevant vendors:\n")

    for v in ranked:
        print(f"#{v['rank']} - {v['company_name']}")
        print(f"   Relevance Score: {v['relevance_score']:.2f}")

        if v.get("trading_name"):
            print(f"   Also known as: {v['trading_name']}")
        if v.get("industry"):
            print(f"   Industry: {v['industry']}")
        if v.get("services"):
            print(f"   Services: {v['services']}")
        if v.get("products"):
            print(f"   Products: {v['products']}")
        if v.get("about"):
            print(f"   About: {v['about']}")
        if v.get("city"):
            print(f"   Location: {v['city']}")
        if v.get("address"):
            print(f"   Address: {v['address']}")
        if v.get("phone"):
            print(f"   Phone: {v['phone']}")
        if v.get("email"):
            print(f"   Email: {v['email']}")
        if v.get("website"):
            print(f"   Website: {v['website']}")
        if v.get("employees"):
            print(f"   Employees: {v['employees']}")
        if v.get("certifications"):
            print(f"   Certifications: {v['certifications']}")

        print(f"\n   Reasoning: {v['reasoning']}")
        print("-" * 70)
