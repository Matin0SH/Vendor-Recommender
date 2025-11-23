"""
Rerank Node - LLM-based reranking with Chain-of-Thought reasoning.
Uses Pydantic for robust JSON parsing and stable candidate_id for lookups.
"""

import re
import json
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import ValidationError

from config import (
    GOOGLE_API_KEY,
    LLM_MODEL,
    LLM_TEMPERATURE,
    RERANKING_PROMPT,
    TOP_K_RERANK,
)
from graph.state import GraphState, RankedVendor, RerankOutputModel


def get_llm():
    """Initialize Gemini LLM for reranking."""
    return ChatGoogleGenerativeAI(
        model=LLM_MODEL,
        google_api_key=GOOGLE_API_KEY,
        temperature=LLM_TEMPERATURE,
    )


def format_candidates_for_prompt(candidates: list) -> str:
    """Format candidates into a readable string for the LLM with stable IDs."""
    formatted = []

    for c in candidates:
        # Use candidate_id as the stable identifier
        parts = [f"### Candidate ID: {c['candidate_id']} - {c['company_name']}"]

        if c.get("trading_name"):
            parts.append(f"- Also known as: {c['trading_name']}")
        if c.get("services"):
            parts.append(f"- Services: {c['services']}")
        if c.get("products"):
            parts.append(f"- Products: {c['products']}")
        if c.get("industry"):
            parts.append(f"- Industry: {c['industry']}")
        if c.get("about"):
            parts.append(f"- About: {c['about']}")
        if c.get("city"):
            parts.append(f"- Location: {c['city']}")
        if c.get("address"):
            parts.append(f"- Address: {c['address']}")
        if c.get("certifications"):
            parts.append(f"- Certifications: {c['certifications']}")
        if c.get("phone"):
            parts.append(f"- Phone: {c['phone']}")
        if c.get("email"):
            parts.append(f"- Email: {c['email']}")
        if c.get("website"):
            parts.append(f"- Website: {c['website']}")
        if c.get("employees"):
            parts.append(f"- Employees: {c['employees']}")

        parts.append(f"- Similarity score: {c.get('similarity_score', 'N/A')}")

        formatted.append("\n".join(parts))

    return "\n\n".join(formatted)


def extract_json_from_text(text: str) -> str:
    """
    Extract JSON object from text that may contain extra content.
    More robust than simple fence stripping.
    """
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()

    # Try to find JSON object using regex - match outermost braces
    # This handles nested objects better
    brace_count = 0
    start_idx = None
    for i, char in enumerate(text):
        if char == '{':
            if start_idx is None:
                start_idx = i
            brace_count += 1
        elif char == '}':
            brace_count -= 1
            if brace_count == 0 and start_idx is not None:
                return text[start_idx:i+1]

    return text


def rerank_node(state: GraphState) -> GraphState:
    """
    Rerank candidates using LLM with Chain-of-Thought reasoning.

    Input: original_query, candidates
    Output: ranked_vendors (with reasoning)
    """
    print("\n[Rerank Node] Analyzing candidates with CoT reasoning...")

    original_query = state["original_query"]
    candidates = state.get("candidates", [])

    if not candidates:
        return {
            **state,
            "ranked_vendors": [],
            "error": "No candidates to rerank",
        }

    # Format candidates for prompt with stable IDs
    candidates_text = format_candidates_for_prompt(candidates)

    # Build prompt with ORIGINAL query (not extracted)
    prompt = RERANKING_PROMPT.format(
        original_query=original_query,
        candidates=candidates_text,
        top_k=TOP_K_RERANK
    )

    # Call LLM
    llm = get_llm()
    print(f"[Rerank Node] Sending {len(candidates)} candidates to LLM for analysis...")

    response = llm.invoke(prompt)

    # Create lookup by candidate_id (stable, string-based)
    candidate_lookup = {str(c["candidate_id"]): c for c in candidates}

    # Parse JSON response with robust extraction
    try:
        json_str = extract_json_from_text(response.content)
        raw_data = json.loads(json_str)

        # Validate with Pydantic
        validated = RerankOutputModel(**raw_data)

        print(f"[Rerank Node] User need analysis: {validated.user_need_analysis}")
        print(f"[Rerank Node] Required services: {validated.required_service_types}")

        # Build ranked vendors list using stable candidate_id
        ranked_vendors: list[RankedVendor] = []

        for r in validated.rankings[:TOP_K_RERANK]:
            candidate = candidate_lookup.get(r.candidate_id)

            if candidate is None:
                print(f"[Rerank Node] WARNING: candidate_id {r.candidate_id} not found, skipping")
                continue

            ranked_vendor: RankedVendor = {
                "rank": r.rank,
                "candidate_id": r.candidate_id,
                "company_name": candidate["company_name"],
                "trading_name": candidate.get("trading_name"),
                "services": candidate.get("services"),
                "products": candidate.get("products"),
                "industry": candidate.get("industry"),
                "about": candidate.get("about"),
                "city": candidate.get("city"),
                "address": candidate.get("address"),
                "phone": candidate.get("phone"),
                "email": candidate.get("email"),
                "website": candidate.get("website"),
                "employees": candidate.get("employees"),
                "certifications": candidate.get("certifications"),
                "relevance_score": r.relevance_score,
                "reasoning": r.reasoning,
            }
            ranked_vendors.append(ranked_vendor)

        print(f"[Rerank Node] Ranked {len(ranked_vendors)} vendors")

        return {
            **state,
            "ranked_vendors": ranked_vendors,
        }

    except json.JSONDecodeError as e:
        print(f"[Rerank Node] ERROR: JSON parse failed: {e}")
        print(f"[Rerank Node] Raw response: {response.content[:500]}...")

    except ValidationError as e:
        print(f"[Rerank Node] ERROR: Pydantic validation failed: {e}")
        print(f"[Rerank Node] Raw response: {response.content[:500]}...")

    except Exception as e:
        print(f"[Rerank Node] ERROR: Unexpected error: {e}")

    # Fallback - return candidates sorted by similarity (highest first)
    print("[Rerank Node] Using fallback: sorting by similarity score")
    sorted_candidates = sorted(candidates, key=lambda x: x["similarity_score"], reverse=True)

    ranked_vendors = []
    for i, c in enumerate(sorted_candidates[:TOP_K_RERANK]):
        ranked_vendors.append({
            "rank": i + 1,
            "candidate_id": c["candidate_id"],
            "company_name": c["company_name"],
            "trading_name": c.get("trading_name"),
            "services": c.get("services"),
            "products": c.get("products"),
            "industry": c.get("industry"),
            "about": c.get("about"),
            "city": c.get("city"),
            "address": c.get("address"),
            "phone": c.get("phone"),
            "email": c.get("email"),
            "website": c.get("website"),
            "employees": c.get("employees"),
            "certifications": c.get("certifications"),
            "relevance_score": c["similarity_score"],  # Use similarity directly
            "reasoning": "Ranked by semantic similarity (LLM reranking failed)",
        })

    return {
        **state,
        "ranked_vendors": ranked_vendors,
        "error": "Reranking parse failed, using similarity fallback",
    }
