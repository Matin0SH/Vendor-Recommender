"""
Extract Node - Parses user query to extract structured job information.
Uses Pydantic for robust JSON parsing.
"""

import re
import json
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import ValidationError

from config import (
    GOOGLE_API_KEY,
    LLM_MODEL,
    LLM_TEMPERATURE,
    EXTRACTION_PROMPT,
)
from graph.state import GraphState, ExtractedInfo, ExtractedInfoModel


def get_llm():
    """Initialize Gemini LLM."""
    return ChatGoogleGenerativeAI(
        model=LLM_MODEL,
        google_api_key=GOOGLE_API_KEY,
        temperature=LLM_TEMPERATURE,
    )


def extract_json_from_text(text: str) -> str:
    """
    Extract JSON object from text that may contain extra content.
    More robust than simple fence stripping.
    """
    # Remove markdown code fences
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()

    # Try to find JSON object using regex
    # Match first complete JSON object {...}
    json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text, re.DOTALL)
    if json_match:
        return json_match.group()

    # Fallback: return cleaned text
    return text


def extract_node(state: GraphState) -> GraphState:
    """
    Extract structured information from user's natural language query.

    Input: original_query
    Output: extracted_info
    """
    print("\n[Extract Node] Analyzing user query...")

    original_query = state["original_query"]

    # Format prompt with user query
    prompt = EXTRACTION_PROMPT.format(query=original_query)

    # Call LLM
    llm = get_llm()
    response = llm.invoke(prompt)

    # Parse JSON response with robust extraction
    try:
        json_str = extract_json_from_text(response.content)
        raw_data = json.loads(json_str)

        # Validate with Pydantic
        validated = ExtractedInfoModel(**raw_data)

        # Enhance optimized query with location if available (and not already present)
        optimized_query = validated.optimized_query
        if validated.location and validated.location.lower() not in optimized_query.lower():
            # Append location to improve retrieval for local vendors
            optimized_query = f"{optimized_query} {validated.location}"

        extracted_info: ExtractedInfo = {
            "job_type": validated.job_type,
            "services_needed": validated.services_needed,
            "location": validated.location,
            "urgency": validated.urgency,
            "additional_context": validated.additional_context,
            "optimized_query": optimized_query,
        }

        print(f"[Extract Node] Job type: {extracted_info['job_type']}")
        print(f"[Extract Node] Services: {extracted_info['services_needed']}")
        if extracted_info['location']:
            print(f"[Extract Node] Location: {extracted_info['location']}")
        print(f"[Extract Node] Optimized query: {extracted_info['optimized_query']}")

        return {
            **state,
            "extracted_info": extracted_info,
            "error": None,
        }

    except json.JSONDecodeError as e:
        print(f"[Extract Node] ERROR: JSON parse failed: {e}")
        print(f"[Extract Node] Raw response: {response.content[:300]}...")

    except ValidationError as e:
        print(f"[Extract Node] ERROR: Pydantic validation failed: {e}")
        print(f"[Extract Node] Raw response: {response.content[:300]}...")

    except Exception as e:
        print(f"[Extract Node] ERROR: Unexpected error: {e}")

    # Fallback - use original query as-is
    print("[Extract Node] Using fallback extraction")
    return {
        **state,
        "extracted_info": {
            "job_type": "unknown",
            "services_needed": [],
            "location": None,
            "urgency": "normal",
            "additional_context": None,
            "optimized_query": original_query,  # Use original query for retrieval
        },
        "error": "Extraction failed, using original query",
    }
