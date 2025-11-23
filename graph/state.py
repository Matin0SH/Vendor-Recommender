"""
State definitions for the vendor recommendation graph.
"""

from typing import TypedDict, Optional
from pydantic import BaseModel, Field, field_validator


# =============================================================================
# Pydantic Models for Structured LLM Output
# =============================================================================

class ExtractedInfoModel(BaseModel):
    """Pydantic model for extraction output parsing."""
    job_type: str = Field(description="Main category of work")
    services_needed: list[str] = Field(description="Specific services required")
    location: Optional[str] = Field(default=None, description="Location if mentioned")
    urgency: str = Field(default="normal", description="urgent, normal, or flexible")
    additional_context: Optional[str] = Field(default=None, description="Other relevant details")
    optimized_query: str = Field(description="Keyword-rich query for semantic search")


class RankedVendorModel(BaseModel):
    """Pydantic model for a single ranked vendor."""
    rank: int = Field(description="Ranking position")
    candidate_id: str = Field(description="Stable ID from retrieval")
    relevance_score: float = Field(description="Score from 0.0 to 1.0")
    reasoning: str = Field(description="Step-by-step reasoning for this ranking")

    @field_validator('candidate_id', mode='before')
    @classmethod
    def coerce_to_string(cls, v):
        return str(v)


class RerankOutputModel(BaseModel):
    """Pydantic model for reranking output parsing."""
    user_need_analysis: str = Field(description="Brief description of user's actual need")
    required_service_types: list[str] = Field(description="List of required service types")
    rankings: list[RankedVendorModel] = Field(description="Ranked list of vendors")


# =============================================================================
# TypedDict State Definitions
# =============================================================================

class ExtractedInfo(TypedDict):
    """Extracted information from user query."""
    job_type: str
    services_needed: list[str]
    location: Optional[str]
    urgency: Optional[str]
    additional_context: Optional[str]
    optimized_query: str


class VendorCandidate(TypedDict):
    """A vendor candidate from retrieval."""
    candidate_id: str  # Stable ID for lookup
    company_name: str
    trading_name: Optional[str]
    services: Optional[str]
    products: Optional[str]
    industry: Optional[str]
    about: Optional[str]
    city: Optional[str]
    address: Optional[str]
    phone: Optional[str]
    email: Optional[str]
    website: Optional[str]
    employees: Optional[str]  # Stored as string to accommodate numeric inputs
    certifications: Optional[str]
    similarity_score: float  # Higher = better (converted from distance)


class RankedVendor(TypedDict):
    """A vendor after reranking with reasoning."""
    rank: int
    candidate_id: str
    company_name: str
    trading_name: Optional[str]
    services: Optional[str]
    products: Optional[str]
    industry: Optional[str]
    about: Optional[str]
    city: Optional[str]
    address: Optional[str]
    phone: Optional[str]
    email: Optional[str]
    website: Optional[str]
    employees: Optional[str]
    certifications: Optional[str]
    relevance_score: float
    reasoning: str


class GraphState(TypedDict):
    """
    State that flows through the recommendation graph.

    Attributes:
        original_query: The user's original natural language query
        extracted_info: Structured extraction from the query
        candidates: Raw candidates from vector retrieval
        ranked_vendors: Final ranked list with reasoning
        error: Any error message if processing fails
    """
    original_query: str
    extracted_info: Optional[ExtractedInfo]
    candidates: Optional[list[VendorCandidate]]
    ranked_vendors: Optional[list[RankedVendor]]
    error: Optional[str]
