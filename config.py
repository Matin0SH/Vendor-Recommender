"""
Configuration for the Vendor Recommender System.

All parameters, model settings, and prompts in one place.
"""

import os
import sys
from dotenv import load_dotenv

load_dotenv()

# =============================================================================
# API CONFIGURATION
# =============================================================================

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Early validation - fail fast with clear error
if not GOOGLE_API_KEY:
    print("ERROR: GOOGLE_API_KEY not found in environment variables.")
    print("Please set it in your .env file or environment:")
    print("  export GOOGLE_API_KEY='your-api-key'")
    print("  # or in .env file:")
    print("  GOOGLE_API_KEY=your-api-key")
    sys.exit(1)

# =============================================================================
# MODEL CONFIGURATION
# =============================================================================

# Embedding model
EMBEDDING_MODEL = "models/gemini-embedding-001"
EMBEDDING_DIMENSIONS = 3072  # Default for gemini-embedding-001

# LLM for extraction and reranking
LLM_MODEL = "gemini-2.0-flash"
LLM_TEMPERATURE = 0.0  # Deterministic outputs

# =============================================================================
# VECTOR STORE CONFIGURATION
# =============================================================================

CHROMA_PERSIST_DIR = "chroma_db"
COLLECTION_NAME = "vendors"

# =============================================================================
# RETRIEVAL CONFIGURATION
# =============================================================================

TOP_K_RETRIEVAL = 30  # Number of candidates to retrieve
TOP_K_RERANK = 10     # Number of final recommendations

# =============================================================================
# FILE PATHS
# =============================================================================

RAW_DATA_PATH = "output/all_results.json"
PROCESSED_DATA_PATH = "output/vendors_processed.json"

# =============================================================================
# PROMPTS
# =============================================================================

EXTRACTION_PROMPT = '''You are an expert job request analyzer specializing in understanding what services users need from contractors and vendors.

Your task: Extract structured information from the user's job request to enable accurate vendor matching.

## Instructions

Analyze the user's request and extract:
1. **job_type**: The main category of work (e.g., "construction", "plumbing", "catering", "IT services")
2. **services_needed**: Specific services required (list of strings)
3. **location**: Any mentioned location or "null" if not specified
4. **urgency**: "urgent", "normal", or "flexible" based on language cues
5. **additional_context**: Any other relevant details
6. **optimized_query**: A clean, keyword-rich query optimized for semantic search

## Optimized Query Rules (very important)
- Use only concepts explicitly present in the user request (or your extracted location below); do NOT invent locations, venues, or contexts.
- If a location is mentioned, include it once; otherwise omit location.
- Keep it concise: 4-12 keywords, no full sentences.
- Focus on service/industry terms, not brand/company names, and do not include user urgency words.
- If the request is commercial/industrial, include that qualifier only if stated or strongly implied.

## Examples

**Example 1:**
User: "I need to dig a hole behind the pub I have, which vendors can do this for me?"

Analysis:
- The user owns a pub and needs excavation/groundwork services
- "Dig a hole" suggests earthworks, excavation, or groundwork contractors
- No specific location mentioned beyond "behind the pub"
- No urgency indicated

Output:
```
job_type: construction
services_needed: excavation, groundwork, digging, earthworks
location: null
urgency: normal
additional_context: Work is for a pub property, outdoor/behind building
optimized_query: excavation groundwork digging earthworks construction contractor
```

**Example 2:**
User: "Emergency! Water pipe burst in my restaurant kitchen in Leeds, need someone NOW"

Analysis:
- Emergency plumbing situation
- Restaurant in Leeds
- Urgent timeline

Output:
```
job_type: plumbing
services_needed: emergency plumbing, pipe repair, water damage, commercial plumbing
location: Leeds
urgency: urgent
additional_context: Commercial kitchen environment, water pipe burst
optimized_query: emergency plumbing pipe repair commercial kitchen restaurant water
```

**Example 3:**
User: "Looking for someone to install fire sprinklers in our new office building"

Analysis:
- Fire safety installation
- Commercial/office building
- New installation (not repair)

Output:
```
job_type: fire protection
services_needed: fire sprinkler installation, fire suppression systems, fire safety
location: null
urgency: normal
additional_context: New office building, commercial installation
optimized_query: fire sprinkler installation fire protection fire suppression commercial building
```

**Example 4 (Quantity surveying):**
User: "Need a quantity surveyor for our housing project in Wellingborough"

Output:
```
job_type: surveying
services_needed: quantity surveying, cost estimation, tendering support
location: Wellingborough
urgency: normal
additional_context: Housing construction project
optimized_query: quantity surveying cost estimation tendering housing project Wellingborough
```

**Example 5 (Fire protection maintenance):**
User: "Service our fire sprinkler system at a Leeds warehouse"

Output:
```
job_type: fire protection
services_needed: fire sprinkler servicing, inspection, fire suppression maintenance
location: Leeds
urgency: normal
additional_context: Warehouse environment
optimized_query: fire sprinkler servicing inspection fire suppression maintenance Leeds warehouse
```

**Example 6 (Security & cleaning FM):**
User: "Need security guards and CCTV monitoring for a retail store, also nightly cleaning"

Output:
```
job_type: facilities management
services_needed: security guarding, CCTV monitoring, retail security, nightly cleaning
location: null
urgency: normal
additional_context: Retail store environment
optimized_query: security guarding CCTV monitoring retail security nightly cleaning
```

**Example 7 (Industrial printing service):**
User: "Looking for maintenance on our industrial coding printers for a food packaging line"

Output:
```
job_type: industrial printing
services_needed: industrial printer maintenance, coding and marking, food packaging line support
location: null
urgency: normal
additional_context: Food packaging production line
optimized_query: industrial printer maintenance coding marking food packaging line
```

## Now analyze this request:

User: "{query}"

Return your response as valid JSON with this exact structure:
{{
  "job_type": "string",
  "services_needed": ["list", "of", "services"],
  "location": "string or null",
  "urgency": "urgent|normal|flexible",
  "additional_context": "string or null",
  "optimized_query": "keyword rich search query"
}}

Return ONLY the JSON. No additional text.
'''


RERANKING_PROMPT = '''You are an expert vendor matching specialist with deep knowledge of contractor services and capabilities.

## Your Task

The user has submitted a job request. You have been given a list of potential vendor candidates retrieved from a database. Your job is to:
1. Understand exactly what the user needs
2. Evaluate each vendor's suitability
3. Rank the top 10 most relevant vendors
4. Provide clear reasoning for each ranking

## User's Original Request

"{original_query}"

## Candidate Vendors

{candidates}

## Evaluation Instructions

Think step by step for each vendor:

### Step 1: Understand the User's Need
- What specific work does the user need done?
- What type of contractor/service provider would handle this?
- Are there any special requirements (location, urgency, certifications)?

### Step 2: Evaluate Each Vendor
For each candidate, consider:
- **Service Match**: Do their services directly address the user's need?
- **Industry Relevance**: Is their industry aligned with the job type?
- **Capability Evidence**: Does their description suggest they can handle this work?
- **Location Proximity**: If the user specified a location, prioritize vendors in or near that city. Use your knowledge of UK geography to assess proximity - vendors in the same region or nearby cities should rank higher than distant ones.

### Step 3: Rank and Justify
- Assign a relevance score (0.0 to 1.0)
- Provide specific reasoning why this vendor is or isn't suitable
- Be critical - only high-scoring vendors should make the top 10

## Scoring Guidelines

- **0.9-1.0**: Perfect match - services directly address the need, and located in/near user's location (if specified)
- **0.7-0.8**: Strong match - clearly relevant services/industry, reasonably close location
- **0.5-0.6**: Partial match - some relevant capabilities, or good service match but distant location
- **0.3-0.4**: Weak match - tangentially related or very far from user's location
- **0.0-0.2**: Poor match - not relevant to the request

## Location Ranking Rule

When the user mentions a location, use it as a **tie-breaker and score modifier**:
- Among vendors with similar service relevance, rank those closer to the user's location higher
- A vendor with excellent services but far away should score lower than one with excellent services nearby
- For example: If user is in Tadcaster (Yorkshire), prefer vendors in Leeds, York, Harrogate over those in London or Bristol

## Output Format

Return a JSON object with your analysis and rankings. IMPORTANT: Use the candidate_id (the number shown as "ID:" for each candidate) to identify vendors - do NOT rely on company names as they may be normalized differently.

{{
  "user_need_analysis": "Brief description of what the user actually needs",
  "required_service_types": ["list", "of", "service", "types"],
  "rankings": [
    {{
      "rank": 1,
      "candidate_id": 0,
      "relevance_score": 0.95,
      "reasoning": "Step-by-step reasoning: 1) User needs X. 2) This vendor provides X service. 3) Their industry (Y) aligns perfectly. 4) They have relevant certifications. Therefore, excellent match."
    }},
    {{
      "rank": 2,
      "candidate_id": 3,
      "relevance_score": 0.82,
      "reasoning": "Detailed reasoning for this ranking..."
    }}
  ]
}}

## Important Notes

- Return ONLY the top {top_k} most relevant vendors
- If fewer than {top_k} are relevant (score > 0.3), return only those that are relevant
- Your reasoning should be specific and reference actual vendor details
- Be honest - if no vendors are good matches, say so in your analysis

Now analyze the candidates and provide your rankings. Return ONLY valid JSON.
'''
