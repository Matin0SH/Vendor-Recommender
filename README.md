# Vendor Recommender System

An intelligent vendor recommendation system powered by **LangGraph** and **Google Gemini**. Given a natural language job request, the system finds and ranks the most suitable vendors from a database of 500+ UK businesses.

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [How It Works](#how-it-works)
- [Data Pipeline](#data-pipeline)
- [API Reference](#api-reference)
- [Troubleshooting](#troubleshooting)

---

## Overview

This system helps users find contractors and service providers by understanding their natural language requests and matching them against a vendor database using semantic search and LLM-powered ranking.

**Key Features:**
- Natural language query understanding
- Semantic vector search using Gemini embeddings
- Chain-of-Thought reasoning for intelligent ranking
- Location-aware recommendations (UK geography)
- Detailed explanations for each recommendation

**Example:**
```
User: "I need to fix my bathroom pipe. My pub is in Tadcaster"

System returns:
#1 - GB GROUP (CORPORATE) LIMITED
    Services: Plumbing, Electrical installation...
    Location: Leeds (nearby)
    Reasoning: User needs plumbing services. This vendor provides plumbing
               and is located in Leeds, which is close to Tadcaster...
```

---

## Architecture

### System Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         VENDOR RECOMMENDER SYSTEM                           │
└─────────────────────────────────────────────────────────────────────────────┘

                              User Query
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            LANGGRAPH WORKFLOW                               │
│                                                                             │
│  ┌──────────────┐      ┌──────────────┐      ┌──────────────┐              │
│  │              │      │              │      │              │              │
│  │   EXTRACT    │─────▶│   RETRIEVE   │─────▶│   RERANK     │              │
│  │    NODE      │      │    NODE      │      │    NODE      │              │
│  │              │      │              │      │              │              │
│  └──────────────┘      └──────────────┘      └──────────────┘              │
│         │                     │                     │                       │
│         ▼                     ▼                     ▼                       │
│  ┌──────────────┐      ┌──────────────┐      ┌──────────────┐              │
│  │ Gemini LLM   │      │  ChromaDB    │      │ Gemini LLM   │              │
│  │ (Extraction) │      │ Vector Store │      │ (Reasoning)  │              │
│  └──────────────┘      └──────────────┘      └──────────────┘              │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
                         Ranked Recommendations
                         (Top 10 with reasoning)
```

### Node Details

```
┌────────────────────────────────────────────────────────────────────────────┐
│                              EXTRACT NODE                                  │
├────────────────────────────────────────────────────────────────────────────┤
│  Input:  "I need to fix my bathroom pipe in Tadcaster"                     │
│                                                                            │
│  Process:                                                                  │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  Gemini LLM analyzes query and extracts:                            │   │
│  │  • job_type: "plumbing"                                             │   │
│  │  • services_needed: ["pipe repair", "bathroom plumbing"]            │   │
│  │  • location: "Tadcaster"                                            │   │
│  │  • urgency: "normal"                                                │   │
│  │  • optimized_query: "pipe repair bathroom plumbing Tadcaster"       │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                            │
│  Output: Structured extraction + optimized search query                    │
└────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────────┐
│                              RETRIEVE NODE                                 │
├────────────────────────────────────────────────────────────────────────────┤
│  Input:  Optimized query from Extract node                                 │
│                                                                            │
│  Process:                                                                  │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  ChromaDB Vector Search:                                            │   │
│  │  1. Embed query using Gemini embeddings                             │   │
│  │  2. Find 30 most similar vendors (cosine similarity)                │   │
│  │  3. Return candidates with metadata                                 │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                            │
│  Output: 30 candidate vendors with similarity scores                       │
└────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────────┐
│                              RERANK NODE                                   │
├────────────────────────────────────────────────────────────────────────────┤
│  Input:  Original query + 30 candidates                                    │
│                                                                            │
│  Process:                                                                  │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  Gemini LLM with Chain-of-Thought reasoning:                        │   │
│  │  1. Analyze what user actually needs                                │   │
│  │  2. Evaluate each vendor's service match                            │   │
│  │  3. Consider location proximity (UK geography)                      │   │
│  │  4. Score 0.0-1.0 with detailed reasoning                           │   │
│  │  5. Return only vendors scoring > 0.3                               │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                            │
│  Output: Top 10 ranked vendors with explanations                           │
└────────────────────────────────────────────────────────────────────────────┘
```

### State Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           GraphState                                    │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  original_query ─────────────────────────────────────────────────────┐  │
│       │                                                              │  │
│       ▼                                                              │  │
│  ┌─────────┐     extracted_info                                      │  │
│  │ EXTRACT │────────────────────┐                                    │  │
│  └─────────┘                    │                                    │  │
│                                 ▼                                    │  │
│                          ┌──────────┐     candidates (30)            │  │
│                          │ RETRIEVE │─────────────────┐              │  │
│                          └──────────┘                 │              │  │
│                                                       ▼              │  │
│                                                 ┌─────────┐          │  │
│                                                 │ RERANK  │◀─────────┘  │
│                                                 └─────────┘             │
│                                                       │                 │
│                                                       ▼                 │
│                                              ranked_vendors (≤10)       │
│                                                                         │
│  error ─────────────────────────────────────────────────────────────────│
│  (populated if any node fails)                                          │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
vendor_recomm/
│
├── config.py                 # Central configuration (API keys, models, prompts)
├── requirements.txt          # Python dependencies
├── .env                      # Environment variables (GOOGLE_API_KEY)
│
├── run_recommender.py        # Main entry point - run recommendations
├── run_preprocessing.py      # Data preprocessing pipeline
│
├── graph/                    # LangGraph workflow
│   ├── __init__.py
│   ├── state.py              # State definitions & Pydantic models
│   ├── workflow.py           # Graph construction & execution
│   └── nodes/
│       ├── __init__.py
│       ├── extract.py        # Node 1: Query extraction (LLM)
│       ├── retrieve.py       # Node 2: Vector search (ChromaDB)
│       └── rerank.py         # Node 3: LLM reranking (CoT)
│
├── preprocessing/            # Data preparation
│   ├── __init__.py
│   ├── preprocess.py         # Combine text fields for embedding
│   └── embeddings.py         # Create embeddings & index to ChromaDB
│
├── output/                   # Data files
│   ├── all_results.json      # Raw vendor data (~500 vendors)
│   └── vendors_processed.json # Processed for embedding
│
├── chroma_db/                # Persisted vector store
│   └── ...
│
└── docs/                     # Reference documentation
    ├── 01_langgraph_semantic_search.md
    ├── 02_langchain_rag_tutorial.md
    └── ...
```

---

## Installation

### Prerequisites

- Python 3.10+
- Google Cloud account with Gemini API access

### Steps

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd vendor_recomm
   ```

2. **Create virtual environment:**
   ```bash
   python -m venv myenv

   # Windows
   myenv\Scripts\activate

   # Linux/Mac
   source myenv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables:**

   Create a `.env` file in the project root:
   ```env
   GOOGLE_API_KEY=your-gemini-api-key-here
   ```

5. **Initialize the vector store (first time only):**
   ```bash
   python run_preprocessing.py
   ```

---

## Configuration

All configuration is centralized in `config.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `EMBEDDING_MODEL` | `models/gemini-embedding-001` | Gemini embedding model |
| `EMBEDDING_DIMENSIONS` | `3072` | Embedding vector size |
| `LLM_MODEL` | `gemini-2.0-flash` | LLM for extraction/reranking |
| `LLM_TEMPERATURE` | `0.0` | Deterministic outputs |
| `TOP_K_RETRIEVAL` | `30` | Candidates from vector search |
| `TOP_K_RERANK` | `10` | Final recommendations |
| `CHROMA_PERSIST_DIR` | `chroma_db` | Vector store location |

---

## Usage

### Interactive Mode

```bash
python run_recommender.py
```

Then enter your queries:
```
Enter your job request:
> I need someone to install fire sprinklers in my warehouse in Leeds

[System processes and returns ranked vendors...]

Enter your job request:
> quit
```

### Single Query Mode

```bash
python run_recommender.py "I need a plumber to fix a burst pipe urgently"
```

### Preprocessing (Rebuild Index)

```bash
# Normal run (incremental update)
python run_preprocessing.py

# Reset and rebuild entire index
python run_preprocessing.py --reset-index

# Skip deduplication
python run_preprocessing.py --no-dedup
```

---

## How It Works

### 1. Extract Node

The Extract node uses Gemini LLM to parse the user's natural language query:

```python
# Input
"Emergency! Water pipe burst in my restaurant kitchen in Leeds"

# Output
{
    "job_type": "plumbing",
    "services_needed": ["emergency plumbing", "pipe repair", "commercial plumbing"],
    "location": "Leeds",
    "urgency": "urgent",
    "optimized_query": "emergency plumbing pipe repair commercial kitchen Leeds"
}
```

### 2. Retrieve Node

Vector similarity search against 500+ indexed vendors:

```
Query embedding ──┐
                  │
                  ▼
            ┌──────────┐
            │ ChromaDB │ ──▶ Top 30 similar vendors
            │  Index   │     (by cosine similarity)
            └──────────┘
```

### 3. Rerank Node

LLM-powered intelligent ranking with Chain-of-Thought reasoning:

**Scoring Guidelines:**
| Score | Meaning |
|-------|---------|
| 0.9-1.0 | Perfect match - services + location aligned |
| 0.7-0.8 | Strong match - relevant services, reasonable location |
| 0.5-0.6 | Partial match - some capabilities or distant location |
| 0.3-0.4 | Weak match - tangentially related |
| 0.0-0.2 | Poor match - not relevant |

**Location Awareness:**
- Vendors near the user's location rank higher
- Uses UK geography knowledge (e.g., Tadcaster → prefer Leeds, York)

---

## Data Pipeline

### Preprocessing Flow

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  all_results    │     │  preprocess.py  │     │   embeddings    │
│    .json        │────▶│  Combine text   │────▶│     .py         │
│  (raw vendors)  │     │    fields       │     │  Create vectors │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                                                        │
                                                        ▼
                                                ┌─────────────────┐
                                                │   ChromaDB      │
                                                │  Vector Store   │
                                                └─────────────────┘
```

### Vendor Data Structure

**Raw data (`all_results.json`):**
```json
{
    "vendor": "Armstrong-Priestley Ltd.",
    "company_name": "ARMSTRONG-PRIESTLEY LIMITED",
    "status": "success",
    "extracted": {
        "services": "Fire Sprinklers, Fire Suppression...",
        "industry": "Public Safety",
        "city": "Leeds",
        "phone": "0113 394 4040",
        "website": "https://www.armstrongpriestley.co.uk/"
    }
}
```

**Processed for embedding (`vendors_processed.json`):**
```json
{
    "id": "1",
    "text": "Company: ARMSTRONG-PRIESTLEY LIMITED\nServices: Fire Sprinklers...\nLocation: Leeds",
    "metadata": {
        "company_name": "ARMSTRONG-PRIESTLEY LIMITED",
        "services": "Fire Sprinklers...",
        "city": "Leeds"
    }
}
```

---

## API Reference

### Core Functions

#### `run_recommendation(query: str) -> dict`

Run the full recommendation pipeline.

```python
from graph.workflow import run_recommendation

result = run_recommendation("I need a plumber in Leeds")

# Result structure:
{
    "original_query": "I need a plumber in Leeds",
    "extracted_info": {...},
    "candidates": [...],  # 30 candidates
    "ranked_vendors": [...],  # Top 10 ranked
    "error": None
}
```

#### `create_graph() -> StateGraph`

Create the LangGraph workflow.

```python
from graph.workflow import create_graph

graph = create_graph()
compiled = graph.compile()
```

### State Types

```python
from graph.state import GraphState, RankedVendor, VendorCandidate

# GraphState fields:
# - original_query: str
# - extracted_info: ExtractedInfo | None
# - candidates: list[VendorCandidate] | None
# - ranked_vendors: list[RankedVendor] | None
# - error: str | None
```

---

## Troubleshooting

### Common Issues

**1. "GOOGLE_API_KEY not found"**
```bash
# Ensure .env file exists with:
GOOGLE_API_KEY=your-api-key
```

**2. "Vector store not found"**
```bash
# Run preprocessing first:
python run_preprocessing.py
```

**3. "Pydantic validation failed" (candidate_id type)**

This was fixed - the system now coerces integer IDs to strings automatically.

**4. "Reranking parse failed, using similarity fallback"**

The LLM response couldn't be parsed. The system falls back to similarity-based ranking. Check:
- API rate limits
- Response format in logs

**5. Low quality recommendations**

Try:
- Increase `TOP_K_RETRIEVAL` in config.py (e.g., 50)
- Check if vendor data covers the requested service type
- Review the extraction output to ensure query is understood correctly

### Debug Mode

Add print statements or check node outputs:

```python
result = run_recommendation("your query")

print("Extracted:", result["extracted_info"])
print("Candidates:", len(result["candidates"]))
print("Ranked:", len(result["ranked_vendors"]))
print("Error:", result["error"])
```

---

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `langchain` | ≥0.3.0 | LLM framework |
| `langchain-google-genai` | ≥2.0.0 | Gemini integration |
| `langchain-chroma` | ≥0.1.0 | Vector store integration |
| `chromadb` | ≥0.5.0 | Vector database |
| `python-dotenv` | ≥1.0.0 | Environment variables |
| `pydantic` | (included) | Data validation |

---


