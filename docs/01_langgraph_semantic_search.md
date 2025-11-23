# LangGraph Semantic Search Tutorial

> Source: https://langchain-ai.github.io/langgraph/how-tos/memory/semantic-search/

## Overview
This guide demonstrates enabling semantic search capabilities in agent memory stores, allowing retrieval of items based on semantic similarity rather than exact matches.

## Core Setup

**Installation Requirements:**
```bash
pip install -U langgraph langchain-openai langchain
```

**Key Configuration:**
Create a store with an index configuration using embeddings. By default, stores are configured without semantic/vector search.

```python
from langchain.embeddings import init_embeddings
from langgraph.store.memory import InMemoryStore

embeddings = init_embeddings("openai:text-embedding-3-small")
store = InMemoryStore(
    index={
        "embed": embeddings,
        "dims": 1536,
    }
)
```

## Basic Operations

**Storing Memories:**
Use `store.put()` to save items with automatic semantic indexing enabled.

**Searching:**
Execute natural language queries against stored memories:
```python
memories = store.search(
    ("user_123", "memories"),
    query="I like food?",
    limit=5
)
```

## Integration Patterns

### Agent Integration
Inject the store into graph nodes to enable contextual memory retrieval before LLM invocations.

### React Agent Pattern
Leverage the `prompt` function parameter in `create_react_agent` to inject relevant memories, and create tools allowing agents to store new memories autonomously.

## Advanced Techniques

**Multi-Vector Indexing:** Index multiple fields separately (e.g., "memory" and "emotional_context") to improve recall precision.

**Field Override:** Specify which fields to embed per-memory using `put(..., index=["field_name"])`.

**Disable Indexing:** Prevent specific memories from semantic indexing using `put(..., index=False)` for system records.
