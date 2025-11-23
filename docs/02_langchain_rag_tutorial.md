# Building RAG Applications with LangChain

> Source: https://python.langchain.com/docs/tutorials/retrievers/

## Overview
This tutorial demonstrates constructing question-answering systems using Retrieval Augmented Generation (RAG). The approach combines a technique known as Retrieval Augmented Generation to enable applications that answer queries about specific source materials.

## Architecture & Core Concepts

### Three-Phase Pipeline

**Indexing Phase:**
1. Load data via Document Loaders (e.g., WebBaseLoader for web content)
2. Split documents using text splitters like RecursiveCharacterTextSplitter
3. Store embeddings in vector databases for semantic search

**Retrieval & Generation Phase:**
- Execute similarity searches against indexed documents
- Pass retrieved context to language models
- Generate answers informed by source material

### Two Implementation Approaches

**1. Agentic RAG**
Uses a tool-based architecture where the LLM decides when to search. The agent can make multiple sequential searches to gather comprehensive context before answering.

Benefits: Flexible search decisions, contextual query generation, multiple searches per query
Trade-offs: Requires two inference calls, less deterministic behavior

**2. Two-Step Chain**
Implements a simpler pattern executing one search and one LLM inference per query.

Benefits: Single inference call, predictable latency
Trade-offs: Always performs searches regardless of need

## Implementation Example

The tutorial provides code for indexing a blog post (~43,000 characters) by splitting it into 66 retrievable chunks, then creating retrieval tools that return both serialized content and metadata.

```python
@tool(response_format="content_and_artifact")
def retrieve_context(query: str):
    """Retrieve information to help answer a query."""
    retrieved_docs = vector_store.similarity_search(query, k=2)
    # Returns formatted content plus document artifacts
```

## Setup Requirements

- LangChain ecosystem packages (langchain, langchain-text-splitters, langchain-community)
- Embeddings model (OpenAI, Anthropic, Google, etc.)
- Vector store (in-memory, Chroma, Pinecone, Qdrant, etc.)
- Chat model (Claude, GPT-4, Gemini, etc.)

The tutorial emphasizes that complete implementations can be achieved in approximately 40 lines of code.
