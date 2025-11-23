# Cross Encoder Reranker in LangChain

> Source: https://python.langchain.com/docs/integrations/document_transformers/cross_encoder_reranker/

## Overview
Cross Encoder Reranker is a technique to enhance RAG system performance by using HuggingFace Cross Encoders to refine the ranking of retrieved documents.

## Why Cross Encoders?

**Bi-Encoder (Embedding Models):**
- Encodes query and documents separately
- Fast: O(1) per document after pre-computation
- Less accurate for fine-grained relevance

**Cross-Encoder:**
- Performs full attention over query-document pairs
- More accurate than bi-encoders
- Slower: requires inference for each query-document pair
- Best used to re-rank top-k results from initial retrieval

## Recommended Models

| Model | Description |
|-------|-------------|
| `BAAI/bge-reranker-base` | Good balance of speed and accuracy |
| `BAAI/bge-reranker-large` | More powerful, slower |
| `cross-encoder/ms-marco-MiniLM-L-6-v2` | Lightweight option |

## Installation

```bash
pip install langchain langchain-community sentence-transformers
```

## Implementation

**Basic Setup:**
```python
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

# Initialize cross encoder
model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")

# Create reranker compressor
compressor = CrossEncoderReranker(model=model, top_n=3)

# Wrap base retriever with compression
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=base_retriever  # Your vector store retriever
)

# Use the compression retriever
results = compression_retriever.invoke("What is the plan for the economy?")
```

## Pipeline Architecture

```
User Query
    │
    ▼
┌─────────────────┐
│ Vector Store    │  Initial retrieval (fast, k=20)
│ Retriever       │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Cross Encoder   │  Re-rank results (accurate, top_n=5)
│ Reranker        │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ LLM Generation  │  Generate answer from top results
└─────────────────┘
```

## Best Practices

1. **Retrieve more, rerank fewer**: Initial retrieval k=20-50, rerank to top_n=3-5
2. **Balance latency vs accuracy**: Cross encoders add latency but improve relevance
3. **Consider model size**: Larger models are more accurate but slower
4. **Cache when possible**: Reranking same documents for similar queries

## Note on Google Gemini
There is no specific Google Gemini reranker in LangChain. Use HuggingFace cross encoders (like BAAI/bge-reranker) for reranking with Gemini embeddings for initial retrieval.
