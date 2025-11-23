# Google Generative AI Embeddings with LangChain

> Source: https://docs.langchain.com/oss/python/integrations/text_embedding/google_generative_ai

## Overview
Google Generative AI Embeddings integrates with LangChain through the `langchain-google-genai` package, enabling access to Google's embedding models like Gemini via the `GoogleGenerativeAIEmbeddings` class.

## Setup Requirements

**Installation:**
```bash
pip install -qU langchain-google-genai
```

**API Key Configuration:**
```python
import os
import getpass

os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter your Google API key: ")
```

## Basic Usage

**Single Query Embedding:**
```python
from langchain_google_genai import GoogleGenerativeAIEmbeddings

embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
vector = embeddings.embed_query("hello, world!")
print(len(vector))  # 3072 dimensions
```

**Batch Processing:**
```python
vectors = embeddings.embed_documents([
    "Today is Monday",
    "Today is Tuesday",
    "Today is April Fools day",
])
print(len(vectors))  # 3 vectors
print(len(vectors[0]))  # 3072 dimensions each
```

## Task Types

The library supports specialized task types for optimization:

| Task Type | Use Case |
|-----------|----------|
| `RETRIEVAL_DOCUMENT` | Embedding documents for search (default for embed_documents) |
| `RETRIEVAL_QUERY` | Embedding queries for search (default for embed_query) |
| `SEMANTIC_SIMILARITY` | Text similarity assessment |
| `CLASSIFICATION` | Preset label classification |
| `CLUSTERING` | Similarity-based grouping |
| `CODE_RETRIEVAL_QUERY` | Natural language code retrieval |
| `QUESTION_ANSWERING` | Q&A optimized retrieval |
| `FACT_VERIFICATION` | Fact-checking retrieval |

**Custom Task Type:**
```python
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    task_type="SEMANTIC_SIMILARITY"
)
```

## Embedding Dimensions

The Gemini embedding model uses Matryoshka Representation Learning (MRL):
- Default output: 3072 dimensions
- Can truncate to: 768, 1536, or 3072 without quality loss
- Smaller dimensions save storage space

```python
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    output_dimensionality=768  # Reduced dimensions
)
```

## Integration with Vector Stores

```python
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore

embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

text = "LangChain is the framework for building context-aware reasoning applications"
vectorstore = InMemoryVectorStore.from_texts([text], embedding=embeddings)

retriever = vectorstore.as_retriever()
retrieved_documents = retriever.invoke("What is LangChain?")
```

## Production Considerations

- **In-memory stores** are ephemeral and only for demos
- For production, use dedicated vector databases:
  - ChromaDB
  - Pinecone
  - Qdrant
  - Milvus
  - FAISS

## Batch API
For high throughput at reduced cost:
- Use Gemini Embeddings with Batch API
- 50% of interactive embedding pricing
- Higher latency but much higher throughput
