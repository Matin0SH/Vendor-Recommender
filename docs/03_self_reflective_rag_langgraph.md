# Self-Reflective RAG with LangGraph

> Source: https://blog.langchain.com/agentic-rag-with-langgraph/

## Overview
This article explains how to implement self-correcting RAG systems using LangGraph, a framework for building LLM state machines.

## Core Motivation
Standard RAG systems embed queries, retrieve documents, and generate answers. However, most LLMs are only periodically trained on a large corpus of public data, they lack recent information and proprietary data. Self-reflective RAG addresses quality issues through automated reasoning and correction loops.

## Key Architectural Approaches

**Cognitive Architectures:**
The article identifies three patterns:
- **Chains**: Simple sequential processing
- **Routing**: LLM-driven selection between alternatives
- **State machines**: Support loops and conditional transitions

## Two Implementation Examples

### Corrective RAG (CRAG)
Features include:
- Lightweight retrieval evaluator to assess the overall quality of retrieved documents
- Web search supplementation when vectorstore retrieval is inadequate
- Knowledge refinement through document partitioning

### Self-RAG
Uses reflection tokens governing retrieval and generation:
- **Retrieve token**: Decides whether to fetch documents
- **ISREL token**: Assesses document relevance
- **ISSUP token**: Validates generation support
- **ISUSE token**: Rates response usefulness

## Implementation Benefits
LangGraph enables flexible workflow design with explicit decision points and feedback loops, reducing complexity in implementing sophisticated RAG patterns while maintaining auditability.

## State Machine Architecture

```
┌─────────┐     ┌─────────────┐     ┌──────────┐
│ Retrieve │────▶│ Grade Docs  │────▶│ Generate │
└─────────┘     └──────┬──────┘     └────┬─────┘
                       │                  │
                       ▼                  ▼
                  Not Relevant?      Hallucination?
                       │                  │
                       ▼                  ▼
                ┌─────────────┐    ┌─────────────┐
                │ Rewrite Query│    │ Re-generate │
                └─────────────┘    └─────────────┘
```

## Resources
The article references implementation notebooks for both CRAG and Self-RAG approaches on the LangGraph GitHub repository.
