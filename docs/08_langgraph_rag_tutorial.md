# LangGraph RAG Tutorial: Complete Overview

> Source: https://docs.langchain.com/oss/python/langgraph/agentic-rag

## Core Architecture

This tutorial demonstrates building a retrieval agent using LangGraph that decides whether to retrieve context or respond directly. The system uses `MessagesState` to manage conversation history and implements conditional routing based on document relevance.

## Key Components

**State Management:**
The implementation relies on `MessagesState`, which maintains a list of chat messages that flow through the graph.

**Node Functions:**

1. **generate_query_or_respond** - Call the model to generate a response based on the current state using the retriever tool via `.bind_tools()`

2. **grade_documents** - Evaluates retrieved documents using structured output with a `GradeDocuments` schema returning "yes" or "no" relevance scores

3. **rewrite_question** - Improves the original user question when documents fail relevance checks

4. **generate_answer** - Produces final responses using retrieved context with a three-sentence maximum constraint

## Graph Construction

```python
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode, tools_condition

workflow = StateGraph(MessagesState)

# Add nodes
workflow.add_node(generate_query_or_respond)
workflow.add_node("retrieve", ToolNode([retriever_tool]))
workflow.add_node(rewrite_question)
workflow.add_node(generate_answer)

# Add edges
workflow.add_edge(START, "generate_query_or_respond")
workflow.add_conditional_edges(
    "generate_query_or_respond",
    tools_condition,
    {"tools": "retrieve", END: END}
)
workflow.add_conditional_edges("retrieve", grade_documents)
workflow.add_edge("generate_answer", END)
workflow.add_edge("rewrite_question", "generate_query_or_respond")

# Compile
graph = workflow.compile()
```

## Processing Pipeline

The workflow executes this sequence:
1. Initial query routing (retrieve or respond)
2. Document retrieval via semantic search
3. Relevance grading with routing to answer generation or question rewriting
4. Final response generation or question refinement loop

## Workflow Diagram

```
┌───────────────────────────────┐
│           START               │
└──────────────┬────────────────┘
               │
               ▼
┌───────────────────────────────┐
│  generate_query_or_respond    │
│  (decide: retrieve or answer) │
└──────────────┬────────────────┘
               │
       ┌───────┴───────┐
       ▼               ▼
   ┌────────┐      ┌───────┐
   │retrieve│      │  END  │
   └───┬────┘      └───────┘
       │
       ▼
┌───────────────────────────────┐
│      grade_documents          │
│   (relevant? yes/no)          │
└──────────────┬────────────────┘
               │
       ┌───────┴───────┐
       ▼               ▼
┌──────────────┐  ┌─────────────────┐
│generate_answer│  │rewrite_question │
└──────┬───────┘  └────────┬────────┘
       │                   │
       ▼                   │
   ┌───────┐               │
   │  END  │               │
   └───────┘               │
                           │
       ┌───────────────────┘
       │ (loop back)
       ▼
┌───────────────────────────────┐
│  generate_query_or_respond    │
└───────────────────────────────┘
```
