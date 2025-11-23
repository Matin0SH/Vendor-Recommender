# Agentic RAG with LangGraph and Qdrant

> Source: https://qdrant.tech/documentation/agentic-rag-langgraph/

## Overview
This tutorial demonstrates building an intelligent agent system that combines document retrieval with web search capabilities using LangGraph and Qdrant.

## Key Concept
Traditional RAG follows a simple path: "query -> retrieve -> generate." Agentic RAG enhances this by introducing AI agents that can orchestrate multiple retrieval steps and intelligently decide which data sources to use.

## Architecture Components

**The System Stack:**
- **AI Agent**: OpenAI's GPT-4o processes queries and selects appropriate tools
- **Embeddings**: OpenAI's text-embedding-3-small model
- **Vector Database**: Qdrant stores and retrieves document embeddings
- **Search Tools**: BraveSearch API for web queries
- **Workflow Engine**: LangGraph manages orchestration

## Workflow Steps

The agent follows this sequence:
1. User submits a query
2. Agent analyzes the request
3. Agent selects relevant tools (documentation retriever or web search)
4. Tools execute and return results
5. Agent generates contextual response
6. Response delivered to user

## Core Implementation Elements

**State Management:**
```python
class State(TypedDict):
    messages: Annotated[list, add_messages]
```

**Tool Creation:**
Three tools power the agent:
- Hugging Face documentation retriever
- Transformers documentation retriever
- Web search functionality

**Tool Integration:**
The ToolNode orchestrates tool execution by mapping tool names to functions, processing LLM tool calls, and formatting results as messages.

**Routing Logic:**
A routing function determines whether the agent should invoke tools or conclude the workflow based on tool_calls presence.

## Graph Construction

```python
from langgraph.graph import StateGraph, END

graph_builder = StateGraph(State)
graph_builder.add_node("agent", agent)
graph_builder.add_node("tools", tool_node)

graph_builder.add_conditional_edges(
    "agent",
    route_tools,
    {"tools": "tools", END: END}
)
graph_builder.add_edge("tools", "agent")
graph_builder.set_entry_point("agent")

graph = graph_builder.compile()
```

## Practical Example
When querying "Are there multilingual models in Transformers?", the agent retrieves specific documentation about models like BERT Multilingual, XLM, M2M100, and MBart.
