# LangGraph Multi-Intent AI Agent

Build a multi-intent AI agent using LangGraph that routes user messages to specialized sub-agents based on intent. A classifier node dispatches each message to a chat agent, a RAG knowledge agent, or a Claude Code coding agent — with a human-in-the-loop approval step before any code changes are made.

## Steps I Did

- Scaffolded the project with `uv` and configured dependencies including `langchain`, `langgraph`, and `langchain-google-genai`
- Defined a custom `State` TypedDict extending LangGraph's message state with `message_intent` and `next_node` fields for routing
- Built a classifier node that uses Claude with structured output (`IntentClassifier` Pydantic model) to label each message as `chat`, `knowledge`, or `code`
- Implemented a RAG agent using `InMemoryVectorStore` with Google Gemini embeddings to answer questions from a local knowledge base
- Added a `prepare_coding_request` node that rewrites the user's message into a precise instruction for Claude Code using conversation history as context
- Wired a human-in-the-loop `accept_coding` node using LangGraph's `interrupt()` — the user can approve, deny, or revise the request before execution
- Invoked the Claude Code CLI as a subprocess (`claude -p ... --permission-mode acceptEdits`) inside a sandboxed `workspace/` directory
- Connected all nodes with conditional edges and compiled the graph with `InMemorySaver` for per-session conversation memory
- Exported a Mermaid graph diagram to `graph.png` for visualization

## What I Learned

- How to build stateful multi-intent pipelines in LangGraph using `StateGraph`, `add_messages`, and conditional edges
- Using `with_structured_output()` to enforce schema-validated responses from an LLM for reliable routing
- Implementing retrieval-augmented generation (RAG) with `InMemoryVectorStore` and embedding models
- How LangGraph's `interrupt()` enables human-in-the-loop flows — pausing graph execution mid-run and resuming with `Command(resume=...)`
- Chaining a preparation node before an approval node to give the user a clean, LLM-refined version of their request to review
- Spawning Claude Code as a subprocess to act as an autonomous coding agent within a scoped workspace directory
