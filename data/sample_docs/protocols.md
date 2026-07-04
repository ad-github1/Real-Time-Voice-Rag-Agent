# Agent Protocols

The MCP interface exposes document question answering as a tool named `query_documents`. MCP clients can call the tool with a question and receive a grounded answer, trace id, and citations.

The A2A interface publishes an agent card and supports task-style question answering. The synchronous endpoint returns a completed answer artifact, while the streaming endpoint emits the same NDJSON events used by the core RAG engine.

The protocol layer is intentionally thin. It delegates retrieval, reasoning, tracing, and citation formatting to the shared RAG engine so that HTTP, MCP, A2A, CLI, and voice transports all behave consistently.
