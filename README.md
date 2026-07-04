# Real-Time Voice RAG Agent

Adapter-based local real-time Voice RAG system with a deterministic offline core, streaming APIs, document-grounded retrieval, live speech integrations, latency/evaluation gates, Docker readiness checks, and strict static quality enforcement.

The project is designed as an AI/backend systems project: the same RAG engine powers CLI queries, FastAPI endpoints, streaming NDJSON, MCP/A2A protocol surfaces, simulated voice turns, and a real microphone-to-speaker voice demo.

## Highlights

- Real voice path: microphone → AssemblyAI streaming ASR → retrieval → Ollama/Gemma or OpenAI reasoning → Cartesia TTS → local speaker.
- Deterministic offline mode for tests, CI, benchmarks, evals, and demos.
- Streaming event model shared across CLI, HTTP, voice, MCP, and A2A.
- Persistent document index with source fingerprinting and rebuild/reload support.
- BM25 and lexical-hybrid retrieval with diagnostics, reranking, retrieval recall, and MRR.
- FastAPI production hardening: API key auth, request IDs, structured errors, upload validation, rate limiting, `/health`, and `/ready`.
- Docker and Docker Compose deployment with readiness healthchecks.
- Benchmark, eval, CPU profile, and memory profile gates.
- Strict quality wall: `compileall`, Ruff format, Ruff lint, strict mypy, pytest, benchmark gate, eval gate, and profile gate.

## Architecture

```mermaid
flowchart LR
    Mic["Microphone / Voice Input"] --> ASR["AssemblyAI Streaming ASR"]
    ASR --> Events["Unified JSON Event Stream"]
    Events --> RAG["RAG Engine"]

    Docs["Documents: MD / TXT / PDF / DOCX / HTML / URL"] --> Index["Persistent Index"]
    Index --> Retriever["BM25 / Lexical-Hybrid Retriever"]
    Retriever --> Reranker["Optional Lexical Reranker"]
    Reranker --> RAG

    RAG --> Reasoner["Template / Ollama-Gemma / OpenAI"]
    Reasoner --> Stream["Answer Delta Stream"]
    Stream --> TTS["Cartesia TTS"]
    TTS --> Speaker["Local Speaker"]

    RAG --> API["FastAPI /query and /query/stream"]
    RAG --> MCP["MCP Tool"]
    RAG --> A2A["A2A Agent Surface"]
    RAG --> Traces["JSONL Traces + OTEL-shaped Export"]
    RAG --> Metrics["Benchmarks / Evals / Profiling"]
