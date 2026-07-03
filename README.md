# Real-Time Voice RAG Agent

A working real-time Voice RAG project that turns static document search into natural voice conversation. The local demo connects microphone input, AssemblyAI real-time transcription, document-grounded RAG retrieval, Ollama/Gemma reasoning, and Cartesia text-to-speech playback through one streaming JSON event pipeline.

The core can also run offline for tests using deterministic local components. Provider adapters are kept separate for LiveKit room transport, AssemblyAI transcription, Ollama/Gemma or OpenAI reasoning, Cartesia TTS, MCP tools, and A2A task interoperability.

## Real Voice Demo

This project supports a working local voice RAG demo:

Mic → AssemblyAI Streaming ASR → RAG retrieval → Ollama/Gemma reasoning → Cartesia TTS → Mac speaker output

### Run locally

```bash
cp .env.example .env
# Fill ASSEMBLYAI_API_KEY and CARTESIA_API_KEY in .env

./scripts/run_voice_live.sh
```

Expected event stream:

```text
asr.transcript
retrieval.completed
retrieval.chunk
answer.delta
tts.audio
answer.completed
```

The live demo uses:

- AssemblyAI for real-time microphone transcription.
- BM25-based local RAG retrieval over sample documents.
- Ollama/Gemma for local reasoning.
- Cartesia for text-to-speech playback.
- ffplay/FFmpeg for local audio output.

## What It Does

- Streams one JSON event envelope across CLI, HTTP, A2A, MCP, and voice transports.
- Retrieves grounded chunks from local documents with deterministic BM25 for tests.
- Ingests Markdown, plain text, PDF, DOCX, HTML/HTM, and web URLs from `urls.txt`.
- Supports optional LlamaIndex retrieval for production vector or hybrid search.
- Streams model deltas from Ollama's local JSON API or OpenAI's Responses API.
- Runs a real local voice path with AssemblyAI ASR and Cartesia TTS.
- Keeps ASR, retrieval, reasoning, TTS, tracing, and transport as separate adapters.
- Records trace JSONL files for latency analysis and eval debugging.

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
PYTHONPATH=src python3 -m voice_rag_agent.cli query "How does the agent keep latency low?"
```

Stream the same turn as NDJSON:

```bash
PYTHONPATH=src python3 -m voice_rag_agent.cli query "What do MCP and A2A expose?" --stream
```

Simulate a voice turn without real ASR/TTS providers:

```bash
PYTHONPATH=src python3 -m voice_rag_agent.cli voice-demo "What services are used for transcription and TTS?"
```

Run the real local voice demo after setting `.env`:

```bash
./scripts/run_voice_live.sh
```

Run evals:

```bash
PYTHONPATH=src python3 -m voice_rag_agent.cli eval
```

Add this:

```md
Summarize latency metrics from trace files:

```bash
PYTHONPATH=src python3 -m voice_rag_agent.cli metrics
PYTHONPATH=src python3 -m voice_rag_agent.cli metrics --mode voice
PYTHONPATH=src python3 -m voice_rag_agent.cli metrics --json

Run the HTTP API after installing the API extra:

```bash
pip install -e ".[api]"
PYTHONPATH=src python3 -m voice_rag_agent.cli serve --host 127.0.0.1 --port 8000
```

## Configuration

Copy `.env.example` to `.env` and fill only your local secrets in `.env`.

Do not commit `.env`.

The offline default is:

```bash
VOICE_RAG_REASONER=template
VOICE_RAG_DATA_DIR=data/sample_docs
```

For Ollama/Gemma:

```bash
ollama pull gemma3:4b
VOICE_RAG_REASONER=ollama
OLLAMA_MODEL=gemma3:4b
```

For OpenAI, set both values explicitly:

```bash
VOICE_RAG_REASONER=openai
OPENAI_API_KEY=...
OPENAI_MODEL=...
```

For the real voice demo:

```bash
ASSEMBLYAI_API_KEY=...
CARTESIA_API_KEY=...
CARTESIA_VOICE_ID=f786b574-daa5-4673-aa0c-cbe3e8534c02
VOICE_RAG_REASONER=ollama
OLLAMA_MODEL=gemma3:4b
```

## Architecture

```mermaid
flowchart LR
    Mic["Microphone or LiveKit audio"] --> ASR["AssemblyAI ASR"]
    ASR --> Events["Streaming JSON events"]
    Events --> RAG["RAG engine"]
    Docs["Documents"] --> Retriever["BM25 or LlamaIndex retriever"]
    Retriever --> RAG
    RAG --> LLM["Template, Ollama/Gemma, or OpenAI"]
    LLM --> TTS["Cartesia TTS"]
    TTS --> Speaker["Local speaker or LiveKit room"]
    RAG --> Trace["JSONL traces"]
    RAG --> API["HTTP / MCP / A2A"]
```

The core package is dependency-light so tests and demos run locally. Production integrations live behind narrow adapters in `src/voice_rag_agent/integrations`, `src/voice_rag_agent/protocols`, and `src/voice_rag_agent/voice`.

## Project Layout

- `src/voice_rag_agent/rag`: document loading, BM25 cache, prompting, streaming RAG engine.
- `src/voice_rag_agent/llm`: deterministic, Ollama, and OpenAI reasoners.
- `src/voice_rag_agent/voice`: ASR/TTS protocols, VAD, audio sink, and voice orchestration.
- `src/voice_rag_agent/protocols`: A2A and MCP surfaces.
- `src/voice_rag_agent/api.py`: FastAPI query and streaming endpoints.
- `data/sample_docs`: small corpus for demos and tests.
- `scripts/run_voice_live.sh`: one-command local voice demo runner.
- `tests`: standard-library unittest suite.

## Latency Strategy

The agent optimizes perceived latency by triggering retrieval only on final ASR transcripts, caching repeated queries, keeping top-k small, streaming answer deltas, and sending sentence-level chunks to TTS as soon as text is ready.

Trace files are written to `traces/<trace_id>.jsonl` when `VOICE_RAG_ENABLE_TRACING=true`. Each trace records event timings and spans for retrieval and reasoning.

## Hybrid Retrieval and Diagnostics

The project supports both BM25 lexical retrieval and hybrid retrieval.

Run BM25 retrieval diagnostics:

```bash
PYTHONPATH=src python3 -m voice_rag_agent.cli retrieve "What services are used for transcription and TTS?"

```

Run hybrid retrieval diagnostics:

```bash
PYTHONPATH=src python3 -m voice_rag_agent.cli retrieve "What services are used for transcription and TTS?" --retriever hybrid
```

Emit JSON diagnostics:

```bash
PYTHONPATH=src python3 -m voice_rag_agent.cli retrieve "What services are used for transcription and TTS?" --retriever hybrid --json
```

Run generation with a selected retriever:

```bash
PYTHONPATH=src python3 -m voice_rag_agent.cli query "What file types are supported for ingestion?" --retriever bm25 --stream
PYTHONPATH=src python3 -m voice_rag_agent.cli query "What file types are supported for ingestion?" --retriever hybrid --stream
```
Hybrid retrieval combines BM25 lexical scoring with lightweight semantic similarity and score fusion. Diagnostics include BM25 score, semantic score, normalized scores, and final fused score.

## Document Upload API

The FastAPI server supports document upload and index management endpoints.

Start the API server:

```bash
PYTHONPATH=src python3 -m voice_rag_agent.cli serve
```

Check API health:

```bash
curl http://127.0.0.1:8000/health
```

List indexed documents:

```bash
curl http://127.0.0.1:8000/documents
```

Upload a document and rebuild the persistent index automatically:

```bash
curl -X POST "http://127.0.0.1:8000/documents/upload?rebuild=true" \
  -F "file=@/path/to/document.md"
```

Check index stats:

```bash
curl http://127.0.0.1:8000/index/stats
```

Manually rebuild the index:

```bash
curl -X POST http://127.0.0.1:8000/index/rebuild
```

The upload API validates supported file types, sanitizes filenames, applies an upload size limit, saves files under the configured data directory, rebuilds the persistent index, and reloads the RAG engine.

## Docker API Deployment

The FastAPI RAG API can run locally through Docker.

Build the API image:

```bash
docker build -t voice-rag-agent-api:local .
```

Run the API with Docker Compose:

```bash
docker compose up --build api
```

Check health:

```bash
curl http://127.0.0.1:8000/health
```

Ask a question:

```bash
curl -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query":"What services are used for transcription and TTS?"}'
```

Stop the API:

```bash
docker compose down
```

The Docker setup mounts local `data/`, `traces/`, and `outputs/` directories, keeps `.env` out of the image, and exposes the FastAPI server on port `8000`.

## OpenTelemetry Trace Export

The project records local JSONL traces for RAG events and spans. These traces can be exported into an OpenTelemetry-shaped JSON payload for observability workflows.

Run a streamed query to generate trace events:

```bash
PYTHONPATH=src python3 -m voice_rag_agent.cli query "How does the agent keep latency low?" --stream
```

Export traces:

```bash
PYTHONPATH=src python3 -m voice_rag_agent.cli traces export
```

Export with JSON summary:

```bash
PYTHONPATH=src python3 -m voice_rag_agent.cli traces export --json
```

Export to a custom path:

```bash
PYTHONPATH=src python3 -m voice_rag_agent.cli traces export \
  --output outputs/custom_otel_traces.json
```

Export only one trace:

```bash
PYTHONPATH=src python3 -m voice_rag_agent.cli traces export \
  --trace-id <trace_id>
```

The exported file is written by default to:

```text
outputs/otel_traces.json
```

The exporter converts existing JSONL span/event records into OpenTelemetry-style `resourceSpans`, preserving trace IDs, span IDs, event payloads, attributes, service name, and export summary metadata.

## Current Status

Working:

1. Offline text RAG demo.
2. Streaming NDJSON query mode.
3. Simulated voice turn through CLI.
4. Real local voice demo with AssemblyAI ASR, Ollama/Gemma reasoning, and Cartesia TTS.
5. MCP and A2A protocol surfaces.
6. FastAPI query and streaming endpoints.
7. PDF, DOCX, HTML/HTM, and URL ingestion support.
8. Latency metrics report CLI with avg, p50, p95, min, and max summaries.
9. Hybrid retrieval with BM25, lightweight semantic similarity, score fusion, and JSON diagnostics.
10. FastAPI document upload and index-management endpoints with safe upload validation and automatic engine reload.
11. Dockerized FastAPI deployment with Compose, healthcheck, mounted data/traces/outputs, and Makefile targets.
12. OpenTelemetry-shaped trace export for JSONL spans/events, with CLI support for service name, trace ID filtering, limits, and JSON/text summaries.

Next improvements:

1. Complete `integrations/livekit_worker.py` for real LiveKit room transport.
2. Add production-grade ingestion controls: file upload API, metadata filters, duplicate detection, and persistent indexes.
3. Promote LlamaIndex or hybrid retrieval as a first-class production retrieval option.
4. Add richer latency metrics for ASR, retrieval, LLM time-to-first-token, TTS first-audio, and full turn latency.
5. Export traces to OpenTelemetry if the `observability` extra is installed.

## Production Wiring

The RAG engine does not care which transport calls it. That makes local evals, HTTP requests, MCP tool calls, A2A tasks, and live voice turns share one behavior.

The current local voice demo uses microphone input and local speaker output. LiveKit room transport is the main remaining production transport layer.






