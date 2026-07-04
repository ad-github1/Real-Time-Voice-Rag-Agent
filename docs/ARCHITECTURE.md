# Architecture Notes

## Runtime Flow

1. Audio arrives from a LiveKit room.
2. VAD gates non-speech frames.
3. AssemblyAI emits partial transcripts for UX and final transcripts for agent turns.
4. The RAG engine retrieves document chunks and emits citations.
5. A reasoner streams answer deltas from the offline template, Ollama/Gemma, or OpenAI.
6. Cartesia streams text-to-speech chunks back to LiveKit.
7. JSONL traces capture retrieval and reasoning spans.

## Why The Core Is Offline First

Real-time voice systems are difficult to debug when every test depends on microphones, rooms, model servers, and paid APIs. This repository keeps a deterministic offline path so retrieval, grounding, event streaming, caching, and evals can be tested quickly.

Provider adapters are deliberately thin. They translate provider-specific events into shared `TranscriptEvent`, `RagEvent`, and TTS byte streams. This keeps orchestration stable while provider SDKs evolve.

## Streaming Contract

Every event has:

```json
{
  "type": "answer.delta",
  "trace_id": "trace_...",
  "created_at_ms": 1760000000000,
  "payload": {}
}
```

This envelope is easy to forward over NDJSON, SSE, WebSocket, LiveKit metadata, MCP tool output, or A2A task subscriptions.
