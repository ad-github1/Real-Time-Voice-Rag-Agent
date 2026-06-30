# Latency Playbook

- Use VAD before ASR to avoid sending silence.
- Emit partial transcript events, but trigger RAG only on final transcripts.
- Keep chunks voice-sized and `top_k` small.
- Cache normalized retrieval queries.
- Stream model deltas immediately.
- Start TTS on complete sentence fragments where the provider supports it.
- Keep protocol adapters thin so they do not block the RAG loop.
- Trace every turn and inspect the gap between transcript finalization and first answer delta.
