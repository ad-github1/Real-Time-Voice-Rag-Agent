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

Expected event stream:

asr.transcript
retrieval.completed
retrieval.chunk
answer.delta
tts.audio
answer.completed

The live demo uses:

AssemblyAI for real-time microphone transcription.
BM25-based local RAG retrieval over sample documents.
Ollama/Gemma for local reasoning.
Cartesia for text-to-speech playback.
ffplay/FFmpeg for local audio output.
