# Voice RAG Architecture

The Real-Time Voice RAG Agent transforms static document querying into a natural spoken conversation. A user speaks in a LiveKit room, speech frames are gated by VAD, AssemblyAI turns speech into final transcripts, the RAG engine retrieves grounded context, a reasoning model drafts the response, and Cartesia TTS streams the answer back as speech.

The production path uses small, composable adapters rather than a single hard-coded provider chain. The transcript source, retriever, reasoner, synthesizer, and transport can each be swapped without changing the orchestration contract. This keeps local development fast and makes provider fallbacks easier to test.

Every user turn emits structured streaming JSON events. Events include transcript updates, retrieval completion, individual source chunks, answer deltas, final answers, and text-to-speech chunks. The same event envelope can be sent as NDJSON over HTTP, Server-Sent Events, WebSockets, MCP tool output, A2A task updates, or LiveKit metadata.
