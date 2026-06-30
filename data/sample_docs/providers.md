# Provider Integration Notes

AssemblyAI is responsible for real-time speech transcription. The adapter should emit partial transcripts for UI feedback and final transcripts for RAG turns. Silero VAD can reduce cost and latency by sending speech frames only when a user is actively speaking.

Ollama runs Gemma locally for private or offline reasoning. The Ollama adapter consumes streaming JSON from the local generate endpoint and forwards deltas through the same answer event stream as other reasoners.

Cartesia is responsible for low-latency text-to-speech. The TTS adapter should stream small audio chunks as soon as text is available and send them to the LiveKit room. Keeping TTS independent from retrieval makes it easier to swap voices, providers, or output codecs.

OpenAI can be used as an optional Responses API reasoner when a hosted model is preferred. The adapter uses an explicit model configured by environment variable so deployments can choose the model available to their account.
