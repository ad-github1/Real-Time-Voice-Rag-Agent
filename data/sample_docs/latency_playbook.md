# Latency Playbook

The agent keeps response latency low by overlapping work wherever the user experience allows it. VAD prevents unnecessary transcription work, final ASR transcripts trigger retrieval immediately, cached retrieval handles repeated questions, and the answer is streamed token by token instead of waiting for a full completion.

Retrieval is optimized with chunk sizes that fit voice answers, a small top-k, and an LRU query cache. In production, LlamaIndex can replace the local BM25 retriever with vector retrieval, hybrid search, reranking, or persisted indexes. The offline BM25 retriever stays in the repository because it is deterministic and useful for tests.

The first spoken response should be short and grounded. The model prompt asks the assistant to lead with the answer, add one supporting detail, and admit when the retrieved context is missing. This reduces time-to-first-audio and avoids long monologues in a real-time voice room.
