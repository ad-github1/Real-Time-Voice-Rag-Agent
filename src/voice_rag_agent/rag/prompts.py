from __future__ import annotations

from .retriever import RetrievedChunk


SYSTEM_PROMPT = """You are a real-time voice RAG agent.
Answer naturally and briefly, but ground every factual claim in the retrieved context.
If the context does not answer the user, say what is missing and ask a focused follow-up.
Use low-latency phrasing: lead with the answer, then add one supporting detail."""


def build_grounded_prompt(query: str, chunks: list[RetrievedChunk]) -> str:
    context = "\n\n".join(
        f"[{index + 1}] {chunk.chunk.metadata.get('title', 'Untitled')} "
        f"({chunk.chunk.metadata.get('path', chunk.chunk.source_path)}):\n{chunk.chunk.text}"
        for index, chunk in enumerate(chunks)
    )
    return f"{SYSTEM_PROMPT}\n\nRetrieved context:\n{context}\n\nUser question: {query}\n\nAnswer:"
