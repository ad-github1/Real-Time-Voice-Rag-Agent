from __future__ import annotations

import asyncio
import json
import queue
import threading
import urllib.error
import urllib.request
from typing import AsyncIterator

from voice_rag_agent.rag.retriever import RetrievedChunk


class OllamaReasoner:
    """Streams JSON chunks from a local Ollama model such as Gemma."""

    name = "ollama"

    def __init__(self, base_url: str, model: str, timeout_s: float = 90.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_s = timeout_s

    async def stream_answer(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        prompt: str,
    ) -> AsyncIterator[str]:
        del query, chunks
        output: queue.Queue[str | BaseException | None] = queue.Queue()
        thread = threading.Thread(target=self._produce, args=(prompt, output), daemon=True)
        thread.start()

        while True:
            item = await asyncio.to_thread(output.get)
            if item is None:
                break
            if isinstance(item, BaseException):
                raise item
            yield item

    def _produce(self, prompt: str, output: queue.Queue[str | BaseException | None]) -> None:
        try:
            payload = json.dumps(
                {
                    "model": self.model,
                    "prompt": prompt,
                    "stream": True,
                    "options": {
                        "temperature": 0.2,
                        "num_ctx": 8192,
                    },
                }
            ).encode("utf-8")
            request = urllib.request.Request(
                f"{self.base_url}/api/generate",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                for raw_line in response:
                    if not raw_line.strip():
                        continue
                    event = json.loads(raw_line.decode("utf-8"))
                    if "response" in event:
                        output.put(str(event["response"]))
                    if event.get("done"):
                        break
        except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
            output.put(
                RuntimeError(
                    "Ollama streaming failed. Start Ollama, pull the configured Gemma model, "
                    "or switch VOICE_RAG_REASONER=template."
                )
            )
            output.put(exc)
        finally:
            output.put(None)
