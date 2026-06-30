from __future__ import annotations

import asyncio
import json
import queue
import threading
import urllib.error
import urllib.request
from typing import Any, AsyncIterator

from voice_rag_agent.rag.retriever import RetrievedChunk


class OpenAIResponsesReasoner:
    """Optional Responses API adapter using standard-library SSE parsing."""

    name = "openai_responses"

    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str = "https://api.openai.com/v1",
        timeout_s: float = 90.0,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
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
                    "input": prompt,
                    "stream": True,
                    "temperature": 0.2,
                }
            ).encode("utf-8")
            request = urllib.request.Request(
                f"{self.base_url}/responses",
                data=payload,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                for raw_line in response:
                    line = raw_line.decode("utf-8").strip()
                    if not line.startswith("data:"):
                        continue
                    data = line.removeprefix("data:").strip()
                    if data == "[DONE]":
                        break
                    event = json.loads(data)
                    delta = _extract_delta(event)
                    if delta:
                        output.put(delta)
        except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
            output.put(
                RuntimeError(
                    "OpenAI streaming failed. Check OPENAI_API_KEY/OPENAI_MODEL, network access, "
                    "or switch VOICE_RAG_REASONER=template."
                )
            )
            output.put(exc)
        finally:
            output.put(None)


def _extract_delta(event: dict[str, Any]) -> str:
    if event.get("type") == "response.output_text.delta":
        return str(event.get("delta", ""))
    if event.get("type") == "response.refusal.delta":
        return str(event.get("delta", ""))
    if "delta" in event and isinstance(event["delta"], str):
        return event["delta"]
    return ""
