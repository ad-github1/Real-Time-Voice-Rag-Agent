from __future__ import annotations

from pathlib import Path
from typing import Any

from voice_rag_agent.config import Settings
from voice_rag_agent.events import async_events_to_ndjson
from voice_rag_agent.llm import build_reasoner
from voice_rag_agent.rag.engine import build_engine


def create_a2a_app(settings: Settings | None = None) -> Any:
    try:
        from fastapi import FastAPI, HTTPException
        from fastapi.responses import StreamingResponse
        from pydantic import BaseModel
    except ImportError as exc:
        raise RuntimeError("Install the api extra to run the A2A server.") from exc

    class Message(BaseModel):
        role: str
        parts: list[dict[str, Any]]

    class TaskRequest(BaseModel):
        id: str | None = None
        message: Message

    resolved = settings or Settings.from_env(Path.cwd())
    engine = build_engine(resolved, reasoner=build_reasoner(resolved))
    app = FastAPI(title="Voice RAG A2A Agent", version="0.1.0")

    @app.get("/.well-known/agent-card.json")
    async def agent_card() -> dict[str, Any]:
        return {
            "name": "Real-Time Voice RAG Agent",
            "description": "Grounded document question answering over text or voice transcripts.",
            "version": "0.1.0",
            "capabilities": {
                "streaming": True,
                "pushNotifications": False,
            },
            "skills": [
                {
                    "id": "voice_rag_query",
                    "name": "Voice RAG Query",
                    "description": "Answers questions from indexed documents with citations.",
                    "inputModes": ["text/plain", "application/json"],
                    "outputModes": ["text/plain", "application/json"],
                }
            ],
        }

    @app.post("/tasks/send")
    async def send_task(request: TaskRequest) -> dict[str, Any]:
        text = _text_from_parts(request.message.parts)
        if not text:
            raise HTTPException(status_code=400, detail="A text part is required.")
        answer = await engine.query(text, trace_id=request.id)
        return {
            "id": answer.trace_id,
            "status": {"state": "completed"},
            "artifacts": [
                {
                    "name": "answer",
                    "parts": [
                        {
                            "kind": "text",
                            "text": answer.text,
                            "metadata": {"citations": answer.citations},
                        }
                    ],
                }
            ],
        }

    @app.post("/tasks/sendSubscribe")
    async def send_subscribe(request: TaskRequest) -> StreamingResponse:
        text = _text_from_parts(request.message.parts)
        if not text:
            raise HTTPException(status_code=400, detail="A text part is required.")

        async def body() -> Any:
            async for line in async_events_to_ndjson(
                engine.stream_query(text, trace_id=request.id)
            ):
                yield line

        return StreamingResponse(body(), media_type="application/x-ndjson")

    return app


def _text_from_parts(parts: list[dict[str, Any]]) -> str:
    texts: list[str] = []
    for part in parts:
        if "text" in part:
            texts.append(str(part["text"]))
        elif part.get("kind") == "text" and "content" in part:
            texts.append(str(part["content"]))
    return "\n".join(texts).strip()
