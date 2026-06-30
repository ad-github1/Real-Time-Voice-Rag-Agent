from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import Settings
from .events import async_events_to_ndjson
from .llm import build_reasoner
from .rag.engine import build_engine


def create_app(settings: Settings | None = None) -> Any:
    try:
        from fastapi import FastAPI, HTTPException
        from fastapi.responses import StreamingResponse
        from pydantic import BaseModel
    except ImportError as exc:
        raise RuntimeError("Install the api extra to run the FastAPI server.") from exc

    class QueryRequest(BaseModel):  # type: ignore[no-redef]
        query: str
        trace_id: str | None = None

    resolved = settings or Settings.from_env(Path.cwd())
    engine = build_engine(resolved, reasoner=build_reasoner(resolved))
    app = FastAPI(title="Real-Time Voice RAG Agent", version="0.1.0")

    @app.get("/health")
    async def health() -> dict[str, str | int]:
        return {
            "status": "ok",
            "env": resolved.env,
            "top_k": resolved.top_k,
            "reasoner": resolved.reasoner,
        }

    @app.post("/query")
    async def query(request: QueryRequest) -> dict[str, Any]:
        try:
            answer = await engine.query(request.query, trace_id=request.trace_id)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {
            "trace_id": answer.trace_id,
            "answer": answer.text,
            "citations": answer.citations,
        }

    @app.post("/query/stream")
    async def stream_query(request: QueryRequest) -> StreamingResponse:
        async def body() -> Any:
            async for line in async_events_to_ndjson(
                engine.stream_query(request.query, trace_id=request.trace_id)
            ):
                yield line

        return StreamingResponse(body(), media_type="application/x-ndjson")

    return app
