from __future__ import annotations

from pathlib import Path
from typing import Any

from voice_rag_agent.config import Settings
from voice_rag_agent.llm import build_reasoner
from voice_rag_agent.rag.engine import build_engine


def create_mcp_server(settings: Settings | None = None) -> Any:
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as exc:
        raise RuntimeError("Install the protocols extra to run the MCP server.") from exc

    resolved = settings or Settings.from_env(Path.cwd())
    engine = build_engine(resolved, reasoner=build_reasoner(resolved))
    mcp = FastMCP("real-time-voice-rag-agent")

    @mcp.tool()  # type: ignore[untyped-decorator]
    async def query_documents(question: str) -> dict[str, Any]:
        """Answer a question against the indexed document corpus."""

        answer = await engine.query(question)
        return {
            "trace_id": answer.trace_id,
            "answer": answer.text,
            "citations": answer.citations,
        }

    return mcp


def run_stdio(settings: Settings | None = None) -> None:
    create_mcp_server(settings).run()
