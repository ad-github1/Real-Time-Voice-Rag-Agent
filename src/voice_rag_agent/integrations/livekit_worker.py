from __future__ import annotations

from pathlib import Path

from voice_rag_agent.config import Settings


def run_worker(settings: Settings | None = None) -> None:
    """Entry point for a LiveKit Agents worker.

    The project keeps this integration intentionally thin: LiveKit owns transport,
    while the RAG engine owns transcript-to-answer behavior.
    """

    resolved = settings or Settings.from_env(Path.cwd())
    if not (resolved.livekit_url and resolved.livekit_api_key and resolved.livekit_api_secret):
        raise RuntimeError("LIVEKIT_URL, LIVEKIT_API_KEY, and LIVEKIT_API_SECRET are required.")

    try:
        from livekit.agents import WorkerOptions, cli
    except ImportError as exc:
        raise RuntimeError("Install the voice extra to run the LiveKit worker.") from exc

    async def entrypoint(ctx: object) -> None:
        del ctx
        raise NotImplementedError(
            "Connect LiveKit room audio to AssemblyAITranscriptSource and CartesiaTTS, "
            "then feed transcript events into VoiceRagPipeline.stream()."
        )

    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
