from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

from .config import Settings
from .evals import run_eval
from .events import async_events_to_ndjson
from .llm import build_reasoner
from .rag.engine import build_engine
from .voice.asr import AssemblyAITranscriptSource, IterableTranscriptSource
from .voice.audio_sink import FfplayPCMFloat32Sink
from .voice.pipeline import VoiceRagPipeline
from .voice.tts import CartesiaTTS, ConsoleTTS


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    settings = _settings_from_args(args)

    if args.command == "query":
        return _query(settings, args.query, stream=args.stream, json_output=args.json)
    if args.command == "voice-demo":
        return _voice_demo(settings, args.utterance)
    if args.command == "voice-live":
        return _voice_live(settings, args.tts)
    if args.command == "eval":
        return _eval(settings, Path(args.cases), Path(args.output) if args.output else None)
    if args.command == "serve":
        return _serve(settings, args.host, args.port)
    if args.command == "mcp":
        return _mcp(settings)
    if args.command == "a2a":
        return _a2a(settings, args.host, args.port)

    parser.print_help()
    return 2


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="voice-rag-agent")
    parser.add_argument("--root", default=str(Path.cwd()), help="Project root directory.")
    parser.add_argument("--data-dir", default=None, help="Document directory to index.")
    parser.add_argument("--reasoner", default=None, help="template, ollama, or openai.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    query = subparsers.add_parser("query", help="Ask a question against the document corpus.")
    query.add_argument("query")
    query.add_argument("--stream", action="store_true", help="Emit event NDJSON.")
    query.add_argument("--json", action="store_true", help="Emit a single JSON answer object.")

    voice = subparsers.add_parser("voice-demo", help="Simulate a final ASR utterance.")
    voice.add_argument("utterance")

    live_voice = subparsers.add_parser("voice-live", help="Run one real microphone voice RAG turn.")
    live_voice.add_argument(
    "--tts",
    choices=["console", "cartesia"],
    default="cartesia",
    help="TTS backend for the live demo.",
    )

    eval_parser = subparsers.add_parser("eval", help="Run grounding checks.")
    eval_parser.add_argument("--cases", default="tests/fixtures/eval_cases.json")
    eval_parser.add_argument("--output", default="outputs/eval_results.json")

    serve = subparsers.add_parser("serve", help="Run the FastAPI HTTP server.")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)

    a2a = subparsers.add_parser("a2a", help="Run the A2A HTTP server.")
    a2a.add_argument("--host", default="127.0.0.1")
    a2a.add_argument("--port", type=int, default=8010)

    subparsers.add_parser("mcp", help="Run the MCP stdio server.")
    return parser


def _settings_from_args(args: argparse.Namespace) -> Settings:
    settings = Settings.from_env(Path(args.root))
    if args.data_dir:
        path = Path(args.data_dir)
        settings = _replace(settings, data_dir=path if path.is_absolute() else settings.root_dir / path)
    if args.reasoner:
        settings = _replace(settings, reasoner=args.reasoner)
    return settings


def _replace(settings: Settings, **changes: object) -> Settings:
    values = settings.__dict__.copy()
    values.update(changes)
    return Settings(**values)


def _engine(settings: Settings):
    return build_engine(settings, reasoner=build_reasoner(settings))


def _query(settings: Settings, query: str, stream: bool, json_output: bool) -> int:
    engine = _engine(settings)
    if stream:
        async def run_stream() -> None:
            async for line in async_events_to_ndjson(engine.stream_query(query)):
                print(line, end="")

        asyncio.run(run_stream())
        return 0

    answer = asyncio.run(engine.query(query))
    if json_output:
        print(json.dumps({"answer": answer.text, "citations": answer.citations}, indent=2))
    else:
        print(answer.text)
        if answer.citations:
            print("\nCitations:")
            for citation in answer.citations:
                print(f"- {citation['title']} ({citation['path']}) score={citation['score']}")
    return 0


def _voice_demo(settings: Settings, utterance: str) -> int:
    engine = _engine(settings)
    pipeline = VoiceRagPipeline(engine, IterableTranscriptSource([utterance]))

    async def run_pipeline() -> None:
        async for event in pipeline.stream():
            print(event.to_json())

    asyncio.run(run_pipeline())
    return 0

def _voice_live(settings: Settings, tts_backend: str) -> int:
    engine = _engine(settings)

    source = AssemblyAITranscriptSource(
        api_key=os.environ.get("ASSEMBLYAI_API_KEY", ""),
        one_turn=True,
    )

    audio_sink = None

    if tts_backend == "cartesia":
        tts = CartesiaTTS(
            api_key=os.environ.get("CARTESIA_API_KEY", ""),
            voice_id=os.environ.get(
                "CARTESIA_VOICE_ID",
                "f786b574-daa5-4673-aa0c-cbe3e8534c02",
            ),
        )
        audio_sink = FfplayPCMFloat32Sink()
    else:
        tts = ConsoleTTS()

    pipeline = VoiceRagPipeline(
        engine=engine,
        transcript_source=source,
        tts=tts,
        audio_sink=audio_sink,
    )

    async def run_pipeline() -> None:
        try:
            async for event in pipeline.stream():
                print(event.to_json())
        finally:
            if audio_sink:
                await audio_sink.close()

    asyncio.run(run_pipeline())
    return 0


def _eval(settings: Settings, cases: Path, output: Path | None) -> int:
    results = run_eval(_engine(settings), settings.root_dir / cases, settings.root_dir / output if output else None)
    passed = sum(1 for result in results if result.passed)
    print(f"{passed}/{len(results)} evals passed")
    for result in results:
        status = "PASS" if result.passed else "FAIL"
        print(f"{status} {result.case_id}: citations={result.citation_count}")
        if result.missing_terms:
            print(f"  missing: {', '.join(result.missing_terms)}")
    return 0 if passed == len(results) else 1


def _serve(settings: Settings, host: str, port: int) -> int:
    try:
        import uvicorn
    except ImportError:
        print("Install the api extra to run the server: pip install -e '.[api]'", file=sys.stderr)
        return 1
    from .api import create_app

    uvicorn.run(create_app(settings), host=host, port=port, reload=False)
    return 0


def _a2a(settings: Settings, host: str, port: int) -> int:
    try:
        import uvicorn
    except ImportError:
        print("Install the api extra to run the A2A server: pip install -e '.[api]'", file=sys.stderr)
        return 1
    from .protocols.a2a import create_a2a_app

    uvicorn.run(create_a2a_app(settings), host=host, port=port, reload=False)
    return 0


def _mcp(settings: Settings) -> int:
    from .protocols.mcp_server import run_stdio

    run_stdio(settings)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
