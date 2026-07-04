from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
from pathlib import Path

from .benchmark import (
    check_benchmark_thresholds,
    benchmark_baseline_summary,
    compare_benchmark_reports,
    format_benchmark_report,
    load_benchmark_queries,
    load_benchmark_report,
    report_to_dict as benchmark_report_to_dict,
    run_query_benchmark,
    run_http_benchmark,
)
from .config import Settings
from .evals import build_eval_report, run_eval
from .events import async_events_to_ndjson
from .llm import build_reasoner
from .metrics import format_metrics_report, report_to_dict, summarize_metrics
from .profiling import (
    check_profile_thresholds,
    format_profile_report,
    profile_report_to_dict,
    run_profile,
)
from .trace_export import (
    export_traces_to_otel_json,
    format_trace_export_summary,
    result_to_dict as trace_export_result_to_dict,
)
from .rag.engine import RagEngine, build_engine
from .rag.index_store import (
    build_persistent_index,
    clean_index,
    format_index_stats,
    index_stats,
)
from .rag.retriever import RetrievedChunk
from .voice.asr import AssemblyAITranscriptSource, IterableTranscriptSource
from .voice.audio_sink import FfplayPCMFloat32Sink
from .voice.pipeline import VoiceRagPipeline
from .voice.tts import CartesiaTTS, ConsoleTTS, SpeechSynthesizer


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    settings = _settings_from_args(args)

    if args.command == "query":
        return _query(settings, args.query, stream=args.stream, json_output=args.json)
    if args.command == "retrieve":
        return _retrieve(settings, args.query, top_k=args.top_k, json_output=args.json)
    if args.command == "voice-demo":
        return _voice_demo(settings, args.utterance)
    if args.command == "voice-live":
        return _voice_live(settings, args.tts)
    if args.command == "metrics":
        return _metrics(settings, mode=args.mode, limit=args.limit, json_output=args.json)
    if args.command == "traces":
        return _traces(
            settings,
            action=args.traces_action,
            output=Path(args.output),
            service_name=args.service_name,
            trace_id=args.trace_id,
            limit=args.limit,
            json_output=args.json,
        )
    if args.command == "profile":
        return _profile(
            settings,
            query=args.query,
            mode=args.mode,
            repeat=args.repeat,
            warmup=args.warmup,
            top_k=args.top_k,
            top_functions=args.top_functions,
            max_avg_ms=args.max_avg_ms,
            max_peak_memory_kb=args.max_peak_memory_kb,
            output_path=Path(args.output) if args.output else None,
            json_output=args.json,
        )
    if args.command == "benchmark":
        return _benchmark(
            settings,
            queries_path=Path(args.queries),
            concurrency=args.concurrency,
            repeat=args.repeat,
            mode=args.mode,
            top_k=args.top_k,
            url=args.url,
            api_key=args.api_key,
            timeout_ms=args.timeout_ms,
            output_path=Path(args.output) if args.output else None,
            baseline_path=Path(args.baseline) if args.baseline else None,
            max_p95_regression=args.max_p95_regression,
            max_error_rate_delta=args.max_error_rate_delta,
            min_qps_ratio=args.min_qps_ratio,
            json_output=args.json,
            max_p95_ms=args.max_p95_ms,
            max_error_rate=args.max_error_rate,
            min_qps=args.min_qps,
        )
    if args.command == "index":
        return _index(settings, action=args.index_action, json_output=args.json)
    if args.command == "eval":
        return _eval(
            settings,
            Path(args.cases),
            Path(args.output) if args.output else None,
            Path(args.report) if args.report else None,
        )
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
    query.add_argument(
        "--retriever",
        choices=["bm25", "hybrid"],
        default=None,
        help="Retrieval mode for this query.",
    )

    retrieve = subparsers.add_parser(
        "retrieve", help="Inspect retrieved chunks without generation."
    )
    retrieve.add_argument("query")
    retrieve.add_argument("--top-k", type=int, default=None, help="Number of chunks to retrieve.")
    retrieve.add_argument("--json", action="store_true", help="Emit retrieval diagnostics as JSON.")
    retrieve.add_argument(
        "--retriever",
        choices=["bm25", "hybrid"],
        default=None,
        help="Retrieval mode for diagnostics.",
    )

    voice = subparsers.add_parser("voice-demo", help="Simulate a final ASR utterance.")
    voice.add_argument("utterance")

    live_voice = subparsers.add_parser("voice-live", help="Run one real microphone voice RAG turn.")
    live_voice.add_argument(
        "--tts",
        choices=["console", "cartesia"],
        default="cartesia",
        help="TTS backend for the live demo.",
    )

    metrics = subparsers.add_parser("metrics", help="Summarize latency metrics from trace files.")
    metrics.add_argument(
        "--mode",
        choices=["all", "text", "voice"],
        default="all",
        help="Filter metrics by event mode.",
    )
    metrics.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only summarize the most recent N metric events.",
    )
    metrics.add_argument("--json", action="store_true", help="Emit metrics report as JSON.")

    traces = subparsers.add_parser("traces", help="Manage and export trace files.")
    traces_subparsers = traces.add_subparsers(dest="traces_action", required=True)

    traces_export = traces_subparsers.add_parser(
        "export",
        help="Export JSONL traces to OpenTelemetry-shaped JSON.",
    )
    traces_export.add_argument(
        "--output",
        default="outputs/otel_traces.json",
        help="Output JSON file.",
    )
    traces_export.add_argument(
        "--service-name",
        default="voice-rag-agent",
        help="OpenTelemetry service.name value.",
    )
    traces_export.add_argument(
        "--trace-id",
        default=None,
        help="Only export one trace ID.",
    )
    traces_export.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only export the most recent N trace records.",
    )
    traces_export.add_argument("--json", action="store_true", help="Emit export summary as JSON.")

    profile_parser = subparsers.add_parser(
        "profile",
        help="Profile retrieval or full query hot paths with CPU and memory metrics.",
    )
    profile_parser.add_argument("query", help="Query to profile.")
    profile_parser.add_argument(
        "--mode",
        choices=["retrieve", "query"],
        default="retrieve",
        help="Profile retrieval-only or full query path.",
    )
    profile_parser.add_argument(
        "--repeat",
        type=int,
        default=100,
        help="Number of measured iterations.",
    )
    profile_parser.add_argument(
        "--warmup",
        type=int,
        default=5,
        help="Number of warmup iterations before profiling.",
    )
    profile_parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Top-k retrieval size for retrieve mode.",
    )
    profile_parser.add_argument(
        "--top-functions",
        type=int,
        default=15,
        help="Number of cumulative CPU hot functions to show.",
    )
    profile_parser.add_argument(
        "--max-avg-ms",
        type=float,
        default=None,
        help="Fail if average profiled duration exceeds this threshold in milliseconds.",
    )
    profile_parser.add_argument(
        "--max-peak-memory-kb",
        type=float,
        default=None,
        help="Fail if peak profiled memory exceeds this threshold in KiB.",
    )
    profile_parser.add_argument(
        "--output",
        default=None,
        help="Optional path to write profile JSON.",
    )
    profile_parser.add_argument("--json", action="store_true", help="Emit profile as JSON.")

    benchmark = subparsers.add_parser(
        "benchmark",
        help="Run concurrent query benchmark and report p50/p95/p99 latency.",
    )
    benchmark.add_argument(
        "--queries",
        default="tests/fixtures/benchmark_queries.json",
        help="JSON or text file containing benchmark queries.",
    )
    benchmark.add_argument(
        "--mode",
        choices=["query", "retrieve", "http"],
        default="query",
        help="Benchmark full query path or retrieval-only hot path.",
    )
    benchmark.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Top-k retrieval size for retrieve benchmark mode.",
    )
    benchmark.add_argument(
        "--url",
        default="http://127.0.0.1:8000/query",
        help="HTTP endpoint for benchmark mode http.",
    )
    benchmark.add_argument(
        "--api-key",
        default=None,
        help="Optional API key for HTTP benchmark mode. Defaults to VOICE_RAG_API_KEY.",
    )
    benchmark.add_argument(
        "--timeout-ms",
        type=float,
        default=5000.0,
        help="Per-request timeout for HTTP benchmark mode.",
    )
    benchmark.add_argument(
        "--concurrency",
        type=int,
        default=4,
        help="Maximum number of in-flight benchmark queries.",
    )
    benchmark.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="Number of times to repeat the query set.",
    )
    benchmark.add_argument(
        "--output",
        default=None,
        help="Optional path to write benchmark report JSON.",
    )
    benchmark.add_argument("--json", action="store_true", help="Emit benchmark report as JSON.")
    benchmark.add_argument(
        "--max-p95-ms",
        type=float,
        default=None,
        help="Fail if benchmark p95 latency exceeds this threshold in milliseconds.",
    )
    benchmark.add_argument(
        "--max-error-rate",
        type=float,
        default=None,
        help="Fail if error rate exceeds this fraction, for example 0.01.",
    )
    benchmark.add_argument(
        "--min-qps",
        type=float,
        default=None,
        help="Fail if benchmark throughput is below this QPS threshold.",
    )
    benchmark.add_argument(
        "--baseline",
        default=None,
        help="Optional previous benchmark JSON report to compare against.",
    )
    benchmark.add_argument(
        "--max-p95-regression",
        type=float,
        default=None,
        help="With --baseline, fail if p95 latency regresses by more than this percent.",
    )
    benchmark.add_argument(
        "--max-error-rate-delta",
        type=float,
        default=0.0,
        help="With --baseline, fail if error rate increases by more than this amount.",
    )
    benchmark.add_argument(
        "--min-qps-ratio",
        type=float,
        default=None,
        help="With --baseline, fail if current QPS divided by baseline QPS is below this ratio.",
    )

    index = subparsers.add_parser("index", help="Manage persistent retrieval index.")
    index_subparsers = index.add_subparsers(dest="index_action", required=True)

    index_build = index_subparsers.add_parser("build", help="Build persistent chunk index.")
    index_build.add_argument("--json", action="store_true", help="Emit build result as JSON.")

    index_stats_parser = index_subparsers.add_parser("stats", help="Show persistent index stats.")
    index_stats_parser.add_argument("--json", action="store_true", help="Emit index stats as JSON.")

    index_clean = index_subparsers.add_parser("clean", help="Delete persistent index files.")
    index_clean.add_argument("--json", action="store_true", help="Emit clean result as JSON.")

    eval_parser = subparsers.add_parser("eval", help="Run grounding checks.")
    eval_parser.add_argument("--cases", default="tests/fixtures/eval_cases.json")
    eval_parser.add_argument("--output", default="outputs/eval_results.json")
    eval_parser.add_argument("--report", default=None, help="Optional Markdown eval report path.")

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
        settings = _replace(
            settings, data_dir=path if path.is_absolute() else settings.root_dir / path
        )

    if args.reasoner:
        settings = _replace(settings, reasoner=args.reasoner)

    retriever = getattr(args, "retriever", None)

    if retriever:
        settings = _replace(settings, retriever=retriever)

    return settings


def _replace(settings: Settings, **changes: object) -> Settings:
    values = settings.__dict__.copy()
    values.update(changes)

    return Settings(**values)


def _engine(settings: Settings) -> RagEngine:
    return build_engine(settings, reasoner=build_reasoner(settings))


def _query(settings: Settings, query: str, stream: bool, json_output: bool) -> int:
    if not query.strip():
        print("Error: Query cannot be empty.", file=sys.stderr)
        return 2

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


def _retrieve(settings: Settings, query: str, top_k: int | None, json_output: bool) -> int:
    if top_k is not None:
        settings = _replace(settings, top_k=top_k)

    engine = _engine(settings)
    results = engine.retrieve(query, top_k=settings.top_k)
    payload = {
        "query": query,
        "retriever": settings.retriever,
        "top_k": settings.top_k,
        "results": [
            _retrieved_chunk_to_dict(index + 1, result) for index, result in enumerate(results)
        ],
    }

    if json_output:
        print(json.dumps(payload, indent=2))
        return 0

    print(f"Retriever: {settings.retriever}")
    print(f"Query: {query}")
    print(f"Results: {len(results)}")
    print("")

    for index, result in enumerate(results, start=1):
        citation = result.citation()
        print(f"{index}. {citation.get('title', '')}")
        print(f"   Score: {citation.get('score')}")
        print(f"   Path: {citation.get('path')}")
        print(f"   Chunk: {citation.get('chunk_index')}")
        print(f"   Preview: {_preview_text(result.chunk.text)}")

        diagnostics = result.diagnostics or {}

        if diagnostics:
            diag_text = ", ".join(
                f"{key}={_format_diag(value)}" for key, value in diagnostics.items()
            )
            print(f"   Diagnostics: {diag_text}")

        print("")

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

    tts: SpeechSynthesizer
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


def _metrics(settings: Settings, mode: str, limit: int | None, json_output: bool) -> int:
    report = summarize_metrics(settings.trace_dir, mode=mode, limit=limit)

    if json_output:
        print(json.dumps(report_to_dict(report), indent=2))
    else:
        print(format_metrics_report(report))

    return 0


def _traces(
    settings: Settings,
    action: str,
    output: Path,
    service_name: str,
    trace_id: str | None,
    limit: int | None,
    json_output: bool,
) -> int:
    if action == "export":
        output_path = output if output.is_absolute() else settings.root_dir / output
        result = export_traces_to_otel_json(
            trace_dir=settings.trace_dir,
            output_path=output_path,
            service_name=service_name,
            trace_id=trace_id,
            limit=limit,
        )

        if json_output:
            print(json.dumps(trace_export_result_to_dict(result), indent=2))
        else:
            print(format_trace_export_summary(result))

        return 0

    print(f"Unknown traces action: {action}", file=sys.stderr)

    return 2


def _profile(
    settings: Settings,
    query: str,
    mode: str,
    repeat: int,
    warmup: int,
    top_k: int | None,
    top_functions: int,
    max_avg_ms: float | None,
    max_peak_memory_kb: float | None,
    output_path: Path | None,
    json_output: bool,
) -> int:
    try:
        report = run_profile(
            engine=_engine(settings),
            query=query,
            mode=mode,
            repeat=repeat,
            warmup=warmup,
            top_k=top_k if top_k is not None else settings.top_k,
            top_functions=top_functions,
        )
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    gate_failures = check_profile_thresholds(
        report,
        max_avg_ms=max_avg_ms,
        max_peak_memory_kb=max_peak_memory_kb,
    )
    payload = profile_report_to_dict(report)
    payload["gate"] = {
        "passed": not gate_failures,
        "failures": gate_failures,
        "thresholds": {
            "max_avg_ms": max_avg_ms,
            "max_peak_memory_kb": max_peak_memory_kb,
        },
    }

    if output_path is not None:
        resolved_output_path = (
            output_path if output_path.is_absolute() else settings.root_dir / output_path
        )
        resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
        resolved_output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if json_output:
        print(json.dumps(payload, indent=2))
    else:
        print(format_profile_report(report))

        if gate_failures:
            print("\nProfile gate: FAIL")
            for failure in gate_failures:
                print(f"- {failure}")
        else:
            print("\nProfile gate: PASS")

        if output_path is not None:
            print(f"\nWrote JSON report: {resolved_output_path}")

    return 1 if gate_failures else 0


def _benchmark(
    settings: Settings,
    queries_path: Path,
    concurrency: int,
    repeat: int,
    mode: str,
    top_k: int | None,
    url: str,
    api_key: str | None,
    timeout_ms: float,
    output_path: Path | None,
    baseline_path: Path | None,
    max_p95_regression: float | None,
    max_error_rate_delta: float | None,
    min_qps_ratio: float | None,
    json_output: bool,
    max_p95_ms: float | None,
    max_error_rate: float | None,
    min_qps: float | None,
) -> int:
    resolved_queries_path = (
        queries_path if queries_path.is_absolute() else settings.root_dir / queries_path
    )
    queries = load_benchmark_queries(resolved_queries_path)

    if mode == "http":
        resolved_api_key = api_key if api_key is not None else os.getenv("VOICE_RAG_API_KEY", "")
        report = asyncio.run(
            run_http_benchmark(
                url=url,
                queries=queries,
                concurrency=concurrency,
                repeat=repeat,
                api_key=resolved_api_key,
                timeout_ms=timeout_ms,
            )
        )
    else:
        report = asyncio.run(
            run_query_benchmark(
                engine=_engine(settings),
                queries=queries,
                concurrency=concurrency,
                repeat=repeat,
                mode=mode,
                top_k=top_k if top_k is not None else settings.top_k,
            )
        )

    payload = benchmark_report_to_dict(report)

    if output_path is not None:
        resolved_output_path = (
            output_path if output_path.is_absolute() else settings.root_dir / output_path
        )
        resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
        resolved_output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    gate_failures = check_benchmark_thresholds(
        report,
        max_p95_ms=max_p95_ms,
        max_error_rate=max_error_rate,
        min_qps=min_qps,
    )

    baseline_failures: list[str] = []
    baseline_comparison: dict[str, object] | None = None

    if baseline_path is not None:
        resolved_baseline_path = (
            baseline_path if baseline_path.is_absolute() else settings.root_dir / baseline_path
        )
        baseline_payload = load_benchmark_report(resolved_baseline_path)
        baseline_failures = compare_benchmark_reports(
            payload,
            baseline_payload,
            max_p95_regression_percent=max_p95_regression,
            max_error_rate_delta=max_error_rate_delta,
            min_qps_ratio=min_qps_ratio,
        )
        baseline_comparison = {
            "baseline_path": str(resolved_baseline_path),
            "passed": not baseline_failures,
            "failures": baseline_failures,
            "thresholds": {
                "max_p95_regression_percent": max_p95_regression,
                "max_error_rate_delta": max_error_rate_delta,
                "min_qps_ratio": min_qps_ratio,
            },
            "summary": benchmark_baseline_summary(payload, baseline_payload),
        }

    all_failures = gate_failures + baseline_failures

    payload["gate"] = {
        "passed": not all_failures,
        "failures": all_failures,
        "thresholds": {
            "max_p95_ms": max_p95_ms,
            "max_error_rate": max_error_rate,
            "min_qps": min_qps,
        },
    }
    payload["baseline_comparison"] = baseline_comparison

    if output_path is not None:
        resolved_output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if json_output:
        print(json.dumps(payload, indent=2))
    else:
        print(format_benchmark_report(report))

        if gate_failures:
            print("\nBenchmark gate: FAIL")
            for failure in gate_failures:
                print(f"- {failure}")
        else:
            print("\nBenchmark gate: PASS")

        if baseline_comparison is not None:
            status = "PASS" if baseline_comparison["passed"] else "FAIL"
            print(f"Baseline comparison: {status}")

            for failure in baseline_failures:
                print(f"- {failure}")

        if output_path is not None:
            print(f"\nWrote JSON report: {resolved_output_path}")

    return 1 if all_failures else 0


def _index(settings: Settings, action: str, json_output: bool) -> int:
    if action == "build":
        result = build_persistent_index(
            data_dir=settings.data_dir,
            index_dir=settings.index_dir,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )

        payload = {
            "index_dir": str(result.index_dir),
            "manifest_path": str(result.manifest_path),
            "chunks_path": str(result.chunks_path),
            "source_count": result.source_count,
            "chunk_count": result.chunk_count,
            "duplicate_chunks_skipped": result.duplicate_chunks_skipped,
            "created_at_ms": result.created_at_ms,
        }

        if json_output:
            print(json.dumps(payload, indent=2))
        else:
            print("Persistent index built successfully.")
            print(f"Index dir: {result.index_dir}")
            print(f"Sources: {result.source_count}")
            print(f"Chunks: {result.chunk_count}")
            print(f"Duplicate chunks skipped: {result.duplicate_chunks_skipped}")

        return 0

    if action == "stats":
        stats = index_stats(settings.index_dir, settings.data_dir)

        if json_output:
            print(json.dumps(stats, indent=2))
        else:
            print(format_index_stats(stats))

        return 0

    if action == "clean":
        removed = clean_index(settings.index_dir)
        payload = {"index_dir": str(settings.index_dir), "removed": removed}

        if json_output:
            print(json.dumps(payload, indent=2))
        else:
            if removed:
                print(f"Removed persistent index: {settings.index_dir}")
            else:
                print(f"No persistent index found at: {settings.index_dir}")

        return 0

    print(f"Unknown index action: {action}", file=sys.stderr)

    return 2


def _eval(settings: Settings, cases: Path, output: Path | None, report: Path | None) -> int:
    results = run_eval(
        _engine(settings),
        settings.root_dir / cases,
        settings.root_dir / output if output else None,
        settings.root_dir / report if report else None,
    )
    eval_report = build_eval_report(results)

    print(f"{eval_report.passed_cases}/{eval_report.total_cases} evals passed")
    print(f"Pass rate: {eval_report.pass_rate:.2%}")
    print(f"Citation coverage: {eval_report.citation_coverage_rate:.2%}")
    print(f"Average term recall: {eval_report.average_term_recall:.2%}")
    print(f"Average retrieval recall: {eval_report.average_retrieval_recall:.2%}")
    print(f"Average retrieval MRR: {eval_report.average_retrieval_mrr:.3f}")
    print(f"Average retrieval latency: {eval_report.average_retrieval_latency_ms:.3f} ms")
    print(f"P95 latency: {eval_report.p95_latency_ms:.3f} ms")

    for result in results:
        status = "PASS" if result.passed else "FAIL"
        print(
            f"{status} {result.case_id}: "
            f"term_recall={result.term_recall:.2%}, "
            f"retrieval_recall={result.retrieval_recall:.2%}, "
            f"retrieval_mrr={result.retrieval_mrr:.3f}, "
            f"citations={result.citation_count}, "
            f"latency_ms={result.latency_ms:.3f}"
        )

        if result.missing_terms:
            print(f"  missing terms: {', '.join(result.missing_terms)}")

        if result.missing_expected_citations:
            print("  missing expected citations: " + ", ".join(result.missing_expected_citations))

    return 0 if eval_report.passed_cases == eval_report.total_cases else 1


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
        print(
            "Install the api extra to run the A2A server: pip install -e '.[api]'", file=sys.stderr
        )
        return 1

    from .protocols.a2a import create_a2a_app

    uvicorn.run(create_a2a_app(settings), host=host, port=port, reload=False)

    return 0


def _mcp(settings: Settings) -> int:
    from .protocols.mcp_server import run_stdio

    run_stdio(settings)

    return 0


def _retrieved_chunk_to_dict(rank: int, result: RetrievedChunk) -> dict[str, object]:
    citation = result.citation()

    return {
        "rank": rank,
        "id": citation.get("id", ""),
        "title": citation.get("title", ""),
        "path": citation.get("path", ""),
        "chunk_index": citation.get("chunk_index", 0),
        "score": citation.get("score", 0),
        "diagnostics": result.diagnostics or {},
        "preview": _preview_text(result.chunk.text),
    }


def _preview_text(text: str, max_chars: int = 260) -> str:
    compact = re.sub(r"\s+", " ", text).strip()

    return compact if len(compact) <= max_chars else f"{compact[: max_chars - 3]}..."


def _format_diag(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"

    return str(value)


if __name__ == "__main__":
    raise SystemExit(main())
