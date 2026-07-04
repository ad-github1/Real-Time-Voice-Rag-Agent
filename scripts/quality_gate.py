from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run(command: list[str]) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")

    print("")
    print("$ " + " ".join(command), flush=True)

    subprocess.run(
        command,
        cwd=ROOT,
        env=env,
        check=True,
    )


def main() -> int:
    (ROOT / "outputs").mkdir(parents=True, exist_ok=True)

    run([sys.executable, "-m", "compileall", "-q", "src", "scripts"])
    run([sys.executable, "-m", "ruff", "format", "--check", "src", "tests", "scripts"])
    run([sys.executable, "-m", "ruff", "check", "src", "tests", "scripts"])
    run([sys.executable, "-m", "mypy", "src", "scripts", "tests"])

    run([sys.executable, "-m", "pytest", "tests", "-q"])

    run(
        [
            sys.executable,
            "-m",
            "voice_rag_agent.cli",
            "benchmark",
            "--mode",
            "retrieve",
            "--queries",
            "tests/fixtures/benchmark_queries.json",
            "--concurrency",
            "8",
            "--repeat",
            "5",
            "--max-p95-ms",
            "50",
            "--max-error-rate",
            "0",
            "--min-qps",
            "100",
            "--output",
            "outputs/quality_benchmark_retrieve.json",
        ]
    )

    run(
        [
            sys.executable,
            "-m",
            "voice_rag_agent.cli",
            "eval",
            "--cases",
            "tests/fixtures/eval_cases.json",
            "--output",
            "outputs/quality_eval_results.json",
            "--report",
            "outputs/quality_eval_report.md",
        ]
    )

    run(
        [
            sys.executable,
            "-m",
            "voice_rag_agent.cli",
            "profile",
            "voice latency streaming",
            "--mode",
            "retrieve",
            "--repeat",
            "50",
            "--warmup",
            "5",
            "--top-functions",
            "10",
            "--max-avg-ms",
            "5",
            "--max-peak-memory-kb",
            "4096",
            "--output",
            "outputs/quality_profile_retrieve.json",
        ]
    )

    print("")
    print("Quality gate passed.")
    print("Static checks:")
    print("- compileall src scripts")
    print("- ruff format --check src tests scripts")
    print("- ruff check src tests scripts")
    print("- mypy src scripts tests")
    print("Generated:")
    print("- outputs/quality_benchmark_retrieve.json")
    print("- outputs/quality_eval_results.json")
    print("- outputs/quality_eval_report.md")
    print("- outputs/quality_profile_retrieve.json")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
