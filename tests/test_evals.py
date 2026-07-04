import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from voice_rag_agent.config import Settings
from voice_rag_agent.evals import build_eval_report, run_eval
from voice_rag_agent.rag.engine import build_engine


ROOT = Path(__file__).resolve().parents[1]


class EvalTests(unittest.TestCase):
    def test_sample_evals_pass(self) -> None:
        with TemporaryDirectory() as tmp:
            settings = Settings.from_env(ROOT)
            settings = _replace(
                settings,
                trace_dir=Path(tmp),
                data_dir=ROOT / "data/sample_docs",
                enable_tracing=False,
            )
            engine = build_engine(settings)
            results = run_eval(engine, ROOT / "tests/fixtures/eval_cases.json")
        self.assertTrue(all(result.passed for result in results))

    def test_eval_report_outputs_json_and_markdown(self) -> None:
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            settings = Settings.from_env(ROOT)
            settings = _replace(
                settings,
                trace_dir=tmp_path / "traces",
                data_dir=ROOT / "data/sample_docs",
                enable_tracing=False,
            )
            engine = build_engine(settings)
            output_path = tmp_path / "eval_results.json"
            report_path = tmp_path / "eval_report.md"

            results = run_eval(
                engine,
                ROOT / "tests/fixtures/eval_cases.json",
                output_path,
                report_path,
            )
            report = build_eval_report(results)

            payload = json.loads(output_path.read_text(encoding="utf-8"))
            markdown = report_path.read_text(encoding="utf-8")

        self.assertEqual(report.total_cases, 3)
        self.assertEqual(report.passed_cases, 3)
        self.assertEqual(payload["summary"]["total_cases"], 3)
        self.assertEqual(payload["summary"]["average_retrieval_recall"], 1.0)
        self.assertIn("retrieval_recall", payload["results"][0])
        self.assertIn("retrieval_mrr", payload["results"][0])
        self.assertIn("retrieval_latency_ms", payload["results"][0])
        self.assertIn("Voice RAG Evaluation Report", markdown)
        self.assertIn("Citation coverage", markdown)
        self.assertIn("P95 latency", markdown)
        self.assertIn("Average Retrieval Recall", markdown)
        self.assertIn("Average Retrieval MRR", markdown)


def _replace(settings: Settings, **changes: object) -> Settings:
    values = settings.__dict__.copy()
    values.update(changes)
    return Settings(**values)


if __name__ == "__main__":
    unittest.main()
