from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from voice_rag_agent.config import Settings
from voice_rag_agent.evals import run_eval
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


def _replace(settings: Settings, **changes: object) -> Settings:
    values = settings.__dict__.copy()
    values.update(changes)
    return Settings(**values)


if __name__ == "__main__":
    unittest.main()
