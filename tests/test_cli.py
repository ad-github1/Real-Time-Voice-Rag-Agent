from pathlib import Path

from pytest import CaptureFixture

from voice_rag_agent.cli import main


def test_cli_rejects_empty_query_without_traceback(
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    code = main(["--root", str(tmp_path), "query", "", "--stream"])

    captured = capsys.readouterr()

    assert code == 2
    assert "Query cannot be empty" in captured.err
    assert "Traceback" not in captured.err
