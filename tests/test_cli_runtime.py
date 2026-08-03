from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from samples2llm.cli import app


runner = CliRunner()


def test_cli_prints_result_when_no_output(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("samples2llm.cli.pack", lambda directory, config: "INLINE_RESULT")

    result = runner.invoke(app, [str(tmp_path), "--style", "json"])

    assert result.exit_code == 0
    assert "INLINE_RESULT" in result.stdout


def test_cli_writes_output_file(monkeypatch, tmp_path: Path) -> None:
    out_file = tmp_path / "out.json"
    monkeypatch.setattr("samples2llm.cli.pack", lambda directory, config: "FILE_RESULT")

    result = runner.invoke(app, [str(tmp_path), "--style", "json", "--output", str(out_file)])

    assert result.exit_code == 0
    assert out_file.read_text(encoding="utf-8") == "FILE_RESULT"
    assert "Written to" in result.stdout


def test_cli_passes_feature_toggles(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    def fake_pack(directory, config):
        captured["directory"] = directory
        captured["tonal"] = config.extraction.enable_tonal
        captured["rhythmic"] = config.extraction.enable_rhythmic
        captured["style"] = config.output.style.value
        return "OK"

    monkeypatch.setattr("samples2llm.cli.pack", fake_pack)

    result = runner.invoke(
        app,
        [str(tmp_path), "--style", "markdown", "--no-tonal", "--no-rhythmic"],
    )

    assert result.exit_code == 0
    assert captured["directory"] == tmp_path
    assert captured["tonal"] is False
    assert captured["rhythmic"] is False
    assert captured["style"] == "markdown"
