from __future__ import annotations

from pathlib import Path

from samples2llm.config.schema import InputConfig
from samples2llm.core.sample_search import search_samples


def test_search_samples_filters_extensions_and_ignored_dirs(tmp_path: Path) -> None:
    keep = tmp_path / "drums" / "kick.wav"
    keep.parent.mkdir(parents=True)
    keep.write_bytes(b"x")

    ignored_dir_file = tmp_path / ".git" / "hidden.wav"
    ignored_dir_file.parent.mkdir(parents=True)
    ignored_dir_file.write_bytes(b"x")

    wrong_ext = tmp_path / "readme.txt"
    wrong_ext.write_text("not audio", encoding="utf-8")

    results = search_samples(tmp_path, InputConfig())

    assert keep in results
    assert ignored_dir_file not in results
    assert wrong_ext not in results


def test_search_samples_respects_custom_extensions(tmp_path: Path) -> None:
    wav_file = tmp_path / "a.wav"
    wav_file.write_bytes(b"x")

    flac_file = tmp_path / "b.flac"
    flac_file.write_bytes(b"x")

    cfg = InputConfig(extensions=[".flac"])
    results = search_samples(tmp_path, cfg)

    assert flac_file in results
    assert wav_file not in results


def test_search_samples_respects_max_file_size_mb(tmp_path: Path) -> None:
    small = tmp_path / "small.wav"
    small.write_bytes(b"x" * 256)

    large = tmp_path / "large.wav"
    large.write_bytes(b"x" * 2048)

    cfg = InputConfig(max_file_size_mb=0.001)  # ~1KB
    results = search_samples(tmp_path, cfg)

    assert small in results
    assert large not in results


def test_search_samples_returns_sorted_paths(tmp_path: Path) -> None:
    b = tmp_path / "b.wav"
    a = tmp_path / "a.wav"
    b.write_bytes(b"x")
    a.write_bytes(b"x")

    results = search_samples(tmp_path, InputConfig())

    assert results == [a, b]
