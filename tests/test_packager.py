from __future__ import annotations

from pathlib import Path

from samples2llm.config.schema import OutputStyle, Samples2LlmConfig
from samples2llm.core.packager import pack


class _DummyRecord:
    def __init__(self, name: str) -> None:
        self.name = name


class _FakeCache:
    def __init__(self, root, extraction_cfg) -> None:
        self.get_map: dict[str, _DummyRecord] = {}
        self.put_calls: list[str] = []
        self.pruned = False
        self.saved = False

    def get(self, file_path, fingerprint):
        return self.get_map.get(file_path.name)

    def put(self, file_path, fingerprint, record):
        self.put_calls.append(file_path.name)

    def prune_to_paths(self, existing_paths):
        self.pruned = True

    def save(self):
        self.saved = True


def test_pack_calls_json_output_path(monkeypatch, tmp_path: Path) -> None:
    cfg = Samples2LlmConfig()
    cfg.output.style = OutputStyle.JSON

    calls: dict[str, object] = {}

    sample_paths = [tmp_path / "a.wav", tmp_path / "b.wav"]

    def fake_search_samples(root, input_cfg):
        calls["search_root"] = root
        calls["search_cfg"] = input_cfg
        return sample_paths

    def fake_generate_tree(paths, root):
        calls["tree_paths"] = paths
        calls["tree_root"] = root
        return "TREE"

    def fake_extract_sample(path, root, extraction_cfg):
        calls.setdefault("extract", []).append((path, root, extraction_cfg))
        return _DummyRecord(path.name)

    def fake_generate_json(records, tree, output_cfg):
        calls["json"] = (records, tree, output_cfg)
        return "JSON_OK"

    fake_cache = _FakeCache(tmp_path, cfg.extraction)

    monkeypatch.setattr("samples2llm.core.packager.search_samples", fake_search_samples)
    monkeypatch.setattr("samples2llm.core.packager.generate_directory_tree", fake_generate_tree)
    monkeypatch.setattr("samples2llm.core.packager.extract_sample", fake_extract_sample)
    monkeypatch.setattr("samples2llm.core.packager.generate_json", fake_generate_json)
    monkeypatch.setattr("samples2llm.core.packager.AnalysisCache", lambda root, extraction_cfg: fake_cache)
    monkeypatch.setattr("samples2llm.core.packager.build_file_fingerprint", lambda p: f"fp:{p.name}")

    out = pack(tmp_path, cfg)

    assert out == "JSON_OK"
    assert calls["tree_paths"] == sample_paths
    assert calls["json"][1] == "TREE"
    assert len(calls["extract"]) == 2
    assert fake_cache.put_calls == ["a.wav", "b.wav"]
    assert fake_cache.pruned is True
    assert fake_cache.saved is True


def test_pack_calls_markdown_output_path(monkeypatch, tmp_path: Path) -> None:
    cfg = Samples2LlmConfig()
    cfg.output.style = OutputStyle.MARKDOWN

    fake_cache = _FakeCache(tmp_path, cfg.extraction)

    monkeypatch.setattr("samples2llm.core.packager.search_samples", lambda root, input_cfg: [tmp_path / "a.wav"])
    monkeypatch.setattr("samples2llm.core.packager.generate_directory_tree", lambda paths, root: "TREE")
    monkeypatch.setattr("samples2llm.core.packager.extract_sample", lambda path, root, extraction_cfg: _DummyRecord(path.name))
    monkeypatch.setattr("samples2llm.core.packager.generate_markdown", lambda records, tree: "MD_OK")
    monkeypatch.setattr("samples2llm.core.packager.AnalysisCache", lambda root, extraction_cfg: fake_cache)
    monkeypatch.setattr("samples2llm.core.packager.build_file_fingerprint", lambda p: f"fp:{p.name}")

    out = pack(tmp_path, cfg)

    assert out == "MD_OK"
    assert fake_cache.put_calls == ["a.wav"]


def test_pack_skips_failed_sample_and_continues(monkeypatch, tmp_path: Path) -> None:
    cfg = Samples2LlmConfig()
    cfg.output.style = OutputStyle.JSON

    paths = [tmp_path / "ok.wav", tmp_path / "bad.wav", tmp_path / "ok2.wav"]

    fake_cache = _FakeCache(tmp_path, cfg.extraction)

    monkeypatch.setattr("samples2llm.core.packager.search_samples", lambda root, input_cfg: paths)
    monkeypatch.setattr("samples2llm.core.packager.generate_directory_tree", lambda sample_paths, root: "TREE")

    def fake_extract(path, root, extraction_cfg):
        if path.name == "bad.wav":
            raise RuntimeError("boom")
        return _DummyRecord(path.name)

    monkeypatch.setattr("samples2llm.core.packager.extract_sample", fake_extract)

    captured: dict[str, object] = {}

    def fake_json(records, tree, output_cfg):
        captured["records"] = records
        return "JSON_OK"

    monkeypatch.setattr("samples2llm.core.packager.generate_json", fake_json)
    monkeypatch.setattr("samples2llm.core.packager.AnalysisCache", lambda root, extraction_cfg: fake_cache)
    monkeypatch.setattr("samples2llm.core.packager.build_file_fingerprint", lambda p: f"fp:{p.name}")

    out = pack(tmp_path, cfg)

    assert out == "JSON_OK"
    assert [r.name for r in captured["records"]] == ["ok.wav", "ok2.wav"]
    assert fake_cache.put_calls == ["ok.wav", "ok2.wav"]


def test_pack_uses_cached_record_without_extract(monkeypatch, tmp_path: Path) -> None:
    cfg = Samples2LlmConfig()
    cfg.output.style = OutputStyle.JSON

    paths = [tmp_path / "cached.wav", tmp_path / "fresh.wav"]
    fake_cache = _FakeCache(tmp_path, cfg.extraction)
    fake_cache.get_map = {"cached.wav": _DummyRecord("cached.wav")}

    monkeypatch.setattr("samples2llm.core.packager.search_samples", lambda root, input_cfg: paths)
    monkeypatch.setattr("samples2llm.core.packager.generate_directory_tree", lambda sample_paths, root: "TREE")
    monkeypatch.setattr("samples2llm.core.packager.AnalysisCache", lambda root, extraction_cfg: fake_cache)
    monkeypatch.setattr("samples2llm.core.packager.build_file_fingerprint", lambda p: f"fp:{p.name}")

    extract_calls: list[str] = []

    def fake_extract(path, root, extraction_cfg):
        extract_calls.append(path.name)
        return _DummyRecord(path.name)

    monkeypatch.setattr("samples2llm.core.packager.extract_sample", fake_extract)

    captured: dict[str, object] = {}

    def fake_json(records, tree, output_cfg):
        captured["records"] = records
        return "JSON_OK"

    monkeypatch.setattr("samples2llm.core.packager.generate_json", fake_json)

    out = pack(tmp_path, cfg)

    assert out == "JSON_OK"
    assert extract_calls == ["fresh.wav"]
    assert [r.name for r in captured["records"]] == ["cached.wav", "fresh.wav"]
    assert fake_cache.put_calls == ["fresh.wav"]
