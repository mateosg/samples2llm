from __future__ import annotations

from pathlib import Path

from samples2llm.config.schema import (
    DeclaredMetadata,
    Dynamics,
    ExtractionConfig,
    FileMetadata,
    Integrity,
    Rhythmic,
    SampleRecord,
    Spatial,
    SpatialWidth,
    Spectral,
    SpectralTrend,
    Tonal,
)
from samples2llm.core.cache import AnalysisCache, build_file_fingerprint


def _record(path: str) -> SampleRecord:
    return SampleRecord(
        path=path,
        filename=path.split("/")[-1],
        directory_context=["Drums"],
        declared=DeclaredMetadata(type_hints=["drum"]),
        file_metadata=FileMetadata(
            format="wav",
            duration_sec=1.0,
            sample_rate=44100,
            bit_depth=16,
            channels=1,
            file_size_bytes=1024,
        ),
        dynamics=Dynamics(
            rms=0.2,
            peak=0.8,
            crest_factor=4.0,
            leading_silence_sec=0.0,
            trailing_silence_sec=0.0,
            attack_time_sec=0.01,
        ),
        spectral=Spectral(
            centroid_hz=1000.0,
            rolloff_hz=2000.0,
            bandwidth_hz=500.0,
            flatness=0.1,
            zero_crossing_rate=0.05,
            centroid_trend=SpectralTrend.STABLE,
            centroid_trend_confidence=0.9,
        ),
        tonal=Tonal(pitch_detected=None, pitch_confidence=None, harmonic_percussive_ratio=0.2),
        rhythmic=Rhythmic(
            bpm_detected=None,
            onset_count=1,
            onset_density=1.0,
            is_static=True,
            tempo_periodicity_strength=0.1,
        ),
        spatial=Spatial(channels=1, width_ratio=None, width_category=SpatialWidth.MONO),
        integrity=Integrity(clipping_detected=False, is_silent=False, channel_redundancy=None),
        derived_tags=["one_shot"],
        confidence_flags=[],
        summary="one-shot, percussive, mono",
        embeddings=None,
    )


def test_build_file_fingerprint_changes_when_file_changes(tmp_path: Path) -> None:
    p = tmp_path / "a.wav"
    p.write_bytes(b"abc")
    fp_1 = build_file_fingerprint(p)

    p.write_bytes(b"abcd")
    fp_2 = build_file_fingerprint(p)

    assert fp_1 != fp_2


def test_cache_roundtrip_hit(tmp_path: Path) -> None:
    p = tmp_path / "a.wav"
    p.write_bytes(b"abc")
    fp = build_file_fingerprint(p)

    cache = AnalysisCache(tmp_path, ExtractionConfig())
    cache.put(p, fp, _record("a.wav"))
    cache.save()

    cache_reloaded = AnalysisCache(tmp_path, ExtractionConfig())
    cached = cache_reloaded.get(p, fp)

    assert cached is not None
    assert cached.path == "a.wav"


def test_cache_miss_when_fingerprint_changes(tmp_path: Path) -> None:
    p = tmp_path / "a.wav"
    p.write_bytes(b"abc")
    fp_1 = build_file_fingerprint(p)

    cache = AnalysisCache(tmp_path, ExtractionConfig())
    cache.put(p, fp_1, _record("a.wav"))
    cache.save()

    p.write_bytes(b"changed")
    fp_2 = build_file_fingerprint(p)

    cache_reloaded = AnalysisCache(tmp_path, ExtractionConfig())
    assert cache_reloaded.get(p, fp_2) is None


def test_cache_invalidated_by_extraction_config_change(tmp_path: Path) -> None:
    p = tmp_path / "a.wav"
    p.write_bytes(b"abc")
    fp = build_file_fingerprint(p)

    cfg_a = ExtractionConfig(enable_tonal=True, enable_rhythmic=True, enable_embeddings=False)
    cfg_b = ExtractionConfig(enable_tonal=False, enable_rhythmic=True, enable_embeddings=False)

    cache_a = AnalysisCache(tmp_path, cfg_a)
    cache_a.put(p, fp, _record("a.wav"))
    cache_a.save()

    cache_b = AnalysisCache(tmp_path, cfg_b)
    assert cache_b.get(p, fp) is None


def test_cache_prune_removes_stale_entries(tmp_path: Path) -> None:
    p1 = tmp_path / "a.wav"
    p2 = tmp_path / "b.wav"
    p1.write_bytes(b"aaa")
    p2.write_bytes(b"bbb")

    cache = AnalysisCache(tmp_path, ExtractionConfig())
    cache.put(p1, build_file_fingerprint(p1), _record("a.wav"))
    cache.put(p2, build_file_fingerprint(p2), _record("b.wav"))
    cache.prune_to_paths([p1])
    cache.save()

    reloaded = AnalysisCache(tmp_path, ExtractionConfig())
    assert reloaded.get(p1, build_file_fingerprint(p1)) is not None
    assert reloaded.get(p2, build_file_fingerprint(p2)) is None


def test_cache_load_ignores_corrupt_json_file(tmp_path: Path) -> None:
    cache_file = tmp_path / ".samples2llm.cache.json"
    cache_file.write_text("{not-valid-json", encoding="utf-8")

    cache = AnalysisCache(tmp_path, ExtractionConfig())

    # Must not raise on construction; corrupt cache is ignored.
    assert cache_file.exists()


def test_cache_get_returns_none_for_invalid_cached_record_payload(tmp_path: Path) -> None:
    p = tmp_path / "a.wav"
    p.write_bytes(b"abc")
    fp = build_file_fingerprint(p)

    cache_file = tmp_path / ".samples2llm.cache.json"
    cache_file.write_text(
        '{\n'
        '  "schema_version": 1,\n'
        '  "config_signature": {"enable_tonal": true, "enable_rhythmic": true, "enable_embeddings": false},\n'
        '  "entries": {\n'
        '    "a.wav": {"fingerprint": "' + fp + '", "record": "not-a-dict"}\n'
        "  }\n"
        "}\n",
        encoding="utf-8",
    )

    cache = AnalysisCache(tmp_path, ExtractionConfig())
    assert cache.get(p, fp) is None
