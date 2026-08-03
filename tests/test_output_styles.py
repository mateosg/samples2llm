from __future__ import annotations

import json

from samples2llm.config.schema import (
    DeclaredMetadata,
    Dynamics,
    FileMetadata,
    Integrity,
    OutputConfig,
    Rhythmic,
    SampleRecord,
    Spatial,
    SpatialWidth,
    Spectral,
    SpectralTrend,
    Tonal,
)
from samples2llm.core.output.json_style import generate_json
from samples2llm.core.output.markdown_style import generate_markdown


def _record(path: str, duration_sec: float = 1.0) -> SampleRecord:
    return SampleRecord(
        path=path,
        filename=path.split("/")[-1],
        directory_context=["Drums"],
        declared=DeclaredMetadata(type_hints=["drum"]),
        file_metadata=FileMetadata(
            format="wav",
            duration_sec=duration_sec,
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
        embeddings={"mfcc": [1.0, 2.0]},
    )


def test_generate_json_excludes_embeddings_by_default() -> None:
    out = generate_json([_record("Drums/kick.wav")], "Drums/\n  kick.wav", OutputConfig())
    payload = json.loads(out)

    assert payload["summary"]["total_samples"] == 1
    assert payload["summary"]["total_duration_sec"] == 1.0
    assert payload["directory_structure"]
    assert "embeddings" not in payload["samples"][0]


def test_generate_json_can_include_embeddings() -> None:
    out = generate_json(
        [_record("Drums/kick.wav")],
        "Drums/\n  kick.wav",
        OutputConfig(include_embeddings=True),
    )
    payload = json.loads(out)

    assert "embeddings" in payload["samples"][0]


def test_generate_markdown_contains_expected_sections() -> None:
    md = generate_markdown([_record("Drums/kick.wav")], "Drums/\n  kick.wav")

    assert "# Sample Library" in md
    assert "## Directory Structure" in md
    assert "## Samples" in md
    assert "| Path | Summary | Tags | Flags |" in md
    assert "Drums/kick.wav" in md
    assert "one-shot, percussive, mono" in md
