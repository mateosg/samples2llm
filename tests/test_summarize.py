"""summarize.py ensambla texto por plantilla a partir de campos ya
calculados -- se puede testear con exactitud total construyendo records
de fixture con valores conocidos y comprobando el string resultante.
"""

from __future__ import annotations

from samples2llm.config.schema import (
    Dynamics,
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
from samples2llm.core.summarize import generate_summary


def _record(**overrides) -> SampleRecord:
    defaults = dict(
        path="x/y.wav",
        filename="y.wav",
        file_metadata=FileMetadata(format="wav", duration_sec=0.3, sample_rate=44100,
                                    bit_depth=16, channels=1, file_size_bytes=1000),
        dynamics=Dynamics(rms=0.1, peak=0.5, crest_factor=5.0, leading_silence_sec=0.0,
                           trailing_silence_sec=0.0, attack_time_sec=0.005),
        spectral=Spectral(centroid_hz=1000.0, rolloff_hz=2000.0, bandwidth_hz=500.0,
                           flatness=0.1, zero_crossing_rate=0.1),
        tonal=Tonal(),
        rhythmic=Rhythmic(onset_count=1, is_static=True),
        spatial=Spatial(channels=1, width_ratio=None, width_category=SpatialWidth.MONO),
        integrity=Integrity(),
    )
    defaults.update(overrides)
    return SampleRecord(**defaults)


def test_one_shot_fast_attack_summary_matches_expected_phrasing():
    record = _record()
    summary = generate_summary(record)
    assert summary.startswith("One-shot (0.3s), fast attack, ")
    assert summary.endswith(".")


def test_length_word_matches_classify_length_exactly():
    """Regresion directa del bug ya corregido: derive_tags y summarize
    usaban logica de duracion distinta y divergian (whoosh: 'loop' en uno,
    sin tag en el otro). Aqui se verifica que summarize usa la MISMA
    fuente de verdad (classify_length) para 'sustained'."""
    record = _record(
        file_metadata=FileMetadata(format="wav", duration_sec=2.0, sample_rate=44100,
                                    bit_depth=16, channels=1, file_size_bytes=1000),
        rhythmic=Rhythmic(onset_count=1, is_static=True),
    )
    summary = generate_summary(record)
    assert summary.startswith("Sustained (2.0s)")


def test_brightness_label_low():
    record = _record(spectral=Spectral(centroid_hz=200.0, rolloff_hz=400.0, bandwidth_hz=100.0,
                                        flatness=0.1, zero_crossing_rate=0.1))
    assert "low brightness" in generate_summary(record)


def test_brightness_label_mid():
    record = _record(spectral=Spectral(centroid_hz=1000.0, rolloff_hz=2000.0, bandwidth_hz=500.0,
                                        flatness=0.1, zero_crossing_rate=0.1))
    assert "mid brightness" in generate_summary(record)


def test_brightness_label_high():
    record = _record(spectral=Spectral(centroid_hz=4000.0, rolloff_hz=6000.0, bandwidth_hz=1000.0,
                                        flatness=0.1, zero_crossing_rate=0.1))
    assert "high brightness" in generate_summary(record)


def test_ascending_trend_phrase_included():
    record = _record(spectral=Spectral(centroid_hz=1000.0, rolloff_hz=2000.0, bandwidth_hz=500.0,
                                        flatness=0.1, zero_crossing_rate=0.1,
                                        centroid_trend=SpectralTrend.ASCENDING,
                                        centroid_trend_confidence=0.9))
    assert "ascending spectral centroid" in generate_summary(record)


def test_stable_trend_phrase_omitted():
    """STABLE explicitamente no genera frase (ver _trend_phrase) -- no
    tiene sentido decir 'stable spectral centroid' en lenguaje natural."""
    record = _record(spectral=Spectral(centroid_hz=1000.0, rolloff_hz=2000.0, bandwidth_hz=500.0,
                                        flatness=0.1, zero_crossing_rate=0.1,
                                        centroid_trend=SpectralTrend.STABLE,
                                        centroid_trend_confidence=0.9))
    assert "spectral centroid" not in generate_summary(record)


def test_tonal_timbre_includes_pitch_in_summary():
    record = _record(tonal=Tonal(pitch_detected="A4", pitch_confidence=0.9, harmonic_percussive_ratio=0.9))
    assert "tonal, pitch ~A4" in generate_summary(record)


def test_harmonic_no_stable_pitch_phrase():
    record = _record(tonal=Tonal(pitch_detected=None, pitch_confidence=None, harmonic_percussive_ratio=0.67))
    assert "harmonic texture, no stable pitch" in generate_summary(record)


def test_percussive_timbre_phrase():
    record = _record(tonal=Tonal(pitch_detected=None, pitch_confidence=None, harmonic_percussive_ratio=0.1))
    assert "percussive, no stable pitch" in generate_summary(record)


def test_multiple_layered_transients_only_mentioned_for_one_shot():
    """Regla explicita del codigo: 'N layered transients' solo aparece si
    length_tag == 'one_shot'. Un loop con muchos onsets no debe decir
    'layered transients' (eso describiria mal un loop ritmico normal)."""
    record = _record(
        file_metadata=FileMetadata(format="wav", duration_sec=4.0, sample_rate=44100,
                                    bit_depth=16, channels=1, file_size_bytes=1000),
        rhythmic=Rhythmic(onset_count=8, is_static=False),
    )
    assert "layered transients" not in generate_summary(record)


def test_multi_hit_one_shot_mentions_transient_count():
    record = _record(rhythmic=Rhythmic(onset_count=3, is_static=True))
    assert "3 layered transients" in generate_summary(record)


def test_wide_stereo_image_phrase_included():
    record = _record(spatial=Spatial(channels=2, width_ratio=0.5, width_category=SpatialWidth.WIDE))
    assert "wide stereo image" in generate_summary(record)


def test_mono_does_not_mention_stereo_image():
    record = _record()  # mono por defecto
    assert "stereo image" not in generate_summary(record)


def test_clipping_detected_phrase_included():
    record = _record(integrity=Integrity(clipping_detected=True))
    assert "clipping detected" in generate_summary(record)


def test_no_clipping_omits_phrase():
    record = _record(integrity=Integrity(clipping_detected=False))
    assert "clipping detected" not in generate_summary(record)


def test_summary_is_capitalized_and_ends_with_period():
    record = _record()
    summary = generate_summary(record)
    assert summary[0].isupper()
    assert summary.endswith(".")


def test_pitch_note_capitalization_preserved_not_lowercased():
    """Bug real encontrado al construir esta suite: el codigo original
    usaba `.capitalize()` sobre el string completo, que ademas de poner
    en mayuscula la primera letra fuerza el RESTO a minusculas -- eso
    corrompia el nombre de nota detectado ("pitch ~A4" quedaba como
    "pitch ~a4", "F#3" como "f#3"). Corregido para tocar solo el primer
    caracter del summary completo."""
    record = _record(tonal=Tonal(pitch_detected="F#3", pitch_confidence=0.9, harmonic_percussive_ratio=0.9))
    summary = generate_summary(record)
    assert "pitch ~F#3" in summary
