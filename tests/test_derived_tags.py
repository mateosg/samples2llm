"""derived_tags.py es logica pura (umbrales sobre numeros ya calculados),
sin dependencia de audio ni de librerias externas -- por eso se puede (y
se debe) testear al 100%, cubriendo cada rama de cada funcion.
"""

from __future__ import annotations

from samples2llm.config.schema import Dynamics, FileMetadata, Rhythmic, Tonal
from samples2llm.core.features.derived_tags import classify_length, classify_timbre, derive_tags


# ---------------------------------------------------------------------
# classify_length -- unica fuente de verdad tras el bug de divergencia
# derived_tags.py / summarize.py de la ronda anterior
# ---------------------------------------------------------------------

def test_short_duration_is_always_one_shot_regardless_of_static_flag():
    assert classify_length(0.5, is_static=True) == "one_shot"
    assert classify_length(0.5, is_static=False) == "one_shot"


def test_long_duration_with_rhythm_is_loop():
    assert classify_length(2.0, is_static=False) == "loop"


def test_long_duration_without_rhythm_is_sustained():
    """Este es exactamente el hueco que causo el bug: 2s + is_static=True
    (el caso whoosh) no era ni loop ni one_shot antes de añadir 'sustained'."""
    assert classify_length(2.0, is_static=True) == "sustained"


def test_boundary_duration_exactly_at_threshold_is_not_one_shot():
    """1.5s es el umbral (duration_sec < 1.5 -> one_shot). Exactamente en
    el umbral debe caer del lado 'no es one_shot' (comparacion estricta <)."""
    assert classify_length(1.5, is_static=False) == "loop"
    assert classify_length(1.5, is_static=True) == "sustained"


def test_boundary_duration_just_below_threshold_is_one_shot():
    assert classify_length(1.499, is_static=False) == "one_shot"


# ---------------------------------------------------------------------
# classify_timbre
# ---------------------------------------------------------------------

def test_no_harmonic_ratio_returns_none():
    tonal = Tonal(pitch_detected=None, pitch_confidence=None, harmonic_percussive_ratio=None)
    assert classify_timbre(tonal) is None


def test_high_ratio_with_confident_pitch_is_tonal():
    tonal = Tonal(pitch_detected="A4", pitch_confidence=0.9, harmonic_percussive_ratio=0.8)
    assert classify_timbre(tonal) == "tonal"


def test_high_ratio_with_low_pitch_confidence_is_harmonic_no_stable_pitch():
    """Caso exacto del bug encontrado con rain.wav: ratio alto pero
    pitch_confidence bajo -> NO debe etiquetarse 'tonal'."""
    tonal = Tonal(pitch_detected="C1", pitch_confidence=0.01, harmonic_percussive_ratio=0.67)
    assert classify_timbre(tonal) == "harmonic_no_stable_pitch"


def test_high_ratio_without_pitch_detected_is_harmonic_no_stable_pitch():
    """Caso exacto del helicoptero: ratio armonico alto (0.67) pero sin
    pitch_detected en absoluto."""
    tonal = Tonal(pitch_detected=None, pitch_confidence=None, harmonic_percussive_ratio=0.67)
    assert classify_timbre(tonal) == "harmonic_no_stable_pitch"


def test_low_ratio_is_percussive_regardless_of_pitch_confidence():
    tonal = Tonal(pitch_detected="A4", pitch_confidence=0.95, harmonic_percussive_ratio=0.2)
    assert classify_timbre(tonal) == "percussive"


def test_ratio_exactly_at_threshold_is_not_tonal_side():
    """_HARMONIC_RATIO_THRESHOLD=0.5, comparacion estricta ratio > 0.5 ->
    exactamente 0.5 debe caer del lado 'percussive'."""
    tonal = Tonal(pitch_detected="A4", pitch_confidence=0.9, harmonic_percussive_ratio=0.5)
    assert classify_timbre(tonal) == "percussive"


def test_pitch_confidence_exactly_at_threshold_counts_as_confident():
    """_MIN_PITCH_CONFIDENCE_FOR_TONAL_TAG=0.3, comparacion >= -> 0.3
    exacto debe contar como 'confiado' (side tonal, no harmonic_no_stable_pitch)."""
    tonal = Tonal(pitch_detected="A4", pitch_confidence=0.3, harmonic_percussive_ratio=0.8)
    assert classify_timbre(tonal) == "tonal"


# ---------------------------------------------------------------------
# derive_tags -- integracion de las dos clasificaciones + low_frequency
# ---------------------------------------------------------------------

def _file_meta(duration_sec: float) -> FileMetadata:
    return FileMetadata(format="wav", duration_sec=duration_sec, sample_rate=44100,
                         bit_depth=16, channels=1, file_size_bytes=1000)


def _dynamics() -> Dynamics:
    return Dynamics(rms=0.1, peak=0.5, crest_factor=5.0,
                     leading_silence_sec=0.0, trailing_silence_sec=0.0)


def test_one_shot_with_single_onset_is_tagged_single_hit():
    tags = derive_tags(
        _file_meta(0.3), _dynamics(), Tonal(), Rhythmic(onset_count=1, is_static=True),
        spectral_centroid_hz=1000.0,
    )
    assert "one_shot" in tags
    assert "single_hit" in tags
    assert "multi_hit" not in tags


def test_one_shot_with_multiple_onsets_is_tagged_multi_hit():
    tags = derive_tags(
        _file_meta(0.3), _dynamics(), Tonal(), Rhythmic(onset_count=3, is_static=True),
        spectral_centroid_hz=1000.0,
    )
    assert "multi_hit" in tags
    assert "single_hit" not in tags


def test_loop_is_not_tagged_single_or_multi_hit():
    """single_hit/multi_hit solo tiene sentido para one_shot -- un loop no
    debe llevar ninguno de los dos, aunque tenga muchos onsets."""
    tags = derive_tags(
        _file_meta(3.0), _dynamics(), Tonal(), Rhythmic(onset_count=8, is_static=False),
        spectral_centroid_hz=1000.0,
    )
    assert "loop" in tags
    assert "single_hit" not in tags
    assert "multi_hit" not in tags


def test_low_centroid_gets_low_frequency_tag():
    tags = derive_tags(
        _file_meta(0.3), _dynamics(), Tonal(), Rhythmic(onset_count=1, is_static=True),
        spectral_centroid_hz=100.0,
    )
    assert "low_frequency" in tags


def test_high_centroid_does_not_get_low_frequency_tag():
    tags = derive_tags(
        _file_meta(0.3), _dynamics(), Tonal(), Rhythmic(onset_count=1, is_static=True),
        spectral_centroid_hz=5000.0,
    )
    assert "low_frequency" not in tags


def test_timbre_tag_included_when_present():
    tonal = Tonal(pitch_detected="A2", pitch_confidence=0.9, harmonic_percussive_ratio=0.9)
    tags = derive_tags(
        _file_meta(0.3), _dynamics(), tonal, Rhythmic(onset_count=1, is_static=True),
        spectral_centroid_hz=1000.0,
    )
    assert "tonal" in tags
