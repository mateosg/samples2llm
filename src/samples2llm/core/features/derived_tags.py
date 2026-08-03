from __future__ import annotations

from samples2llm.config.schema import Dynamics, FileMetadata, Rhythmic, Tonal

_LOOP_MIN_DURATION_SEC = 1.5
_LOW_FREQ_CENTROID_HZ = 400
_HARMONIC_RATIO_THRESHOLD = 0.5

# Encontrado probando contra ESC-50 (audio real, no sintetico): "rain.wav"
# salia con pitch_confidence=0.01 (basicamente ruido) pero se etiquetaba
# "tonal" igualmente, porque solo se comprobaba pitch_detected is not None,
# sin mirar la confianza. 0.3 es provisional -- separa correctamente los
# 2 casos reales observados (rain=0.01 debe fallar, siren=0.541 debe pasar)
# pero no esta calibrado contra un dataset mayor.
_MIN_PITCH_CONFIDENCE_FOR_TONAL_TAG = 0.3


def classify_length(duration_sec: float, is_static: bool) -> str:
    """Unica fuente de verdad para esta clasificacion -- usada tanto por
    derive_tags como por summarize.generate_summary. Estaban implementadas
    por separado y divergieron: un sonido de 2s con un solo onset (un
    whoosh, riser, drone) no encajaba en ninguna de las dos categorias
    originales (duracion>=1.5 + ritmico -> loop; duracion<1.5 -> one_shot),
    y cada modulo lo resolvia de forma distinta e inconsistente entre si.
    'sustained' cubre ese hueco: largo pero sin pulso repetido."""
    if duration_sec < _LOOP_MIN_DURATION_SEC:
        return "one_shot"
    if not is_static:
        return "loop"
    return "sustained"


def classify_timbre(tonal: Tonal) -> str | None:
    """Unica fuente de verdad, igual que classify_length -- evita repetir el
    error de tener la misma decision implementada por separado en dos
    modulos (ya paso con la clasificacion de duracion, ver classify_length)."""
    ratio = tonal.harmonic_percussive_ratio
    if ratio is None:
        return None
    is_confidently_pitched = (
        tonal.pitch_detected is not None
        and (tonal.pitch_confidence or 0) >= _MIN_PITCH_CONFIDENCE_FOR_TONAL_TAG
    )
    if ratio > _HARMONIC_RATIO_THRESHOLD and is_confidently_pitched:
        return "tonal"
    if ratio > _HARMONIC_RATIO_THRESHOLD:
        return "harmonic_no_stable_pitch"
    return "percussive"


def derive_tags(
    file_metadata: FileMetadata,
    dynamics: Dynamics,
    tonal: Tonal,
    rhythmic: Rhythmic,
    spectral_centroid_hz: float,
) -> list[str]:
    """Heuristica pura basada en umbrales, sin ML. Intencionadamente
    conservadora: es preferible que un tag falte a que sea incorrecto,
    ya que este campo se presenta al LLM como una afirmacion, no como
    una probabilidad (a diferencia de confidence_flags)."""
    tags: list[str] = []

    length_tag = classify_length(file_metadata.duration_sec, rhythmic.is_static)
    tags.append(length_tag)
    if length_tag == "one_shot" and rhythmic.onset_count is not None:
        tags.append("multi_hit" if rhythmic.onset_count >= 2 else "single_hit")

    timbre_tag = classify_timbre(tonal)
    if timbre_tag is not None:
        tags.append(timbre_tag)

    if spectral_centroid_hz < _LOW_FREQ_CENTROID_HZ:
        tags.append("low_frequency")

    return tags
