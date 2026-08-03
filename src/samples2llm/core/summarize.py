"""
Resumen en lenguaje natural ensamblado por PLANTILLA a partir de campos ya
calculados -- deliberadamente NO intenta nombrar el tipo de sonido
("whoosh", "kick", etc.), porque eso requeriria un clasificador entrenado
que este proyecto no tiene. Ver discusion en el chat sobre la propuesta
externa: esta es la parte de esa propuesta que SI es alcanzable sin
dependencias nuevas -- la parte de clasificacion semantica no lo es.

PROVISIONAL: los umbrales de categorizacion (attack rapido/medio/lento,
brillo alto/medio/bajo) estan calibrados a ojo contra 2-3 samples
sinteticos, no contra una libreria real. Es el tipo de cosa que ya nos
mordio una vez con el heuristico de BPM -- revisar con datos reales antes
de confiar en esto para decisiones automatizadas.
"""

from __future__ import annotations

from samples2llm.config.schema import SampleRecord, SpectralTrend
from samples2llm.core.features.derived_tags import classify_length, classify_timbre

_FAST_ATTACK_SEC = 0.01
_SLOW_ATTACK_SEC = 0.1

_LOW_BRIGHTNESS_HZ = 500
_HIGH_BRIGHTNESS_HZ = 3000


def _attack_label(attack_sec: float | None) -> str:
    if attack_sec is None:
        return "unknown attack"
    if attack_sec < _FAST_ATTACK_SEC:
        return "fast attack"
    if attack_sec < _SLOW_ATTACK_SEC:
        return "medium attack"
    return "slow attack"


def _brightness_label(centroid_hz: float) -> str:
    if centroid_hz < _LOW_BRIGHTNESS_HZ:
        return "low"
    if centroid_hz < _HIGH_BRIGHTNESS_HZ:
        return "mid"
    return "high"


def _trend_phrase(trend: SpectralTrend | None) -> str | None:
    if trend is None or trend == SpectralTrend.STABLE:
        return None
    return f"{trend.value} spectral centroid"


def generate_summary(record: SampleRecord) -> str:
    parts: list[str] = []

    duration = record.file_metadata.duration_sec
    length_tag = classify_length(duration, record.rhythmic.is_static)
    length_word = length_tag.replace("_", "-")
    parts.append(f"{length_word} ({duration}s)")

    parts.append(_attack_label(record.dynamics.attack_time_sec))

    brightness = _brightness_label(record.spectral.centroid_hz)
    parts.append(f"{brightness} brightness")

    trend_phrase = _trend_phrase(record.spectral.centroid_trend)
    if trend_phrase:
        parts.append(trend_phrase)

    timbre_tag = classify_timbre(record.tonal)
    if timbre_tag == "tonal":
        parts.append(f"tonal, pitch ~{record.tonal.pitch_detected}")
    elif timbre_tag == "percussive":
        parts.append("percussive, no stable pitch")
    elif timbre_tag == "harmonic_no_stable_pitch":
        parts.append("harmonic texture, no stable pitch")

    if record.rhythmic.onset_count and record.rhythmic.onset_count >= 2 and length_tag == "one_shot":
        parts.append(f"{record.rhythmic.onset_count} layered transients")

    if record.spatial.width_category.value != "mono":
        parts.append(f"{record.spatial.width_category.value} stereo image")

    if record.integrity.clipping_detected:
        parts.append("clipping detected")

    text = ", ".join(parts)
    # NO usar str.capitalize(): ademas de poner en mayuscula la primera
    # letra, fuerza el RESTO del string a minusculas -- eso corrompia
    # nombres de nota detectados (p.ej. "pitch ~A4" pasaba a "pitch ~a4",
    # "F#3" a "f#3"). Solo se debe tocar el primer caracter.
    if text:
        text = text[0].upper() + text[1:]
    return text + "."
