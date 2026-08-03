from __future__ import annotations

import librosa
import numpy as np

from samples2llm.config.schema import Spectral, SpectralTrend

# Umbral de pendiente relativa antes de considerar que hay tendencia real
# (no solo ruido de frame a frame). Se compara el cambio total estimado por
# el ajuste lineal contra el propio valor medio del centroide.
# PROVISIONAL: calibrado a ojo contra 2 casos sinteticos (ver
# examples/sample_library), no contra una libreria real. Revisar si en uso
# real "stable" sale con muy poca frecuencia o al reves.
_TREND_RELATIVE_THRESHOLD = 0.15
_MIN_FRAMES_FOR_TREND = 8


def _classify_trend(centroid_frames: np.ndarray, sr: int, hop_length: int) -> tuple[SpectralTrend | None, float | None]:
    n = centroid_frames.shape[-1]
    if n < _MIN_FRAMES_FOR_TREND:
        return None, None

    times = librosa.frames_to_time(np.arange(n), sr=sr, hop_length=hop_length)
    values = centroid_frames.flatten()

    # Ajuste lineal centroid ~ a*t + b
    slope, intercept = np.polyfit(times, values, 1)
    predicted = slope * times + intercept
    ss_res = np.sum((values - predicted) ** 2)
    ss_tot = np.sum((values - np.mean(values)) ** 2)
    r_squared = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    mean_value = float(np.mean(values))
    total_change = slope * (times[-1] - times[0])
    relative_change = total_change / mean_value if mean_value > 0 else 0.0

    if abs(relative_change) < _TREND_RELATIVE_THRESHOLD:
        trend = SpectralTrend.STABLE
    elif relative_change > 0:
        trend = SpectralTrend.ASCENDING
    else:
        trend = SpectralTrend.DESCENDING

    return trend, round(max(r_squared, 0.0), 4)


def extract_spectral(y: np.ndarray, sr: int) -> Spectral:
    if y.size == 0:
        return Spectral(centroid_hz=0.0, rolloff_hz=0.0, bandwidth_hz=0.0,
                         flatness=0.0, zero_crossing_rate=0.0)

    hop_length = 512
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    flatness = librosa.feature.spectral_flatness(y=y)
    zcr = librosa.feature.zero_crossing_rate(y=y)

    trend, trend_confidence = _classify_trend(centroid, sr, hop_length)

    return Spectral(
        centroid_hz=round(float(np.mean(centroid)), 2),
        rolloff_hz=round(float(np.mean(rolloff)), 2),
        bandwidth_hz=round(float(np.mean(bandwidth)), 2),
        flatness=round(float(np.mean(flatness)), 6),
        zero_crossing_rate=round(float(np.mean(zcr)), 6),
        centroid_trend=trend,
        centroid_trend_confidence=trend_confidence,
    )
