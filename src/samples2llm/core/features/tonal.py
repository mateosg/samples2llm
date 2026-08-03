from __future__ import annotations

import librosa
import numpy as np

from samples2llm.config.schema import Tonal

_FMIN = librosa.note_to_hz("C1")
_FMAX = librosa.note_to_hz("C7")
_MIN_VOICED_RATIO = 0.1  # por debajo de esto, consideramos que no hay tono claro
_PYIN_FRAME_LENGTH = 4096  # evita warning de periodos insuficientes para fmin=C1 a 44.1kHz


def extract_tonal(y: np.ndarray, sr: int) -> Tonal:
    if y.size < sr * 0.05:  # menos de 50ms, no hay suficiente señal para pyin
        return Tonal()

    try:
        f0, voiced_flag, voiced_probs = librosa.pyin(
            y,
            fmin=_FMIN,
            fmax=_FMAX,
            sr=sr,
            frame_length=_PYIN_FRAME_LENGTH,
        )
    except Exception:
        # pyin puede fallar en casos aislados (p. ej. error interno numba).
        # No debemos perder el sample completo por un fallo puntual de pitch.
        f0 = np.array([])
        voiced_flag = np.array([], dtype=bool)
        voiced_probs = np.array([])

    pitch_detected = None
    pitch_confidence = None
    voiced_ratio = float(np.mean(voiced_flag)) if voiced_flag.size else 0.0

    if voiced_ratio >= _MIN_VOICED_RATIO:
        voiced_f0 = f0[voiced_flag]
        if voiced_f0.size > 0 and not np.all(np.isnan(voiced_f0)):
            median_f0 = float(np.nanmedian(voiced_f0))
            pitch_detected = librosa.hz_to_note(median_f0)
            pitch_confidence = round(float(np.mean(voiced_probs[voiced_flag])), 4)

    harmonic_percussive_ratio = None
    try:
        y_harmonic, y_percussive = librosa.effects.hpss(y)
        harmonic_energy = float(np.sum(y_harmonic ** 2))
        percussive_energy = float(np.sum(y_percussive ** 2))
        total = harmonic_energy + percussive_energy
        if total > 0:
            harmonic_percussive_ratio = round(harmonic_energy / total, 4)
    except Exception:
        # hpss puede fallar en señales extremadamente cortas o silenciosas;
        # no queremos que un sample problematico tumbe todo el pipeline.
        pass

    return Tonal(
        pitch_detected=pitch_detected,
        pitch_confidence=pitch_confidence,
        harmonic_percussive_ratio=harmonic_percussive_ratio,
    )
