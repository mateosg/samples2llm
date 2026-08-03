from __future__ import annotations

import numpy as np

from samples2llm.config.schema import Dynamics

_SILENCE_THRESHOLD = 1e-4


def extract_dynamics(y: np.ndarray, sr: int) -> Dynamics:
    """y: waveform mono, ya cargada. sr: sample rate."""
    if y.size == 0:
        return Dynamics(rms=0.0, peak=0.0, crest_factor=0.0,
                         leading_silence_sec=0.0, trailing_silence_sec=0.0)

    rms = float(np.sqrt(np.mean(y.astype(np.float64) ** 2)))
    peak = float(np.max(np.abs(y)))
    crest_factor = round(peak / rms, 4) if rms > 0 else 0.0

    above = np.abs(y) > _SILENCE_THRESHOLD
    nonzero_idx = np.flatnonzero(above)

    if nonzero_idx.size == 0:
        leading_silence_sec = round(len(y) / sr, 6)
        trailing_silence_sec = leading_silence_sec
    else:
        leading_silence_sec = round(nonzero_idx[0] / sr, 6)
        trailing_silence_sec = round((len(y) - 1 - nonzero_idx[-1]) / sr, 6)

    # Tiempo de ataque: desde el primer sample audible hasta el pico maximo.
    attack_time_sec = None
    if nonzero_idx.size > 0:
        peak_idx = int(np.argmax(np.abs(y)))
        start_idx = int(nonzero_idx[0])
        if peak_idx >= start_idx:
            attack_time_sec = round((peak_idx - start_idx) / sr, 6)

    return Dynamics(
        rms=round(rms, 6),
        peak=round(peak, 6),
        crest_factor=crest_factor,
        leading_silence_sec=leading_silence_sec,
        trailing_silence_sec=trailing_silence_sec,
        attack_time_sec=attack_time_sec,
    )
