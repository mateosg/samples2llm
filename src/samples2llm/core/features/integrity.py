from __future__ import annotations

import numpy as np

from samples2llm.config.schema import Integrity

_CLIPPING_THRESHOLD = 0.999
_SILENCE_RMS_THRESHOLD = 1e-5


def extract_integrity(y_channels: np.ndarray, sr: int) -> Integrity:
    """y_channels: array de forma (channels, samples) SIN mezclar a mono --
    a diferencia de los otros extractores, aqui necesitamos los canales
    por separado para detectar clipping y redundancia estereo real."""
    if y_channels.ndim == 1:
        y_channels = y_channels[np.newaxis, :]

    clipping_detected = bool(np.any(np.abs(y_channels) >= _CLIPPING_THRESHOLD))

    overall_rms = float(np.sqrt(np.mean(y_channels.astype(np.float64) ** 2)))
    is_silent = overall_rms < _SILENCE_RMS_THRESHOLD

    channel_redundancy = None
    if y_channels.shape[0] == 2:
        channel_redundancy = bool(np.allclose(y_channels[0], y_channels[1], atol=1e-4))

    return Integrity(
        clipping_detected=clipping_detected,
        is_silent=is_silent,
        channel_redundancy=channel_redundancy,
    )
