from __future__ import annotations

import numpy as np

from samples2llm.config.schema import Spatial, SpatialWidth

# PROVISIONAL: umbrales calibrados a ojo, no contra una libreria real.
# width_ratio = energia del canal "side" (L-R) / energia total (mid+side).
# 0 = canales identicos, ~0.5 = L/R decorrelacionados con energia similar.
_NARROW_THRESHOLD = 0.02
_WIDE_THRESHOLD = 0.15


def extract_spatial(y_channels: np.ndarray) -> Spatial:
    """y_channels: array (channels, samples), SIN mezclar a mono."""
    channels = y_channels.shape[0]

    if channels == 1:
        return Spatial(channels=1, width_ratio=None, width_category=SpatialWidth.MONO)

    left, right = y_channels[0], y_channels[1]
    mid = (left + right) / 2
    side = (left - right) / 2

    mid_energy = float(np.mean(mid ** 2))
    side_energy = float(np.mean(side ** 2))
    total = mid_energy + side_energy

    width_ratio = round(side_energy / total, 4) if total > 0 else 0.0

    if width_ratio < _NARROW_THRESHOLD:
        category = SpatialWidth.NARROW
    elif width_ratio < _WIDE_THRESHOLD:
        category = SpatialWidth.MEDIUM
    else:
        category = SpatialWidth.WIDE

    return Spatial(channels=channels, width_ratio=width_ratio, width_category=category)
