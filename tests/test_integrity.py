"""integrity.py tambien es matematica pura (umbrales fijos sobre la señal
cruda) -- ground truth exacto por construccion en todos los casos."""

from __future__ import annotations

import numpy as np

from samples2llm.core.features.integrity import extract_integrity


def test_signal_below_clipping_threshold_not_flagged():
    y = np.array([[0.9, -0.9, 0.5]], dtype=np.float32)
    result = extract_integrity(y, sr=22050)
    assert result.clipping_detected is False


def test_signal_at_clipping_threshold_is_flagged():
    """0.999 es exactamente el umbral (>=) definido en el modulo -- caso
    borde deliberado, no un valor arbitrario."""
    y = np.array([[0.999, 0.0, -0.5]], dtype=np.float32)
    result = extract_integrity(y, sr=22050)
    assert result.clipping_detected is True


def test_signal_above_clipping_threshold_is_flagged():
    y = np.array([[0.1, -1.0, 0.2]], dtype=np.float32)
    result = extract_integrity(y, sr=22050)
    assert result.clipping_detected is True


def test_fully_silent_signal_is_flagged_as_silent():
    y = np.zeros((1, 5000), dtype=np.float32)
    result = extract_integrity(y, sr=22050)
    assert result.is_silent is True


def test_loud_signal_is_not_flagged_as_silent():
    y = np.full((1, 5000), 0.3, dtype=np.float32)
    result = extract_integrity(y, sr=22050)
    assert result.is_silent is False


def test_mono_input_has_no_channel_redundancy_field():
    """Con 1 solo canal no hay "otro canal" con el que comparar --
    channel_redundancy debe quedar en None, no en True ni False."""
    y = np.array([[0.1, 0.2, 0.3]], dtype=np.float32)
    result = extract_integrity(y, sr=22050)
    assert result.channel_redundancy is None


def test_identical_stereo_channels_flagged_as_redundant():
    """L y R EXACTAMENTE iguales por construccion -> redundancy=True,
    caso inequivoco (esto es literalmente un archivo mono duplicado)."""
    mono = np.linspace(-0.5, 0.5, 4000, dtype=np.float32)
    y = np.stack([mono, mono])
    result = extract_integrity(y, sr=22050)
    assert result.channel_redundancy is True


def test_decorrelated_stereo_channels_not_flagged_as_redundant():
    """L y R generados con ruido independiente (semillas distintas) ->
    no deben ser considerados redundantes."""
    rng = np.random.default_rng(1)
    left = rng.uniform(-0.5, 0.5, 4000).astype(np.float32)
    right = rng.uniform(-0.5, 0.5, 4000).astype(np.float32)
    y = np.stack([left, right])
    result = extract_integrity(y, sr=22050)
    assert result.channel_redundancy is False


def test_mono_1d_input_is_handled_without_crashing():
    """extract_integrity documenta que puede recibir un array 1D (se
    normaliza internamente con newaxis) -- verificado explicitamente,
    no asumido."""
    y = np.array([0.1, 0.2, -0.9995], dtype=np.float32)
    result = extract_integrity(y, sr=22050)
    assert result.clipping_detected is True
    assert result.channel_redundancy is None
