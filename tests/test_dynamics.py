"""dynamics.py es matematica pura sobre la waveform (RMS, peak, silencios).
No depende de ninguna heuristica de libreria -- por eso aqui SI se puede
exigir un ground truth exacto, calculado a mano, no aproximado.
"""

from __future__ import annotations

import numpy as np
import pytest

from samples2llm.core.features.dynamics import extract_dynamics


def test_empty_signal_returns_zeroed_record():
    result = extract_dynamics(np.array([], dtype=np.float32), sr=22050)
    assert result.rms == 0.0
    assert result.peak == 0.0
    assert result.crest_factor == 0.0


def test_constant_amplitude_square_wave_has_crest_factor_one():
    """Una señal de amplitud CONSTANTE (cuadrada, sin silencio) tiene
    peak == rms por definicion matematica -> crest_factor == 1.0 exacto."""
    y = np.array([0.5, -0.5] * 1000, dtype=np.float32)
    result = extract_dynamics(y, sr=22050)
    assert result.rms == pytest.approx(0.5, abs=1e-6)
    assert result.peak == pytest.approx(0.5, abs=1e-6)
    assert result.crest_factor == pytest.approx(1.0, abs=1e-4)


def test_single_impulse_rms_and_peak_are_computable_by_hand():
    """1 muestra a 1.0 entre 9999 ceros: rms = sqrt(1/10000) = 0.01 exacto,
    peak = 1.0 exacto, crest_factor = peak/rms = 100.0 exacto."""
    y = np.zeros(10000, dtype=np.float32)
    y[5000] = 1.0
    result = extract_dynamics(y, sr=22050)
    assert result.rms == pytest.approx(0.01, abs=1e-5)
    assert result.peak == pytest.approx(1.0, abs=1e-6)
    assert result.crest_factor == pytest.approx(100.0, abs=0.5)


def test_leading_and_trailing_silence_measured_exactly():
    """500 ceros + pulso de 1000 muestras + 300 ceros, a sr=1000Hz:
    leading_silence_sec = 500/1000 = 0.5s, trailing = 300/1000 = 0.3s,
    ambos exactos por construccion (no aproximados)."""
    sr = 1000
    pulse = np.ones(1000, dtype=np.float32) * 0.5
    y = np.concatenate([np.zeros(500, dtype=np.float32), pulse, np.zeros(300, dtype=np.float32)])
    result = extract_dynamics(y, sr=sr)
    assert result.leading_silence_sec == pytest.approx(0.5, abs=1e-3)
    assert result.trailing_silence_sec == pytest.approx(0.3, abs=1e-3)


def test_fully_silent_signal_has_leading_equal_trailing_equal_full_duration():
    sr = 1000
    y = np.zeros(2000, dtype=np.float32)
    result = extract_dynamics(y, sr=sr)
    assert result.leading_silence_sec == pytest.approx(2.0, abs=1e-3)
    assert result.trailing_silence_sec == pytest.approx(2.0, abs=1e-3)
    assert result.rms == 0.0


def test_attack_time_measured_exactly_for_linear_ramp():
    """Rampa lineal de 0 a 1.0 en exactamente 100 muestras, sr=1000Hz:
    el pico esta en la muestra 99 (la ultima de la rampa) y el primer
    sample audible (>1e-4) esta cerca de la muestra 1 -- attack_time_sec
    debe ser ~99/1000 = 0.099s, calculable a mano."""
    sr = 1000
    ramp = np.linspace(0, 1.0, 100, dtype=np.float32)
    y = np.concatenate([ramp, np.zeros(50, dtype=np.float32)])
    result = extract_dynamics(y, sr=sr)
    assert result.attack_time_sec == pytest.approx(0.099, abs=0.003)


def test_instantaneous_attack_for_single_sample_impulse():
    """Impulso instantaneo (1 sola muestra no-cero): el primer sample
    audible ES el pico -> attack_time_sec debe ser exactamente 0.0."""
    y = np.zeros(1000, dtype=np.float32)
    y[10] = 0.8
    result = extract_dynamics(y, sr=22050)
    assert result.attack_time_sec == pytest.approx(0.0, abs=1e-6)
