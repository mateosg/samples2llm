"""tonal.py: ground truth mas fuerte disponible es un tono puro a 440Hz --
el diapason estandar (A4), un hecho externo objetivo, no una suposicion.
Si un seno puro a 440Hz no se detecta como A4 con alta confianza, algo
esta mal en el pipeline de deteccion de pitch, independientemente de lo
que "parezca razonable"."""

from __future__ import annotations

import numpy as np

from samples2llm.core.features.tonal import extract_tonal
from tests.conftest import sine_wave, white_noise


def test_pure_440hz_tone_detected_as_a4_with_high_confidence():
    y = sine_wave(440.0, duration_sec=1.0, sr=22050, amplitude=0.6)
    result = extract_tonal(y, sr=22050)
    assert result.pitch_detected == "A4"
    assert result.pitch_confidence > 0.7


def test_pure_tone_has_high_harmonic_percussive_ratio():
    """Un tono puro sostenido es, por definicion, harmonico -- casi toda
    su energia debe caer del lado "harmonic" de la separacion hpss."""
    y = sine_wave(440.0, duration_sec=1.0, sr=22050, amplitude=0.6)
    result = extract_tonal(y, sr=22050)
    assert result.harmonic_percussive_ratio > 0.7


def test_white_noise_does_not_yield_confident_pitch():
    """Ruido blanco no tiene una frecuencia fundamental real -- no debe
    reportarse un pitch con alta confianza. (Este es exactamente el bug
    encontrado antes con rain.wav: pitch_confidence=0.01 pero se
    etiquetaba igual "tonal" en otro modulo -- aqui se verifica que el
    propio extractor no miente sobre la confianza, sea la que sea)."""
    y = white_noise(duration_sec=1.0, sr=22050, amplitude=0.4)
    result = extract_tonal(y, sr=22050)
    if result.pitch_detected is not None:
        assert result.pitch_confidence < 0.3


def test_signal_shorter_than_50ms_returns_empty_tonal():
    """Por debajo del umbral minimo para pyin (50ms), no se debe intentar
    detectar pitch -- debe devolver el record vacio, no un valor
    inventado sobre datos insuficientes."""
    y = sine_wave(440.0, duration_sec=0.02, sr=22050)
    result = extract_tonal(y, sr=22050)
    assert result.pitch_detected is None
    assert result.pitch_confidence is None
    assert result.harmonic_percussive_ratio is None


def test_octave_up_tone_detected_as_different_note_than_base_tone():
    """880Hz es exactamente una octava por encima de 440Hz (A5 vs A4) --
    verifica que el detector distingue octavas, no solo la clase de nota."""
    low = extract_tonal(sine_wave(440.0, 1.0, sr=22050, amplitude=0.6), sr=22050)
    high = extract_tonal(sine_wave(880.0, 1.0, sr=22050, amplitude=0.6), sr=22050)
    assert low.pitch_detected == "A4"
    assert high.pitch_detected == "A5"


def test_silence_does_not_crash_and_yields_no_confident_pitch():
    y = np.zeros(22050, dtype=np.float32)
    result = extract_tonal(y, sr=22050)
    # No debe reventar; si por ruido de cuantizacion detecta "voiced" en
    # algun frame, no debe ser con confianza alta sobre silencio puro.
    if result.pitch_detected is not None:
        assert result.pitch_confidence < 0.5


def test_pyin_failure_does_not_crash_and_returns_safe_tonal(monkeypatch):
    def fail_pyin(*args, **kwargs):
        raise RuntimeError("simulated pyin failure")

    monkeypatch.setattr("samples2llm.core.features.tonal.librosa.pyin", fail_pyin)

    y = sine_wave(440.0, duration_sec=1.0, sr=22050, amplitude=0.6)
    result = extract_tonal(y, sr=22050)

    assert result.pitch_detected is None
    assert result.pitch_confidence is None
    # hpss debe seguir ejecutando aunque pyin falle.
    assert result.harmonic_percussive_ratio is not None
