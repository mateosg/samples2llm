"""Fixtures y generadores de señal compartidos.

Filosofia de esta suite: para los modulos que son matematica pura sobre la
señal (dynamics, integrity, spatial) se generan señales con una propiedad
CONOCIDA DE ANTEMANO por construccion (ej. un tono puro a 440Hz, un array
de ceros con un pico exacto en la muestra N) y se comprueba el resultado
contra ese valor exacto -- no contra "lo que parece razonable". Para los
modulos que dependen de heuristicas de libreria (deteccion de pitch,
tendencia espectral) se usan señales sinteticas de propiedad conocida
(seno a frecuencia exacta, barrido ascendente/descendente) como la mejor
aproximacion disponible a un ground truth real, documentando la tolerancia
usada y por que.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
ESC50_DIR = ROOT / "examples" / "esc50_real"
LIBRARY_DIR = ROOT / "examples" / "sample_library"


def sine_wave(freq_hz: float, duration_sec: float, sr: int = 22050, amplitude: float = 0.5) -> np.ndarray:
    """Tono puro. Frecuencia y amplitud son EXACTAS por construccion."""
    t = np.linspace(0, duration_sec, int(sr * duration_sec), endpoint=False)
    return (amplitude * np.sin(2 * np.pi * freq_hz * t)).astype(np.float32)


def chirp(f0_hz: float, f1_hz: float, duration_sec: float, sr: int = 22050, amplitude: float = 0.5) -> np.ndarray:
    """Barrido lineal de frecuencia f0->f1. Sentido de la pendiente conocido
    por construccion (ascendente si f1>f0, descendente si f1<f0)."""
    import librosa

    t = np.linspace(0, duration_sec, int(sr * duration_sec), endpoint=False)
    phase = 2 * np.pi * (f0_hz * t + (f1_hz - f0_hz) * t**2 / (2 * duration_sec))
    return (amplitude * np.sin(phase)).astype(np.float32)


def white_noise(duration_sec: float, sr: int = 22050, amplitude: float = 0.3, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (amplitude * rng.uniform(-1, 1, int(sr * duration_sec))).astype(np.float32)


def silence(duration_sec: float, sr: int = 22050) -> np.ndarray:
    return np.zeros(int(sr * duration_sec), dtype=np.float32)


def padded_pulse(pulse: np.ndarray, leading_zeros: int, trailing_zeros: int) -> np.ndarray:
    """Señal con un numero EXACTO y conocido de muestras en silencio antes
    y despues de `pulse` -- para testear leading/trailing_silence_sec."""
    return np.concatenate([np.zeros(leading_zeros, dtype=np.float32), pulse, np.zeros(trailing_zeros, dtype=np.float32)])


@pytest.fixture
def sr() -> int:
    return 22050
