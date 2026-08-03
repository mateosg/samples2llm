from __future__ import annotations

import librosa
import numpy as np

from samples2llm.config.schema import Rhythmic

# Por debajo de esta duracion no tiene sentido buscar un tempo estable
# (un one-shot de kick de 400ms no tiene "bpm"). onset_count SI se calcula
# incluso por debajo de este umbral -- es util para detectar capas/golpes
# multiples dentro de un one-shot corto.
_MIN_DURATION_FOR_TEMPO_SEC = 1.5

# delta/wait de onset_detect por defecto (delta=0.07, sin wait minimo) son
# demasiado sensibles a fluctuaciones de amplitud en señales con contenido
# ruidoso -- un solo kick de percusion se detectaba como "3 onsets" y una
# textura de ruido continuo como "6 onsets", ambos falsos positivos.
# Verificado empiricamente (no por defecto de la libreria) contra 3 casos
# con resultado esperado conocido: delta=0.2 + wait de 30ms entre onsets
# corrige ambos casos sin perder el conteo correcto en el caso multi-hit
# real. PROVISIONAL igualmente -- son 3 casos sinteticos, no una libreria real.
_HOP_LENGTH = 512
_ONSET_DELTA = 0.2
_ONSET_MIN_GAP_MS = 30

# Rango de tempo plausible para la busqueda del pico de autocorrelacion.
# Fuera de este rango de lags no tiene sentido buscar "el" pico dominante.
_MIN_TEMPO_BPM = 40.0
_MAX_TEMPO_BPM = 240.0


def _tempo_periodicity_strength(onset_env: np.ndarray, sr: int) -> float | None:
    """Fuerza CRUDA del pico de autocorrelacion del onset envelope dentro
    del rango de tempo plausible, normalizada por la autocorrelacion en
    lag 0 (energia total). 0 = sin periodicidad detectable, 1 = periodicidad
    perfecta dentro del rango.

    Investigado y verificado que esto NO distingue "musical" de
    "mecanico mas no musical": helicopter.wav (0.92) y rain.wav (0.90)
    puntuan mas alto que loop_Cmaj_120bpm.wav (0.85, el unico loop musical
    real disponible), y clock_tick.wav (0.46, reloj real periodico) puntua
    mas bajo que ruido de textura (0.70). Por eso se expone como campo
    crudo de confianza (Rhythmic.tempo_periodicity_strength), no como un
    filtro que oculte o valide bpm_detected -- ver discusion en el chat.
    """
    if onset_env is None or len(onset_env) < 2 or not np.any(onset_env):
        return None

    acf = librosa.autocorrelate(onset_env, max_size=len(onset_env))
    if acf[0] <= 0:
        return None

    min_lag = int(np.ceil(60.0 * sr / (_MAX_TEMPO_BPM * _HOP_LENGTH)))
    max_lag = int(np.floor(60.0 * sr / (_MIN_TEMPO_BPM * _HOP_LENGTH)))
    max_lag = min(max_lag, len(acf) - 1)
    min_lag = max(min_lag, 1)
    if min_lag >= max_lag:
        return None

    peak = float(np.max(acf[min_lag:max_lag + 1]))
    strength = peak / float(acf[0])
    return round(min(max(strength, 0.0), 1.0), 4)


def extract_rhythmic(y: np.ndarray, sr: int, duration_sec: float) -> Rhythmic:
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=_HOP_LENGTH)
    wait_frames = int(sr * (_ONSET_MIN_GAP_MS / 1000) / _HOP_LENGTH)
    onsets = librosa.onset.onset_detect(
        onset_envelope=onset_env, sr=sr, hop_length=_HOP_LENGTH,
        delta=_ONSET_DELTA, wait=wait_frames,
    )
    onset_count = len(onsets)
    onset_density = round(onset_count / duration_sec, 4) if duration_sec > 0 else 0.0

    # tempo_periodicity_strength se calcula SIEMPRE que haya señal, al
    # margen de la compuerta de duracion/onset_count de bpm_detected --
    # es una medida independiente, no una confirmacion de bpm_detected.
    periodicity_strength = _tempo_periodicity_strength(onset_env, sr)

    if duration_sec < _MIN_DURATION_FOR_TEMPO_SEC or onset_count < 2:
        return Rhythmic(
            bpm_detected=None,
            onset_count=onset_count,
            onset_density=onset_density,
            is_static=True,
            tempo_periodicity_strength=periodicity_strength,
        )

    tempo, _ = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
    bpm_detected = round(float(np.asarray(tempo).item()), 2) if tempo is not None else None

    return Rhythmic(
        bpm_detected=bpm_detected,
        onset_count=onset_count,
        onset_density=onset_density,
        is_static=False,
        tempo_periodicity_strength=periodicity_strength,
    )
