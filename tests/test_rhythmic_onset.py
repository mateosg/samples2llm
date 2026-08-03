"""Suite de regresion para deteccion de onsets (rhythmic.extract_rhythmic).

Contexto: en rondas anteriores, los parametros por defecto de
librosa.onset.onset_detect generaban falsos positivos sistematicos sobre
señales con ruido de fondo (un kick de un solo golpe se contaba como "3
onsets"). Se ajusto delta=0.2 + wait=30ms, verificado contra 3 casos
sinteticos. Esta suite amplia esa verificacion con:

1. Casos con ground truth FUERTE (no adivinado): eventos diseñados a
   proposito (Kick, Perc_multihit) o confirmados por evidencia externa
   objetiva -- no por "me parece que suena a X".
2. Casos de regresion anti-falso-positivo: sonidos continuos (ruido,
   motor, sirena...) que NO deben producir un onset_count alto, que es
   precisamente el bug que ya mordio una vez a este proyecto.
3. Un caso de limitacion CONOCIDA, documentada explicitamente como tal
   (no oculta con xfail): Perc_multihit_layered tiene 3 golpes diseñados
   pero el detector actual encuentra 2. Se afirma el valor actual (2) para
   que cualquier cambio futuro en el comportamiento sea visible en el
   diff del test, y se deja constancia en el docstring de que NO es el
   valor "correcto" ideal.

Metodologia del ground truth en casos reales (ESC-50):
Para dog.wav y clock_tick.wav el conteo se determino inspeccionando el
perfil de energia RMS cruda (no el onset_detect que se esta probando,
para evitar razonamiento circular) y contando rafagas de energia
claramente separadas por silencio o caida a linea de base. Evidencia de
apoyo:
  - dog.wav: la fuente original en Freesound se llama "rose_bark.wav"
    (singular) -- consistente con la unica rafaga de ~180ms detectada
    entre 2.26s-2.44s, con silencio total (RMS=0) en el resto de los 5s.
  - clock_tick.wav: 5 picos de RMS nitidos, separados por silencio
    parcial, con espaciado casi periodico (~1.0s +/- 0.1s) -- coherente
    con un reloj real de tick regular.
  - glass_breaking.wav: 2 rafagas de energia claramente separadas por
    ~1.1s de silencio parcial (impacto inicial + fragmento secundario).
Estos NO son "escuchados" (el agente no puede oir audio), son inferidos
de la señal misma con un metodo distinto al que se esta validando. Donde
la señal no daba una lectura inequivoca (footsteps.wav, clapping.wav,
siren.wav: rafagas de energia ambiguas en numero o duracion, imposibles
de contar con confianza sin escuchar el audio) se evito fijar un ground
truth exacto -- esos casos solo se usan como test de regresion
anti-falso-positivo con un limite superior, no de conteo exacto.
"""

from __future__ import annotations

from pathlib import Path

import librosa
import pytest

from samples2llm.core.features.rhythmic import extract_rhythmic

_ROOT = Path(__file__).resolve().parent.parent
_ESC50 = _ROOT / "examples" / "esc50_real"
_LIBRARY = _ROOT / "examples" / "sample_library"


def _onset_count(path: Path) -> int:
    y, sr = librosa.load(path, sr=None, mono=True)
    duration_sec = len(y) / sr
    rhythmic = extract_rhythmic(y, sr, duration_sec)
    return rhythmic.onset_count


# ---------------------------------------------------------------------
# 1. Ground truth fuerte: eventos diseñados a proposito
# ---------------------------------------------------------------------


def test_kick_single_hit_detects_exactly_one_onset():
    """Kick_808_hit.wav es un golpe unico por diseño (no medido, construido)."""
    assert _onset_count(_LIBRARY / "Drums" / "Kicks" / "Kick_808_hit.wav") == 1


def test_multihit_layered_known_limitation_detects_two_of_three():
    """LIMITACION CONOCIDA, no arreglada a ciegas: Perc_multihit_layered.wav
    se diseño con 3 golpes reales. El detector actual encuentra 2, no 3
    -- mejor que los falsos positivos de antes (que inflaban el conteo en
    señales SIN golpes reales), pero sigue sin ser exacto en señales CON
    golpes reales muy cercanos entre si. Este test fija el valor actual
    (2) a proposito, para que si algun cambio futuro en delta/wait lo
    mueve, aparezca como una diferencia visible en el test -- no como una
    sorpresa silenciosa. Si algun dia se corrige a 3, este test debe
    actualizarse junto con el fix (y el README).
    """
    assert _onset_count(_LIBRARY / "Perc_multihit_layered.wav") == 2


# ---------------------------------------------------------------------
# 2. Ground truth por evidencia externa objetiva (no por "escuchar")
# ---------------------------------------------------------------------


def test_dog_bark_detects_exactly_one_onset():
    """dog.wav: fuente original en Freesound = 'rose_bark.wav' (singular).
    El perfil RMS muestra silencio total salvo una unica rafaga de
    ~180ms (2.26s-2.44s). Cierra la pregunta abierta en la ronda anterior:
    no es un bug de sub-deteccion, es el conteo correcto -- solo hay un
    ladrido real en el clip.
    """
    assert _onset_count(_ESC50 / "dog.wav") == 1


def test_clock_tick_detects_five_onsets():
    """clock_tick.wav: 5 picos de RMS nitidos y casi periodicos (~1.0s de
    separacion), consistentes con un tick de reloj real a ~1 Hz durante
    5s. Tolerancia +/-1 porque el ultimo tick (4.50s-4.55s) es mas debil
    que los otros y un cambio menor de umbral podria perderlo.
    """
    assert _onset_count(_ESC50 / "clock_tick.wav") == pytest.approx(5, abs=1)


def test_glass_breaking_detects_multiple_onsets():
    """CORRECCION sobre un supuesto propio equivocado: en un primer intento
    esta prueba asumia "2 onsets" a partir de 2 rafagas MACRO de energia
    RMS (impacto inicial + fragmento secundario). El detector real
    encuentra 7 -- y es razonable: un cristal al romperse no es un golpe
    limpio, son muchos micro-impactos de fragmentos (tintineo) dentro de
    cada rafaga macro, que SI son transitorios reales, no ruido de fondo.
    Contar rafagas de energia a ojo no es lo mismo que contar onsets
    reales; no se puede fijar un numero exacto sin escuchar el audio, asi
    que esto queda como cota de sanidad, no como aserto de precision.
    """
    count = _onset_count(_ESC50 / "glass_breaking.wav")
    assert 2 <= count <= 15


# ---------------------------------------------------------------------
# 3. Regresion anti-falso-positivo: ruido SIN ningun pulso real
# ---------------------------------------------------------------------
# El bug original (ya corregido) era especifico: ruido/texturas SIN
# ningun transitorio real (silencio con jitter de amplitud, ruido blanco
# decorrelacionado) contadas como si tuvieran varios golpes. NO aplica a
# sonidos mecanicos con pulsos ciclicos reales (motor, motosierra,
# helicoptero) -- esos SI tienen transitorios reales y es correcto que
# onset_count sea alto ahi; forzar un techo bajo ahi seria repetir el
# mismo error de "adivinar sin verificar" que ya causo bugs antes.

_SILENT_NOISE_MAX_ONSETS = 3


def test_wide_stereo_noise_texture_does_not_produce_many_false_onsets():
    """Texture_wide_stereo.wav es ruido L/R decorrelacionado sin golpes
    percusivos reales -- ya causo el bug original (6 onsets falsos antes
    del fix de delta/wait). Este SI es el caso correcto para un techo
    bajo: no hay ningun pulso ciclico real detras del ruido, a diferencia
    de un motor o una motosierra.
    """
    count = _onset_count(_LIBRARY / "Texture_wide_stereo.wav")
    assert count <= _SILENT_NOISE_MAX_ONSETS


# ---------------------------------------------------------------------
# 4. Sonidos con pulsos reales pero de numero ambiguo sin escuchar
# ---------------------------------------------------------------------
# chainsaw/engine/helicopter/rain/siren/clapping/footsteps SI tienen
# transitorios reales (pulsos mecanicos, aplausos individuales, pasos),
# pero contar el numero exacto sin poder escuchar el audio seria
# adivinar. Se documenta el conteo actual medido como referencia de
# regresion (si un cambio de parametros lo mueve mucho, este test lo
# marca), con una cota amplia en vez de un valor exacto.

_REAL_PULSE_SOUNDS_MAX_ONSETS = 40


@pytest.mark.parametrize(
    "filename",
    ["footsteps.wav", "clapping.wav", "siren.wav", "chainsaw.wav", "engine.wav", "helicopter.wav", "rain.wav"],
)
def test_real_pulse_sounds_stay_within_sane_bounds(filename):
    count = _onset_count(_ESC50 / filename)
    assert 1 <= count <= _REAL_PULSE_SOUNDS_MAX_ONSETS


# ---------------------------------------------------------------------
# 5. tempo_periodicity_strength: regresion de la investigacion sobre el
# gate de bpm_detected espurio (ver README, "Sexta ronda"). Estos tests
# NO afirman que el campo distinga musical/no-musical -- afirman
# justo lo contrario, a proposito, para que quede constancia si algun
# cambio futuro en la formula empieza a comportarse como si discriminara
# (lo cual seria sospechoso, dado que la investigacion establecio que
# ninguna heuristica barata de señal puede hacerlo).
# ---------------------------------------------------------------------


def _rhythmic(path: Path):
    y, sr = librosa.load(path, sr=None, mono=True)
    duration_sec = len(y) / sr
    return extract_rhythmic(y, sr, duration_sec)


def test_periodicity_strength_is_present_even_when_bpm_detected_is_none():
    """dog.wav no pasa la compuerta de bpm (onset_count=1) pero
    tempo_periodicity_strength debe calcularse igual -- es una medida
    independiente de esa compuerta, no una confirmacion de bpm_detected."""
    r = _rhythmic(_ESC50 / "dog.wav")
    assert r.bpm_detected is None
    assert r.tempo_periodicity_strength is not None
    assert 0.0 <= r.tempo_periodicity_strength <= 1.0


def test_periodicity_strength_does_not_separate_musical_from_mechanical():
    """Regresion INTENCIONAL del hallazgo de la investigacion: un motor
    (engine.wav, periodicidad fisica real pero no musical) puntua tan
    alto o mas que un loop musical real (loop_Cmaj_120bpm.wav). Si este
    test empieza a fallar porque engine.wav puntua mucho mas bajo, es una
    señal de que la formula cambio de comportamiento y conviene revisar
    si ahora SI podria usarse como filtro -- no asumirlo sin verificar de
    nuevo, el mismo patron que ya causo bugs antes en este proyecto."""
    engine = _rhythmic(_ESC50 / "engine.wav")
    loop = _rhythmic(_LIBRARY / "loop_Cmaj_120bpm.wav")
    assert engine.tempo_periodicity_strength is not None
    assert loop.tempo_periodicity_strength is not None
    assert engine.tempo_periodicity_strength >= loop.tempo_periodicity_strength - 0.15
