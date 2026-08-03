"""
Equivalente a fileProcess.ts / fileProcessContent.ts en repomix: la etapa
que toma un archivo "crudo" y produce su representacion procesada final.

Diferencia de fondo con repomix: alli el "processing" transforma texto en
texto (quitar comentarios, comprimir). Aqui el "processing" transforma
señal en estructura (numeros con semantica), asi que en vez de un pipeline
de transforms encadenados tenemos un fan-out: la misma waveform se pasa a
varios extractores independientes que no se afectan entre si.

TODO (no implementado en este esqueleto, ver conversacion):
- Paralelizacion real con ProcessPoolExecutor, analoga al worker pool de
  calculateFileMetrics.ts. Librosa es CPU-bound, tiene sentido en cuanto
  se procesen carpetas grandes.

Nota: el cache por hash de archivo ya esta implementado en
core/packager.py + core/cache.py (equivalente conceptual a
tokenCountCache.ts), de modo que este modulo se mantiene centrado solo en
la extraccion de features.
"""

from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

from samples2llm.config.schema import ExtractionConfig, SampleRecord
from samples2llm.core.context_parse import directory_context, directory_type_hints, parse_declared_metadata
from samples2llm.core.features.derived_tags import derive_tags
from samples2llm.core.features.dynamics import extract_dynamics
from samples2llm.core.features.integrity import extract_integrity
from samples2llm.core.features.metadata import extract_file_metadata
from samples2llm.core.features.rhythmic import extract_rhythmic
from samples2llm.core.features.spatial import extract_spatial
from samples2llm.core.features.spectral import extract_spectral
from samples2llm.core.features.tonal import extract_tonal
from samples2llm.core.summarize import generate_summary
from samples2llm.config.schema import DeclaredMetadata, Rhythmic, Tonal


_BPM_MISMATCH_TOLERANCE = 0.03  # 3% de margen antes de considerarlo discrepancia

# type_hints (ver context_parse._TYPE_KEYWORDS) que sugieren que el sample
# esta pensado para encajar en un contexto musical con tempo (loop, capa
# de instrumento con groove...). Se excluye "one_shot" a proposito: un
# one-shot puede ser perfectamente un FX/foley sin tempo real, asi que su
# presencia no corrobora nada sobre bpm_detected. Se comprueba tanto en
# declared.type_hints (nombre de archivo) como en directory_type_hints()
# (nombre de carpeta) -- ver _has_musical_context.
_MUSICAL_TYPE_HINTS = frozenset({
    "kick", "snare", "hihat", "bass", "loop", "stem", "pad", "vocal",
})


def _has_musical_context(declared: DeclaredMetadata, directory_ctx: list[str]) -> bool:
    """True si hay alguna corroboracion DECLARADA (no medida) de que el
    sample esta pensado para tener un tempo musical real: un bpm en el
    nombre, un type_hint de los que implican groove/instrumentacion (en
    el nombre de archivo O en el nombre de carpeta), o directory_context
    dando la misma señal. Ver _build_confidence_flags: se usa para no
    presentar bpm_detected como si fuera fiable cuando no hay ninguna
    señal externa que lo respalde -- la investigacion en el chat confirmo
    que no existe una caracteristica de la propia señal que separe "tempo
    musical real" de "periodicidad fisica real pero no musical" (motor,
    helicoptero...).

    Antes solo miraba declared.type_hints (nombre de archivo) -- ampliado
    tras probar contra una libreria real (KSHMR Sounds Vol.4) donde la
    categoria a veces solo esta en el nombre de carpeta ("Drum Loops",
    "Vocals"), no repetida en cada archivo individual."""
    if declared.bpm is not None:
        return True
    if any(hint in _MUSICAL_TYPE_HINTS for hint in declared.type_hints):
        return True
    folder_hints = directory_type_hints(directory_ctx)
    return any(hint in _MUSICAL_TYPE_HINTS for hint in folder_hints)


def _bpm_values_match(declared_bpm: float, detected_bpm: float) -> bool:
    """El error mas comun en deteccion de tempo es el "octave error":
    el detector devuelve el doble o la mitad del tempo real (p.ej. detecta
    143 cuando el tempo real -y declarado- es ~71.5, o al reves). Por eso
    comparamos contra bpm, bpm*2 y bpm/2, no solo contra el valor literal.
    """
    for factor in (1.0, 2.0, 0.5):
        expected = declared_bpm * factor
        if abs(detected_bpm - expected) / expected <= _BPM_MISMATCH_TOLERANCE:
            return True
    return False


def _build_confidence_flags(record_kwargs: dict) -> list[str]:
    """Contrasta lo DECLARADO contra lo MEDIDO y señala discrepancias.
    Logica minima a proposito -- se amplia segun haga falta."""
    flags: list[str] = []
    declared = record_kwargs["declared"]
    rhythmic: Rhythmic = record_kwargs["rhythmic"]
    tonal: Tonal = record_kwargs["tonal"]

    if declared.bpm is not None and rhythmic.is_static:
        flags.append("bpm_undetected_static_sample")
        flags.append("declared_bpm_likely_project_tempo_not_sample_tempo")
    elif declared.bpm is not None and rhythmic.bpm_detected is not None:
        if not _bpm_values_match(declared.bpm, rhythmic.bpm_detected):
            flags.append("declared_bpm_mismatch_with_detected_bpm")

    if declared.key is not None and tonal.pitch_detected is None:
        flags.append("declared_key_not_confirmed_by_analysis")

    # bpm_detected siempre devuelve un numero cuando pasa la compuerta de
    # duracion/onset_count (librosa.beat.beat_track no distingue "tempo
    # musical real" de "periodicidad fisica real pero no musical" -- ver
    # rhythmic._tempo_periodicity_strength). Sin corroboracion declarada
    # (bpm o type_hint musical en el nombre), se marca en vez de dejar que
    # el valor se use como si fuera fiable por defecto.
    if rhythmic.bpm_detected is not None and not _has_musical_context(
        declared, record_kwargs["directory_context"]
    ):
        flags.append("bpm_detected_without_musical_context")

    return flags


def extract_sample(file_path: Path, root_dir: Path, config: ExtractionConfig) -> SampleRecord:
    # Cargamos dos veces con proposito distinto: y_channels sin mezclar
    # (para integridad estereo real) y y_mono para el resto de analisis,
    # donde mezclar a mono es lo estandar en MIR.
    y_channels, sr = sf.read(str(file_path), always_2d=True)
    y_channels = y_channels.T  # sf.read da (samples, channels); queremos (channels, samples)
    y_mono = librosa.to_mono(y_channels) if y_channels.shape[0] > 1 else y_channels[0]
    y_mono = y_mono.astype(np.float32)

    file_metadata = extract_file_metadata(file_path)
    dynamics = extract_dynamics(y_mono, sr)
    spectral = extract_spectral(y_mono, sr)
    integrity = extract_integrity(y_channels, sr)
    spatial = extract_spatial(y_channels)

    tonal = extract_tonal(y_mono, sr) if config.enable_tonal else Tonal()
    rhythmic = (
        extract_rhythmic(y_mono, sr, file_metadata.duration_sec)
        if config.enable_rhythmic
        else Rhythmic()
    )

    declared = parse_declared_metadata(file_path)
    tags = derive_tags(file_metadata, dynamics, tonal, rhythmic, spectral.centroid_hz)

    record_kwargs = dict(
        path=str(file_path.relative_to(root_dir)),
        filename=file_path.name,
        directory_context=directory_context(file_path, root_dir),
        declared=declared,
        file_metadata=file_metadata,
        dynamics=dynamics,
        spectral=spectral,
        tonal=tonal,
        rhythmic=rhythmic,
        spatial=spatial,
        integrity=integrity,
        derived_tags=tags,
    )
    record_kwargs["confidence_flags"] = _build_confidence_flags(record_kwargs)

    record = SampleRecord(**record_kwargs)
    record.summary = generate_summary(record)
    return record
