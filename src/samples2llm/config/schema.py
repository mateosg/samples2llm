"""
Esquema de datos de samples2llm.

Equivalente directo a `src/config/configSchema.ts` en repomix, pero con dos
diferencias de fondo respecto al original:

1. Repomix solo necesita un schema de CONFIGURACION (como se comporta la
   herramienta). Aqui ademas necesitamos un schema de DATOS DE SALIDA
   (que campos tiene cada sample), porque a diferencia del "contenido de
   codigo" de repomix, aqui la salida no es texto libre sino una estructura
   con muchos campos numericos con semantica propia.

2. Repomix trata cada archivo de forma homogenea (todo es texto). Aqui
   distinguimos explicitamente entre informacion DECLARADA (deducida del
   nombre/ruta, puede estar mal o ser ambigua) e informacion MEDIDA
   (calculada analizando la señal, no puede "mentir" pero puede fallar
   a detectarse). Ver DeclaredMetadata vs. el resto de bloques.

pydantic aqui cumple el mismo rol que valibot en el proyecto original.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Output style enum -- equivalente a repomixOutputStyleSchema en configSchema.ts
# ---------------------------------------------------------------------------

class OutputStyle(str, Enum):
    JSON = "json"
    MARKDOWN = "markdown"


DEFAULT_OUTPUT_FILENAME: dict[OutputStyle, str] = {
    OutputStyle.JSON: "samples2llm-output.json",
    OutputStyle.MARKDOWN: "samples2llm-output.md",
}


# ---------------------------------------------------------------------------
# Per-sample data contract (niveles 0-7 acordados en la conversacion)
# ---------------------------------------------------------------------------

class DeclaredMetadata(BaseModel):
    """Nivel 0. Extraido por heuristica del nombre/ruta. NO verificado contra
    la señal real -- puede ser incorrecto o estar ausente. Ver
    core/context_parse.py para la logica de extraccion (TODO)."""

    key: Optional[str] = None
    bpm: Optional[float] = None
    variation: Optional[str] = None  # p.ej. "v2", "RR3"
    type_hints: list[str] = Field(default_factory=list)  # p.ej. ["bass", "sub"]


class FileMetadata(BaseModel):
    """Nivel 1. Metadata de archivo, practicamente gratis (soundfile)."""

    format: str
    duration_sec: float
    sample_rate: int
    bit_depth: Optional[int] = None
    channels: int
    file_size_bytes: int


class Dynamics(BaseModel):
    """Nivel 2. Amplitud / dinamica (numpy sobre la waveform)."""

    rms: float
    peak: float
    crest_factor: float
    leading_silence_sec: float
    trailing_silence_sec: float
    attack_time_sec: Optional[float] = None


class SpectralTrend(str, Enum):
    ASCENDING = "ascending"
    DESCENDING = "descending"
    STABLE = "stable"


class Spectral(BaseModel):
    """Nivel 3. Features espectrales (librosa.feature).

    centroid_hz/etc. son promedios sobre todo el archivo -- pierden por
    completo la forma temporal del sonido (un whoosh y un tono estable
    pueden tener el mismo centroide medio). centroid_trend intenta
    recuperar esa informacion sin serializar la serie temporal completa:
    resume la evolucion del brillo en una etiqueta categorica + pendiente.
    """

    centroid_hz: float
    rolloff_hz: float
    bandwidth_hz: float
    flatness: float
    zero_crossing_rate: float

    centroid_trend: Optional[SpectralTrend] = None
    centroid_trend_confidence: Optional[float] = None  # ver spectral.py: R^2 del ajuste lineal


class Tonal(BaseModel):
    """Nivel 4. Solo relevante si el sample tiene tono definido."""

    pitch_detected: Optional[str] = None  # notacion tipo "A2"
    pitch_confidence: Optional[float] = None
    harmonic_percussive_ratio: Optional[float] = None


class Rhythmic(BaseModel):
    """Nivel 5.

    onset_count/onset_density se calculan SIEMPRE, independientemente de la
    duracion -- son utiles incluso en one-shots cortos (p.ej. distinguir un
    golpe simple de una capa de 2-3 transitorios superpuestos). bpm_detected
    en cambio solo tiene sentido con duracion suficiente y pulso repetido,
    ver _MIN_DURATION_FOR_TEMPO_SEC en rhythmic.py."""

    bpm_detected: Optional[float] = None
    onset_count: Optional[int] = None
    onset_density: Optional[float] = None
    is_static: bool = False  # True si no se detecta pulso ritmico repetido (bpm)

    # Fuerza de periodicidad CRUDA (0-1), via autocorrelacion del onset
    # envelope -- ver rhythmic.py. Deliberadamente SIN umbral binario: la
    # investigacion mostro que ningun umbral separa "tempo musical real"
    # de "periodicidad fisica real pero no musical" (un motor tiene un
    # pico de autocorrelacion tan nitido como un loop real, p.ej.
    # helicopter.wav=0.92 > loop_Cmaj_120bpm.wav=0.85 medido). Por eso se
    # expone el numero crudo en vez de intentar convertirlo en un booleano
    # "es_musical" que estaria adivinando. Se calcula siempre que haya
    # señal suficiente, incluso cuando bpm_detected es None -- es
    # informacion independiente de la compuerta de duracion/onset_count.
    tempo_periodicity_strength: Optional[float] = None


class SpatialWidth(str, Enum):
    MONO = "mono"
    NARROW = "narrow"
    MEDIUM = "medium"
    WIDE = "wide"


class Spatial(BaseModel):
    """Sin nivel asignado en el diseño original -- añadido tras revisar la
    propuesta externa. mid/side energy ratio: 0 = canales identicos
    (redundancia total), ~1 = canales totalmente decorrelacionados."""

    channels: int
    width_ratio: Optional[float] = None  # None si mono
    width_category: SpatialWidth = SpatialWidth.MONO


class Integrity(BaseModel):
    """Nivel 7. Equivalente conceptual a securityCheck.ts, pero para audio:
    no buscamos secretos, buscamos archivos "rotos" o mal formados."""

    clipping_detected: bool = False
    is_silent: bool = False
    channel_redundancy: Optional[bool] = None  # True si stereo con canales identicos


class SampleRecord(BaseModel):
    """Registro completo de un sample. Este es el contrato de datos central
    del proyecto -- el equivalente a ProcessedFile en repomix, pero mucho
    mas rico porque el "contenido" no es texto sino un conjunto de
    features derivadas."""

    path: str
    filename: str
    directory_context: list[str] = Field(default_factory=list)

    declared: DeclaredMetadata = Field(default_factory=DeclaredMetadata)
    file_metadata: FileMetadata
    dynamics: Dynamics
    spectral: Spectral
    tonal: Tonal
    rhythmic: Rhythmic
    spatial: Spatial
    integrity: Integrity

    derived_tags: list[str] = Field(default_factory=list)  # nivel 6
    confidence_flags: list[str] = Field(default_factory=list)

    # Resumen en lenguaje natural, ensamblado por plantilla a partir de los
    # campos categoricos de arriba -- NO es salida de un clasificador ML.
    # Ver core/summarize.py. Deliberadamente no incluye afirmaciones tipo
    # "esto es un whoosh": eso requeriria un modelo entrenado, que este
    # proyecto no tiene (ver discusion en el chat).
    summary: Optional[str] = None

    # Campo opcional y separado a proposito -- ver discusion sobre MFCCs:
    # utiles para similitud/ML pero ilegibles para un LLM razonando en
    # lenguaje natural. Nunca se mezclan con los campos "legibles" de arriba.
    embeddings: Optional[dict] = None


# ---------------------------------------------------------------------------
# Config global de la herramienta -- equivalente a repomixConfigBaseSchema
# ---------------------------------------------------------------------------

class InputConfig(BaseModel):
    extensions: list[str] = Field(
        default_factory=lambda: [".wav", ".aiff", ".aif", ".flac", ".mp3", ".ogg"]
    )
    max_file_size_mb: Optional[float] = None


class OutputConfig(BaseModel):
    file_path: Optional[str] = None
    style: OutputStyle = OutputStyle.JSON
    include_embeddings: bool = False


class ExtractionConfig(BaseModel):
    """Permite desactivar niveles de analisis caros si solo se necesita
    metadata basica -- equivalente en espiritu al --compress de repomix,
    pero aqui el control es a nivel de que analisis correr, no de cuanto
    comprimir texto."""

    enable_tonal: bool = True
    enable_rhythmic: bool = True
    enable_embeddings: bool = False


class Samples2LlmConfig(BaseModel):
    input: InputConfig = Field(default_factory=InputConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    extraction: ExtractionConfig = Field(default_factory=ExtractionConfig)
