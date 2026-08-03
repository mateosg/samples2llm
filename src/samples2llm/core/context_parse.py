"""
Nivel 0: metadata DECLARADA, extraida del nombre de archivo y la ruta.

Sin equivalente directo en repomix -- es la pieza que añadimos porque en
librerias de samples profesionales la carpeta y el nombre ya clasifican el
sample (ver discusion en el chat). Es heuristica por definicion: un regex no
puede saber si "128bpm" en un nombre de archivo es el tempo real del sample
o el tempo del proyecto donde se uso. Por eso los resultados de este modulo
van siempre en el campo `declared`, nunca mezclados con `measured`.

Este modulo SI esta implementado de forma funcional (no es solo un stub),
porque es relativamente barato y autocontenido comparado con el analisis
de señal.
"""

from __future__ import annotations

import re
from pathlib import Path

from samples2llm.config.schema import DeclaredMetadata

_BPM_RE = re.compile(r"(?<![a-zA-Z0-9])(\d{2,3})\s*[-_]?bpm(?![a-zA-Z])", re.IGNORECASE)

# Segundo patron de BPM, encontrado al probar contra una libreria real
# (KSHMR Sounds Vol.4): una convencion de nomenclatura profesional muy
# comun es "(TEMPO, KEY)" entre parentesis SIN la palabra "bpm" pegada,
# p.ej. "KSHMR Vintage Record Loop 01 (81, Fm) - ...wav". El patron de
# arriba no lo capturaba -- 7 de los 24 archivos de la muestra real
# perdian su bpm declarado por esto (no es un caso raro, es un patron
# comun en librerias comerciales). Requiere que el numero vaya seguido de
# coma y luego una letra de nota (A-G) para no confundir cualquier
# numero entre parentesis con un tempo -- p.ej. no dispara con "(v2)" ni
# con un numero de variacion aislado.
_BPM_PAREN_RE = re.compile(r"\((\d{2,3})\s*,\s*(?=[A-Ga-g])")

# Notas musicales tipo "Am", "A#m", "Cmaj", "F#min", "Bb"
_KEY_RE = re.compile(
    r"(?<![a-zA-Z0-9])([A-G])([#b]?)\s*(maj|min|m)?(?![a-zA-Z0-9])"
)

_VARIATION_RE = re.compile(r"(?<![a-zA-Z0-9])((?:v|rr|round)\d{1,2})(?![a-zA-Z0-9])", re.IGNORECASE)

# Diccionario minimo de tipo/instrumento. Deliberadamente pequeño en el
# esqueleto -- se amplia con el tiempo o se sustituye por un diccionario
# externo cargado desde config.
#
# Las formas en plural (kicks, snares, hats, loops, stems, pads, vocals)
# se añadieron al probar contra nombres de CARPETA de una libreria real
# (KSHMR Sounds Vol.4: "Drum Loops", "SYNTH - Lead Loops", "Vocals"...) --
# los nombres de carpeta de categoria casi siempre pluralizan, a
# diferencia de los nombres de archivo individuales que suelen ir en
# singular ("Kick_808_hit.wav"). Sin esto, directory_type_hints() de mas
# abajo no habria matcheado nada en la mayoria de carpetas reales.
_TYPE_KEYWORDS: dict[str, list[str]] = {
    "kick": ["kick", "kicks", "bd"],
    "snare": ["snare", "snares", "sd"],
    "hihat": ["hihat", "hat", "hats", "hh"],
    "bass": ["bass", "sub", "subs", "808"],
    "loop": ["loop", "loops"],
    "one_shot": ["oneshot", "one-shot", "one_shot", "hit", "hits"],
    "stem": ["stem", "stems"],
    "pad": ["pad", "pads"],
    "vocal": ["vox", "vocal", "vocals", "acapella"],
}


def _tokenize(name: str) -> list[str]:
    return re.split(r"[\s_\-\.]+", name)


def parse_declared_metadata(file_path: Path) -> DeclaredMetadata:
    """Extrae bpm/key/variacion/type_hints del nombre de archivo por regex,
    y coteja palabras clave de tipo/instrumento contra un diccionario simple.
    No toca la señal de audio en absoluto -- es puro parsing de texto.
    """
    name = file_path.stem
    tokens = [t.lower() for t in _tokenize(name)]

    bpm_match = _BPM_RE.search(name)
    if bpm_match:
        bpm = float(bpm_match.group(1))
    else:
        # Fallback: convencion "(TEMPO, KEY)" sin la palabra "bpm" -- ver
        # comentario junto a _BPM_PAREN_RE. Solo se usa si el patron
        # principal (con "bpm" explicito) no encontro nada, para no
        # arriesgar falsos positivos donde ya hay una lectura mas segura.
        bpm_paren_match = _BPM_PAREN_RE.search(name)
        bpm = float(bpm_paren_match.group(1)) if bpm_paren_match else None

    key_match = _KEY_RE.search(name)
    key = None
    if key_match:
        note, accidental, quality = key_match.groups()
        key = f"{note}{accidental}{quality or ''}"

    variation_match = _VARIATION_RE.search(name)
    variation = variation_match.group(1) if variation_match else None

    type_hints = [
        label
        for label, keywords in _TYPE_KEYWORDS.items()
        if any(kw in tokens for kw in keywords)
    ]

    return DeclaredMetadata(key=key, bpm=bpm, variation=variation, type_hints=type_hints)


def directory_type_hints(directory_components: list[str]) -> list[str]:
    """Aplica el mismo diccionario de tipo/instrumento que
    parse_declared_metadata() usa sobre el nombre de archivo, pero sobre
    los componentes de CARPETA. Separado en su propia funcion (no fusionado
    dentro de parse_declared_metadata) porque directory_context ya se
    calcula aparte con directory_context() y requiere el root_dir, que
    parse_declared_metadata no recibe -- ver core/extract.py para donde se
    combinan ambos resultados.

    Encontrado necesario al probar contra una libreria real (KSHMR Sounds
    Vol.4): carpetas como "Drum Loops" o "Vocals" declaran el tipo del
    contenido sin que el nombre de archivo individual lo repita siempre.
    """
    tokens: list[str] = []
    for component in directory_components:
        tokens.extend(t.lower() for t in _tokenize(component))

    return [
        label
        for label, keywords in _TYPE_KEYWORDS.items()
        if any(kw in tokens for kw in keywords)
    ]


def directory_context(file_path: Path, root_dir: Path) -> list[str]:
    """Devuelve los componentes de carpeta entre root_dir y el archivo,
    p.ej. ['Bass', 'Sub'] para Bass/Sub/archivo.wav."""
    try:
        relative = file_path.relative_to(root_dir)
    except ValueError:
        relative = file_path
    return list(relative.parts[:-1])
