from __future__ import annotations

from pathlib import Path

import soundfile as sf

from samples2llm.config.schema import FileMetadata


def extract_file_metadata(file_path: Path) -> FileMetadata:
    """Lee solo la cabecera del archivo (soundfile.info), sin decodificar
    todas las muestras -- es el equivalente al primer nivel "barato",
    analogo a leer solo stats de un archivo en fileCollect.ts."""
    info = sf.info(str(file_path))
    subtype = info.subtype or ""
    bit_depth = None
    for candidate in (8, 16, 24, 32, 64):
        if str(candidate) in subtype:
            bit_depth = candidate
            break

    return FileMetadata(
        format=info.format.lower(),
        duration_sec=round(info.frames / info.samplerate, 6) if info.samplerate else 0.0,
        sample_rate=info.samplerate,
        bit_depth=bit_depth,
        channels=info.channels,
        file_size_bytes=file_path.stat().st_size,
    )
