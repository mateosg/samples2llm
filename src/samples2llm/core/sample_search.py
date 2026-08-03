"""
Equivalente a src/core/file/fileSearch.ts en repomix.

Repomix usa globby (respeta .gitignore automaticamente) + minimatch para
patrones include/ignore explicitos. Aqui replicamos el mismo patron con
pathlib + fnmatch, mas simple porque no necesitamos igualar el comportamiento
exacto de git.

TODO (pendiente de implementar, no bloqueante para el esqueleto):
- Respetar .gitignore ademas de .samplesignore (parsear con `pathspec`,
  la libreria Python mas cercana a la semantica de git que usa repomix).
- Soporte de patrones --include / --ignore por CLI, igual que repomix.
"""

from __future__ import annotations

from pathlib import Path

from samples2llm.config.defaults import DEFAULT_AUDIO_EXTENSIONS, DEFAULT_IGNORE_DIRS
from samples2llm.config.schema import InputConfig


def search_samples(root_dir: str | Path, config: InputConfig) -> list[Path]:
    """Recorre root_dir y devuelve las rutas de todos los archivos de audio
    validos, excluyendo directorios de sistema/VCS.

    Analogo al `searchFiles` de repomix, sin el soporte de .gitignore todavia
    (ver TODO arriba).
    """
    root = Path(root_dir)
    extensions = {ext.lower() for ext in (config.extensions or DEFAULT_AUDIO_EXTENSIONS)}

    results: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if any(part in DEFAULT_IGNORE_DIRS for part in path.parts):
            continue
        if path.suffix.lower() not in extensions:
            continue
        if config.max_file_size_mb is not None:
            size_mb = path.stat().st_size / (1024 * 1024)
            if size_mb > config.max_file_size_mb:
                continue
        results.append(path)

    return sorted(results)
