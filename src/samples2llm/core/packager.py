"""
Equivalente directo a src/core/packager.ts: la funcion `pack` que orquesta
todo el pipeline. En repomix esta funcion inyecta dependencias
(`defaultDeps`) para poder testear cada etapa aislada -- aqui se mantiene
la misma idea de forma mas ligera (funciones puras importadas directamente,
en Python es menos habitual el patron de inyeccion explicita que en TS).

TODO: paralelizar extract_sample con ProcessPoolExecutor cuando el numero
de samples sea grande (analogo al worker pool de metrics/calculateFileMetrics.ts
y de fileProcess.ts en repomix).
"""

from __future__ import annotations

from pathlib import Path

from samples2llm.config.schema import OutputStyle, Samples2LlmConfig
from samples2llm.core.cache import AnalysisCache, build_file_fingerprint
from samples2llm.core.extract import extract_sample
from samples2llm.core.output.json_style import generate_json
from samples2llm.core.output.markdown_style import generate_markdown
from samples2llm.core.sample_search import search_samples
from samples2llm.core.tree_generate import generate_directory_tree
from samples2llm.shared.logger import get_logger

logger = get_logger(__name__)


def pack(root_dir: str | Path, config: Samples2LlmConfig) -> str:
    root = Path(root_dir)
    cache = AnalysisCache(root, config.extraction)

    logger.info("Searching samples...")
    sample_paths = search_samples(root, config.input)
    logger.info(f"{len(sample_paths)} samples found")

    directory_tree = generate_directory_tree(sample_paths, root)

    records = []
    cache_hits = 0
    cache_misses = 0
    for i, path in enumerate(sample_paths, start=1):
        logger.info(f"Processing ({i}/{len(sample_paths)}): {path.name}")

        fingerprint: str | None = None
        try:
            fingerprint = build_file_fingerprint(path)
        except Exception as exc:
            logger.warning(f"Could not compute fingerprint for {path}: {exc}")

        if fingerprint is not None:
            cached_record = cache.get(path, fingerprint)
            if cached_record is not None:
                records.append(cached_record)
                cache_hits += 1
                logger.info(f"Cache hit: {path.name}")
                continue

        try:
            record = extract_sample(path, root, config.extraction)
            records.append(record)
            cache_misses += 1
            if fingerprint is not None:
                cache.put(path, fingerprint, record)
        except Exception as exc:
            # Un sample corrupto o en un formato que soundfile/librosa no
            # puede leer no debe tumbar todo el pack -- lo saltamos y
            # seguimos, igual que repomix hace "best-effort" con la
            # compresion de archivos individuales.
            logger.warning(f"Could not process {path}: {exc}")

    cache.prune_to_paths(sample_paths)
    cache.save()
    logger.info(f"Cache summary: {cache_hits} hits / {cache_misses} misses")

    if config.output.style == OutputStyle.JSON:
        return generate_json(records, directory_tree, config.output)
    return generate_markdown(records, directory_tree)
