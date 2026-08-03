from __future__ import annotations

import json

from samples2llm.config.schema import OutputConfig, SampleRecord


def generate_json(
    records: list[SampleRecord],
    directory_tree: str,
    output_config: OutputConfig,
) -> str:
    """Misma forma de alto nivel que el JSON de repomix: un resumen, el
    arbol de directorios UNA sola vez (no repetido por archivo), y la lista
    de registros. Ver repomix-self.json generado antes en la conversacion
    para el precedente exacto (top-level keys: fileSummary, directoryStructure,
    files)."""
    payload = {
        "summary": {
            "total_samples": len(records),
            "total_duration_sec": round(sum(r.file_metadata.duration_sec for r in records), 3),
        },
        "directory_structure": directory_tree,
        "samples": [
            r.model_dump(exclude={"embeddings"} if not output_config.include_embeddings else set())
            for r in records
        ],
    }
    return json.dumps(payload, indent=2, ensure_ascii=False)
