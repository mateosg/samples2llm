from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from samples2llm.config.schema import ExtractionConfig, SampleRecord

_CACHE_SCHEMA_VERSION = 1
_CACHE_FILENAME = ".samples2llm.cache.json"


def build_file_fingerprint(path: Path) -> str:
    """Build a content fingerprint for cache validation.

    Includes file size and mtime for quick human debugging plus a SHA-256
    digest for robust invalidation when content changes.
    """
    stat = path.stat()
    sha256 = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            sha256.update(chunk)
    return f"{stat.st_size}:{stat.st_mtime_ns}:{sha256.hexdigest()}"


class AnalysisCache:
    def __init__(self, root_dir: Path, extraction_config: ExtractionConfig) -> None:
        self.root_dir = root_dir
        self.cache_path = root_dir / _CACHE_FILENAME
        self._config_signature = extraction_config.model_dump(mode="json")
        self._entries: dict[str, dict[str, Any]] = {}
        self._load()

    def _load(self) -> None:
        if not self.cache_path.exists():
            return
        try:
            raw = json.loads(self.cache_path.read_text(encoding="utf-8"))
        except Exception:
            # Corrupt cache should not block processing.
            return

        if raw.get("schema_version") != _CACHE_SCHEMA_VERSION:
            return
        if raw.get("config_signature") != self._config_signature:
            return

        entries = raw.get("entries")
        if isinstance(entries, dict):
            self._entries = entries

    def get(self, file_path: Path, fingerprint: str) -> SampleRecord | None:
        rel = file_path.relative_to(self.root_dir).as_posix()
        entry = self._entries.get(rel)
        if not entry:
            return None
        if entry.get("fingerprint") != fingerprint:
            return None

        record_data = entry.get("record")
        if not isinstance(record_data, dict):
            return None
        try:
            return SampleRecord.model_validate(record_data)
        except Exception:
            return None

    def put(self, file_path: Path, fingerprint: str, record: SampleRecord) -> None:
        rel = file_path.relative_to(self.root_dir).as_posix()
        self._entries[rel] = {
            "fingerprint": fingerprint,
            "record": record.model_dump(mode="json"),
        }

    def prune_to_paths(self, existing_paths: list[Path]) -> None:
        keep = {p.relative_to(self.root_dir).as_posix() for p in existing_paths}
        self._entries = {k: v for k, v in self._entries.items() if k in keep}

    def save(self) -> None:
        payload = {
            "schema_version": _CACHE_SCHEMA_VERSION,
            "config_signature": self._config_signature,
            "entries": self._entries,
        }
        self.cache_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
