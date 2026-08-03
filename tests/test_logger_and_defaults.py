from __future__ import annotations

from samples2llm.config.defaults import (
    DEFAULT_AUDIO_EXTENSIONS,
    DEFAULT_IGNORE_DIRS,
    IGNORE_CONTROL_FILE_NAMES,
)
from samples2llm.shared.logger import get_logger


def test_defaults_include_expected_audio_extensions() -> None:
    for ext in (".wav", ".aiff", ".flac", ".mp3", ".ogg", ".m4a"):
        assert ext in DEFAULT_AUDIO_EXTENSIONS


def test_defaults_include_expected_ignored_dirs_and_control_files() -> None:
    assert ".git" in DEFAULT_IGNORE_DIRS
    assert "Ableton Project Info" in DEFAULT_IGNORE_DIRS
    assert ".gitignore" in IGNORE_CONTROL_FILE_NAMES
    assert ".samplesignore" in IGNORE_CONTROL_FILE_NAMES


def test_get_logger_is_idempotent_and_uses_info_level() -> None:
    logger_a = get_logger("samples2llm.test.logger")
    handlers_before = len(logger_a.handlers)

    logger_b = get_logger("samples2llm.test.logger")

    assert logger_a is logger_b
    assert len(logger_b.handlers) == handlers_before
    assert logger_b.level == 20  # logging.INFO
