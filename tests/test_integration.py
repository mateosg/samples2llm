"""Pipeline completo end-to-end sobre TODOS los samples disponibles
(sinteticos + ESC-50 real), en vez de solo modulos aislados. Objetivo:
detectar lo que un test unitario no puede -- interacciones entre modulos,
como el bug historico de este proyecto (derived_tags.py y summarize.py
decidiendo lo mismo con logica distinta y divergiendo).
"""

from __future__ import annotations

import glob

import pytest

from samples2llm.config.schema import ExtractionConfig
from samples2llm.core.extract import extract_sample
from tests.conftest import ESC50_DIR, LIBRARY_DIR

ALL_SAMPLE_FILES = sorted(glob.glob(str(ESC50_DIR / "*.wav"))) + sorted(
    glob.glob(str(LIBRARY_DIR / "**" / "*.wav"), recursive=True)
)


@pytest.mark.parametrize("file_path", ALL_SAMPLE_FILES, ids=lambda p: p.split("/")[-1])
def test_extract_sample_runs_without_exceptions_on_every_sample(file_path):
    """No es una prueba de "el resultado es correcto", es una prueba de
    "el pipeline no revienta" sobre cada archivo disponible -- una
    regresion en un modulo (excepcion no capturada, NaN propagado a
    pydantic, etc.) se detecta aqui aunque los tests unitarios pasen."""
    from pathlib import Path

    root = Path(file_path).parent
    record = extract_sample(Path(file_path), root, ExtractionConfig())
    assert record.file_metadata.duration_sec > 0
    assert record.summary is not None
    assert len(record.summary) > 0


@pytest.mark.parametrize("file_path", ALL_SAMPLE_FILES, ids=lambda p: p.split("/")[-1])
def test_length_tag_and_summary_length_word_are_consistent(file_path):
    """Regresion DIRECTA del bug ya corregido en la ronda anterior: la
    palabra de duracion en el `summary` (one-shot/loop/sustained) debe
    coincidir siempre con el tag correspondiente en `derived_tags`. Si
    algun cambio futuro vuelve a separar la logica entre los dos modulos,
    este test lo detecta inmediatamente en cualquiera de los samples
    disponibles, no solo en el caso que origino el bug."""
    from pathlib import Path

    root = Path(file_path).parent
    record = extract_sample(Path(file_path), root, ExtractionConfig())

    length_tags_present = [t for t in record.derived_tags if t in ("one_shot", "loop", "sustained")]
    assert len(length_tags_present) == 1
    length_tag = length_tags_present[0]
    length_word = length_tag.replace("_", "-")
    assert record.summary.lower().startswith(length_word)


@pytest.mark.parametrize("file_path", ALL_SAMPLE_FILES, ids=lambda p: p.split("/")[-1])
def test_timbre_tag_and_summary_timbre_phrase_are_consistent(file_path):
    """Misma logica que el test anterior pero para classify_timbre --
    verifica que ambos modulos usan la misma fuente de verdad tras el fix
    del bug encontrado con rain.wav/helicopter.wav."""
    from pathlib import Path

    root = Path(file_path).parent
    record = extract_sample(Path(file_path), root, ExtractionConfig())

    summary_lower = record.summary.lower()
    if "tonal" in record.derived_tags:
        assert "tonal, pitch" in summary_lower
    elif "harmonic_no_stable_pitch" in record.derived_tags:
        assert "harmonic texture, no stable pitch" in summary_lower
    elif "percussive" in record.derived_tags:
        assert "percussive, no stable pitch" in summary_lower


def test_tonal_tag_never_applied_without_confident_pitch_across_all_samples():
    """Regresion directa del bug de rain.wav: ningun sample real debe
    quedar etiquetado 'tonal' si su pitch_confidence esta por debajo del
    umbral minimo (0.3) -- se comprueba contra los 10 samples de ESC-50 y
    los sinteticos, no solo el caso que origino el fix."""
    from pathlib import Path

    for file_path in ALL_SAMPLE_FILES:
        root = Path(file_path).parent
        record = extract_sample(Path(file_path), root, ExtractionConfig())
        if "tonal" in record.derived_tags:
            assert record.tonal.pitch_confidence is not None
            assert record.tonal.pitch_confidence >= 0.3, (
                f"{file_path}: etiquetado 'tonal' con confianza "
                f"{record.tonal.pitch_confidence} -- regresion del bug de rain.wav"
            )


def test_declared_bpm_mismatch_flag_on_known_synthetic_sample():
    """loop_Cmaj_120bpm.wav declara 120bpm en el nombre -- si el detector
    mide algo claramente distinto (ni 120, ni el doble, ni la mitad), debe
    aparecer el flag de discrepancia. No fijamos el bpm_detected exacto
    (es una heuristica de libreria), solo que el sistema de flags
    reacciona coherentemente a lo que sea que se detecte."""
    from pathlib import Path

    file_path = LIBRARY_DIR / "loop_Cmaj_120bpm.wav"
    record = extract_sample(file_path, LIBRARY_DIR, ExtractionConfig())
    assert record.declared.bpm == 120.0
    if record.rhythmic.bpm_detected is not None:
        matches = any(
            abs(record.rhythmic.bpm_detected - 120.0 * factor) / (120.0 * factor) <= 0.03
            for factor in (1.0, 2.0, 0.5)
        )
        if not matches:
            assert "declared_bpm_mismatch_with_detected_bpm" in record.confidence_flags


def test_bpm_without_musical_context_flag_on_unlabeled_real_sound():
    """rain.wav (ESC-50): sin bpm ni type_hint musical en el nombre, pero
    bpm_detected SI produce un numero (librosa.beat.beat_track siempre
    devuelve algo si pasa la compuerta de duracion/onset_count). Debe
    quedar marcado -- es exactamente el caso que motivo esta decision de
    diseño (rain.wav dando bpm_detected=126.05 sin ninguna corroboracion)."""
    from pathlib import Path

    file_path = ESC50_DIR / "rain.wav"
    record = extract_sample(file_path, ESC50_DIR, ExtractionConfig())
    assert record.declared.bpm is None
    if record.rhythmic.bpm_detected is not None:
        assert "bpm_detected_without_musical_context" in record.confidence_flags


def test_bpm_without_musical_context_flag_absent_when_declared_bpm_present():
    """loop_Cmaj_120bpm.wav SI declara bpm en el nombre -- el flag de "sin
    contexto musical" no debe aparecer (haya o no discrepancia con lo
    medido, eso ya lo cubre declared_bpm_mismatch_with_detected_bpm por
    separado)."""
    file_path = LIBRARY_DIR / "loop_Cmaj_120bpm.wav"
    record = extract_sample(file_path, LIBRARY_DIR, ExtractionConfig())
    assert "bpm_detected_without_musical_context" not in record.confidence_flags


def test_bpm_without_musical_context_flag_absent_when_musical_type_hint_present():
    """Bass_Sub_Amin_128bpm_v2.wav tiene type_hint 'bass' Y bpm declarado
    -- cualquiera de los dos deberia bastar para no marcar el flag de
    "sin contexto musical" (aunque en este caso ademas hay discrepancia
    de bpm, cubierta por otro flag distinto)."""
    file_path = LIBRARY_DIR / "Bass" / "Sub" / "Bass_Sub_Amin_128bpm_v2.wav"
    record = extract_sample(file_path, LIBRARY_DIR, ExtractionConfig())
    assert "bpm_detected_without_musical_context" not in record.confidence_flags


def test_tempo_periodicity_strength_present_across_all_samples_with_signal():
    """tempo_periodicity_strength debe estar en [0, 1] (o None solo si no
    hay señal suficiente para calcularlo) en todo el set disponible --
    regresion de integracion, no solo del caso aislado."""
    for file_path in ALL_SAMPLE_FILES:
        from pathlib import Path

        root = Path(file_path).parent
        record = extract_sample(Path(file_path), root, ExtractionConfig())
        strength = record.rhythmic.tempo_periodicity_strength
        if strength is not None:
            assert 0.0 <= strength <= 1.0, f"{file_path}: {strength}"


def test_bpm_without_musical_context_flag_suppressed_by_folder_name(tmp_path):
    """rain.wav (ESC-50) dispara bpm_detected_without_musical_context
    cuando esta suelto en la raiz (sin bpm ni type_hint en el nombre, ver
    test_bpm_without_musical_context_flag_on_unlabeled_real_sound). Si el
    MISMO archivo se coloca dentro de una carpeta 'Loops/' -- sin tocar
    el nombre de archivo -- el flag debe desaparecer, porque ahora hay
    corroboracion via directory_context. Encontrado necesario al probar
    contra una libreria real (KSHMR Sounds Vol.4) donde la categoria a
    veces solo esta en el nombre de carpeta."""
    import shutil

    nested = tmp_path / "Loops"
    nested.mkdir()
    shutil.copy(ESC50_DIR / "rain.wav", nested / "rain.wav")

    record = extract_sample(nested / "rain.wav", tmp_path, ExtractionConfig())
    assert record.directory_context == ["Loops"]
    if record.rhythmic.bpm_detected is not None:
        assert "bpm_detected_without_musical_context" not in record.confidence_flags


def test_bpm_without_musical_context_flag_still_fires_with_unrelated_folder_name(tmp_path):
    """Contraste con el test anterior: una carpeta que NO es una de las
    keywords musicales (p.ej. 'Misc') no debe suprimir el flag -- para
    evitar falsos negativos por cualquier nombre de carpeta."""
    import shutil

    nested = tmp_path / "Misc"
    nested.mkdir()
    shutil.copy(ESC50_DIR / "rain.wav", nested / "rain.wav")

    record = extract_sample(nested / "rain.wav", tmp_path, ExtractionConfig())
    if record.rhythmic.bpm_detected is not None:
        assert "bpm_detected_without_musical_context" in record.confidence_flags
