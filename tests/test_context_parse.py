"""context_parse.py es puro parsing de texto por regex -- ground truth
100% exacto y controlado por el propio test (yo defino el nombre de
archivo, se que deberia extraer)."""

from __future__ import annotations

from pathlib import Path

from samples2llm.core.context_parse import directory_context, directory_type_hints, parse_declared_metadata


def _parse(filename: str):
    return parse_declared_metadata(Path(filename))


def test_bpm_extracted_from_filename():
    result = _parse("Bass_Sub_Amin_128bpm_v2.wav")
    assert result.bpm == 128.0


def test_bpm_with_hyphen_separator_extracted():
    result = _parse("Loop_140-bpm_dark.wav")
    assert result.bpm == 140.0


def test_bpm_paren_tempo_key_convention_extracted():
    """Convencion "(TEMPO, KEY)" sin la palabra bpm -- encontrada al
    probar contra la libreria real KSHMR Sounds Vol.4, donde 7 de 24
    archivos usaban este formato y perdian el bpm declarado antes de
    este fix."""
    result = _parse("KSHMR Vintage Record Loop 01 (81, Fm) - Steel Guitar Dreams Chords.wav")
    assert result.bpm == 81.0
    assert result.key == "Fm"


def test_bpm_paren_convention_with_sharp_key():
    result = _parse("KSHMR Vintage Record Loop 02 (90, A#m) - Somber Strings.wav")
    assert result.bpm == 90.0


def test_bpm_paren_convention_with_mode_name_after_key():
    """'F Phrygian' -- el modo se pierde (solo se captura la nota 'F'),
    pero el bpm si debe extraerse. Limitacion conocida y aceptada, no es
    el objetivo de este fix."""
    result = _parse("KSHMR Synth Lead Loop 02 (72, F Phrygian).wav")
    assert result.bpm == 72.0


def test_explicit_bpm_suffix_takes_priority_over_paren_convention():
    """Si el nombre tiene AMBOS patrones, el explicito ('80BPM') gana --
    es la lectura mas segura, el fallback solo se usa cuando no hay
    patron explicito."""
    result = _parse("Loop (140, Am) 128bpm.wav")
    assert result.bpm == 128.0


def test_bare_number_in_parens_without_note_letter_not_read_as_bpm():
    """'(2, 4)' no debe leerse como bpm=2 -- el patron exige que tras la
    coma venga una letra de nota (A-G), no cualquier cosa."""
    result = _parse("Sample (2, 4) take3.wav")
    assert result.bpm is None


def test_no_bpm_in_filename_returns_none():
    result = _parse("Kick_808_hit.wav")
    assert result.bpm is None


def test_number_that_is_not_bpm_pattern_not_misread_as_bpm():
    """'128' sin 'bpm' pegado no debe interpretarse como tempo."""
    result = _parse("Sample_128_v3.wav")
    assert result.bpm is None


def test_minor_key_extracted():
    result = _parse("Bass_Sub_Amin_128bpm_v2.wav")
    assert result.key == "Amin"


def test_major_key_extracted():
    result = _parse("loop_Cmaj_120bpm.wav")
    assert result.key == "Cmaj"


def test_sharp_key_extracted():
    result = _parse("Pad_Fsmin_drone.wav".replace("Fs", "F#"))
    assert result.key == "F#min"


def test_variation_v_pattern_extracted():
    result = _parse("Bass_Sub_Amin_128bpm_v2.wav")
    assert result.variation == "v2"


def test_variation_round_robin_pattern_extracted():
    result = _parse("Snare_RR3.wav")
    assert result.variation == "RR3"


def test_no_variation_returns_none():
    result = _parse("Kick_808_hit.wav")
    assert result.variation is None


def test_type_hint_kick_detected_from_keyword():
    result = _parse("Kick_808_hit.wav")
    assert "kick" in result.type_hints


def test_type_hint_bass_detected_from_sub_keyword():
    """'sub' es uno de los keywords listados para 'bass' en _TYPE_KEYWORDS."""
    result = _parse("Bass_Sub_Amin_128bpm_v2.wav")
    assert "bass" in result.type_hints


def test_multiple_type_hints_can_coexist():
    result = _parse("Bass_Sub_Amin_128bpm_v2.wav")
    # "Sub" dispara bass; no hay palabra "one-shot"/"loop" aqui, pero
    # confirmamos que el campo es una lista y bass esta presente sin
    # que otros hints inventados aparezcan.
    assert result.type_hints == ["bass"]


def test_no_matching_keywords_returns_empty_type_hints():
    result = _parse("mystery_texture_47.wav")
    assert result.type_hints == []


def test_full_metadata_combo_all_fields_at_once():
    result = _parse("Snare_Amin_140bpm_v3.wav")
    assert result.bpm == 140.0
    assert result.key == "Amin"
    assert result.variation == "v3"
    assert "snare" in result.type_hints


def test_directory_context_extracts_folder_components_between_root_and_file():
    root = Path("/samples")
    file_path = Path("/samples/Bass/Sub/Bass_Sub_Amin_128bpm_v2.wav")
    assert directory_context(file_path, root) == ["Bass", "Sub"]


def test_directory_context_empty_when_file_is_directly_in_root():
    root = Path("/samples")
    file_path = Path("/samples/Kick_808_hit.wav")
    assert directory_context(file_path, root) == []


def test_directory_context_handles_file_outside_root_gracefully():
    """Si el archivo no esta bajo root_dir (relative_to lanza ValueError),
    el codigo cae a usar el path completo -- se verifica que no revienta."""
    root = Path("/other/place")
    file_path = Path("/samples/Kick_808_hit.wav")
    result = directory_context(file_path, root)
    assert isinstance(result, list)


def test_directory_type_hints_matches_plural_folder_name():
    """'Drum Loops' (plural, nombre de carpeta real de KSHMR) debe
    disparar el hint 'loop' -- el diccionario original solo tenia la
    forma singular, que no matcheaba nombres de carpeta de categoria."""
    assert directory_type_hints(["Drum Loops", "Drum Loops - Main"]) == ["loop"]


def test_directory_type_hints_matches_vocals_folder():
    assert "vocal" in directory_type_hints(["Vocals", "Ethnic and World", "Tribal"])


def test_directory_type_hints_empty_when_no_keyword_folder():
    assert directory_type_hints(["FX", "Lazers & Glitches", "Lazers"]) == []


def test_directory_type_hints_empty_list_input_returns_empty():
    assert directory_type_hints([]) == []
