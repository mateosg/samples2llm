"""spectral.py depende de librosa.feature.spectral_centroid, que no tiene
un "valor exacto calculable a mano" para audio real -- pero SI podemos
generar señales cuya DIRECCION de tendencia es conocida por construccion
(un barrido de frecuencia ascendente NO puede dar centroide descendente,
salvo bug). Eso es lo que se verifica: la direccion, no el valor exacto.
"""

from __future__ import annotations

from samples2llm.config.schema import SpectralTrend
from samples2llm.core.features.spectral import extract_spectral
from tests.conftest import chirp, sine_wave, silence


def test_ascending_chirp_detected_as_ascending_trend():
    """Barrido de 500Hz a 8000Hz en 2s -- el centroide DEBE subir con el
    tiempo por construccion de la señal."""
    y = chirp(500, 8000, duration_sec=2.0, sr=22050)
    result = extract_spectral(y, sr=22050)
    assert result.centroid_trend == SpectralTrend.ASCENDING
    assert result.centroid_trend_confidence > 0.5


def test_descending_chirp_detected_as_descending_trend():
    """Barrido de 8000Hz a 500Hz en 2s -- caso inverso al anterior, mismo
    razonamiento (este es el caso 'whoosh' que ya se probo antes en el
    chat con FX_whoosh_descending.wav)."""
    y = chirp(8000, 500, duration_sec=2.0, sr=22050)
    result = extract_spectral(y, sr=22050)
    assert result.centroid_trend == SpectralTrend.DESCENDING
    assert result.centroid_trend_confidence > 0.5


def test_constant_tone_detected_as_stable_trend():
    """Tono puro a frecuencia FIJA durante 2s -- el centroide no deberia
    tener una tendencia direccional real, solo jitter de frame a frame."""
    y = sine_wave(1000, duration_sec=2.0, sr=22050)
    result = extract_spectral(y, sr=22050)
    assert result.centroid_trend == SpectralTrend.STABLE


def test_short_signal_below_min_frames_has_no_trend():
    """Señal demasiado corta para _MIN_FRAMES_FOR_TREND (8 frames) ->
    debe devolver None, no adivinar una tendencia sobre pocos datos."""
    y = sine_wave(1000, duration_sec=0.02, sr=22050)
    result = extract_spectral(y, sr=22050)
    assert result.centroid_trend is None
    assert result.centroid_trend_confidence is None


def test_empty_signal_returns_zeroed_record_without_crashing():
    result = extract_spectral(silence(0.0, sr=22050), sr=22050)
    assert result.centroid_hz == 0.0
    assert result.centroid_trend is None


def test_low_frequency_tone_has_lower_centroid_than_high_frequency_tone():
    """No es un valor exacto (el centroide no es igual a la frecuencia
    fundamental), pero el ORDEN debe respetarse: un tono a 200Hz no puede
    tener un centroide espectral mayor que uno a 5000Hz."""
    low = extract_spectral(sine_wave(200, 1.0, sr=22050), sr=22050)
    high = extract_spectral(sine_wave(5000, 1.0, sr=22050), sr=22050)
    assert low.centroid_hz < high.centroid_hz


def test_pure_tone_has_low_spectral_flatness():
    """Un tono puro concentra casi toda la energia en una frecuencia --
    flatness (medida de "que tan parecido a ruido blanco") debe ser baja."""
    result = extract_spectral(sine_wave(440, 1.0, sr=22050), sr=22050)
    assert result.flatness < 0.1


def test_white_noise_has_higher_spectral_flatness_than_pure_tone():
    """Ruido blanco reparte energia por todo el espectro -> flatness
    notablemente mas alta que un tono puro. Comparacion relativa, no
    valor absoluto exacto (no hay una formula cerrada simple para
    flatness de ruido finito)."""
    from tests.conftest import white_noise

    tone = extract_spectral(sine_wave(440, 1.0, sr=22050), sr=22050)
    noise = extract_spectral(white_noise(1.0, sr=22050), sr=22050)
    assert noise.flatness > tone.flatness
