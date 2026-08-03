"""spatial.py: width_ratio = energia(side) / (energia(mid)+energia(side)),
con mid=(L+R)/2, side=(L-R)/2. Es matematica pura y por tanto invertible:
para lograr un mid=A y side=B EXACTOS y conocidos de antemano basta
construir L=A+B, R=A-B (arrays constantes). Esto permite verificar el
valor de width_ratio con precision exacta, no solo su categoria.
"""

from __future__ import annotations

import numpy as np
import pytest

from samples2llm.config.schema import SpatialWidth
from samples2llm.core.features.spatial import extract_spatial


def _constant_stereo_with_mid_side(mid: float, side: float, n: int = 1000) -> np.ndarray:
    left = np.full(n, mid + side, dtype=np.float64)
    right = np.full(n, mid - side, dtype=np.float64)
    return np.stack([left, right])


def test_mono_input_has_no_width_ratio():
    y = np.zeros((1, 1000), dtype=np.float32)
    result = extract_spatial(y)
    assert result.channels == 1
    assert result.width_ratio is None
    assert result.width_category == SpatialWidth.MONO


def test_identical_channels_have_width_ratio_exactly_zero():
    """L == R por construccion -> side=0 en todas las muestras ->
    width_ratio EXACTAMENTE 0.0, no solo "bajo"."""
    mono = np.linspace(-0.5, 0.5, 4000)
    y = np.stack([mono, mono])
    result = extract_spatial(y)
    assert result.width_ratio == pytest.approx(0.0, abs=1e-9)
    assert result.width_category == SpatialWidth.NARROW


def test_width_ratio_at_exact_value_below_narrow_threshold():
    """mid=10, side=1 -> ratio = 1/(100+1) = 0.0099..., por debajo del
    umbral NARROW (0.02). Valor calculado a mano, no aproximado."""
    y = _constant_stereo_with_mid_side(mid=10.0, side=1.0)
    result = extract_spatial(y)
    expected_ratio = round(1.0 / (100.0 + 1.0), 4)
    assert result.width_ratio == pytest.approx(expected_ratio, abs=1e-4)
    assert result.width_category == SpatialWidth.NARROW


def test_width_ratio_at_exact_value_in_medium_range():
    """mid=3, side=1 -> ratio = 1/(9+1) = 0.1 exacto, dentro del rango
    MEDIUM (0.02 <= ratio < 0.15)."""
    y = _constant_stereo_with_mid_side(mid=3.0, side=1.0)
    result = extract_spatial(y)
    assert result.width_ratio == pytest.approx(0.1, abs=1e-4)
    assert result.width_category == SpatialWidth.MEDIUM


def test_width_ratio_at_exact_value_in_wide_range():
    """mid=2, side=1 -> ratio = 1/(4+1) = 0.2 exacto, por encima del
    umbral WIDE (0.15)."""
    y = _constant_stereo_with_mid_side(mid=2.0, side=1.0)
    result = extract_spatial(y)
    assert result.width_ratio == pytest.approx(0.2, abs=1e-4)
    assert result.width_category == SpatialWidth.WIDE


def test_fully_decorrelated_independent_noise_lands_near_half():
    """L y R generados con ruido gaussiano independiente e identica
    varianza: estadisticamente mid_energy ~ side_energy -> ratio ~ 0.5.
    Ya validado antes en la conversacion (~0.50 medido); aqui se fija
    como test de regresion permanente con tolerancia estadistica (N grande
    para que la varianza del estimador sea pequeña)."""
    rng = np.random.default_rng(42)
    n = 200_000
    left = rng.normal(0, 1.0, n)
    right = rng.normal(0, 1.0, n)
    y = np.stack([left, right])
    result = extract_spatial(y)
    assert result.width_ratio == pytest.approx(0.5, abs=0.02)
    assert result.width_category == SpatialWidth.WIDE


def test_silent_stereo_signal_does_not_crash_on_division_by_zero():
    """mid_energy=side_energy=0 -> el modulo debe devolver 0.0, no NaN ni
    excepcion (total>0 se comprueba explicitamente en el codigo)."""
    y = np.zeros((2, 1000), dtype=np.float32)
    result = extract_spatial(y)
    assert result.width_ratio == 0.0
    assert result.width_category == SpatialWidth.NARROW
