"""Regression tests for the unified (parquet + resampling) band simulator.

Every sensor is simulated by the same engine. We check that:
- output matches an independent recomputation of the SRF-weighted average from
  the same parquet SRF (guards the numeric result, incl. the old 10x bug);
- band centers respect the 95% coverage filter (all within the input range);
- values are physical (a weighted average never exceeds the max input).
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rotina_simulacaobandas_python import resampling
from rotina_simulacaobandas_python.config.sensor_config import SENSOR_CONFIGS
from rotina_simulacaobandas_python.core import spectra_simulation

SRC = Path(__file__).resolve().parents[1] / "src"
EXAMPLE_CSV = SRC / "example" / "GLORIA_Rrs.csv"
GRID = np.arange(400, 901)


def _load_spectra():
    data = pd.read_csv(EXAMPLE_CSV)
    spectra = pd.DataFrame(
        data[[f"Rrs_{w}" for w in GRID]].T.values, index=list(GRID)
    ).clip(lower=0).fillna(0.0)
    return spectra, data["GLORIA_ID"].tolist()


def _reference(srf_name, spectra):
    """Independent SRF-weighted average from the parquet, on the input grid."""
    srf = resampling.get_srf(srf_name)  # normalized per band
    srf = srf.fillna(0).reindex(GRID.astype(float), fill_value=0)  # zero out-of-support NaN
    vals = spectra.values  # wl x points
    return {band: srf[band].to_numpy() @ vals for band in srf.columns}


@pytest.mark.parametrize("sensor_id", list(SENSOR_CONFIGS))
def test_matches_reference_and_is_physical(sensor_id):
    spectra, points = _load_spectra()
    sim = spectra_simulation.SatelliteBandSimulator()
    out = sim.simulate(sensor_id, spectra, points)

    assert out.shape[0] == len(points)
    assert out.shape[1] >= 1  # at least one band survives the coverage filter

    centers = [float(c.replace("Band_", "").replace("nm", "")) for c in out.columns]
    assert all(400 <= c <= 900 for c in centers)  # coverage filter respected

    vals = out.to_numpy(dtype=float)
    assert np.isfinite(vals).all()
    assert vals.max() <= spectra.values.max() + 1e-9  # physical (weighted mean)

    # Numeric result matches an independent recomputation for the kept bands.
    ref = _reference(SENSOR_CONFIGS[sensor_id]["srf"], spectra)
    for center, col in zip(centers, out.columns):
        expected = ref[center]
        np.testing.assert_allclose(
            out[col].to_numpy(dtype=float), expected, rtol=1e-6, atol=1e-9,
            err_msg=f"{sensor_id} {col} deviates from the SRF-weighted reference",
        )


def test_output_is_numeric_without_wave_column():
    spectra, points = _load_spectra()
    sim = spectra_simulation.SatelliteBandSimulator()
    out = sim.simulate("oli_l8", spectra, points)
    assert "Wave" not in out.columns
    for col in out.columns:
        assert np.issubdtype(out[col].dtype, np.floating)


def test_hyperspectral_band_counts_stable():
    """Coverage filter yields a stable band count from a 400-900 nm input."""
    spectra, points = _load_spectra()
    sim = spectra_simulation.SatelliteBandSimulator()
    expected = {"enmap": 78, "prisma": 53, "hico": 83, "pace": 241}
    for sensor_id, n in expected.items():
        assert sim.simulate(sensor_id, spectra, points).shape[1] == n
