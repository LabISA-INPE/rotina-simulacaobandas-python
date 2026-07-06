"""Regression tests for the band-simulation core.

Independently recompute the reference formula (FAC = SRF/sum(SRF); band =
sum(FAC * Rrs), matching dmaciel123/BandSimulation) and assert the routine
reproduces it. This is the test that would have caught the historical 10x
inflation bug.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rotina_simulacaobandas_python.core import spectra_simulation

SRC = Path(__file__).resolve().parents[1] / "src"
SRF_DIR = SRC / "SRF"
EXAMPLE_CSV = SRC / "example" / "GLORIA_Rrs.csv"

GRID = np.arange(400, 901)  # 400-900 nm at 1 nm


def _load_spectra():
    data = pd.read_csv(EXAMPLE_CSV)
    rrs_cols = [f"Rrs_{w}" for w in GRID]
    spectra = pd.DataFrame(data[rrs_cols].T.values, index=list(GRID))
    # same cleaning the routine applies: negatives -> 0, NaN -> 0
    spectra = spectra.clip(lower=0).fillna(0.0)
    return spectra, data["GLORIA_ID"].tolist()


def _reference_band(srf_df, col_idx, spectra_clean):
    """SRF-weighted average, aligned to the 400-900 grid by integer wavelength."""
    wl = pd.to_numeric(srf_df.iloc[:, 0], errors="coerce").values
    v = pd.to_numeric(srf_df.iloc[:, col_idx], errors="coerce").values
    m = ~(np.isnan(wl) | np.isnan(v))
    wl, v = wl[m], v[m]
    in_range = (wl >= 400) & (wl <= 900)
    wl, v = wl[in_range], v[in_range]
    fac = v / v.sum()

    grid_pos = {w: i for i, w in enumerate(GRID)}
    band = np.zeros(spectra_clean.shape[1])
    for f, w in zip(fac, wl):
        idx = grid_pos.get(int(round(w)))
        if idx is not None:
            band += f * spectra_clean[idx]
    return band


# sensors whose SRF files are simple single-sheet 400-900 grids
SENSORS = [
    ("oli", "oli_SRF.xlsx", [1, 2, 3, 4, 5], None),
    ("etm", "L7_RSR_Ok.xlsx", [1, 2, 3, 4], None),
    ("tm", "L5_RSR.xlsx", [1, 2, 3, 4], None),
    ("olci", "olci_FRE.xlsx", list(range(1, 20)), (400, 900)),
]


@pytest.mark.parametrize("sensor_id,srf_file,band_indices,wl_range", SENSORS)
def test_matches_reference_formula(sensor_id, srf_file, band_indices, wl_range):
    spectra, points = _load_spectra()
    spectra_clean = spectra.values

    sim = spectra_simulation.SatelliteBandSimulator(srf_folder=str(SRF_DIR))
    out = sim.simulate(sensor_id, spectra, points)
    band_cols = [c for c in out.columns if c.startswith("Band_")]

    srf = pd.read_excel(SRF_DIR / srf_file, sheet_name=0)
    for bi, col in enumerate(band_cols):
        expected = _reference_band(srf, band_indices[bi], spectra_clean)
        got = out[col].to_numpy(dtype=float)
        np.testing.assert_allclose(
            got, expected, rtol=1e-6, atol=1e-9,
            err_msg=f"{sensor_id} {col} deviates from reference (10x bug regression?)",
        )


def test_output_is_numeric_without_wave_column():
    spectra, points = _load_spectra()
    sim = spectra_simulation.SatelliteBandSimulator(srf_folder=str(SRF_DIR))
    out = sim.simulate("oli", spectra, points)
    assert "Wave" not in out.columns  # the bogus per-point Wave column is gone
    for col in out.columns:
        assert np.issubdtype(out[col].dtype, np.floating)  # numeric, not strings


def test_values_are_physical_not_inflated():
    """Simulated Rrs must stay in the input's magnitude range (guards the 10x bug)."""
    spectra, points = _load_spectra()
    sim = spectra_simulation.SatelliteBandSimulator(srf_folder=str(SRF_DIR))
    out = sim.simulate("oli", spectra, points)
    band_cols = [c for c in out.columns if c.startswith("Band_")]
    max_band = out[band_cols].to_numpy(dtype=float).max()
    max_input = spectra.values.max()
    # a weighted average can't exceed the max input value (x10 would blow past it)
    assert max_band <= max_input + 1e-9


# Sensors courtesy of Bruno Rech (rs_tools). From a 400-900 nm input, the 95%
# coverage filter keeps only bands fully within range (band counts are stable).
BRUNO_EXPECTED = {"enmap": 78, "prisma": 53, "hico": 83, "pace": 241}


@pytest.mark.parametrize("sensor_id,n_bands", list(BRUNO_EXPECTED.items()))
def test_bruno_sensors(sensor_id, n_bands):
    spectra, points = _load_spectra()
    sim = spectra_simulation.SatelliteBandSimulator(srf_folder=str(SRF_DIR))
    out = sim.simulate(sensor_id, spectra, points)

    assert out.shape == (len(points), n_bands)
    centers = [float(c.replace("Band_", "").replace("nm", "")) for c in out.columns]
    assert all(400 <= c <= 900 for c in centers)  # coverage filter respected

    vals = out.to_numpy(dtype=float)
    assert vals.max() <= spectra.values.max() + 1e-9  # physical (weighted average)
    assert np.isfinite(vals).all()
