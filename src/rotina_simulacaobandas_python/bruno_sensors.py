"""
Hyperspectral / extra sensor simulation, courtesy of Bruno Rech.

The spectral response functions (``response_functions/SRF_*.parquet``) and the
resampling method in this module come from the ``rs_tools`` package by
**Bruno Rech** (National Institute for Space Research, INPE),
https://github.com/b-rech/rs_tools — used here with permission under its MIT
license (see ``response_functions/LICENSE.rs_tools``).

EnMAP and PRISMA SRFs are Gaussian curves fitted from each channel's central
wavelength and FWHM; the other sensors use SRFs from NASA's RSR repository
(https://oceancolor.gsfc.nasa.gov/resources/docs/rsr_tables/).

``get_srf`` and ``resample_sensor`` below are adapted from ``rs_tools`` with only
minor changes (load path + a wavelength-alignment safeguard).
"""

from __future__ import annotations

from importlib import resources

import numpy as np
import pandas as pd

# rs_tools sensor id -> human-readable name (for the routine's sensor registry).
BRUNO_SENSORS = {
    "hsi-enmap": "HSI (EnMAP)",
    "hyc-prisma": "HYC (PRISMA)",
    "hico-iss": "HICO (ISS)",
    "oci-pace": "OCI (PACE)",
}


# --- adapted from rs_tools.transformations (Bruno Rech, MIT) ------------------

def get_srf(sensor: str, mode: str = "rf") -> pd.DataFrame | np.ndarray:
    """Load a sensor's SRF (rows = wavelength, cols = band centers).

    mode='rf' returns response functions normalized so each band sums to 1;
    mode='cw' returns the central wavelengths.
    """
    fname = f"SRF_{sensor.upper().replace('-', '_')}.parquet"
    try:
        fpath = resources.files(
            "rotina_simulacaobandas_python.response_functions"
        ).joinpath(fname)
        srf_df = pd.read_parquet(fpath, engine="pyarrow")
        srf_df.columns = srf_df.columns.map(float)
    except Exception as exc:
        raise NotImplementedError(f"Sensor {sensor} not supported") from exc

    if mode == "rf":
        return srf_df.div(srf_df.sum(axis=0), axis=1)
    if mode == "cw":
        return srf_df.columns.to_numpy()
    raise ValueError("Select a valid mode")


def resample_sensor(field_data: pd.DataFrame, bands, sensor: str) -> pd.DataFrame:
    """Resample spectra (rows = stations, cols = wavelength) to a sensor's bands.

    Only bands whose 95% cumulative-energy interval falls entirely within the
    input wavelength range are simulated (partially-covered bands are dropped).
    """
    srf = get_srf(sensor)
    srf[np.isnan(srf)] = 0

    lim_dict = {"band": [], "wl_min": [], "wl_max": []}
    for band in srf.columns:
        sum_min = srf[band][::-1].cumsum()
        sum_max = srf[band].cumsum()
        prop = 0
        pmin = 0.95
        while prop < 0.95:
            lmin = abs(sum_min - pmin).idxmin()
            lmax = abs(sum_max - pmin).idxmin()
            pmin += 0.001
            prop = srf[band].loc[lmin:lmax].sum()
        lim_dict["band"].append(band)
        lim_dict["wl_min"].append(lmin)
        lim_dict["wl_max"].append(lmax)

    srf_limits = pd.DataFrame(lim_dict)
    wl_min, wl_max = bands.min(), bands.max()
    filtered_bands = srf_limits.loc[
        (srf_limits.wl_min >= wl_min) & (srf_limits.wl_max <= wl_max)
    ].band.tolist()

    # Align the SRF rows to exactly the input wavelengths (safeguard against
    # positional mismatch), then resample by matrix multiplication.
    srf = srf.reindex(bands, fill_value=0).loc[:, filtered_bands]
    resampled_matrix = np.asarray(field_data) @ np.asarray(srf)

    resampled = pd.DataFrame(
        resampled_matrix, columns=srf.columns, index=field_data.index
    )
    resampled.index.rename("station", inplace=True)
    return resampled


# --- routine-facing wrapper ---------------------------------------------------

def simulate(spectra: pd.DataFrame, point_names, sensor_id: str) -> pd.DataFrame:
    """Run a Bruno-provided sensor and return the routine's standard shape.

    Args:
        spectra: DataFrame with wavelengths as index and one column per station
            (same object the rest of the routine uses).
        point_names: station names (row index of the result).
        sensor_id: an rs_tools sensor id (see ``BRUNO_SENSORS``).

    Returns:
        DataFrame indexed by point_names with one ``Band_<center>nm`` column per
        simulated band.
    """
    bands = spectra.index.to_numpy(dtype=float)
    field_data = spectra.T  # stations x wavelengths
    field_data.columns = bands

    resampled = resample_sensor(field_data, bands, sensor_id)
    resampled.columns = [f"Band_{c:g}nm" for c in resampled.columns]
    resampled.index = point_names
    return resampled
