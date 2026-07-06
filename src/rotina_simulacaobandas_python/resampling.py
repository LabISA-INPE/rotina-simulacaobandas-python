"""
Spectral resampling engine — the single method used for every sensor.

The approach and the original spectral response functions come from the
``rs_tools`` package by **Bruno Rech** (National Institute for Space Research,
INPE), https://github.com/b-rech/rs_tools, used under its MIT license
(retained at ``response_functions/LICENSE.rs_tools``). The whole routine is
standardized on this method and on parquet SRFs.

Each band is the SRF-weighted average of the input spectrum
(``FAC = SRF / sum(SRF)``; ``band = sum(FAC * Rrs)``). Only bands whose 95%
cumulative-energy interval lies entirely within the input wavelength range are
simulated (partially-covered bands are dropped rather than truncated/biased).

SRFs are stored as ``response_functions/SRF_<ID>.parquet`` with wavelengths as
the row index and band-center wavelengths as (float) column names. Sensors from
Bruno's package use his data directly; the LabISA-only sensors (ETM, TM,
SuperDove, CBERS, Amazonia) were converted to this same format, with each band's
center taken as the centroid of its SRF.
"""

from __future__ import annotations

from importlib import resources

import numpy as np
import pandas as pd


def get_srf(srf_name: str, mode: str = "rf") -> pd.DataFrame | np.ndarray:
    """Load an SRF parquet (rows = wavelength, cols = band centers).

    mode='rf' returns response functions normalized so each band sums to 1;
    mode='cw' returns the central wavelengths.
    """
    fname = srf_name if srf_name.endswith(".parquet") else f"{srf_name}.parquet"
    try:
        fpath = resources.files(
            "rotina_simulacaobandas_python.response_functions"
        ).joinpath(fname)
        srf_df = pd.read_parquet(fpath, engine="pyarrow")
        srf_df.columns = srf_df.columns.map(float)
    except Exception as exc:
        raise NotImplementedError(f"SRF {srf_name} not found") from exc

    if mode == "rf":
        return srf_df.div(srf_df.sum(axis=0), axis=1)
    if mode == "cw":
        return srf_df.columns.to_numpy()
    raise ValueError("Select a valid mode")


def resample_sensor(field_data: pd.DataFrame, bands, srf_name: str) -> pd.DataFrame:
    """Resample spectra (rows = stations, cols = wavelength) onto a sensor's bands.

    Adapted from ``rs_tools.transformations.resample_sensor`` (Bruno Rech, MIT),
    with the SRF loaded by name and reindexed onto the input grid.
    """
    srf = get_srf(srf_name)
    srf[np.isnan(srf)] = 0

    # For each band, find the 95% cumulative-energy wavelength interval.
    lim = {"band": [], "wl_min": [], "wl_max": []}
    for band in srf.columns:
        sum_min = srf[band][::-1].cumsum()
        sum_max = srf[band].cumsum()
        prop, pmin = 0, 0.95
        while prop < 0.95:
            lmin = abs(sum_min - pmin).idxmin()
            lmax = abs(sum_max - pmin).idxmin()
            pmin += 0.001
            prop = srf[band].loc[lmin:lmax].sum()
        lim["band"].append(band)
        lim["wl_min"].append(lmin)
        lim["wl_max"].append(lmax)

    limits = pd.DataFrame(lim)
    wl_min, wl_max = bands.min(), bands.max()
    keep = limits.loc[
        (limits.wl_min >= wl_min) & (limits.wl_max <= wl_max)
    ].band.tolist()

    # Align SRF rows to exactly the input wavelengths, then resample by matmul.
    srf = srf.reindex(bands, fill_value=0).loc[:, keep]
    resampled = np.asarray(field_data) @ np.asarray(srf)

    out = pd.DataFrame(resampled, columns=srf.columns, index=field_data.index)
    out.index.rename("station", inplace=True)
    return out


def simulate(spectra: pd.DataFrame, point_names, srf_name: str) -> pd.DataFrame:
    """Simulate one sensor; return points x ``Band_<center>nm`` (routine shape).

    Args:
        spectra: wavelengths as index, one column per station (the routine's
            standard spectra object).
        point_names: station names for the result index.
        srf_name: SRF parquet basename, e.g. ``"SRF_MSI_S2A"``.
    """
    bands = spectra.index.to_numpy(dtype=float)
    field_data = spectra.T
    field_data.columns = bands

    resampled = resample_sensor(field_data, bands, srf_name)
    resampled.columns = [f"Band_{c:g}nm" for c in resampled.columns]
    resampled.index = point_names
    return resampled
