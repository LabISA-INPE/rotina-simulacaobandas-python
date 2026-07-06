# Band Simulation — review & improvement list

Comparison of the Python routine (`src/rotina_simulacaobandas_python/`) against the
original R reference (`dmaciel123/BandSimulation`, `R/spectra_simulation.R`) on the
shared GLORIA example. Line refs are in `core/spectra_simulation.py` unless noted.

**The reference formula (identical for every sensor):**
`FAC = SRF / sum(SRF)` over 400–900 nm, then `band = Σ(FAC · Rrs)` — an SRF-weighted
average of the spectrum. No scaling, no extra factors. Output units = input Rrs units.

---

## A. Correctness bugs (they change the numbers)

### A1. Every output is inflated 10× — CRITICAL  ·  `core:154`
```python
band_value = band_value * 10   # comment: "Apply scaling factor ... (divide by 10^11)"
```
There is no `×10` in the reference. Verified on the GLORIA example for OLI — the routine
returns **exactly 10× the reference** for all 5 bands (ratio 10.000 across the board).
The comment ("divide by 10^11") doesn't even match the code (multiply by 10). **Fix: delete
this line.** This also means the LIA app currently plots Rrs 10× too high; fixing the
routine fixes LIA automatically.

### A2. MODIS is offered but cannot run  ·  `config/sensor_config.py:78,85`
Config points at `modis_srf.xlsx`, which **does not exist** in `src/SRF/` (the R package
used `modis_srf.rda`), and the sensor is `enabled: False`. Yet the LIA frontend lists MODIS
as a choice → selecting it fails/returns empty. Also its `wave_centers` include 1240/1640/2130
nm, which **cannot be simulated from 400–900 nm input**. Fix: either add the MODIS SRF file
(converted from the official RSR) and drop the SWIR bands, or remove MODIS from the UI.

### A3. Fabricated stations by cycling  ·  `utils/output_handler.py:64–73, 78`
`convert_to_wave_format` fills requested-but-missing station columns by **copying real
stations** (`source_idx = i % actual_gid_count`), and `save_all_results` defaults
`target_gid_count=1000`. So with 985 real stations you silently get 15 duplicated "GID_986…"
columns presented as data. (LIA avoids this by passing the real count, but the routine's own
save path fabricates.) Fix: never pad beyond the real station count.

### A4. Meaningless per-point `Wave` column  ·  `core:175`
```python
result_df.insert(0, 'Wave', [wave_centers[i] if i < len(wave_centers) else 0
                             for i in range(len(point_names))])
```
Rows are *points*, not wavelengths, so mapping point `i → wave_centers[i]` is nonsense (and
zero-filled once points outnumber bands). It's dropped later in `convert_to_wave_format`, so
it's dead + confusing. Fix: remove.

---

## B. Output type & robustness

### B1. Band values returned as strings  ·  `core:179`
`result_df[col] = result_df[col].apply(lambda x: f"{x:.16f}" ...)` turns numeric results into
strings, forcing every consumer (incl. LIA) to re-parse with `pd.to_numeric`. Fix: keep them
numeric; format only at final CSV/JSON serialization.

### B2. Brittle wavelength alignment  ·  `core:93,121`
SRF wavelengths are `.astype(int)` (truncation, not round) and matched to the spectrum by
**exact equality**. It works today only because every SRF file is a clean 1-nm integer grid
(verified: OLCI/MSI/SuperDove/ETM/TM/CBERS/Amazônia all 1-nm). Any future sub-nm SRF (e.g.
442.5 nm) would silently misalign and mis-normalize. Fix: interpolate the SRF onto the
spectrum's wavelength grid (`np.interp`) instead of integer exact-match.

### B3. Bands with response beyond 900 nm are truncated (scientific caveat)
Because input Rrs stops at 900 nm, any band whose SRF extends past 900 (e.g. MSI B8/B8a, OLI
B5, SuperDove NIR) has its response clipped and `FAC` renormalized over the *partial* curve →
a biased band value. This is inherited from the R reference (it also filters ≤900), but it
should be **documented per sensor**, and ideally those partial-coverage bands flagged in the
output.

### B4. Silent failure swallowing  ·  `_load_all_srf`, `simulate`, `data_processor`
Broad `except Exception: continue/print(...)` on SRF load and per-band simulation means a
missing file or bad column yields empty/NaN with only a stdout warning. Fix: use logging and
surface a real error for a requested sensor that can't be produced.

---

## C. Performance & structure

### C1. Triple nested Python loop → one matrix multiply  ·  `core:107–170`
The core is `for band: for srf_wavelength: np.where(...); for point: Σ`. That's
O(bands · |SRF| · points) in pure Python. It can be a single vectorized step: build a weight
matrix `W (n_bands × n_wavelengths)` aligned to the spectrum grid (rows = normalized FAC),
then `bands = W @ spectra_values` (`n_bands × n_points`) in one `np.dot`. Faster, and the
formula becomes readable/auditable.

### C2. Round-trip cruft
Insert-then-drop `Wave` (A4), numeric→string→numeric (B1), and 10 near-identical legacy
per-sensor wrappers (`olci()`, `msi()`, …) that all just call `simulate(id, …)` — collapse to
the generic `simulate`.

---

## D. Per-sensor labeling accuracy (verify against official specs)

- **MODIS**: `wave_centers` 1240/1640/2130 are SWIR — impossible from 400–900 input; drop them.
- **CBERS-04 MUX / 04A MUX**: labeled `[490,560,665,865]`, but MUX B5–B8 centers are ≈
  485/555/660/830 nm — the 865 (NIR) label looks off; confirm and relabel.
- **OLI**: config name "Landsat-8/9" but the SRF is Landsat-8 only; L9/OLI-2 differs slightly.
- **MSI**: B1 labeled 440; Sentinel-2 B1 (coastal) is 443.
- **OLCI**: 19 bands (Oa1–Oa19) is correct for ≤900 nm (Oa20/Oa21 at 940/1020 rightly excluded).

## E. Testing & packaging

- **No tests exist** despite `pytest`/`coverage` in `pyproject.toml`. Add a regression test
  that checks a few band values against reference numbers from the R package (golden file) —
  this would have caught A1 immediately.
- The routine is otherwise well-structured (config-driven sensors, clean separation), so the
  fixes above are mostly small and local.

---

## Suggested order of work
1. **A1** (delete `×10`) + add a regression test vs reference values — biggest impact, tiny change.
2. **A3, A4, B1** — output correctness/type cleanups.
3. **C1** — vectorize the core (also makes B2 interpolation natural).
4. **A2 + D** — MODIS + per-sensor labels/spec accuracy (needs official SRF/spec confirmation).
