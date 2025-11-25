# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A Python library for simulating satellite sensor bands from hyperspectral remote sensing data. Converts continuous spectral data (400-900 nm) into discrete satellite band measurements for various sensors including Sentinel-2 (MSI), Sentinel-3 (OLCI), Landsat (OLI, ETM+, TM), Planet SuperDove, and MODIS.

## Development Commands

### Environment Setup
```bash
# Using Poetry (recommended)
poetry install
poetry shell

# Using pip
pip install -e .
```

### Testing
```bash
# Run all tests
poetry run pytest

# Run with coverage
poetry run pytest --cov=src/rotina_simulacaobandas_python --cov-report=html

# Run specific test file (if exists)
poetry run pytest tests/test_<filename>.py -v
```

### Running the Simulation
```bash
# From project root
cd src/rotina_simulacaobandas_python
python main.py
```

## Architecture

### Core Components

The codebase follows a configuration-driven architecture with clear separation of concerns:

**Configuration Layer** ([config/sensor_config.py](src/rotina_simulacaobandas_python/config/sensor_config.py))
- Central configuration file defining all sensor parameters
- Each sensor config includes: file path, file type, band indices, wavelength centers, and optional wavelength ranges
- Adding new sensors only requires updating this config file
- Disabled sensors (like MODIS without SRF file) are marked with `enabled: False`

**SatelliteBandSimulator** ([core/spectra_simulation.py](src/rotina_simulacaobandas_python/core/spectra_simulation.py))
- Central simulation engine that loads SRF files from Excel/CSV based on config
- `_load_all_srf()`: Automatically loads all enabled sensors from config
- `simulate(sensor_id, spectra, point_names, variant)`: Generic simulation method for any sensor
- Legacy methods (olci(), msi(), oli(), etc.): Backward-compatible wrapper methods
- All simulation logic uses `_simulate_bands_direct_optimized()` algorithm

**DataLoader** ([utils/data_loader.py](src/rotina_simulacaobandas_python/utils/data_loader.py))
- Loads GLORIA CSV input data containing hyperspectral measurements
- Expected format: `GLORIA_ID` column followed by `Rrs_<wavelength>` columns (e.g., Rrs_400, Rrs_401, ..., Rrs_900)

**DataProcessor** ([utils/data_processor.py](src/rotina_simulacaobandas_python/utils/data_processor.py))
- Orchestrates simulation workflows using config-driven sensor list
- Automatically builds available sensors from SENSOR_CONFIGS
- Key methods:
  - `process_spectra()`: Extracts wavelengths 400-900nm, transposes data, cleans NaN/negative values
  - `run_sensor_simulation()`: Runs single sensor, handles variants (e.g., MSI s2a/s2b)
  - `run_multiple_sensors()`: Accepts list of sensor strings or dicts with `{'sensor': 'msi', 'variant': 's2b'}`
  - `run_all_simulations()`: Runs all enabled sensors with all variants

**OutputHandler** ([utils/output_handler.py](src/rotina_simulacaobandas_python/utils/output_handler.py))
- Converts simulation results from band-per-column to wave-per-row format
- `convert_to_wave_format()`: Transposes data, creates `Wave` column + `GID_N` columns
- `save_all_results()`: Saves each sensor result as `<sensor_name>_simulation.csv` in results directory

### Data Flow

1. Load hyperspectral data (wavelengths × points)
2. Process: extract 400-900nm range, transpose, clean invalid values
3. Simulate: for each sensor, convolve SRF with spectra using weighted sum
4. Format: transpose results (bands → rows, points → columns)
5. Save: CSV files in results/ directory

### Band Simulation Algorithm

The `_simulate_bands_direct_optimized()` method implements the core convolution:
1. Load SRF data for the target sensor band
2. Filter SRF to valid wavelengths and optional wavelength range
3. Normalize SRF values by their sum (FAC = SRF / Σ(SRF))
4. Match SRF wavelengths with input spectra wavelengths
5. For each point: band_value = Σ(FAC × spectra) × 10
6. Handle NaN/negative values by replacing with 0.0

### Path Conventions

- Main script imports assume running from [src/rotina_simulacaobandas_python/](src/rotina_simulacaobandas_python/) directory
- SRF data loads directly from Excel/CSV files in `../SRF/` directory (no pickle files needed)
- Example data path: `../example/GLORIA_Rrs.csv`
- Output directory: `results/` (created automatically)

### Supported Sensors

| Sensor ID | Full Name | Variants | Bands | Method |
|-----------|-----------|----------|-------|--------|
| olci | OLCI (Sentinel-3) | None | 19 | olci() |
| msi | MSI (Sentinel-2) | s2a, s2b | 9 | msi() |
| oli | OLI (Landsat-8/9) | None | 5 | oli() |
| etm | ETM+ (Landsat-7) | None | 4 | etm() |
| tm | TM (Landsat-5) | None | 4 | tm() |
| superdove | SuperDove (Planet) | None | 8 | superdove() |
| modis | MODIS (Aqua/Terra) | None | 16 | modis() |
| cbers04_mux | CBERS-04 MUX | None | 4 | cbers04_mux() |
| cbers04a_mux | CBERS-04A MUX | None | 4 | cbers04a_mux() |
| cbers04a_wpm | CBERS-04A WPM | None | 5 | cbers04a_wpm() |
| amazonia1_wfi | Amazonia-1 WFI | None | 8 | amazonia1_wfi() |

## Adding New Sensors

To add a new sensor, simply update [config/sensor_config.py](src/rotina_simulacaobandas_python/config/sensor_config.py):

```python
'new_sensor_id': {
    'name': 'Sensor Full Name',
    'file': 'sensor_srf_file.xlsx',
    'file_type': 'excel',  # or 'csv'
    'sheet': 0,  # for Excel files
    'band_indices': list(range(1, 5)),  # SRF column indices
    'wave_centers': [440, 560, 665, 865],  # Band wavelengths
    'wavelength_range': (400, 900),  # Optional filter
    'enabled': True  # Set to False to disable
}
```

No code changes needed! The simulator will automatically:
1. Load the SRF file
2. Create the simulation method
3. Add it to available sensors

## Important Notes

- Python 3.11+ required (specified in pyproject.toml)
- All numeric outputs use 16-decimal precision to avoid scientific notation
- The codebase handles invalid spectral data by replacing NaN/negative values with 0.0
- MSI is the only sensor with variants; handle the dict return value appropriately
- Results are padded to target_gid_count (default 1000) by cycling through actual data
- Configuration-driven: Add sensors by editing config only, no code changes needed
