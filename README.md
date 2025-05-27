# Satellite Band Simulation

A Python library for simulating satellite sensor bands from hyperspectral remote sensing data. This tool converts continuous spectral data (400-900 nm) into discrete satellite band measurements for various sensors including Sentinel-2 (MSI), Sentinel-3 (OLCI), Landsat (OLI, ETM+, TM), Planet SuperDove, and MODIS.

## 🛰️ Supported Satellites

| Satellite | Sensor | Bands | Wavelength Range | Method |
|-----------|--------|-------|------------------|--------|
| Sentinel-3 | OLCI | 19 | 400-900 nm | `olci()` |
| Sentinel-2A/2B | MSI | 9 | 400-900 nm | `msi()` |
| Landsat-8/9 | OLI | 5 | 400-900 nm | `oli()` |
| Landsat-7 | ETM+ | 4 | 400-900 nm | `etm()` |
| Landsat-5 | TM | 4 | 400-900 nm | `tm()` |
| Planet | SuperDove | 8 | 400-900 nm | `superdove()` |
| Aqua/Terra | MODIS | 16 | 400-900 nm | `modis()` |

## 📋 Requirements

- Python 3.8+
- pandas >= 1.3.0
- numpy >= 1.21.0

## 🚀 Installation

### Using Poetry (Recommended)

```bash
# Clone the repository
git clone https://github.com/yourusername/rotina-simulacaobandas-python.git
cd rotina-simulacaobandas-python

# Install with Poetry
poetry install
poetry shell
```

### Using pip

```bash
pip install -e .
```

## 📁 Project Structure

```
rotina-simulacaobandas-python/
├── src/
│   └── rotina_simulacaobandas_python/
│       ├── core/
│       │   └── spectra_simulation.py    # Main simulation class
│       └── utils/
│           └── formatters.py            # Output formatting utilities
├── data-raw/                            # Spectral Response Functions (SRF)
│   ├── s3_srf.pkl                      # Sentinel-3 OLCI
│   ├── s2_srf.pkl                      # Sentinel-2A MSI
│   ├── s2b_srf.pkl                     # Sentinel-2B MSI
│   ├── l8_srf.pkl                      # Landsat-8 OLI
│   ├── l7_srf.pkl                      # Landsat-7 ETM+
│   ├── l5_srf.pkl                      # Landsat-5 TM
│   ├── planet_srf.pkl                  # Planet SuperDove
│   └── modis_srf.pkl                   # MODIS
├── example/
│   └── GLORIA_Rrs.csv                  # Sample input data
├── results/                            # Output directory
└── main.py                             # Example usage script
```
