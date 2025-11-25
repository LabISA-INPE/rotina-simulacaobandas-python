# Satellite Band Simulation

A Python library for simulating satellite sensor bands from hyperspectral remote sensing data. This tool converts continuous spectral data (400-900 nm) into discrete satellite band measurements using Spectral Response Functions (SRF) for various international and Brazilian satellite sensors.

## ✨ Key Features

- **Configuration-driven architecture**: Add new sensors by editing config only, no code changes needed
- **11 satellite sensors supported**: International (Sentinel, Landsat, Planet) and Brazilian (CBERS, Amazonia-1) satellites
- **Automatic data handling**: Processes NaN values, negative reflectances, and missing wavelengths
- **Flexible workflows**: Run single sensors, multiple sensors, or all sensors at once
- **Variant support**: Handle sensor variants (e.g., Sentinel-2A vs 2B) automatically
- **Production-ready output**: High-precision CSV files with proper formatting

## 🛰️ Supported Satellites

| Satellite | Sensor | Bands | Wavelength Range | Status |
|-----------|--------|-------|------------------|--------|
| Sentinel-3 | OLCI | 19 | 400-900 nm | ✅ Active |
| Sentinel-2A/2B | MSI | 9 | 400-900 nm | ✅ Active |
| Landsat-8/9 | OLI | 5 | 400-900 nm | ✅ Active |
| Landsat-7 | ETM+ | 4 | 400-900 nm | ✅ Active |
| Landsat-5 | TM | 4 | 400-900 nm | ✅ Active |
| Planet | SuperDove | 8 | 400-900 nm | ✅ Active |
| CBERS-04 | MUX | 4 | 400-900 nm | ✅ Active |
| CBERS-04A | MUX | 4 | 400-900 nm | ✅ Active |
| CBERS-04A | WPM | 5 (incl. PAN) | 400-900 nm | ✅ Active |
| Amazonia-1 | WFI | 8 | 400-900 nm | ✅ Active |

## 📋 Requirements

- Python 3.11+
- pandas >= 2.2.3
- numpy >= 2.2.6
- openpyxl >= 3.0.0

## 🚀 Installation

### Using Poetry (Recommended)

```bash
# Clone the repository
git clone https://github.com/LabISA-INPE/rotina-simulacaobandas-python.git
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
│   ├── SRF/                              # Spectral Response Functions (Excel/CSV)
│   │   ├── olci_FRE.xlsx                # Sentinel-3 OLCI
│   │   ├── Spectral Response - Sentinel 2.xlsx  # Sentinel-2A/2B MSI
│   │   ├── oli_SRF.xlsx                 # Landsat-8/9 OLI
│   │   ├── L7_RSR_Ok.xlsx               # Landsat-7 ETM+
│   │   ├── L5_RSR.xlsx                  # Landsat-5 TM
│   │   ├── Superdove.csv                # Planet SuperDove
│   │   ├── CBERS_04_MUX.xlsx            # CBERS-04 MUX
│   │   ├── cbers_04a_mux.xlsx           # CBERS-04A MUX
│   │   ├── CBERS_04A_WPM.xlsx           # CBERS-04A WPM
│   │   └── amazonia_1.xlsx              # Amazonia-1 WFI
│   └── rotina_simulacaobandas_python/
│       ├── config/
│       │   └── sensor_config.py         # Sensor configuration (add sensors here!)
│       ├── core/
│       │   └── spectra_simulation.py    # Main simulation engine
│       ├── utils/
│       │   ├── data_loader.py           # Input data loading
│       │   ├── data_processor.py        # Simulation orchestration
│       │   └── output_handler.py        # Result formatting & saving
│       └── main.py                      # Example usage script
├── example/
│   └── GLORIA_Rrs.csv                   # Sample hyperspectral data
└── results/                             # Output directory (auto-created)
```

## 🎯 Quick Start

```bash
# Navigate to source directory
cd src/rotina_simulacaobandas_python

# Run example simulations
python main.py
```

This will:
1. Load sample GLORIA hyperspectral data
2. Run simulations for all supported sensors
3. Save results to `results/` directory

## 💻 Usage Examples

### Basic Usage - Single Sensor

```python
from core.spectra_simulation import SatelliteBandSimulator
from utils.data_loader import DataLoader
from utils.data_processor import DataProcessor

# Initialize components
simulator = SatelliteBandSimulator()
loader = DataLoader()
processor = DataProcessor()

# Load and process data
data = loader.load_gloria_data("../example/GLORIA_Rrs.csv")
spectra, point_names = processor.process_spectra(data)

# Run OLI simulation (legacy method)
oli_result = simulator.oli(spectra, point_names)
```

### Using the New Generic API

```python
# Simulate any sensor by ID
cbers_result = simulator.simulate('cbers04_mux', spectra, point_names)

# Simulate specific variant
s2a_result = simulator.simulate('msi', spectra, point_names, variant='s2a')
```

### Run Multiple Selected Sensors

```python
from utils.output_handler import OutputHandler

output_handler = OutputHandler("results")

# Select specific sensors to run
selected_sensors = [
    'oli',                                    # Single sensor
    'olci',                                   # Another sensor
    {'sensor': 'msi', 'variant': 's2b'},     # Specific variant
    'cbers04_mux',                           # Brazilian satellite
]

# Run simulations
results = processor.run_multiple_sensors(
    simulator, spectra, point_names, selected_sensors
)

# Save all results
output_handler.save_all_results(results, point_names)
```

### Run All Available Sensors

```python
# Automatically runs all enabled sensors with all variants
all_results = processor.run_all_simulations(simulator, spectra, point_names)
output_handler.save_all_results(all_results, point_names)
```

## 🔧 Adding New Sensors

The configuration-driven architecture makes adding sensors simple. Just edit `src/rotina_simulacaobandas_python/config/sensor_config.py`:

```python
SENSOR_CONFIGS = {
    # ... existing sensors ...

    'your_sensor_id': {
        'name': 'Your Sensor Name',
        'file': 'your_srf_file.xlsx',
        'file_type': 'excel',  # or 'csv'
        'sheet': 0,  # for Excel files
        'band_indices': list(range(1, 5)),  # Column indices in SRF file
        'wave_centers': [440, 560, 665, 865],  # Band wavelengths (nm)
        'wavelength_range': (400, 900),  # Optional: filter SRF wavelengths
        'enabled': True
    }
}
```
