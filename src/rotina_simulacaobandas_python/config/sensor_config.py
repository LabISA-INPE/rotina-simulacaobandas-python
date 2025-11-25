"""
Sensor configuration for all supported satellite sensors.

Each sensor configuration includes:
- name: Full sensor name
- file: SRF data file name
- file_type: 'excel' or 'csv'
- sheet: Excel sheet name or index (for Excel files)
- band_indices: List of band column indices in the SRF file
- wave_centers: Wavelength centers for each band (nm)
- wavelength_range: Optional tuple (min, max) to filter SRF wavelengths
- variants: Optional dict of sensor variants (e.g., S2A/S2B)
"""

SENSOR_CONFIGS = {
    'olci': {
        'name': 'OLCI (Sentinel-3)',
        'file': 'olci_FRE.xlsx',
        'file_type': 'excel',
        'sheet': 0,
        'band_indices': list(range(1, 20)),
        'wave_centers': [400, 412, 442, 490, 510, 560, 620, 665, 673, 681,
                        708, 753, 761, 764, 767, 778, 865, 885, 900],
        'wavelength_range': (400, 900)
    },

    'msi': {
        'name': 'MSI (Sentinel-2)',
        'file': 'Spectral Response - Sentinel 2.xlsx',
        'file_type': 'excel',
        'variants': {
            's2a': {'sheet': 'Spectral Responses (S2A)'},
            's2b': {'sheet': 'Spectral Responses (S2B)'}
        },
        'band_indices': list(range(1, 10)),
        'wave_centers': [440, 490, 560, 665, 705, 740, 783, 842, 865],
        'wavelength_range': (400, 900)
    },

    'oli': {
        'name': 'OLI (Landsat-8/9)',
        'file': 'oli_SRF.xlsx',
        'file_type': 'excel',
        'sheet': 0,
        'band_indices': list(range(1, 6)),
        'wave_centers': [440, 490, 560, 665, 865]
    },

    'etm': {
        'name': 'ETM+ (Landsat-7)',
        'file': 'L7_RSR_Ok.xlsx',
        'file_type': 'excel',
        'sheet': 0,
        'band_indices': list(range(1, 5)),
        'wave_centers': [490, 560, 665, 865]
    },

    'tm': {
        'name': 'TM (Landsat-5)',
        'file': 'L5_RSR.xlsx',
        'file_type': 'excel',
        'sheet': 0,
        'band_indices': list(range(1, 5)),
        'wave_centers': [490, 560, 665, 865]
    },

    'superdove': {
        'name': 'SuperDove (Planet)',
        'file': 'Superdove.csv',
        'file_type': 'csv',
        'band_indices': list(range(1, 9)),
        'wave_centers': [443, 490, 531, 565, 610, 665, 705, 865],
        'wavelength_range': (400, 900)
    },

    'modis': {
        'name': 'MODIS (Aqua/Terra)',
        'file': 'modis_srf.xlsx',  # Note: This file needs to be created
        'file_type': 'excel',
        'sheet': 0,
        'band_indices': list(range(1, 17)),
        'wave_centers': [412, 443, 469, 488, 531, 551, 555, 645, 667, 678,
                        748, 859, 869, 1240, 1640, 2130],
        'wavelength_range': (400, 900),
        'enabled': False  # Disabled until file is created
    },

    'cbers04_mux': {
        'name': 'CBERS-04 MUX',
        'file': 'CBERS_04_MUX.xlsx',
        'file_type': 'excel',
        'sheet': 0,
        'band_indices': list(range(1, 5)),
        'wave_centers': [490, 560, 665, 865]
    },

    'cbers04a_mux': {
        'name': 'CBERS-04A MUX',
        'file': 'cbers_04a_mux.xlsx',
        'file_type': 'excel',
        'sheet': 0,
        'band_indices': list(range(1, 5)),
        'wave_centers': [490, 560, 665, 865]
    },

    'cbers04a_wpm': {
        'name': 'CBERS-04A WPM',
        'file': 'CBERS_04A_WPM.xlsx',
        'file_type': 'excel',
        'sheet': 0,
        'band_indices': list(range(1, 6)),
        'wave_centers': ['PAN', 490, 560, 665, 865],
        'wavelength_range': (400, 900)
    },

    'amazonia1_wfi': {
        'name': 'Amazonia-1 WFI',
        'file': 'amazonia_1.xlsx',
        'file_type': 'excel',
        'sheet': 0,
        'band_indices': list(range(1, 9)),
        'wave_centers': ['RO_490', 'RO_560', 'RO_665', 'RO_865',
                        'LO_490', 'LO_560', 'LO_665', 'LO_865'],
        'wavelength_range': (400, 900)
    }
}
