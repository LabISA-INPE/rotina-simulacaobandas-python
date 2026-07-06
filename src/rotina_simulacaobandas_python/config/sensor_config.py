"""
Sensor registry — every sensor maps to a parquet SRF and is simulated by the
single resampling engine (``resampling.py``). See that module for the method and
attribution (courtesy of Bruno Rech, rs_tools).

Band centers are determined by each SRF file (Bruno's nominal centers for his
sensors; SRF centroids for the converted LabISA sensors), and the effective set
of output bands depends on the input wavelength range (95% coverage filter).

MODIS is intentionally omitted (not implemented).
"""

SENSOR_CONFIGS = {
    # --- Bruno Rech / rs_tools sensors (NASA-RSR / Gaussian SRFs) ---
    'msi_s2a': {'name': 'MSI (Sentinel-2A)', 'srf': 'SRF_MSI_S2A'},
    'msi_s2b': {'name': 'MSI (Sentinel-2B)', 'srf': 'SRF_MSI_S2B'},
    'olci': {'name': 'OLCI (Sentinel-3A)', 'srf': 'SRF_OLCI_S3A'},
    'olci_s3b': {'name': 'OLCI (Sentinel-3B)', 'srf': 'SRF_OLCI_S3B'},
    'oli': {'name': 'OLI (Landsat-8)', 'srf': 'SRF_OLI_L8'},
    'enmap': {'name': 'HSI (EnMAP) - hyperspectral', 'srf': 'SRF_HSI_ENMAP'},
    'prisma': {'name': 'HYC (PRISMA) - hyperspectral', 'srf': 'SRF_HYC_PRISMA'},
    'hico': {'name': 'HICO (ISS) - hyperspectral', 'srf': 'SRF_HICO_ISS'},
    'pace': {'name': 'OCI (PACE) - hyperspectral', 'srf': 'SRF_OCI_PACE'},

    # --- LabISA sensors (SRFs converted to parquet, centers = SRF centroid) ---
    'etm': {'name': 'ETM+ (Landsat-7)', 'srf': 'SRF_ETM_L7'},
    'tm': {'name': 'TM (Landsat-5)', 'srf': 'SRF_TM_L5'},
    'superdove': {'name': 'SuperDove (Planet)', 'srf': 'SRF_SUPERDOVE'},
    'cbers04_mux': {'name': 'CBERS-04 MUX', 'srf': 'SRF_CBERS04_MUX'},
    'cbers04a_mux': {'name': 'CBERS-04A MUX', 'srf': 'SRF_CBERS04A_MUX'},
    'cbers04a_wpm': {'name': 'CBERS-04A WPM', 'srf': 'SRF_CBERS04A_WPM'},
    'amazonia_ro': {'name': 'Amazonia-1 WFI (right optics)', 'srf': 'SRF_AMAZONIA1_WFI_RO'},
    'amazonia_lo': {'name': 'Amazonia-1 WFI (left optics)', 'srf': 'SRF_AMAZONIA1_WFI_LO'},
}
