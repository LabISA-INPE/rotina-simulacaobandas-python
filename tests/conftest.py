import pytest
import pandas as pd
import numpy as np

@pytest.fixture
def sample_spectra():
    wavelengths = list(range(400, 901))
    n_stations = 3

    # Create realistic spectral data
    spectra_data = []
    for i in range(n_stations):
        # Simulate water reflectance with some noise
        base_reflectance  = np.exp(-(np.array(wavelengths) - 550)**2 / (2 * 50**2)) * 0.05
        noise = np.random.normal(0, 0.001, len(wavelengths))
        spectrum = np.maximum(base_reflectance + noise, 0)
        spectra_data.append(spectrum)

    # Create DataFrame
    spectra_df = pd.DataFrame(
        np.array(spectra_data).T,
        index=wavelengths
    )

    return spectra_df

@pytest.fixture
def limited_spectra():
    wavelengths = list(range(450, 851))
    n_stations = 3
    
    spectra_data = []
    for i in range(n_stations):
        spectrum = np.exp(-(np.array(wavelengths) - 600)**2 / (2 * 100**2)) * 0.03
        spectra_data.append(spectrum)
    
    spectra_df = pd.DataFrame(
        np.array(spectra_data).T,
        index=wavelengths
    )

    return spectra_df

@pytest.fixture
def sample_point_names():
    return ['GID_1', 'GID_2', 'GID_3']

@pytest.fixture
def mock_srf_data(tmp_path):
    srf_dir = tmp_path / 'data-raw'
    srf_dir.mkdir()

    # Create mock SRF data for each sensor
    sensor = {        
        's3_srf.pkl': 22,    
        's2_srf.pkl': 10,    
        's2b_srf.pkl': 10,    
        'l8_srf.pkl': 6,       
        'l7_srf.pkl': 5,       
        'l5_srf.pkl': 5,      
        'planet_srf.pkl': 9,   
        'modis_srf.pkl': 17    
    }

    for filename, n_cols in sensor.items():
        # Craete wavelengths columns
        wavelengths = list(range(400, 901))

        if 'modis' in filename:
            srf_data = {'wavelength': wavelengths}
        else:
            srf_data = {'Wavelength': wavelengths}

        for i in range(1, n_cols):
            # Create realistic SRF curves 
            center = 400 + (500 / max(1, n_cols-2)) * (i-1)  # Distribute across spectrum
            srf_values = np.exp(-(np.array(wavelengths) - center)**2 / (2 * 30**2))
            
            if 's3' in filename:
                srf_data[f'B{i}'] = srf_values
            elif 's2' in filename:
                if 's2b' in filename:
                    srf_data[f'S2B_SR_AV_B{i}'] = srf_values
                else:
                    srf_data[f'S2A_SR_AV_B{i}'] = srf_values
            elif 'l8' in filename:
                srf_data[f'OB{i}'] = srf_values
            elif 'l7' in filename:
                srf_data[f'ETMB{i}'] = srf_values
            elif 'l5' in filename:
                srf_data[f'TMB{i}'] = srf_values
            elif 'planet' in filename:
                srf_data[f'Band_{i}'] = srf_values
            elif 'modis' in filename:
                    modis_bands = ['RSR_412', 'RSR_443', 'RSR_469', 'RSR_488', 'RSR_531', 
                                'RSR_551', 'RSR_555', 'RSR_645', 'RSR_667', 'RSR_678', 
                                'RSR_748', 'RSR_859', 'RSR_869', 'RSR_1240', 'RSR_1640', 'RSR_2130']
                    
                    if i-1 < len(modis_bands):
                        srf_data[modis_bands[i-1]] = srf_values
        
        df = pd.DataFrame(srf_data)
        df.to_pickle(srf_dir / filename)

    return str(srf_dir)
