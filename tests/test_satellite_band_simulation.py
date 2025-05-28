import pandas as pd
import numpy as np
import pytest 
from src.rotina_simulacaobandas_python.core.spectra_simulation import SatelliteBandSimulator

class TestSatelliteBandSimulator:
    def test_initialization(self, mock_srf_data):
        simulator = SatelliteBandSimulator(data_folder=mock_srf_data)

        # Check that all SRF data is loaded
        assert 's3' in simulator.srf_data
        assert 's2a' in simulator.srf_data
        assert 's2b' in simulator.srf_data
        assert 'l8' in simulator.srf_data
        assert 'l7' in simulator.srf_data
        assert 'l5' in simulator.srf_data
        assert 'planet' in simulator.srf_data
        assert 'modis' in simulator.srf_data

        # Check data type
        for key, data in simulator.srf_data.items():
            assert isinstance(data, pd.DataFrame)
            assert len(data) > 0
    
    def test_etm_simulation(self, mock_srf_data, sample_spectra, sample_point_names):
        simulator = SatelliteBandSimulator(data_folder=mock_srf_data)

        result = simulator.etm(sample_spectra, sample_point_names)

        # Check output structure
        assert isinstance(result, pd.DataFrame)
        assert 'Wave' in result.columns
        assert len(result) == 4
        assert list(result['Wave']) == [490, 560, 665, 865]

        # Check that all point names are present
        for name in sample_point_names:
            assert name in result.columns
        
        # Check that values are numeric and positive
        for col in sample_point_names:
            assert result[col].dtype in [np.float64, np.float32]
            assert all(result[col] >= 0)
        
    def test_msi_simulation(self, mock_srf_data, sample_spectra, sample_point_names):
        simulator = SatelliteBandSimulator(data_folder=mock_srf_data)

        result = simulator.msi(sample_spectra, sample_point_names)

        assert isinstance(result, dict)
        assert 's2a' in result
        assert 's2b' in result

        for sensor in ['s2a', 's2b']:
            df = result[sensor]
            assert isinstance(df, pd.DataFrame)
            assert 'Wave' in df.columns
            assert len(df) == 9
            assert list(df['Wave']) == [440, 490, 560, 665, 705, 740, 783, 842, 865]

            # Check point names
            for name in sample_point_names:
                assert name in df.columns
    
    def test_olci_simulation(self, mock_srf_data, sample_spectra, sample_point_names):
        simulator = SatelliteBandSimulator(data_folder=mock_srf_data)

        result = simulator.olci(sample_spectra, sample_point_names)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 19

        expected_waves = [400, 412, 442, 490, 510, 560, 620, 665, 673, 681, 
                         708, 753, 761, 764, 767, 778, 865, 885, 900]
        
        assert list(result['Wave']) == expected_waves

        # OLCI should have good coverage across 400-900nm
        assert min(result['Wave']) == 400
        assert max(result['Wave']) == 900

    def test_oli_simulation(self, mock_srf_data, sample_spectra, sample_point_names):
        simulator = SatelliteBandSimulator(data_folder=mock_srf_data)

        result = simulator.oli(sample_spectra, sample_point_names)

        # OLI-specific tests
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 5
        assert list(result['Wave']) == [440, 490, 560, 665, 865]
    
    def test_tm_simulation(self, mock_srf_data, sample_spectra, sample_point_names):
        simulator = SatelliteBandSimulator(data_folder=mock_srf_data)

        result = simulator.tm(sample_spectra, sample_point_names)

        # TM-specific tests
        assert isinstance(result, pd.DataFrame)
        assert len(result) ==   4
        assert list(result['Wave']) == [490, 560, 665, 865]

    def test_superdove_simulation(self, mock_srf_data, sample_spectra, sample_point_names):
        simulator = SatelliteBandSimulator(data_folder=mock_srf_data)

        result = simulator.superdove(sample_spectra, sample_point_names)

        # SuperDove-specific tests
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 8
        assert list(result['Wave']) == [443, 490, 531, 565, 610, 665, 705, 865]

    def test_modis_simulation(self, mock_srf_data, sample_spectra, sample_point_names):
        simulator = SatelliteBandSimulator(data_folder=mock_srf_data)

        result = simulator.modis(sample_spectra, sample_point_names)
        
        # MODIS-specific tests
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 16 
        expected_waves = [412, 443, 469, 488, 531, 551, 555, 645, 667, 678, 
                         748, 859, 869, 1240, 1640, 2130]
        assert list(result['Wave']) == expected_waves

    def test_missing_wavelengths_coverage(self, mock_srf_data, sample_point_names):
        """Test handling of spectra with missing wavelengths to cover the else branch"""
        simulator = SatelliteBandSimulator(data_folder=mock_srf_data)
        
        # Create spectra with LIMITED wavelength range (missing some wavelengths)
        limited_wavelengths = list(range(450, 851))  # Missing 400-449 and 851-900
        n_stations = len(sample_point_names)
        
        # Create limited spectral data
        limited_spectra_data = []
        for i in range(n_stations):
            # Simple linear spectral data
            spectrum = np.linspace(0.01, 0.05, len(limited_wavelengths))
            limited_spectra_data.append(spectrum)
        
        # Create DataFrame with limited wavelength range
        limited_spectra = pd.DataFrame(
            np.array(limited_spectra_data).T,
            index=limited_wavelengths
        )
        
        # Run ETM simulation - this should trigger the missing wavelength handling
        result = simulator.etm(limited_spectra, sample_point_names)
        
        # Verify the simulation still works despite missing wavelengths
        assert isinstance(result, pd.DataFrame)
        assert 'Wave' in result.columns
        assert len(result) == 4  # ETM has 4 bands
        assert list(result['Wave']) == [490, 560, 665, 865]
        
        # Check that results are reasonable (not all zeros)
        for name in sample_point_names:
            assert name in result.columns
            assert all(result[name] >= 0)
            assert any(result[name] > 0)


    