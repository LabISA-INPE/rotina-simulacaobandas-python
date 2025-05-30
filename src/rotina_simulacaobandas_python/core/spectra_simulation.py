import pandas as pd
import numpy as np

class SatelliteBandSimulator:
    def __init__(self, data_folder='./data-raw'):
        self.srf_data = {
            's3': pd.read_pickle(f"{data_folder}/s3_srf.pkl"),
            's2a': pd.read_pickle(f"{data_folder}/s2_srf.pkl"),
            's2b': pd.read_pickle(f"{data_folder}/s2b_srf.pkl"),
            'l8': pd.read_pickle(f"{data_folder}/l8_srf.pkl"),
            'l7': pd.read_pickle(f"{data_folder}/l7_srf.pkl"),
            'l5': pd.read_pickle(f"{data_folder}/l5_srf.pkl"),
            'planet': pd.read_pickle(f"{data_folder}/planet_srf.pkl"),
            'modis': pd.read_pickle(f"{data_folder}/modis_srf.pkl")
        }

    def _simulate_bands_direct_optmized(self, spectra, srf_data, band_indices, wave_centers, point_names, wavelength_range=None):
        # Convert to numpy for faster operations
        spectra_values = spectra.values
        spectra_wavelengths = spectra.index.values
        
        # Initialize result array
        n_bands = len(band_indices)
        n_points = len(point_names)
        results = np.zeros(n_bands, n_points)
        
        # Process each band
        for band_idx, srf_col_idx in enumerate(band_indices):
            # Extract SRF for this band
            srf_wavelengths = srf_data.iloc[:, 0].values.astype(int)
            srf_values = srf_data.iloc[:, srf_col_idx].values

            # Apply wavelengths filtering
            if wavelength_range is not None:
                min_wave, max_wave = wavelength_range
                mask = (srf_wavelengths >= min_wave) & (srf_wavelengths <= max_wave)
                srf_wavelengths - srf_wavelengths[mask]
                srf_values = srf_values[mask]

            # Calculate FAC (normalization)
            srf_sum = np.sum(srf_values)
            if srf_sum > 0:
                fac_values = srf_values / srf_sum

                # Find matching wavelengths between spectra and SRF
                # Use numpy searchsorted for fast lookup
                valid_indices = []
                valid_fac = []

                for i, wl in enumerate(srf_wavelengths):
                    spec_idx = np.where(spectra_wavelengths == wl)[0][0]
                    valid_indices.append(spec_idx)
                    valid_fac.append(fac_values[i])
                
                if valid_indices:
                    valid_indices = np.array(valid_indices)
                    valid_fac = np.array(valid_fac)

                    # Vectorized computation across all points
                    selected_spectra = spectra_values[valid_indices, :]  # (valid_wavelengths, points)
                    results[band_idx, :] = np.sum(valid_fac[:, np.newaxis] * selected_spectra, axis=0)

        # Create result DataFrame
        result_df = pd.DataFrame(results.T, columns=[f'Band_{wave}nm' for wave in wave_centers])
        result_df.insert(0, 'Wave', [wave_centers[i] if i < len(wave_centers) else 0 for i in range(len(point_names))])
        result_df.index = point_names
        result_df = result_df.reset_index()
        result_df = result_df.rename(columns={'index': 'Point_ID'})
    
        return result_df            

    def olci(self, spectra, point_names):
        # Use bands 1-19
        band_indices = list(range(1, 20))
        wave_centers = [400, 412, 442, 490, 510, 560, 620, 665, 673, 681, 
                       708, 753, 761, 764, 767, 778, 865, 885, 900]
    
        return self._simulate_band_direct(
            spectra, self.srf_data['s3'], band_indices, wave_centers,
            point_names, wavelength_range=(400, 900)
        )
    
    def msi(self, spectra, point_names):
        # S2A and S2B bands 1-9
        band_indices = list(range(1, 10))
        wave_centers = [440, 490, 560, 665, 705, 740, 783, 842, 865]

        s2a_result = self._simulate_band_direct(
            spectra, self.srf_data['s2a'], band_indices, wave_centers,
            point_names, wavelength_range=(400, 900)
        )

        s2b_result = self._simulate_band_direct(
            spectra, self.srf_data['s2b'], band_indices, wave_centers,
            point_names, wavelength_range=(400, 900)
        )

        return {'s2a': s2a_result, 's2b': s2b_result}
    
    def oli(self, spectra, point_names):
        # OLI bands 1-5
        band_indices = list(range(1, 6))
        wave_centers = [440, 490, 560, 665, 865]

        return self._simulate_band_direct(
            spectra, self.srf_data['l8'], band_indices, wave_centers, 
            point_names
        )
    
    def etm(self, spectra, point_names):
        # ETM bands 1-4
        band_indices = list(range(1, 5))
        wave_centers = [490, 560, 665, 865]

        return self._simulate_band_direct(
            spectra, self.srf_data['l7'], band_indices, wave_centers, 
            point_names
        )

    def tm(self, spectra, point_names):
        # TM bands 1-4
        band_indices = list(range(1, 5))
        wave_centers = [490, 560, 665, 865]

        return self._simulate_band_direct(
            spectra, self.srf_data['l5'], band_indices, wave_centers,
            point_names
        )

    def superdove(self, spectra, point_names):
        # SuperDove bands 1-8
        band_indices = list(range(1, 9))
        wave_centers = [443, 490, 531, 565, 610, 665, 705, 865]

        return self._simulate_band_direct(
            spectra, self.srf_data['planet'], band_indices, wave_centers,
            point_names, wavelength_range=(400, 900)
        )
    
    def modis(self, spectra, point_names):
        band_indices = list(range(1, 17))
        wave_centers = [412, 443, 469, 488, 531, 551, 555, 645, 667, 678, 
                       748, 859, 869, 1240, 1640, 2130] 
        
        return self._simulate_band_direct(
            spectra, self.srf_data['modis'], band_indices, wave_centers, 
            point_names, wavelength_range=(400, 900)
        )
    