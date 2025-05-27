import pandas as pd
import numpy as np

class SateliteBandSimulator:
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

    def _simulate_band_direct(self, spectra, srf_data, band_indices, wave_centers, point_names, wavelength_range=None):
        # Create DataFrame with wavelengths 
        wavelengths = list(range(400, 901))
        espec = pd.DataFrame(index=wavelengths)

        # Add spectra columns
        for i, col_name in enumerate(point_names):
            espec[col_name] = spectra.iloc[:, i].values

        # Initialize result dataframe
        n_bands = len(band_indices)
        result_df = pd.DataFrame(index=range(n_bands))

        # Process each band
        for band_idx, srf_col_idx in enumerate(band_indices):
            # Extract SRF for this band
            srf_band = srf_data.iloc[:, [0, srf_col_idx]].copy()

            # Apply wavelength filtering if specified
            if wavelength_range is not None:
                min_wave, max_wave = wavelength_range
                srf_band = srf_band[(srf_band.iloc[:, 0] >= min_wave) & (srf_band.iloc[:, 0] <= max_wave)]

            # Calculate FAC (correction factor)
            srf_values = srf_band.iloc[:, 1]
            fac_values = srf_values / srf_values.sum()

            # For each station/point
            for col_name in point_names:
                # Get spectra values for wavelengths that match SRF wavelengths
                srf_wavelengths = srf_band.iloc[:, 0].values.astype(int)

                # Extract corresponding spectra values
                spectra_values = []
                for wl in srf_wavelengths:
                    if wl in espec.index:
                        spectra_values.append(espec.loc[wl ,col_name])
                    else:
                        spectra_values.append(0)
                
                spectra_values = np.array(spectra_values)

                # Calculaate weighted sum
                band_value = np.nansum(fac_values.values * spectra_values)
                result_df.loc[band_idx, col_name] = band_value
        
        # Add wavelengths centers
        result_df.insert(0, 'Wave', wave_centers)

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
    