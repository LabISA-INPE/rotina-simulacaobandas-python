import pandas as pd
import numpy as np
from rotina_simulacaobandas_python.config.sensor_config import SENSOR_CONFIGS


class SatelliteBandSimulator:
    def __init__(self, srf_folder='../SRF'):
        """
        Initialize the satellite band simulator.

        Args:
            srf_folder: Path to folder containing SRF (Spectral Response Function) files
        """
        self.srf_folder = srf_folder
        self.srf_data = {}
        self._load_all_srf()

    def _load_all_srf(self):
        """Load all SRF files based on sensor configuration."""
        for sensor_id, config in SENSOR_CONFIGS.items():
            # Skip disabled sensors
            if config.get('enabled', True) is False:
                continue

            try:
                if 'variants' in config:
                    # Load variants (e.g., S2A and S2B)
                    for variant_id, variant_config in config['variants'].items():
                        key = f"{sensor_id}_{variant_id}"
                        self.srf_data[key] = self._load_srf_file(config, variant_config)
                else:
                    # Load single sensor
                    self.srf_data[sensor_id] = self._load_srf_file(config)
            except FileNotFoundError as e:
                print(f"Warning: Could not load SRF file for {sensor_id}: {e}")
            except Exception as e:
                print(f"Error loading {sensor_id}: {e}")

    def _load_srf_file(self, config, variant_config=None):
        """
        Load a single SRF file.

        Args:
            config: Sensor configuration dict
            variant_config: Optional variant-specific configuration

        Returns:
            DataFrame containing SRF data
        """
        filepath = f"{self.srf_folder}/{config['file']}"

        if config['file_type'] == 'excel':
            sheet = variant_config.get('sheet') if variant_config else config.get('sheet', 0)
            return pd.read_excel(filepath, sheet_name=sheet)
        elif config['file_type'] == 'csv':
            return pd.read_csv(filepath)
        else:
            raise ValueError(f"Unsupported file type: {config['file_type']}")

    def _simulate_bands_direct_optimized(self, spectra, srf_data, band_indices, wave_centers, point_names, wavelength_range=None):
        """
        Core band simulation algorithm using optimized numpy operations.

        Args:
            spectra: DataFrame with wavelengths as index and spectra values as columns
            srf_data: DataFrame containing Spectral Response Function data
            band_indices: List of band column indices in SRF data
            wave_centers: List of wavelength centers for each band
            point_names: List of point/station names
            wavelength_range: Optional tuple (min_wavelength, max_wavelength) to filter

        Returns:
            DataFrame with simulated band values

        The simulated value for a band is the SRF-weighted average of the input
        spectrum: ``FAC = SRF / sum(SRF)`` then ``band = sum(FAC * Rrs)``. This
        matches the original R reference (dmaciel123/BandSimulation) exactly.
        Implemented as a single matrix product ``W @ spectra`` (W = per-band
        normalized weights aligned to the spectrum's wavelength grid).
        """
        # Spectrum grid and cleaned values (drop NaN -> 0, clip negatives -> 0).
        grid = spectra.index.values.astype(float)
        spectra_values = np.nan_to_num(np.maximum(spectra.values.astype(float), 0.0), nan=0.0)

        n_bands = len(band_indices)
        n_wavelengths = len(grid)

        # Build the weight matrix: one normalized SRF per band, resampled onto the
        # spectrum grid. Bands with no usable SRF stay all-zero (-> 0 output).
        weights = np.zeros((n_bands, n_wavelengths))
        srf_wavelengths_raw = srf_data.iloc[:, 0].values

        for band_idx, srf_col_idx in enumerate(band_indices):
            srf_values_raw = srf_data.iloc[:, srf_col_idx].values

            valid = ~(pd.isna(srf_wavelengths_raw) | pd.isna(srf_values_raw))
            srf_wl = srf_wavelengths_raw[valid].astype(float)
            srf_v = srf_values_raw[valid].astype(float)

            if wavelength_range is not None:
                min_wave, max_wave = wavelength_range
                in_range = (srf_wl >= min_wave) & (srf_wl <= max_wave)
                srf_wl, srf_v = srf_wl[in_range], srf_v[in_range]

            if srf_wl.size == 0:
                continue

            # Resample the SRF onto the spectrum grid (0 outside its support).
            # np.interp needs ascending x.
            order = np.argsort(srf_wl)
            resampled = np.interp(grid, srf_wl[order], srf_v[order], left=0.0, right=0.0)

            total = resampled.sum()
            if total > 0:
                weights[band_idx] = resampled / total

        # bands (n_bands x n_points) = weights (n_bands x n_wl) @ spectra (n_wl x n_points)
        results = weights @ spectra_values

        band_names = [f'Band_{wave}nm' for wave in wave_centers]
        return pd.DataFrame(results.T, columns=band_names, index=point_names)

    def simulate(self, sensor_id, spectra, point_names, variant=None):
        """
        Generic simulation method for any configured sensor.

        Args:
            sensor_id: Sensor identifier (e.g., 'oli', 'msi', 'olci')
            spectra: DataFrame with wavelengths as index and spectra values as columns
            point_names: List of point/station names
            variant: Optional variant identifier (e.g., 's2a', 's2b' for MSI)

        Returns:
            DataFrame with simulated band values or dict of DataFrames for multi-variant sensors
        """
        if sensor_id not in SENSOR_CONFIGS:
            raise ValueError(f"Unknown sensor: {sensor_id}")

        config = SENSOR_CONFIGS[sensor_id]

        # Check if sensor is disabled
        if config.get('enabled', True) is False:
            raise ValueError(f"Sensor {sensor_id} is disabled. Check configuration.")

        # Handle sensors with variants
        if 'variants' in config:
            if variant:
                # Simulate specific variant
                srf_key = f"{sensor_id}_{variant}"
                if srf_key not in self.srf_data:
                    raise ValueError(f"Variant {variant} not available for {sensor_id}")

                return self._simulate_bands_direct_optimized(
                    spectra,
                    self.srf_data[srf_key],
                    config['band_indices'],
                    config['wave_centers'],
                    point_names,
                    config.get('wavelength_range')
                )
            else:
                # Simulate all variants
                results = {}
                for variant_id in config['variants'].keys():
                    srf_key = f"{sensor_id}_{variant_id}"
                    results[variant_id] = self._simulate_bands_direct_optimized(
                        spectra,
                        self.srf_data[srf_key],
                        config['band_indices'],
                        config['wave_centers'],
                        point_names,
                        config.get('wavelength_range')
                    )
                return results
        else:
            # Single sensor simulation
            return self._simulate_bands_direct_optimized(
                spectra,
                self.srf_data[sensor_id],
                config['band_indices'],
                config['wave_centers'],
                point_names,
                config.get('wavelength_range')
            )

    # Legacy methods for backward compatibility
    def olci(self, spectra, point_names):
        """OLCI (Sentinel-3) simulation - legacy method."""
        return self.simulate('olci', spectra, point_names)

    def msi(self, spectra, point_names):
        """MSI (Sentinel-2) simulation - legacy method."""
        return self.simulate('msi', spectra, point_names)

    def oli(self, spectra, point_names):
        """OLI (Landsat-8/9) simulation - legacy method."""
        return self.simulate('oli', spectra, point_names)

    def etm(self, spectra, point_names):
        """ETM+ (Landsat-7) simulation - legacy method."""
        return self.simulate('etm', spectra, point_names)

    def tm(self, spectra, point_names):
        """TM (Landsat-5) simulation - legacy method."""
        return self.simulate('tm', spectra, point_names)

    def superdove(self, spectra, point_names):
        """SuperDove (Planet) simulation - legacy method."""
        return self.simulate('superdove', spectra, point_names)

    def modis(self, spectra, point_names):
        """MODIS simulation - legacy method."""
        return self.simulate('modis', spectra, point_names)

    def cbers04_mux(self, spectra, point_names):
        """CBERS-04 MUX simulation - legacy method."""
        return self.simulate('cbers04_mux', spectra, point_names)

    def cbers04a_mux(self, spectra, point_names):
        """CBERS-04A MUX simulation - legacy method."""
        return self.simulate('cbers04a_mux', spectra, point_names)

    def cbers04a_wpm(self, spectra, point_names):
        """CBERS-04A WPM simulation - legacy method."""
        return self.simulate('cbers04a_wpm', spectra, point_names)

    def amazonia1_wfi(self, spectra, point_names):
        """Amazonia-1 WFI simulation - legacy method."""
        return self.simulate('amazonia1_wfi', spectra, point_names)
