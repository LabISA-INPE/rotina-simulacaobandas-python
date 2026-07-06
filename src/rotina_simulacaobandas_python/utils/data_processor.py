import pandas as pd
from rotina_simulacaobandas_python.config.sensor_config import SENSOR_CONFIGS


class DataProcessor:
    def __init__(self):
        """Initialize DataProcessor with sensor configurations from config."""
        self.available_sensors = {
            sensor_id: {'name': config['name']}
            for sensor_id, config in SENSOR_CONFIGS.items()
        }

    def process_spectra(self, data):
        """
        Process input spectra data from GLORIA format.

        Args:
            data: DataFrame with GLORIA_ID and Rrs_<wavelength> columns

        Returns:
            Tuple of (spectra DataFrame, point_names list)
        """
        # Extract point names
        point_names = data['GLORIA_ID'].tolist()

        # Get available Rrs columns
        available_rrs = [col for col in data.columns if col.startswith("Rrs_")]

        if not available_rrs:
            raise ValueError("No Rrs columns found in data")

        # Extract wavelengths from available columns
        wl_pattern = "Rrs_"
        available_wavelengths = sorted([int(col[len(wl_pattern):]) for col in available_rrs])

        # Use only wavelengths in 400-900 range
        wavelengths = [wl for wl in available_wavelengths if 400 <= wl <= 900]
        rrs_columns = [f"Rrs_{wl}" for wl in wavelengths]

        # Select Rrs columns and transpose
        rrs_data = data[rrs_columns].T

        # Create spectra DataFrame with wavelengths as index
        spectra = pd.DataFrame(rrs_data.values, index=wavelengths)

        # Clean data
        spectra = self._clean_spectra_data(spectra)

        # Print info about the dataset
        current_stations = len(point_names)
        print(f"Total stations in GLORIA dataset: {current_stations}")
        print(f"Using all {current_stations} stations from dataset")

        return spectra, point_names

    def _clean_spectra_data(self, spectra):
        """
        Clean spectra data by handling negative values and NaN.

        Args:
            spectra: DataFrame with spectra values

        Returns:
            Cleaned spectra DataFrame
        """
        # Replace negative values with 0
        negative_mask = spectra < 0
        spectra[negative_mask] = 0.0

        # Handle NaN values
        nan_mask = spectra.isna()
        nan_count = nan_mask.sum().sum()

        if nan_count > 0:
            spectra = spectra.fillna(0.0)
            print(f"Filled {nan_count} NaN values with 0.0")

        return spectra

    def run_sensor_simulation(self, simulator, spectra, point_names, sensor_id, variant=None):
        """Run simulation for a single sensor.

        Returns ``{sensor_id: DataFrame}``. ``variant`` is deprecated (MSI
        variants are now their own sensor ids) and ignored.
        """
        if sensor_id not in self.available_sensors:
            raise ValueError(
                f"Unknown sensor: {sensor_id}. "
                f"Available sensors: {list(self.available_sensors.keys())}"
            )
        try:
            result = simulator.simulate(sensor_id, spectra, point_names)
        except Exception as e:
            raise RuntimeError(f"Error in {sensor_id} simulation: {e}") from e
        return {sensor_id: result}

    def run_multiple_sensors(self, simulator, spectra, point_names, sensor_ids):
        """Run simulations for a list of sensor ids."""
        results = {}
        for sensor_id in sensor_ids:
            try:
                results.update(
                    self.run_sensor_simulation(simulator, spectra, point_names, sensor_id)
                )
                print(f"✓ Completed simulation for {sensor_id}")
            except Exception as e:
                print(f"✗ Failed simulation for {sensor_id}: {e}")
        return results

    def run_all_simulations(self, simulator, spectra, point_names):
        """Run simulations for all available sensors."""
        return self.run_multiple_sensors(
            simulator, spectra, point_names, list(self.available_sensors)
        )
