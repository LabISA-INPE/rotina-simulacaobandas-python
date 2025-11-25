import pandas as pd
import numpy as np
from rotina_simulacaobandas_python.config.sensor_config import SENSOR_CONFIGS


class DataProcessor:
    def __init__(self):
        """Initialize DataProcessor with sensor configurations from config."""
        # Build available sensors from config
        self.available_sensors = {}
        for sensor_id, config in SENSOR_CONFIGS.items():
            # Skip disabled sensors
            if config.get('enabled', True) is False:
                continue

            self.available_sensors[sensor_id] = {
                'name': config['name'],
                'variants': list(config['variants'].keys()) if 'variants' in config else None,
                'method': sensor_id  # Method name matches sensor_id
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
        """
        Run simulation for a single sensor.

        Args:
            simulator: SatelliteBandSimulator instance
            spectra: Spectra DataFrame
            point_names: List of point/station names
            sensor_id: Sensor identifier
            variant: Optional variant identifier

        Returns:
            Dictionary with sensor results
        """
        if sensor_id not in self.available_sensors:
            raise ValueError(f"Unknown sensor: {sensor_id}. Available sensors: {list(self.available_sensors.keys())}")

        sensor_config = self.available_sensors[sensor_id]
        method_name = sensor_config['method']

        try:
            # Get the simulation method from the simulator
            sim_method = getattr(simulator, method_name)
            result = sim_method(spectra, point_names)

            # Handle sensors with variants (like MSI)
            if sensor_config['variants'] and isinstance(result, dict):
                if variant:
                    if variant not in result:
                        raise ValueError(f"Variant {variant} not available for {sensor_id}. Available variants: {list(result.keys())}")
                    return {f"{sensor_id}_{variant}": result[variant]}
                else:
                    # Return all variants with proper naming
                    return {f"{sensor_id}_{var}": data for var, data in result.items()}
            else:
                # Single result sensor
                return {sensor_id: result}

        except AttributeError:
            raise ValueError(f"Simulation method '{method_name}' not found in simulator")
        except Exception as e:
            raise RuntimeError(f"Error in {sensor_id} simulation: {str(e)}")

    def run_multiple_sensors(self, simulator, spectra, point_names, sensor_requests):
        """
        Run simulations for multiple sensors.

        Args:
            simulator: SatelliteBandSimulator instance
            spectra: Spectra DataFrame
            point_names: List of point/station names
            sensor_requests: List of sensor IDs or dicts with 'sensor' and 'variant' keys

        Returns:
            Dictionary with all simulation results
        """
        simulation_results = {}

        for request in sensor_requests:
            if isinstance(request, str):
                sensor_id = request
                variant = None
            elif isinstance(request, dict):
                sensor_id = request.get('sensor')
                variant = request.get('variant')
            else:
                print(f"Warning: Invalid sensor request format: {request}")
                continue

            try:
                result = self.run_sensor_simulation(simulator, spectra, point_names, sensor_id, variant)
                simulation_results.update(result)
                print(f"✓ Completed simulation for {sensor_id}" + (f" ({variant})" if variant else ""))
            except Exception as e:
                print(f"✗ Failed simulation for {sensor_id}: {e}")

        return simulation_results

    def run_all_simulations(self, simulator, spectra, point_names):
        """
        Run simulations for all available sensors.

        Args:
            simulator: SatelliteBandSimulator instance
            spectra: Spectra DataFrame
            point_names: List of point/station names

        Returns:
            Dictionary with all simulation results
        """
        all_sensors = []
        for sensor_id, config in self.available_sensors.items():
            if config['variants']:
                # Add each variant as a separate request
                for variant in config['variants']:
                    all_sensors.append({'sensor': sensor_id, 'variant': variant})
            else:
                all_sensors.append(sensor_id)

        return self.run_multiple_sensors(simulator, spectra, point_names, all_sensors)
