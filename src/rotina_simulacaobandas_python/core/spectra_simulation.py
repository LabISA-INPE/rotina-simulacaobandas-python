"""Satellite band simulator.

Thin wrapper over the shared resampling engine (``resampling.py``). Every sensor
is simulated the same way, from a parquet SRF (see ``config.sensor_config``).
"""

from rotina_simulacaobandas_python import resampling
from rotina_simulacaobandas_python.config.sensor_config import SENSOR_CONFIGS


class SatelliteBandSimulator:
    def __init__(self, srf_folder=None):
        # srf_folder is accepted for backward compatibility but no longer used:
        # SRFs are parquet files bundled in the package (loaded on demand).
        self.srf_folder = srf_folder

    def simulate(self, sensor_id, spectra, point_names, variant=None):
        """Simulate a sensor.

        Args:
            sensor_id: sensor identifier (see ``SENSOR_CONFIGS``).
            spectra: DataFrame with wavelengths as index and one column per station.
            point_names: station names for the result index.
            variant: deprecated/ignored (MSI variants are now their own sensor ids).

        Returns:
            DataFrame indexed by point_names with one ``Band_<center>nm`` column
            per simulated band.
        """
        if sensor_id not in SENSOR_CONFIGS:
            raise ValueError(f"Unknown sensor: {sensor_id}")
        return resampling.simulate(spectra, point_names, SENSOR_CONFIGS[sensor_id]['srf'])
