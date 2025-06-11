import pandas as pd
import numpy as np

class DataProcessor:
    def process_spectra(self, data, target_stations=1000):
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
        
        # Extend data if needed
        point_names, spectra = self._extend_data_if_needed(point_names, spectra, target_stations)
                
        return spectra, point_names
    
    def _clean_spectra_data(self, spectra):
        # Replace negative values with 0
        negative_mask = spectra < 0
        spectra[negative_mask] = 0.0
        
        # Handle NaN values
        nan_mask = spectra.isna()
        nan_count = nan_mask.sum().sum()
        
        if nan_count > 0:
            spectra = spectra.fillna(0.0)
        
        return spectra
    
    def _extend_data_if_needed(self, point_names, spectra, target_stations):
        current_stations = len(point_names)
        print(f"Total stations in GLORIA dataset: {current_stations}")
        
        if current_stations < target_stations:
            # Extend point_names
            for i in range(current_stations, target_stations):
                point_names.append(f"PLACEHOLDER_STATION_{i+1}")
            
            # Extend spectra with duplicated real data
            additional_cols = target_stations - current_stations
            
            if current_stations > 0:
                # Repeat existing data cyclically
                repeat_indices = np.tile(np.arange(current_stations), 
                                       (additional_cols // current_stations) + 1)[:additional_cols]
                additional_data = spectra.iloc[:, repeat_indices].values
                
                additional_df = pd.DataFrame(additional_data, 
                                           index=spectra.index,
                                           columns=range(current_stations, target_stations))
                spectra = pd.concat([spectra, additional_df], axis=1)
                
                print(f"Extended spectra with duplicated real data")
            else:
                print("Warning: No original data to duplicate")
        
        elif current_stations >= target_stations:
            print(f"Using all {current_stations} stations from GLORIA dataset")
        
        return point_names, spectra
    
    def run_all_simulations(self, simulator, spectra, point_names):
        simulation_results = {}
        
        simulations = [
            ('MSI', lambda: simulator.msi(spectra, point_names)),
            ('OLI', lambda: simulator.oli(spectra, point_names)),
            ('ETM', lambda: simulator.etm(spectra, point_names)),
            ('TM', lambda: simulator.tm(spectra, point_names)),
            ('OLCI', lambda: simulator.olci(spectra, point_names)),
            ('SuperDove', lambda: simulator.superdove(spectra, point_names)),
            ('MODIS', lambda: simulator.modis(spectra, point_names))
        ]
        
        for sim_name, sim_func in simulations:
            try:
                result = sim_func()
                
                if sim_name == 'MSI':
                    simulation_results['msi_s2a'] = result['s2a']
                    simulation_results['msi_s2b'] = result['s2b']
                else:
                    simulation_results[sim_name.lower()] = result
                    
            except Exception as e:
                print(f"Error in {sim_name} simulation: {e}")
        
        return simulation_results