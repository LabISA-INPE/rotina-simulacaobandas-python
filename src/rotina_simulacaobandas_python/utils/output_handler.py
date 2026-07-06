import pandas as pd
import os

class OutputHandler:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self._create_output_directory()
    
    def _create_output_directory(self):
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
    
    def convert_to_wave_format(self, df, point_names, sensor_name="", target_gid_count=None):
        df_copy = df.copy()
        
        # Remove the 'Wave' column if it exists
        if 'Wave' in df_copy.columns:
            df_copy = df_copy.drop('Wave', axis=1)
        
        # Get band columns (should be 'Band_XXXnm' format)
        data_columns = [col for col in df_copy.columns if col.startswith('Band_')]
        
        if not data_columns:
            return pd.DataFrame()
        
        # Extract wave centers from column names
        # Keep both numeric and string wavelength identifiers
        wave_centers = []
        valid_columns = []
        for col in data_columns:
            wave_str = col.replace('Band_', '').replace('nm', '')
            try:
                # Try to parse as integer
                wave_centers.append(int(wave_str))
                valid_columns.append(col)
            except ValueError:
                # If not numeric, keep the string identifier (e.g., 'PAN', 'RO_490')
                wave_centers.append(wave_str)
                valid_columns.append(col)

        # Transpose the data so bands become rows and points become columns
        # Use only valid columns that have corresponding wave_centers
        df_transposed = df_copy[valid_columns].T
        
        # Number of real stations available. `target_gid_count` may cap it lower,
        # but we NEVER fabricate columns beyond the real data (the old code cycled
        # through existing stations to pad up to target, inventing fake stations).
        available = df_transposed.shape[1]
        if target_gid_count is None:
            gid_count = available
        else:
            gid_count = min(available, target_gid_count)

        # Create the result dictionary (one GID column per real station).
        result_data = {'Wave': wave_centers}
        for i in range(gid_count):
            result_data[f'GID_{i+1}'] = df_transposed.iloc[:, i].values

        # Create DataFrame
        result_df = pd.DataFrame(result_data)
        result_df.index = range(1, len(result_df) + 1)
        
        return result_df
    
    def save_all_results(self, simulation_results, point_names, target_gid_count=1000):
        for sensor_name, result_df in simulation_results.items():
            try:
                converted_df = self.convert_to_wave_format(
                    result_df, point_names, sensor_name, target_gid_count
                )
                if not converted_df.empty:
                    output_path = f"{self.output_dir}/{sensor_name}_simulation.csv"
                    converted_df.to_csv(output_path, index=False)
                else:
                    print(f"Warning: {sensor_name} results are empty")
            except Exception as e:
                print(f"Error saving {sensor_name} results: {e}")