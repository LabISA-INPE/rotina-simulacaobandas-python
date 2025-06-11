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
        wave_centers = []
        for col in data_columns:
            wave_str = col.replace('Band_', '').replace('nm', '')
            try:
                wave_centers.append(int(wave_str))
            except ValueError:
                print(f"Warning: Could not parse wavelength from column {col}")
        
        # Transpose the data so bands become rows and points become columns
        df_transposed = df_copy[data_columns].T
        
        # If target_gid_count is None, use the actual number of points
        if target_gid_count is None:
            target_gid_count = len(point_names)
        
        # Use the minimum of available data and target count
        actual_gid_count = min(df_transposed.shape[1], len(point_names), target_gid_count)
        
        # Create the result dictionary
        result_data = {'Wave': wave_centers}
        
        # Add GID columns with actual data
        for i in range(target_gid_count):
            if i < actual_gid_count:
                # Use actual data
                column_data = df_transposed.iloc[:, i].values
                result_data[f'GID_{i+1}'] = column_data
            else:
                # For stations beyond available data, use the pattern from real data
                if actual_gid_count > 0:
                    # Cycle through existing data
                    source_idx = i % actual_gid_count
                    column_data = df_transposed.iloc[:, source_idx].values
                    result_data[f'GID_{i+1}'] = column_data
                else:
                    # Fallback to zeros if no data available
                    result_data[f'GID_{i+1}'] = [0.0] * len(wave_centers)
        
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