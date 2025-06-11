import pandas as pd
import os

class DataLoader:
    def load_gloria_data(self, data_path):
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Error: File {data_path} not found")
        
        return pd.read_csv(data_path)