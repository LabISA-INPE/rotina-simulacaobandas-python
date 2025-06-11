import os
from core.spectra_simulation import SatelliteBandSimulator
from utils.data_loader import DataLoader
from utils.data_processor import DataProcessor
from utils.output_handler import OutputHandler

def main():
    # Configuration
    data_path = "../example/GLORIA_Rrs.csv"
    output_dir = "results"
    target_stations = 1000
    
    # Initialize components
    simulator = SatelliteBandSimulator()
    data_loader = DataLoader()
    data_processor = DataProcessor()
    output_handler = OutputHandler(output_dir)
    
    # Load and process data
    print("Loading GLORIA data...")
    data = data_loader.load_gloria_data(data_path)
    
    print("Processing spectra...")
    spectra, point_names = data_processor.process_spectra(data, target_stations)
    
    # Run simulations
    print("Running satellite band simulations...")
    simulation_results = data_processor.run_all_simulations(simulator, spectra, point_names)
    
    # Save results
    print("Saving results...")
    output_handler.save_all_results(simulation_results, point_names, target_stations)
    
    print(f"Results saved to {output_dir}/ directory.")
    print("Simulation completed!")

if __name__ == '__main__':
    main()