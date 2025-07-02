import os
from core.spectra_simulation import SatelliteBandSimulator
from utils.data_loader import DataLoader
from utils.data_processor import DataProcessor
from utils.output_handler import OutputHandler

def main():
    data_path = "../example/GLORIA_Rrs.csv"
    output_dir = "results"
    
    # Initialize components
    simulator = SatelliteBandSimulator()
    data_loader = DataLoader()
    data_processor = DataProcessor()
    output_handler = OutputHandler(output_dir)
    
    # Load and process data
    print("Loading GLORIA data...")
    data = data_loader.load_gloria_data(data_path)
    
    print("Processing spectra...")
    spectra, point_names = data_processor.process_spectra(data)
    
    # Get actual number of stations from the processed data
    actual_stations = len(point_names)
    print(f"Working with {actual_stations} stations")

    # Run single sensor
    print("Running OLI simulation")
    #oli_results = data_processor.run_sensor_simulation(simulator, spectra, point_names, "oli")
    #output_handler.save_all_results(oli_results, point_names, actual_stations)

    # Run multiple selected sensors
    print("Running Multiple Sensors")
    selected_sensors = [
        'oli',
        'olci',
        {'sensor': 'msi', 'variant': 's2b'},
        'modis'
    ]
    #multi_results = data_processor.run_multiple_sensors(simulator, spectra, point_names, selected_sensors)
    #output_handler.save_all_results(multi_results, point_names, actual_stations)

    # Run all sensors
    print("Running all satellite band simulation")
    simulation_results = data_processor.run_all_simulations(simulator, spectra, point_names)
    output_handler.save_all_results(simulation_results, point_names, actual_stations)

    print(f"Results saved to {output_dir}/ directory.")
    print("Simulation completed!")

if __name__ == '__main__':
    main()