"""To Evaluate 02_build_causl_model"""
import csv

def _02_build_causal_model():
    # Read the CSV file into a dictionary
    def csv_to_dict(file_path):
        with open(file_path, 'r') as f:
            reader = csv.DictReader(f)
            data_dict = []
            for row in reader:
                converted_row = {}
                for key, value in row.items():
                    try:
                        # Try to convert to float
                        converted_value = float(value)
                    except ValueError:
                        # If fails (e.g., a string like a timestamp), keep as-is
                        converted_value = value
                    converted_row[key] = converted_value
                data_dict.append(converted_row)
        return data_dict

    file_path = 'cat797f_egt_causal_data.csv'
    observation_input = csv_to_dict(file_path)

    causal_relationships = """
        [
            # Air intake system relationships
            ('altitude', 'air_filter_pressure'),
            ('air_filter_pressure', 'egt_turbo_inlet'),
            ('air_filter_pressure', 'fuel_consumption'),
            
            # Primary mechanical relationships
            ('engine_load', 'engine_rpm'),
            ('engine_load', 'fuel_consumption'),
            ('engine_rpm', 'air_filter_pressure'),
            
            # Environmental influences
            ('altitude', 'engine_load'),
            ('ambient_temp', 'coolant_temp'),
            ('ambient_temp', 'egt_turbo_inlet'),
            
            # Fuel and combustion chain
            ('fuel_consumption', 'egt_turbo_inlet'),
            ('engine_load', 'egt_turbo_inlet'),
            
            # Cooling system relationships
            ('coolant_temp', 'egt_turbo_inlet'),
            ('engine_rpm', 'coolant_temp')
        ]
        """ 

    # Combine into the final data dictionary
    data = {
        "observation": observation_input,
        "causal_relationships": causal_relationships,
        "causal_model_type": "invertible", # {invertible, non-invertible}
        "model_path": "",
        "model_name": "causal-model"
    }

    return on_receive(data)

# Please write here what function to evaluate
from _02_build_causal_model import on_receive
if __name__ == "__main__":
    print(_02_build_causal_model())