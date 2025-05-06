""" Out of the MetaAgent"""

import csv

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

# Combine into the final data dictionary
data = {
    "observation": observation_input,
    "model_path": "causal_model.pkl"
}

""" In MetaAgent"""

# Import necessary libraries
import pandas as pd
from dowhy import gcm
from datetime import datetime
import pickle
import warnings

# Suppress all warnings
warnings.filterwarnings("ignore")

def on_receive(data: dict) -> dict:
    try:
        # Set a fixed random seed for reproducibility
        gcm.util.general.set_random_seed(0)

        # Step 0: Read the observation dataset into a pandas DataFrame
        observation = pd.DataFrame(data['observation'])

        # Step 1: Load the pre-trained causal model from file
        with open(data['model_path'], 'rb') as file:
            causal_model = pickle.load(file)

        # Step 2: Evaluate the causal model without producing plots
        summary_evaluation = gcm.evaluate_causal_model(
            causal_model, 
            observation, 
            compare_mechanism_baselines=True
        )

        # Capture current timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Return successful evaluation result
        result = {
            "timestamp": timestamp,
            "status": "success",
            "message": "Successful evaluation.",
            "summary_evaluation": summary_evaluation
        }
    except Exception as e:
        # In case of any exception, return error information
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        result = {
            "timestamp": timestamp,
            "status": "error",
            "message": str(e),
            "summary_evaluation": None
        }

    return result

result = on_receive(data)
print(result)
print("###########")
print(result["summary_evaluation"])