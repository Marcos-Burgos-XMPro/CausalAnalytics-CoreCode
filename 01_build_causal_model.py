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

""" In MetaAgent"""

# Install Libraries
import networkx as nx
import pandas as pd
from dowhy import gcm
import pickle
import ast
import os
from datetime import datetime
import json

def on_receive(data: dict) -> dict:
    """
    Handles incoming data, trains the appropriate causal model type 
    (invertible or non-invertible), and returns model training status information.

    Args:
        data (dict): A dictionary containing:
        - observation (str): Telemetry data from the CSV file
        - causal_relationships (str): String representation of causal graph edges
        - causal_model_type (str): Model type, either "invertible" or "non-invertible"
        - model_path (str): Directory path where the trained model should be saved

    Returns:
        dict: Result dictionary containing:
        - timestamp (str): Timestamp of process completion
        - status (str): "success" or "error"
        - message (str): Description of the outcome
        - causal_model_type (str): Model type used (echoed from input)
        - saved_path (str): Path where model was saved, or None if error
    """
    try:
        # Retrieve input parameters from the data dictionary
        causal_model_type = data.get("causal_model_type")
        # Set a fixed random seed for reproducibility
        gcm.util.general.set_random_seed(0)

        # --- Step 0: Read the test dataset into a pandas DataFrame
        # Option 1: If the data is a JSON string that needs to be deserialized
        if isinstance(data['observation'], str):
            deserialized_data = json.loads(data['observation'])  # In Meta Agent
        # Option 2: If the data is already a dictionary
        else:
            deserialized_data = data['observation']  # In Local
        observation = pd.DataFrame(deserialized_data)

        # --- Step 1: Define Causal Model ---
        # Create a directed graph representing the causal relationships
        causal_relationship = ast.literal_eval(data["causal_relationships"].strip())
        causal_graph = nx.DiGraph(causal_relationship)

        # Create the structural causal model object
        if causal_model_type == "non-invertible":
            causal_model = gcm.StructuralCausalModel(causal_graph)
        elif causal_model_type == "invertible":
            causal_model = gcm.InvertibleStructuralCausalModel(causal_graph)

        # Automatically assign generative models (causal mechanisms)
        summary_auto_assignment = gcm.auto.assign_causal_mechanisms(causal_model, observation)

        # --- Step 2: Fit Causal Models to Data ---
        gcm.fit(causal_model, observation)

        # --- Step 3: Save the fitted model to a file
        model_save_path = os.path.join(data["model_path"], f'{data["model_name"]}.pkl')
        with open(model_save_path, 'wb') as file:
            pickle.dump(causal_model, file)

        # If all above steps are successful
        # Add this before creating the result dictionary
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # e.g., '2025-04-14 14:25:00'
        
        result = {
            "timestamp": timestamp,
            "status": "success",
            "message": "Model saved successfully.",
            "causal_model_type": data.get("causal_model_type"),
            "saved_path": model_save_path
        }

    except Exception as e:
        # If there is any exception
        # Add this before creating the result dictionary
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # e.g., '2025-04-14 14:25:00'

        result = {
            "timestamp": timestamp,
            "status": "error",
            "message": str(e),
            "causal_model_type": data.get("causal_model_type"),
            "saved_path": None
        }

    return result

print(on_receive(data))