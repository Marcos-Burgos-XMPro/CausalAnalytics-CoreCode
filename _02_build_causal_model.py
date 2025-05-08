"""
Default script template for the Python Meta Action Agent.

When importing packages, follow the format below to add a comment at the end of declaration 
and specify a version or a package name when the import name is different from expected python package.
This allows the agent to install the correct package version during configuration:
e.g. import paho.mqtt as np  # version=2.1.0 package=paho-mqtt

This script provides a structure for implementing on_create, on_receive, and on_destroy functions.
It includes a basic example using 'foo' and 'bar' concepts to demonstrate functionality.
Each function should return a dictionary object with result data, or None if no result is needed.
"""

def on_create(data: dict) -> dict | None:
    return None

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
        - model_name (str): Name of the model to be saved

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

def on_destroy() -> dict | None:
    return None