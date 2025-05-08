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

# Import necessary libraries
import pandas as pd
from dowhy import gcm
from datetime import datetime
import pickle
import warnings
import json

# Suppress all warnings
warnings.filterwarnings("ignore")

def on_receive(data: dict) -> dict:
    """
    Processes an input dictionary to evaluate a pre-trained causal model using observational data.

    This function:
    - Deserializes the observational data (if in JSON string format) or uses it directly (if already in dict/list form),
    - Loads a pre-trained causal model from the specified pickle file path,
    - Evaluates the model using the `dowhy.gcm.evaluate_causal_model` function with mechanism baseline comparison,
    - Returns a dictionary containing the evaluation results, timestamp, and status.

    Parameters:
        data (dict): A dictionary containing:
            - 'observation' (str or list[dict]): Observational data either as a JSON string or as a list of dictionaries.
            - 'model_path' (str): Path to the pickle file containing the pre-trained causal model.

    Returns:
        dict: A result dictionary with the following keys:
            - 'timestamp' (str): The time at which the evaluation was performed.
            - 'status' (str): "success" if the evaluation was successful; otherwise "error".
            - 'message' (str): A message describing the result or the error.
            - 'summary_evaluation' (dict or None): The evaluation summary if successful; otherwise None.
    """
    try:
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

def on_destroy() -> dict | None:
    return None