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
from dowhy import gcm
from datetime import datetime
import pickle
import warnings
import json
import ast
import pandas as pd

# Suppress all warnings
warnings.filterwarnings("ignore")

def on_receive(data: dict) -> dict:
    """
    Perform a counterfactual query using a pre-trained causal model.

    This function receives input data including a path to a serialized causal model,
    a counterfactual input specifying the desired changes in treatment variables,
    and an observation containing real-world feature values. It performs counterfactual 
    sampling using the DoWhy GCM library and returns the resulting counterfactual 
    outputs along with metadata.

    Parameters:
    -----------
    data : dict
        A dictionary containing the following keys:
        - "model_path": str
            Path to the pickle file of a pre-trained invertible causal model.
        - "counterfactual_input": str
            A string representing a list of (variable, value) tuples to intervene on, 
            e.g., "[('altitude', 5), ('ambient_temp', 3)]".
        - "observation": str
            A JSON-formatted string representing the observed data as a dictionary 
            with variable names as keys and lists of values, including all 
            treatments, features, and outcome variables.

    Returns:
    --------
    dict
        A dictionary with the following keys:
        - "timestamp": str
            Timestamp of when the evaluation was performed.
        - "status": str
            Either "success" or "error" depending on execution outcome.
        - "message": str
            Descriptive message about the result or error.
        - "counterfactual_input": str
            The original counterfactual input string.
        - "observation": str
            The original observation string.
        - "counterfactual_output": str or None
            JSON-formatted list of counterfactual samples if successful, else None.

    Notes:
    ------
    - Sets a fixed random seed to ensure reproducibility of results.
    - Assumes that the causal model is invertible and compatible with the input format.
    - All features used in the causal model must be present in the observation data.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        # Retrieve input parameters from the data dictionary
        model_path = data.get("model_path")
        counterfactual_user_input = ast.literal_eval(data.get("counterfactual_input").strip())
        observation = pd.DataFrame(data=json.loads(data.get("observation")))

        # Set a fixed random seed for reproducibility
        gcm.util.general.set_random_seed(0)

        # Step 1: Load the pre-trained causal model from file
        with open(model_path, 'rb') as file:
            causal_model = pickle.load(file)
        
        # Step 2: Format intervention input as a dictionary of lambda functions
        counterfactual = {key: (lambda value: lambda variable: value)(val) for key, val in counterfactual_user_input}

        # Step 3: Causal Query - Counterfactual Result
        counterfactuals_sample = gcm.counterfactual_samples(causal_model, counterfactual, observed_data=observation)
        counterfactuals_sample_json = counterfactuals_sample.to_json(orient='records')

        # Return successful evaluation result
        result = {
            "timestamp": timestamp,
            "status": "success",
            "message": "Successful Counterfactual.",
            "counterfactual_input": data.get("counterfactual_input"),
            "observation": json.dumps(data.get("observation")),
            "counterfactual_output": json.dumps(counterfactuals_sample_json)
        }

    except Exception as e:
        # In case of any exception, return error information
        result = {
            "timestamp": timestamp,
            "status": "error",
            "message": str(e),
            "counterfactual_input": data.get("counterfactual_input"),
            "observation": json.dumps(data.get("observation")),
            "counterfactual_output": None
        }

    return result

def on_destroy() -> dict | None:
    return None