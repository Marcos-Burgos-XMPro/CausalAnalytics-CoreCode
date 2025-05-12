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

# Suppress all warnings
warnings.filterwarnings("ignore")

def on_receive(data: dict) -> dict:
    """
    Executes a causal intervention on a pre-trained model and returns simulated outcomes.

    This function performs an interventional query (either atomic or shift) on a loaded causal model,
    using user-provided intervention variables. It generates a set of samples that represent the
    model’s behavior under the specified intervention.

    Parameters:
    -----------
    data : dict
        A dictionary containing the following keys:
        - "model_path" (str): Path to the serialized causal model (pickle file).
        - "intervention_input" (str): A string representation of a list of (variable, value) tuples
          indicating which variables to intervene on and their assigned values, e.g.,
          "[('altitude', 5), ('ambient_temp', 3)]".
        - "num_samples_to_draw" (int): Number of samples to generate from the interventional distribution.
        - "intervention_type" (str): The type of intervention, must be either "atomic" (value replacement)
          or "shift" (value adjustment).

    Returns:
    --------
    dict
        A dictionary with the following keys:
        - "timestamp" (str): Time the function was executed.
        - "status" (str): "success" if execution was successful; "error" otherwise.
        - "message" (str): Description of success or error message.
        - "intervention_input" (str): The original intervention input received.
        - "intervention_type" (str): The type of intervention performed.
        - "intervention_output" (str or None): A JSON string of the simulated intervention samples,
          or None if an error occurred.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        # Retrieve input parameters from the data dictionary
        model_path = data.get("model_path")
        intervention_user_input = ast.literal_eval(data.get("intervention_input").strip())
        num_samples_to_draw = data.get("num_samples_to_draw")
        intervention_type = data.get("intervention_type")
        
        # Set a fixed random seed for reproducibility
        gcm.util.general.set_random_seed(0)

        # Step 1: Load the pre-trained causal model from file
        with open(model_path, 'rb') as file:
            causal_model = pickle.load(file)
        
        # Step 2: Format intervention input as a dictionary of lambda functions
        if intervention_type == "atomic":
            intervention = {key: (lambda value: lambda variable: value)(val) for key, val in intervention_user_input}
        elif intervention_type == "shift":
            intervention = {key: (lambda value: lambda variable: variable + value)(val) for key, val in intervention_user_input}

        # Step 3: Causal Query - Interventional Sample
        intervention_sample = gcm.interventional_samples(causal_model, intervention, num_samples_to_draw=num_samples_to_draw)
        intervention_sample_json = intervention_sample.to_json(orient='records')

        # Return successful evaluation result
        result = {
            "timestamp": timestamp,
            "status": "success",
            "message": "Successful Intervention.",
            "intervention_input": data.get("intervention_input"),
            "intervention_type": intervention_type,
            "intervention_output": json.dumps(intervention_sample_json)
        }

    except Exception as e:
        # In case of any exception, return error information
        result = {
            "timestamp": timestamp,
            "status": "error",
            "message": str(e),
            "intervention_input": data.get("intervention_input"),
            "intervention_type": intervention_type,
            "intervention_output": None
        }

    return result

def on_destroy() -> dict | None:
    return None