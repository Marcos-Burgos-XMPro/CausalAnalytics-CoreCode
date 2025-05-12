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
import pandas as pd
import numpy as np

# Suppress all warnings
warnings.filterwarnings("ignore")

def on_receive(data: dict) -> dict:
    """
    Perform anomaly attribution on a specified node in a causal model using bootstrap-based 
    confidence intervals and return attribution results in raw, percentage, and interval formats.

    This function expects a data dictionary that contains:
    - the file path to a serialized causal model,
    - the name of the anomalous node (target variable),
    - and a JSON-encoded dictionary of anomaly data observations.

    The function loads the causal model, performs anomaly attribution on the specified node 
    using the provided observation, and calculates attribution confidence intervals.
    The resulting attribution scores are also converted to percentages representing their
    relative importance in explaining the anomaly.

    Args:
        data (dict): A dictionary with the following keys:
            - "model_path" (str): Path to the pickle file of the trained causal model.
            - "anomalous_node" (str): The node in the graph where the anomaly has been detected.
            - "anomaly_data" (str): JSON-formatted string representing a dictionary of one-row 
                                     observations (with all causal variables as keys and single-element lists as values).

    Returns:
        dict: A dictionary containing:
            - "timestamp" (str): Time when the evaluation was executed.
            - "status" (str): Either "success" or "error".
            - "message" (str): Descriptive message for result status.
            - "anomalous_node" (str): The node under analysis.
            - "anomaly_data" (str): The input anomaly observation in JSON format.
            - "anomaly_attribution" (str or None): JSON-formatted raw attribution scores (median values), sorted.
            - "anomaly_attribution_pct" (str or None): JSON-formatted attribution scores as percentages, sorted.
            - "anomaly_attribution_confidence" (str or None): JSON-formatted 95% confidence intervals per cause.

    Example:
        >>> data = {
        ...     "model_path": "invertible_causal_model.pkl",
        ...     "anomalous_node": "egt_turbo_inlet",
        ...     "anomaly_data": '{"altitude": [1], "ambient_temp": [2], "engine_load": [3], '
        ...                      '"engine_rpm": [4], "air_filter_pressure": [5], "coolant_temp": [6], '
        ...                      '"fuel_consumption": [7], "egt_turbo_inlet": [8]}'
        ... }
        >>> result = on_receive(data)
    """
    def convert_to_percentage(value_dictionary: dict) -> dict:
        total_absolute_sum = np.sum([abs(v) for v in value_dictionary.values()])
        if total_absolute_sum == 0:
            # Avoid division by zero
            return {k: 0 for k in value_dictionary}
        return {k: abs(v) / total_absolute_sum * 100 for k, v in value_dictionary.items()}
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    try:
        # Retrieve input parameters from the data dictionary
        model_path = data.get("model_path")
        anomalous_node = data.get("anomalous_node")
        anomaly_data = pd.DataFrame(data=json.loads(data.get("anomaly_data")))

        # Set a fixed random seed for reproducibility
        gcm.util.general.set_random_seed(0)

        # Step 1: Load the pre-trained causal model from file
        with open(model_path, 'rb') as file:
            causal_model = pickle.load(file)

        # Step 2: Causal Query - Anomaly Attribution
        attribution_scores_median,  attribution_scores_intervals = gcm.confidence_intervals(
            gcm.bootstrap_sampling(gcm.attribute_anomalies,
                                   causal_model,
                                   anomalous_node, 
                                   anomaly_samples=anomaly_data))
        
        attribution_scores = attribution_scores_median
        attribution_scores_pct = convert_to_percentage(attribution_scores)

        # --- Prepare Output Dictionary (sorted descending by value) ---
        attribution_scores_dict = dict(
            sorted(
                ((treatment, round(value, 2)) for treatment, value in attribution_scores.items()),
                key=lambda item: item[1],
                reverse=True
            )
        )

        attribution_scores_pct_dict = dict(
            sorted(
                ((treatment, round(value, 2)) for treatment, value in attribution_scores_pct.items()),
                key=lambda item: item[1],
                reverse=True
            )
        )

        attribution_scores_intervals_dict = dict((treatment, [round(x, 2) for x in value.tolist()]) for treatment, value in attribution_scores_intervals.items())
        
        # Return successful evaluation result
        result = {
            "timestamp": timestamp,
            "status": "success",
            "message": "Successful Calculation of Anomaly Attribution",
            "anomalous_node": data.get("anomalous_node"),
            "anomaly_data": json.dumps(data.get("anomaly_data")),
            "anomaly_attribution": json.dumps(attribution_scores_dict),
            "anomaly_attribution_pct": json.dumps(attribution_scores_pct_dict),
            "anomaly_attribution_confidence": json.dumps(attribution_scores_intervals_dict)
        }

    except Exception as e:
        # In case of any exception, return error information
        result = {
            "timestamp": timestamp,
            "status": "error",
            "message": str(e),
            "anomalous_node": data.get("anomalous_node"),
            "anomaly_data": json.dumps(data.get("anomaly_data")),
            "anomaly_attribution": None,
            "anomaly_attribution_pct": None,
            "anomaly_attribution_confidence": None
        }

    return result

def on_destroy() -> dict | None:
    return None