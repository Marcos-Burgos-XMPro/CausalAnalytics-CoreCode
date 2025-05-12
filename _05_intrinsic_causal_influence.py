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
import numpy as np

# Suppress all warnings
warnings.filterwarnings("ignore")

def on_receive(data: dict) -> dict:
    """
    Evaluate intrinsic causal influences for a specified target node using a pre-trained causal model.

    This function loads a causal model from the given file path, then calculates the intrinsic causal 
    influence of each treatment node on the specified target node using bootstrap sampling. It returns 
    the raw influence values, their percentage contributions, and the associated confidence intervals.

    Args:
        data (dict): A dictionary containing:
            - "model_path" (str): Path to the serialized causal model (Pickle format).
            - "target_node" (str): Name of the node in the causal graph for which intrinsic influence is evaluated.
            - "num_samples_randomization" (int): Number of samples randomization

    Returns:
        dict: A dictionary containing:
            - "timestamp" (str): Timestamp of execution.
            - "status" (str): "success" or "error" based on execution outcome.
            - "message" (str): A success message or error details.
            - "target_node" (str): The node analyzed.
            - "num_samples_randomization" (int): Number of samples randomization
            - "intrinsic_influence" (str): JSON string of a dictionary mapping treatment nodes to their 
              intrinsic influence values (sorted descending).
            - "intrinsic_influence_pct" (str): JSON string of a dictionary mapping treatment nodes to their 
              percentage influence contribution (sorted descending).
            - "intrinsic_influence_intervals" (str): JSON string of a dictionary mapping treatment nodes 
              to confidence intervals [lower_bound, upper_bound].
    """
    def convert_to_percentage(value_dictionary: dict) -> dict:
        total_absolute_sum = np.sum([abs(v) for v in value_dictionary.values()])
        if total_absolute_sum == 0:
            # Avoid division by zero
            return {k: 0 for k in value_dictionary}
        return {k: abs(v) / total_absolute_sum * 100 for k, v in value_dictionary.items()}

    # Capture timestamp early
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    try:
        # Safely access required keys
        target_node = data.get("target_node")
        model_path = data.get("model_path")
        num_samples_randomization = data.get("num_samples_randomization")

        # Set a fixed random seed for reproducibility
        gcm.util.general.set_random_seed(0)

        # Step 1: Load the pre-trained causal model from file
        with open(model_path, 'rb') as file:
            causal_model = pickle.load(file)

        # Step 2: Causal Query - Intrinsic Causal Influence
        intrinsic_influence_median,  intrinsic_influence_intervals = gcm.confidence_intervals(
            gcm.bootstrap_sampling(gcm.intrinsic_causal_influence,
                                   causal_model,
                                   target_node=target_node, 
                                   num_samples_randomization=num_samples_randomization))
        
        intrinsic_influence = intrinsic_influence_median
        intrinsic_influence_pct = convert_to_percentage(intrinsic_influence)

        # --- Prepare Output Dictionary (sorted descending by value) ---
        intrinsic_influence_dict = dict(
            sorted(
                ((treatment, round(value, 2)) for treatment, value in intrinsic_influence.items()),
                key=lambda item: item[1],
                reverse=True
            )
        )

        intrinsic_influence_pct_dict = dict(
            sorted(
                ((treatment, round(value, 2)) for treatment, value in intrinsic_influence_pct.items()),
                key=lambda item: item[1],
                reverse=True
            )
        )
        intrinsic_influence_intervals_dict = dict((treatment, [round(x, 2) for x in value.tolist()]) for treatment, value in intrinsic_influence_intervals.items())

        # Return successful evaluation result
        result = {
            "timestamp": timestamp,
            "status": "success",
            "message": "Intrinsic influences calculated successfully.",
            "target_node": target_node,
            "num_samples_randomization": num_samples_randomization,
            "intrinsic_influence": json.dumps(intrinsic_influence_dict),
            "intrinsic_influence_pct": json.dumps(intrinsic_influence_pct_dict),
            "intrinsic_influence_intervals": json.dumps(intrinsic_influence_intervals_dict)
        }

    except Exception as e:
        # In case of any exception, return error information
        result = {
            "timestamp": timestamp,
            "status": "error",
            "message": str(e),
            "target_node": data.get("target_node", ""),
            "num_samples_randomization": num_samples_randomization,
            "intrinsic_influence": None,
            "intrinsic_influence_pct": None,
            "intrinsic_influence_intervals": None
        }

    return result

def on_destroy() -> dict | None:
    return None