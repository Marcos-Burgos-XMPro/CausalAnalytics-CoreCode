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
    Computes and returns the direct causal arrow strengths for a specified target node 
    from a pre-trained causal model using bootstrap sampling.

    The function performs the following:
    1. Loads a causal model from the specified file path.
    2. Estimates the direct arrow strengths from treatment nodes to the target node.
    3. Computes confidence intervals and percentage contributions of the arrow strengths.
    4. Returns all results in a structured dictionary, including rounded values and timestamp.

    Parameters:
    ----------
    data : dict
        A dictionary containing:
            - "model_path" (str): Path to the serialized causal model file (.pkl).
            - "target_node" (str): Name of the target node in the causal graph.

    Returns
    -------
    dict
        A dictionary with the following keys:
            - "timestamp": Timestamp when the computation was executed.
            - "status": "success" or "error" depending on process outcome.
            - "message": Success or error message.
            - "target_node": The node for which the analysis was conducted.
            - "arrow_strength_edge": JSON-encoded dictionary of edge-based arrow strengths 
                                      with format "(source, target)" → strength.
            - "arrow_strength_edge_pct": JSON-encoded dictionary of edge-based strengths 
                                          as percentages of total influence.
            - "arrow_strengths_edge_intervals": JSON-encoded dictionary of edge-based 
                                                 95% confidence intervals for arrow strengths.
            - "arrow_strength_node": JSON-encoded dictionary of treatment nodes and their 
                                     direct influence strengths on the target node.
            - "arrow_strength_node_pct": JSON-encoded dictionary of treatment nodes and 
                                         their percentage contributions.
            - "arrow_strengths_node_intervals": JSON-encoded dictionary of node-level 
                                                95% confidence intervals.
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

        # Set a fixed random seed for reproducibility
        gcm.util.general.set_random_seed(0)

        # Step 1: Load the pre-trained causal model from file
        with open(model_path, 'rb') as file:
            causal_model = pickle.load(file)

        # Step 2: Causal Query - Direct Arrow Strength
        arrow_strengths_median, arrow_strengths_intervals = gcm.confidence_intervals(
            gcm.bootstrap_sampling(gcm.arrow_strength,
                                   causal_model,
                                   target_node=target_node))
        
        arrow_strengths = arrow_strengths_median
        arrow_strengths_pct = convert_to_percentage(arrow_strengths)

        # Node Version
        # --- Prepare Output Dictionary (sorted descending by value) ---
        arrow_strengths_node = dict(
            sorted(
                ((treatment, round(value, 2)) for (treatment, _), value in arrow_strengths.items()),
                key=lambda item: item[1],
                reverse=True
            )
        )
        arrow_strengths_pct_node = dict(
            sorted(
                ((treatment, round(value, 2)) for (treatment, _), value in arrow_strengths_pct.items()),
                key=lambda item: item[1],
                reverse=True
            )
        )
        arrow_strengths_intervals_node = dict((treatment, [round(x, 2) for x in value.tolist()]) for (treatment, _), value in arrow_strengths_intervals.items())

        #Edge version
        arrow_strengths_edge_str = {f"({k[0]}, {k[1]})": round(v, 2) for k, v in arrow_strengths.items()}
        arrow_strengths_pct_edge_str = {f"({k[0]}, {k[1]})": round(v, 2) for k, v in arrow_strengths_pct.items()}
        arrow_strengths_intervals_edge_str = {f"({k[0]}, {k[1]})": [round(x, 2) for x in v.tolist()] for k, v in arrow_strengths_intervals.items()}

        # Return successful evaluation result
        result = {
            "timestamp": timestamp,
            "status": "success",
            "message": "Arrow strengths calculated successfully.",
            "target_node": target_node,
            "arrow_strength_edge": json.dumps(arrow_strengths_edge_str),
            "arrow_strength_edge_pct": json.dumps(arrow_strengths_pct_edge_str),
            "arrow_strengths_edge_intervals": json.dumps(arrow_strengths_intervals_edge_str),
            "arrow_strength_node": json.dumps(arrow_strengths_node),
            "arrow_strength_node_pct": json.dumps(arrow_strengths_pct_node),
            "arrow_strengths_node_intervals": json.dumps(arrow_strengths_intervals_node)
        }

    except Exception as e:
        # In case of any exception, return error information
        result = {
            "timestamp": timestamp,
            "status": "error",
            "message": str(e),
            "target_node": data.get("target_node", ""),
            "arrow_strength_edge": None,
            "arrow_strength_edge_pct": None,
            "arrow_strengths_edge_intervals": None,
            "arrow_strength_node": None,
            "arrow_strength_node_pct": None,
            "arrow_strengths_node_intervals": None
        }

    return result

def on_destroy() -> dict | None:
    return None