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
import ast
from datetime import datetime
from dowhy.gcm.falsify import falsify_graph
import json

def on_receive(data: dict) -> dict:
    """
    Evaluate the falsifiability and validity of a user-defined causal graph against observational data.

    This function takes in a dictionary containing observational data and a string-formatted list of
    causal relationships. It constructs a causal graph using NetworkX, applies falsifiability tests 
    using `dowhy.gcm.falsify_graph`, and returns a structured result with the test outcomes and interpretation.

    Parameters:
    -----------
    data : dict
        A dictionary with the following required keys:
        - "observation": A list of dictionaries (typically converted from a CSV file) representing 
          tabular observational data, where each dictionary is a row with column names as keys.
        - "causal_relationships": A string representing a Python list of tuple pairs, 
          each defining a directed edge in the causal graph (e.g., "('A', 'B')").

    Returns:
    --------
    dict
        A dictionary containing:
        - "timestamp": The time the function was executed.
        - "status": "success" if the test ran without error, "error" otherwise.
        - "message": Status description or error message.
        - "causal_relationships": The input causal relationships.
        - "falsifiable": Boolean indicating if the causal model makes testable predictions.
        - "falsified": Boolean indicating if the testable predictions contradict the data.
        - "explanation": Human-readable interpretation of the falsifiability result.

    Notes:
    ------
    - The causal relationships must be meaningful and concrete enough to allow conditional 
      independence testing.
    - The `falsify_graph` method evaluates if the graph structure holds up against statistical 
      tests derived from observed dependencies and independencies in the data.
    - The input data is expected to be pre-processed (e.g., numeric values where applicable).

    Example:
    --------
    >>> data = {
    ...     "observation": csv_to_dict("cat797f_egt_causal_data.csv"),
    ...     "causal_relationships": "[('engine_load', 'fuel_consumption'), ('fuel_consumption', 'egt_turbo_inlet')]"
    ... }
    >>> result = on_receive(data)
    >>> print(result["falsifiable"], result["falsified"])
    """
    # Set a fixed random seed for reproducibility
    gcm.util.general.set_random_seed(0)
    try:
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
        causal_graph = nx.DiGraph(ast.literal_eval(data["causal_relationships"].strip()))

        # --- Step 2: Refute the Causal Graph
        result = falsify_graph(causal_graph, data=observation, show_progress_bar=False)

        explanation=""
        if result.falsifiable==False and result.falsified==False:
            explanation = """
            Your causal graph is too vague to test properly. Add more specific causal relationships to create a model that makes more concrete predictions.
            """
        elif result.falsifiable==True and result.falsified==False:
            explanation = """
            Great! Your causal graph is both meaningful and supported by data. Proceed with confidence using this model for further analysis.
            """
        elif result.falsifiable==False and result.falsified==True:
            explanation = """
            Unexpected error in testing procedure. Review your methodology and code implementation, as this result combination shouldn't occur.
            """
        elif result.falsifiable==True and result.falsified==True:
            explanation = """
            Your causal model is incorrect. Revise your graph structure by examining which specific 
            conditional independence assumptions failed and adjust the causal relationships accordingly.
            """

        # If all above steps are successful
        # Add this before creating the result dictionary
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # e.g., '2025-04-14 14:25:00'
        
        result = {
            "timestamp": timestamp,
            "status": "success",
            "message": "Assessment Completed",
            "causal_relationships": data.get("causal_relationships"),
            "falsifiable": result.falsifiable,
            "falsified": result.falsified,
            "explanation": explanation
        }

    except Exception as e:
        # If there is any exception
        # Add this before creating the result dictionary
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # e.g., '2025-04-14 14:25:00'

        result = {
            "timestamp": timestamp,
            "status": "error",
            "message": str(e),
            "causal_relationships": data.get("causal_relationships"),
            "falsifiable": None,
            "falsified": None,
            "explanation": None
        }

    return result

def on_destroy() -> dict | None:
    return None