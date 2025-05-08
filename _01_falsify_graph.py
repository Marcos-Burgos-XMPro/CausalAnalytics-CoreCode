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
}

""" In MetaAgent"""

# Install Libraries
import networkx as nx
import pandas as pd
from dowhy import gcm
import ast
from datetime import datetime
from dowhy.gcm.falsify import falsify_graph

def on_receive(data: dict) -> dict:
    # Set a fixed random seed for reproducibility
    gcm.util.general.set_random_seed(0)
    try:
        # --- Step 0: Read the test dataset into a pandas DataFrame
        observation = pd.DataFrame(data['observation'])

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

print(on_receive(data))