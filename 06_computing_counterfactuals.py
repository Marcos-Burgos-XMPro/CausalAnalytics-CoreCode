""" Out of the MetaAgent"""

# Combine into the final data dictionary
data = {
    "model_path": "invertible_causal_model.pkl",
    "counterfactual_input":  "[('altitude', 5),('ambient_temp', 3)]",
    "observation": {"altitude": [1,2], 
                        "ambient_temp": [2,3], 
                        "engine_load": [2,4], 
                        "engine_rpm": [2,5], 
                        "air_filter_pressure": [2,5], 
                        "coolant_temp": [2,5], 
                        "fuel_consumption": [2,5], 
                        "egt_turbo_inlet": [2,5]}
} # Observation must contain all features, treatments and outcomes

""" In MetaAgent """

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
    # Set a fixed random seed for reproducibility
    gcm.util.general.set_random_seed(0)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        # Retrieve input parameters from the data dictionary
        model_path = data.get("model_path")
        counterfactual_user_input = ast.literal_eval(data.get("counterfactual_input").strip())
        observation = pd.DataFrame(data=data.get("observation"))

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

result = on_receive(data)
print(result)