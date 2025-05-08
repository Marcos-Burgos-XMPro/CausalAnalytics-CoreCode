""" Out of the MetaAgent"""

# Combine into the final data dictionary
data = {
    "model_path": "causal_model.pkl",
    "intervention_input":  "[('altitude', 5),('ambient_temp', 3)]",
    "num_samples_to_draw": 5,
    "intervention_type": "atomic" # {atomic, shift}
}

""" In MetaAgent """

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

result = on_receive(data)
print(result)