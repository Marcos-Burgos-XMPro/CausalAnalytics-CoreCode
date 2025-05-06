""" Out of the MetaAgent"""

# Combine into the final data dictionary
data = {
    "model_path": "invertible_causal_model.pkl",
    "anomalous_node": 'egt_turbo_inlet',
    "anomaly_data": {"altitude": [1], 
                        "ambient_temp": [2], 
                        "engine_load": [3], 
                        "engine_rpm": [4], 
                        "air_filter_pressure": [5], 
                        "coolant_temp": [6], 
                        "fuel_consumption": [7], 
                        "egt_turbo_inlet": [8]}
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
        anomalous_node = data.get("anomalous_node")
        anomaly_data = pd.DataFrame(data=data.get("anomaly_data"))

        # Step 1: Load the pre-trained causal model from file
        with open(model_path, 'rb') as file:
            causal_model = pickle.load(file)

        # Step 2: Causal Query - Anomaly Attribution
        attribution_scores = gcm.attribute_anomalies(causal_model, anomalous_node, anomaly_samples=anomaly_data)
        print(attribution_scores)
        print("AFTER")
        attribution_scores_df = pd.DataFrame(attribution_scores)
        attribution_scores_json = attribution_scores_df.to_json(orient='records')

        # Return successful evaluation result
        result = {
            "timestamp": timestamp,
            "status": "success",
            "message": "Successful Calculation of Anomaly Attribution",
            "anomalous_node": data.get("anomalous_node"),
            "anomaly_data": json.dumps(data.get("anomaly_data")),
            "anomaly_attribution": json.dumps(attribution_scores_json)
        }

    except Exception as e:
        # In case of any exception, return error information
        result = {
            "timestamp": timestamp,
            "status": "error",
            "message": str(e),
            "anomalous_node": data.get("anomalous_node"),
            "anomaly_data": json.dumps(data.get("anomaly_data")),
            "anomaly_attribution": None
        }

    return result

result = on_receive(data)
print(result)