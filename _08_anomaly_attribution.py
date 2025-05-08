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
import pandas as pd
import numpy as np

# Suppress all warnings
warnings.filterwarnings("ignore")

def on_receive(data: dict) -> dict:
    def convert_to_percentage(value_dictionary: dict) -> dict:
        total_absolute_sum = np.sum([abs(v) for v in value_dictionary.values()])
        if total_absolute_sum == 0:
            # Avoid division by zero
            return {k: 0 for k in value_dictionary}
        return {k: abs(v) / total_absolute_sum * 100 for k, v in value_dictionary.items()}
    
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

result = on_receive(data)
print(result)