""" Out of the MetaAgent"""

# Combine into the final data dictionary
data = {
    "model_path": "causal_model.pkl",
    "target_node": 'egt_turbo_inlet'
}

""" In MetaAgent """

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

        # Step 2: Causal Query - Intrinsic Causal Influence
        intrinsic_influence_median,  intrinsic_influence_intervals = gcm.confidence_intervals(
            gcm.bootstrap_sampling(gcm.intrinsic_causal_influence,
                                   causal_model,
                                   target_node=target_node, 
                                   num_samples_randomization=100))
        
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
        print(intrinsic_influence_intervals_dict)

        # Return successful evaluation result
        result = {
            "timestamp": timestamp,
            "status": "success",
            "message": "Intrinsic influences calculated successfully.",
            "target_node": target_node,
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
            "intrinsic_influence": None,
            "intrinsic_influence_pct": None,
            "intrinsic_influence_intervals": None
        }

    return result

result = on_receive(data)
print(result)