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

        # Step 2: Causal Query - Direct Arrow Strength
        arrow_strengths_median, arrow_strengths_intervals = gcm.confidence_intervals(
            gcm.bootstrap_sampling(gcm.arrow_strength,
                                   causal_model,
                                   target_node=target_node))
        
        arrow_strengths = arrow_strengths_median
        arrow_strengths_pct = convert_to_percentage(arrow_strengths)

        # --- Prepare Output Dictionary (sorted descending by value) ---
        arrow_strengths_dict = dict(
            sorted(
                ((treatment, round(value, 2)) for (treatment, _), value in arrow_strengths.items()),
                key=lambda item: item[1],
                reverse=True
            )
        )
        arrow_strengths_pct_dict = dict(
            sorted(
                ((treatment, round(value, 2)) for (treatment, _), value in arrow_strengths_pct.items()),
                key=lambda item: item[1],
                reverse=True
            )
        )
        arrow_strengths_intervals_dict = dict((treatment, [round(x, 2) for x in value.tolist()]) for (treatment, _), value in arrow_strengths_intervals.items())

        # Return successful evaluation result
        result = {
            "timestamp": timestamp,
            "status": "success",
            "message": "Arrow strengths calculated successfully.",
            "target_node": target_node,
            "arrow_strength": json.dumps(arrow_strengths_dict),
            "arrow_strength_pct": json.dumps(arrow_strengths_pct_dict),
            "arrow_strengths_intervals": json.dumps(arrow_strengths_intervals_dict)
        }

    except Exception as e:
        # In case of any exception, return error information
        result = {
            "timestamp": timestamp,
            "status": "error",
            "message": str(e),
            "target_node": data.get("target_node", ""),
            "arrow_strength": None,
            "arrow_strength_pct": None,
            "arrow_strengths_intervals": None
        }

    return result

result = on_receive(data)
print(result)