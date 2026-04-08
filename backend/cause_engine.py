import json
import os

from bayesian_engine import bayesian_inference

# Load pattern dataset
DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "source_patterns.json")

with open(DATA_PATH, "r") as f:
    SOURCE_PATTERNS = json.load(f)

if not SOURCE_PATTERNS:
    raise ValueError("source_patterns.json is empty")


def detect_causes(predictions, rolling_baseline):

    if not predictions:
        raise ValueError("predictions cannot be empty")

    if not rolling_baseline:
        raise ValueError("rolling_baseline cannot be empty")

    spike_ratios = {}

    for pollutant, value in predictions.items():

        if pollutant not in rolling_baseline:
            raise ValueError(f"Missing baseline for pollutant: {pollutant}")

        baseline = rolling_baseline[pollutant]

        if baseline <= 0:
            raise ValueError(f"Invalid baseline for pollutant: {pollutant}")

        spike_ratios[pollutant] = value / baseline

    if not spike_ratios:
        raise ValueError("No spike ratios computed")

    ranked = bayesian_inference(spike_ratios, SOURCE_PATTERNS)

    if len(ranked) < 2:
        raise ValueError("Insufficient sources for inference")

    primary = ranked[0]
    secondary = ranked[1]

    return {
        "primary_source": {
            "source": primary["source"],
            "confidence": primary["probability"]
        },
        "secondary_source": {
            "source": secondary["source"],
            "confidence": secondary["probability"]
        }
    }