import json
import os
import random
import statistics

BASE_DIR = os.path.dirname(__file__)

def load_json(name):
    with open(os.path.join(BASE_DIR, "data", name)) as f:
        return json.load(f)

CONFIG = load_json("uncertainty_config.json")


def validate_key(data, key, context):
    if key not in data:
        raise ValueError(f"{context} missing key: {key}")
    return data[key]


def add_noise(value, std_fraction):
    std = value * std_fraction
    return random.gauss(value, std)


def simulate_predictions(predictions):

    samples = validate_key(CONFIG, "num_samples", "uncertainty_config")
    std_frac = validate_key(CONFIG, "noise_std_fraction", "uncertainty_config")

    simulations = []

    for _ in range(samples):

        perturbed = {
            k: add_noise(v, std_frac)
            for k, v in predictions.items()
        }

        simulations.append(perturbed)

    return simulations


def compute_uncertainty_range(values):
    return {
        "mean": statistics.mean(values),
        "min": min(values),
        "max": max(values)
    }