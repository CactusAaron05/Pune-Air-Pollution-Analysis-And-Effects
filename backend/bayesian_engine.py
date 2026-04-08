import json
import os
import math

BASE_DIR = os.path.dirname(__file__)

def load_json(filename):
    path = os.path.join(BASE_DIR, "data", filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing config file: {filename}")
    with open(path) as f:
        return json.load(f)

PRIORS = load_json("source_priors.json")
LIKELIHOOD_CONFIG = load_json("likelihood_config.json")


def validate_key(data, key, context):
    if key not in data:
        raise ValueError(f"{context} missing key: {key}")
    return data[key]


def gaussian_likelihood(x, mean, std_dev):
    if std_dev <= 0:
        raise ValueError("std_dev must be positive")

    exponent = -((x - mean) ** 2) / (2 * (std_dev ** 2))
    return math.exp(exponent)


def compute_likelihood(spike_ratios, pollutants):

    if not pollutants:
        raise ValueError("Pollutant list cannot be empty")

    distribution = validate_key(LIKELIHOOD_CONFIG, "distribution", "likelihood_config")

    params = validate_key(LIKELIHOOD_CONFIG, "parameters", "likelihood_config")

    mean_ref = validate_key(params, "mean_reference", "likelihood_config")
    std_dev = validate_key(params, "std_dev", "likelihood_config")

    likelihood_values = []

    for p in pollutants:
        if p not in spike_ratios:
            raise ValueError(f"Missing spike ratio for pollutant: {p}")

        value = spike_ratios[p]

        if distribution == "gaussian":
            likelihood = gaussian_likelihood(value, mean_ref, std_dev)
        else:
            raise ValueError(f"Unsupported distribution: {distribution}")

        likelihood_values.append(likelihood)

    # joint likelihood (independence assumption)
    joint = 1.0
    for v in likelihood_values:
        joint *= v

    return joint


def bayesian_inference(spike_ratios, source_patterns):

    results = []

    for source, config in source_patterns.items():

        pollutants = validate_key(config, "pollutants", "source_patterns")

        prior = validate_key(
            validate_key(PRIORS, source, "source_priors"),
            "prior",
            "source_priors"
        )

        likelihood = compute_likelihood(spike_ratios, pollutants)

        posterior = prior * likelihood

        results.append({
            "source": source,
            "posterior": posterior
        })

    total = sum(r["posterior"] for r in results)

    if total <= 0:
        raise ValueError("Posterior normalization failed")

    for r in results:
        r["probability"] = r["posterior"] / total

    ranked = sorted(results, key=lambda x: x["probability"], reverse=True)

    return ranked