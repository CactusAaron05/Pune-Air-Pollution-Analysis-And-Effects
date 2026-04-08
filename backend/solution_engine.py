import json
import os

BASE_DIR = os.path.dirname(__file__)

def load_json(name):
    with open(os.path.join(BASE_DIR, "data", name)) as f:
        return json.load(f)

SOLUTIONS = load_json("solutions_dataset.json")
REGION_PROFILES = load_json("region_profiles.json")
CONFIG = load_json("system_config.json")
TIME_STRATEGY = load_json("time_strategy.json")
LONG_TERM_POLICY = load_json("long_term_policy.json")
SOURCE_PATTERNS = load_json("source_patterns.json")


# ───────────────── HELPERS ─────────────────

def validate_key(data, key, context):
    if key not in data:
        raise ValueError(f"{context} missing key: {key}")
    return data[key]


def get_peak_horizon(predictions):
    return max(predictions, key=predictions.get)

def determine_time_window(predictions, health_risk):

    # extract values
    aqi_values = [predictions[h] for h in ["1h", "3h", "6h"]]

    risk_levels = [
        health_risk[h]["interpretation"]["risk_level"]
        for h in ["1h", "3h", "6h"]
    ]

    # ── CASE 1: sustained severe ──
    if all(r == "Severe" for r in risk_levels):
        return "next 6h (sustained high pollution)"

    # ── CASE 2: worsening ──
    if aqi_values[0] < aqi_values[1] < aqi_values[2]:
        return "next 6h (worsening conditions)"

    # ── CASE 3: improving but still high ──
    if aqi_values[0] > aqi_values[1] > aqi_values[2]:
        if risk_levels[0] in ["Moderate", "Severe"]:
            return "next few hours (gradual improvement)"

    # ── DEFAULT ──
    peak = max(predictions, key=predictions.get)
    return f"next {peak}"


def compute_feasibility(action, profile):
    constraints = action.get("constraints", {})

    if not constraints:
        return 1.0

    score = 0
    total = 0

    for key, allowed in constraints.items():
        total += 1
        if profile.get(key) in allowed:
            score += 1
        else:
            score += 0.5  # partial

    return score / total if total else 1.0


def compute_pollutant_alignment(source, dominant_pollutant):
    pattern = SOURCE_PATTERNS.get(source, {})
    pollutants = pattern.get("pollutants", [])

    if dominant_pollutant in pollutants:
        return 1.0
    return 0.5


def severity_multiplier(level):
    return {
        "Minimal": 0.5,
        "Mild": 0.8,
        "Moderate": 1.0,
        "Severe": 1.3
    }.get(level, 1.0)


# ───────────────── MAIN ENGINE ─────────────────

def generate_solutions(region, predictions, causes, health_risk):

    region_profile = validate_key(REGION_PROFILES, region, "region_profiles")

    peak = max(predictions, key=predictions.get)
    time_window = determine_time_window(predictions, health_risk)   

    cause_block = validate_key(causes, peak, "causes")
    primary = validate_key(cause_block, "primary_source", "cause_block")
    secondary = cause_block.get("secondary_source")

    risk_block = validate_key(health_risk, peak, "health_risk")
    interpretation = validate_key(risk_block, "interpretation", "health_risk")

    risk_level = interpretation["risk_level"]
    dominant_pollutant = risk_block["dominant_pollutant"]

    severity_factor = severity_multiplier(risk_level)

    allowed_types = TIME_STRATEGY.get(peak, {}).get("allowed_action_types", ["short"])

    candidate_actions = []

    def process_source(source_obj, weight):

        if not source_obj:
            return

        source = source_obj["source"]
        confidence = source_obj["confidence"]

        if source not in SOLUTIONS:
            return

        alignment = compute_pollutant_alignment(source, dominant_pollutant)

        for action_type in ["short_term", "long_term"]:

            if action_type == "long_term" and "long" not in allowed_types:
                continue

            for action in SOLUTIONS[source][action_type]:

                feasibility = compute_feasibility(action, region_profile)

                impact = action["effectiveness"]

                score = (
                    impact
                    * feasibility
                    * confidence
                    * alignment
                    * severity_factor
                    * weight
                )

                explanation = (
                    f"{action['action']} → "
                    f"targets {source}, "
                    f"dominant pollutant: {dominant_pollutant}, "
                    f"feasibility: {round(feasibility,2)}, "
                    f"confidence: {round(confidence,2)}"
                )

                candidate_actions.append({
                    "type": "short" if action_type == "short_term" else "long",
                    "action": action["action"],
                    "score": score,
                    "explanation": explanation,
                    "source": source
                })

    # Primary has more weight
    process_source(primary, 1.0)
    process_source(secondary, 0.7)

    ranked = sorted(candidate_actions, key=lambda x: x["score"], reverse=True)

    max_short = CONFIG["selection_policy"]["max_short_term"]
    max_long = CONFIG["selection_policy"]["max_long_term"]

    short_term = []
    long_term = []
    explanations = []

    for item in ranked:

        if item["type"] == "short" and len(short_term) < max_short:
            short_term.append(item["action"])
            explanations.append(item["explanation"])

        elif item["type"] == "long" and len(long_term) < max_long:
            long_term.append(item["action"])
            explanations.append(item["explanation"])

        if len(short_term) >= max_short and len(long_term) >= max_long:
            break

    return {
        "where": region,
        "when": time_window,
        "why": f"{primary['source']} + {secondary['source']}" if secondary else primary["source"],
        "short_term": short_term,
        "long_term": long_term,
        "explanation": explanations
    }