import numpy as np

# WHO Air Quality Guidelines (2021) + Epidemiological studies (GBD, EPA)
import json
import os
import numpy as np

HEALTH_IMPACT_PATH = os.path.join(
    os.path.dirname(__file__),
    "data",
    "health_impact_dataset.json"
)

with open(HEALTH_IMPACT_PATH, "r") as f:
    HEALTH_IMPACT_DATA = json.load(f)

# Load dataset
DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "health_dataset.json")

with open(DATA_PATH, "r") as f:
    HEALTH_DATA = json.load(f)




def get_pollutant_health_data(pollutant):
    return HEALTH_DATA[pollutant].copy()

def compute_excess_exposure(pollutant, predicted_value):
    """
    ΔC = max(0, predicted - baseline)

    Returns excess exposure for a pollutant
    """
    data = get_pollutant_health_data(pollutant)
    baseline = data["baseline"]["value"]

    delta_c = max(0, predicted_value - baseline)

    return delta_c

def compute_relative_risk(pollutant, predicted_value):
    """
    Correct CRF model:

    RR = (rr_per_unit)^(ΔC / delta_c_unit)
    """

    data = get_pollutant_health_data(pollutant)

    rr_per_unit = data["rr_per_unit"]
    delta_c_unit = data["delta_c_unit"]
    baseline = data["baseline"]["value"]

    # ΔC (excess exposure)
    delta_c = max(0, predicted_value - baseline)

    # CRF (correct)
    rr = rr_per_unit ** (delta_c / delta_c_unit)

    return rr


def interpret_health_risk(rr, pollutant):
    """
    Epidemiologically correct interpretation using % risk increase
    """

    # Load impact data (separate dataset — correct design)
    impact_data = HEALTH_IMPACT_DATA.get(pollutant, {})
    effects = impact_data.get("short_term_effects", [])

    # % risk increase
    risk_increase = (rr - 1) * 100

    # Primary effect (safe fallback)
    primary_effect = effects[0] if effects else "health effects"

    # --- Classification + Message (linked, no mismatch) ---
    if risk_increase <= 0:
        level = "Minimal"
        message = "Air quality is within safe limits."

    elif risk_increase <= 5:
        level = "Minimal"
        message = f"Negligible increase in risk. No significant {primary_effect} expected."

    elif risk_increase <= 15:
        level = "Mild"
        message = f"Low increase in risk (~{round(risk_increase,1)}%). Mild {primary_effect} possible in sensitive groups."

    elif risk_increase <= 30:
        level = "Moderate"
        message = f"Moderate increase in risk (~{round(risk_increase,1)}%). {primary_effect} may affect general population."

    else:
        level = "Severe"
        message = f"High increase in risk (~{round(risk_increase,1)}%). Significant {primary_effect} likely across population."

    return {
        "risk_level": level,
        "risk_increase": float(risk_increase),  # tied directly to model output (not arbitrary)
        "message": message
    }
def compute_health_risk(predictions):
    """
    Full Health Engine

    Input:
    predictions = {
        "1h": {"PM2.5": ..., "PM10": ..., "NO2": ..., "CO": ..., "O3": ...},
        "3h": {...},
        "6h": {...}
    }

    Output:
    Structured health risk per horizon
    """

    results = {}

    for horizon, pollutants in predictions.items():

        # extract duration (1h, 3h, 6h)
        

        horizon_result = {
        "relative_risk": {},
        "excess_exposure": {},
        "total_relative_risk": None,   # ADD THIS
        "dominant_pollutant": None,
        "interpretation": None,
        "health_impact": None
    }

        contributions = {}

        # --- PER POLLUTANT COMPUTATION ---
        for pollutant, value in pollutants.items():

            data = get_pollutant_health_data(pollutant)

            
            baseline = data["baseline"]["value"]

            # ΔC
            delta_c = max(0, value - baseline)

           # Use stable exposure scaling
            # Controlled linear growth (scientifically better)
           

            # instantaneous exposure (correct for your dataset)
            rr = compute_relative_risk(pollutant, value)
            # Store
            horizon_result["relative_risk"][pollutant] = float(rr)
            horizon_result["excess_exposure"][pollutant] = float(delta_c)

            # contribution (use same scaled exposure)
            contributions[pollutant] = np.log(rr)

        # --- DOMINANT POLLUTANT ---
        dominant_pollutant = max(contributions, key=contributions.get)
        horizon_result["dominant_pollutant"] = dominant_pollutant

        # --- AGGREGATED RR (combined effect) ---
        # Prevent multi-pollutant explosion
        total_rr = np.exp(sum(contributions.values()))
        horizon_result["total_relative_risk"] = float(total_rr)

        horizon_result["total_risk_interpretation"] = (
        f"Combined exposure risk is {round((total_rr - 1)*100,1)}% higher than baseline"
)

        # --- INTERPRETATION ---
        dominant_rr = horizon_result["relative_risk"][dominant_pollutant]

        interpretation = interpret_health_risk(dominant_rr, dominant_pollutant)

        horizon_result["interpretation"] = interpretation

        dominant_data = get_pollutant_health_data(dominant_pollutant)

        impact_data = HEALTH_IMPACT_DATA.get(dominant_pollutant, {})

        horizon_result["health_impact"] = {
    "pollutant": dominant_pollutant,

   "short_term_effects": list(impact_data.get("short_term_effects", [])),
"long_term_effects": list(impact_data.get("long_term_effects", [])),
"sensitive_groups": list(impact_data.get("sensitive_groups", [])),
"sources": list(impact_data.get("sources", []))
}

        results[horizon] = horizon_result

    return results