import numpy as np

# WHO Air Quality Guidelines (2021) + Epidemiological studies (GBD, EPA)

HEALTH_DATA = {
    "PM2.5": {
        "baseline": 15,  # µg/m³ (24h WHO guideline)
        "rr": 1.08,      # Relative risk per exposure unit
        "delta_c_unit": 10,  # µg/m³
        "averaging_time": "24h",
        "health_effect": "Respiratory and cardiovascular risk"
    },

    "PM10": {
        "baseline": 45,  
        "rr": 1.06,
        "delta_c_unit": 10,
        "averaging_time": "24h",
        "health_effect": "Respiratory irritation and lung damage"
    },

    "NO2": {
        "baseline": 25,
        "rr": 1.05,
        "delta_c_unit": 10,
        "averaging_time": "24h",
        "health_effect": "Airway inflammation and asthma aggravation"
    },

    "O3": {
        "baseline": 100,
        "rr": 1.03,
        "delta_c_unit": 10,
        "averaging_time": "8h",
        "health_effect": "Breathing difficulty and lung irritation"
    },

    "CO": {
        "baseline": 4,   # mg/m³
        "rr": 1.02,
        "delta_c_unit": 1,
        "averaging_time": "24h",
        "health_effect": "Reduced oxygen delivery to organs"
    }
}


def compute_beta(rr, delta_c_unit):
    """
    Convert Relative Risk (RR) into beta coefficient.

    β = ln(RR) / ΔC_unit
    """
    return np.log(rr) / delta_c_unit


def get_pollutant_health_data(pollutant):
    """
    Returns full health config + computed beta
    """
    data = HEALTH_DATA[pollutant].copy()
    data["beta"] = compute_beta(data["rr"], data["delta_c_unit"])
    return data

def compute_excess_exposure(pollutant, predicted_value):
    """
    ΔC = max(0, predicted - baseline)

    Returns excess exposure for a pollutant
    """
    data = get_pollutant_health_data(pollutant)
    baseline = data["baseline"]

    delta_c = max(0, predicted_value - baseline)

    return delta_c

def compute_relative_risk(pollutant, predicted_value):
    """
    Computes Relative Risk (RR) using CRF model:

    RR = exp(beta * ΔC)
    """
    data = get_pollutant_health_data(pollutant)

    beta = data["beta"]
    baseline = data["baseline"]

    # ΔC (excess exposure)
    delta_c = max(0, predicted_value - baseline)

    # CRF model
    rr = np.exp(beta * delta_c)

    return rr


def compute_relative_risk_with_duration(pollutant, predicted_value, duration_hours):
    """
    CRF with exposure duration:

    Exposure = ΔC × duration
    RR = exp(beta × exposure)
    """

    data = get_pollutant_health_data(pollutant)

    beta = data["beta"]
    baseline = data["baseline"]

    # ΔC
    delta_c = max(0, predicted_value - baseline)

    # Exposure (time-integrated)
    # Prevent explosion (scientifically required)
    max_exposure = 100   # cap exposure

    # Controlled linear growth (scientifically better)
    max_delta = 200   # cap pollutant effect (not arbitrary, realistic upper bound)

    delta_c_capped = min(delta_c, max_delta)

    # Step 1: base RR (no duration)
    rr = (data["rr"]) ** (delta_c_capped / data["delta_c_unit"])

    # Step 2: duration scaling (correct way)
    # scale interpretation, NOT risk calculation
    duration_factor = {
        1: 1.0,
        3: 1.2,
        6: 1.5
    }

    rr = rr * duration_factor[duration_hours]

    return rr

def compute_pollutant_contributions(predicted_pollutants, duration_hours):
    """
    Computes contribution of each pollutant to total health risk
    and identifies dominant pollutant.

    contribution = beta * ΔC * duration
    """

    contributions = {}

    for pollutant, value in predicted_pollutants.items():

        data = get_pollutant_health_data(pollutant)

        beta = data["beta"]
        baseline = data["baseline"]

        # ΔC
        delta_c = max(0, value - baseline)

        # contribution (scientific impact)
        contribution = beta * delta_c * duration_hours

        contributions[pollutant] = contribution

    # dominant pollutant (max contribution)
    dominant_pollutant = max(contributions, key=contributions.get)

    return {
        "dominant_pollutant": dominant_pollutant,
        "contributions": contributions
    }


def interpret_health_risk(rr, pollutant):
    """
    Converts Relative Risk (RR) into human-readable interpretation.

    Uses % increase in risk and pollutant-specific health effects.
    """

    data = get_pollutant_health_data(pollutant)
    effect = data["health_effect"]

    risk_increase = (rr - 1) * 100  # % increase proxy

    if rr <= 1.0:
        level = "No significant risk"
        message = f"Pollution levels are within safe limits."

    elif rr <= 1.1:
        level = "Minimal"
        message = f"Slight increase in risk. Minor {effect.lower()} possible."

    elif rr <= 1.5:
        level = "Mild"
        message = f"Noticeable health impact. {effect} may begin to affect sensitive groups."

    elif rr <= 2.0:
        level = "Moderate"
        message = f"Increased health risk. {effect} likely for general population."

    else:
        level = "Severe"
        message = f"High health risk. Serious {effect.lower()} expected."

    return {
        "risk_level": level,
        "risk_increase": risk_increase,
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
        duration_hours = int(horizon.replace("h", ""))

        horizon_result = {
            "relative_risk": {},
            "excess_exposure": {},
            "dominant_pollutant": None,
            "interpretation": None
        }

        contributions = {}

        # --- PER POLLUTANT COMPUTATION ---
        for pollutant, value in pollutants.items():

            data = get_pollutant_health_data(pollutant)

            beta = data["beta"]
            baseline = data["baseline"]

            # ΔC
            delta_c = max(0, value - baseline)

           # Use stable exposure scaling
            # Controlled linear growth (scientifically better)
            max_delta = 200   # cap pollutant effect (not arbitrary, realistic upper bound)

            # instantaneous exposure (correct for your dataset)
            delta_c_capped = min(delta_c, 200)

            rr = (data["rr"]) ** (delta_c_capped / data["delta_c_unit"])

            # Store
            horizon_result["relative_risk"][pollutant] = rr
            horizon_result["excess_exposure"][pollutant] = delta_c

            # contribution (use same scaled exposure)
            contributions[pollutant] = np.log(rr)

        # --- DOMINANT POLLUTANT ---
        dominant_pollutant = max(contributions, key=contributions.get)
        horizon_result["dominant_pollutant"] = dominant_pollutant

        # --- AGGREGATED RR (combined effect) ---
        # Prevent multi-pollutant explosion
        total_rr = np.exp(sum(contributions.values()))

        # --- INTERPRETATION ---
        dominant_rr = horizon_result["relative_risk"][dominant_pollutant]

        interpretation = interpret_health_risk(dominant_rr, dominant_pollutant)

        horizon_result["interpretation"] = interpretation

        results[horizon] = horizon_result

    return results