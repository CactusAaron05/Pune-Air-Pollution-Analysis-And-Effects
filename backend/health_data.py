import json
import os
import numpy as np

# ─────────────────────────────────────────────
# LOAD DATASETS
# ─────────────────────────────────────────────

HEALTH_IMPACT_PATH = os.path.join(
    os.path.dirname(__file__),
    "data",
    "health_impact_dataset.json"
)

with open(HEALTH_IMPACT_PATH, "r") as f:
    HEALTH_IMPACT_DATA = json.load(f)

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "health_dataset.json")

with open(DATA_PATH, "r") as f:
    HEALTH_DATA = json.load(f)


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def get_pollutant_health_data(pollutant):
    return HEALTH_DATA[pollutant].copy()


def compute_excess_exposure(pollutant, predicted_value):
    data = get_pollutant_health_data(pollutant)
    baseline = data["baseline"]["value"]
    return max(0, predicted_value - baseline)


def compute_relative_risk(pollutant, predicted_value):
    data = get_pollutant_health_data(pollutant)

    rr_per_unit = data["rr_per_unit"]
    delta_c_unit = data["delta_c_unit"]
    baseline = data["baseline"]["value"]

    delta_c = max(0, predicted_value - baseline)

    return rr_per_unit ** (delta_c / delta_c_unit)


def interpret_health_risk(rr, pollutant):
    impact_data = HEALTH_IMPACT_DATA.get(pollutant, {})
    effects = impact_data.get("short_term_effects", [])

    risk_increase = (rr - 1) * 100
    primary_effect = effects[0] if effects else "health effects"

    if risk_increase <= 0:
        level = "Minimal"
        message = "Air quality is within safe limits."

    elif risk_increase <= 5:
        level = "Minimal"
        message = f"Negligible increase in risk. No significant {primary_effect} expected."

    elif risk_increase <= 20:
        level = "Mild"
        message = f"Low increase in risk (~{round(risk_increase,1)}%). Mild {primary_effect} possible in sensitive groups."

    elif risk_increase <= 40:
        level = "Moderate"
        message = f"Moderate increase in risk (~{round(risk_increase,1)}%). {primary_effect} may affect general population."

    else:
        level = "Severe"
        message = f"High increase in risk (~{round(risk_increase,1)}%). Significant {primary_effect} likely across population."

    return {
        "risk_level": level,
        "risk_increase": float(risk_increase),
        "message": message
    }


# ─────────────────────────────────────────────
# MAIN ENGINE
# ─────────────────────────────────────────────

def compute_health_risk(predictions):

    results = {}

    for horizon, pollutants in predictions.items():

        duration = int(horizon.replace("h", ""))

        horizon_result = {
            "relative_risk": {},
            "excess_exposure": {},
            "total_relative_risk": None,
            "pollutant_contribution": None,
            "dominant_pollutant": None,
            "interpretation": None,
            "health_impact": None
        }

        contributions = {}

        # ── PER POLLUTANT ──
        for pollutant, value in pollutants.items():

            data = get_pollutant_health_data(pollutant)
            baseline = data["baseline"]["value"]

            delta_c = max(0, value - baseline)

            base_rr = compute_relative_risk(pollutant, value)

            # duration scaling (log-linear)
            rr = np.exp(np.log(base_rr) * (duration / 24))

            horizon_result["relative_risk"][pollutant] = float(rr)
            horizon_result["excess_exposure"][pollutant] = float(delta_c)

            # interaction damping (PM overlap)
            weight = 0.6 if pollutant in ["PM2.5", "PM10"] else 1.0

            contributions[pollutant] = weight * np.log(rr)

        # ── TOTAL RISK ──
        total_log = sum(contributions.values())
        total_rr = np.exp(total_log)

        horizon_result["total_relative_risk"] = float(total_rr)

        # ── CONTRIBUTION (%) ──
        pollutant_contribution = {}

        if total_log <= 0:
            for p in contributions:
                pollutant_contribution[p] = 0.0
        else:
            for p, val in contributions.items():
                pollutant_contribution[p] = float((val / total_log) * 100)

        horizon_result["pollutant_contribution"] = pollutant_contribution

        # ── DOMINANT POLLUTANT ──
        dominant_pollutant = max(
            pollutant_contribution,
            key=pollutant_contribution.get
        )
        horizon_result["dominant_pollutant"] = dominant_pollutant

        # ── INTERPRETATION (TOTAL RISK) ──
        interpretation = interpret_health_risk(total_rr, dominant_pollutant)
        horizon_result["interpretation"] = interpretation

        horizon_result["total_risk_interpretation"] = (
            f"Combined exposure risk is {round((total_rr - 1)*100,1)}% higher than baseline"
        )

        # ── MULTI-POLLUTANT HEALTH IMPACT ──
        significant_pollutants = [
            p for p, v in pollutant_contribution.items() if v >= 10
        ]

        short_term_effects = set()
        long_term_effects = set()
        sensitive_groups = set()
        sources = set()

        for p in significant_pollutants:
            impact_data = HEALTH_IMPACT_DATA.get(p, {})

            short_term_effects.update(impact_data.get("short_term_effects", []))
            long_term_effects.update(impact_data.get("long_term_effects", []))
            sensitive_groups.update(impact_data.get("sensitive_groups", []))
            sources.update(impact_data.get("sources", []))

        horizon_result["health_impact"] = {
            "affected_pollutants": significant_pollutants,
            "short_term_effects": list(short_term_effects),
            "long_term_effects": list(long_term_effects),
            "sensitive_groups": list(sensitive_groups),
            "sources": list(sources)
        }

        results[horizon] = horizon_result

    return results