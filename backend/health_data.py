import json
import os
import numpy as np

# ───────────────── LOAD DATA ─────────────────
BASE_DIR = os.path.dirname(__file__)

with open(os.path.join(BASE_DIR, "data/health_dataset.json")) as f:
    HEALTH_DATA = json.load(f)

with open(os.path.join(BASE_DIR, "data/health_impact_dataset.json")) as f:
    HEALTH_IMPACT_DATA = json.load(f)


# ───────────────── HELPERS ─────────────────
def get_data(pollutant):
    if pollutant not in HEALTH_DATA:
        raise ValueError(f"Missing pollutant data: {pollutant}")
    return HEALTH_DATA[pollutant]


def compute_rr(pollutant, value):
    data = get_data(pollutant)

    baseline = data["baseline"]["value"]
    rr_per_unit = data["rr_per_unit"]
    delta_unit = data["delta_c_unit"]

    delta = max(0, value - baseline)

    return rr_per_unit ** (delta / delta_unit)


def compute_exposure_weight(delta):
    """
    Normalize exposure contribution
    Prevents domination by single pollutant
    """
    return np.log1p(delta)


def interpret(total_rr):

    # ── STEP 1: LOG-SCALED RISK ──
    raw_increase = max(0, total_rr - 1)

    # logarithmic scaling (prevents exaggeration)
    scaled = np.log1p(raw_increase) * 100

    increase = float(scaled)

    # ── STEP 2: LEVEL CLASSIFICATION ──
    if increase <= 5:
        level = "Minimal"
    elif increase <= 15:
        level = "Mild"
    elif increase <= 35:
        level = "Moderate"
    else:
        level = "Severe"

    return level, increase


# ───────────────── CORE ENGINE ─────────────────
def compute_health_risk(predictions):

    results = {}

    for horizon, pollutants in predictions.items():

        rr_values = {}
        deltas = {}
        weights = {}

        # ── STEP 1: INDIVIDUAL RR ──
        for p, value in pollutants.items():

            data = get_data(p)
            baseline = data["baseline"]["value"]

            delta = max(0, value - baseline)

            rr = compute_rr(p, value)

            rr_values[p] = float(rr)
            deltas[p] = delta

        # ── STEP 2: NORMALIZED WEIGHTS ──
        raw_weights = {p: compute_exposure_weight(d) for p, d in deltas.items()}

        total_weight = sum(raw_weights.values())

        if total_weight == 0:
            weights = {p: 0 for p in raw_weights}
        else:
            weights = {p: w / total_weight for p, w in raw_weights.items()}

        # ── STEP 3: COMBINE RISKS ──
        log_total = 0

        for p in rr_values:
            log_total += weights[p] * np.log(rr_values[p])

        total_rr = float(np.exp(log_total))

        # ── STEP 4: CONTRIBUTION ──
        contribution = {}

        if log_total <= 0:
            contribution = {p: 0 for p in rr_values}
        else:
            for p in rr_values:
                contribution[p] = float(
                    (weights[p] * np.log(rr_values[p])) / log_total * 100
                )

        # ── STEP 5: DOMINANT ──
        dominant = max(contribution, key=contribution.get)

        # ── STEP 6: INTERPRET ──
        level, increase = interpret(total_rr)

        # ── STEP 7: HEALTH IMPACT ──
        impact = HEALTH_IMPACT_DATA.get(dominant, {})

        results[horizon] = {
            "relative_risk": rr_values,
            "excess_exposure": deltas,
            "total_relative_risk": total_rr,
            "pollutant_contribution": contribution,
            "dominant_pollutant": dominant,
            "interpretation": {
                "risk_level": level,
                "risk_increase": increase,
                "message": f"Health risk increased by {round(increase,1)}%"
            },
            "health_impact": {
                "short_term_effects": impact.get("short_term_effects", []),
                "long_term_effects": impact.get("long_term_effects", []),
                "sensitive_groups": impact.get("sensitive_groups", []),
                "sources": impact.get("sources", [])
            }
        }

    return results