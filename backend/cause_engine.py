import json
import os
import numpy as np

BASE_DIR = os.path.dirname(__file__)

with open(os.path.join(BASE_DIR, "data/source_patterns.json")) as f:
    SOURCE_PATTERNS = json.load(f)

with open(os.path.join(BASE_DIR, "data/source_priors.json")) as f:
    BASE_PRIORS = json.load(f)

with open(os.path.join(BASE_DIR, "data/region_profiles.json")) as f:
    REGION_PROFILES = json.load(f)


# ───────────────── NORMALIZE ─────────────────
def normalize(d):
    total = sum(d.values())
    if total == 0:
        return {k: 0 for k in d}
    return {k: v / total for k, v in d.items()}


# ───────────────── REGION-AWARE PRIORS ─────────────────
def compute_region_priors(region):

    profile = REGION_PROFILES.get(region)

    if not profile:
        return {k: v["prior"] for k, v in BASE_PRIORS.items()}

    priors = {}

    for source, base in BASE_PRIORS.items():

        score = base["prior"]

        # ── TRAFFIC ──
        if source == "traffic_emission":
            if profile["road_capacity"] == "high":
                score *= 1.3
            if profile["public_transport"] == "low":
                score *= 1.2

        # ── ROAD DUST ──
        elif source == "road_dust":
            if profile["construction_activity"] == "high":
                score *= 1.4

        # ── INDUSTRIAL ──
        elif source == "industrial":
            if "industrial" in profile["area_type"]:
                score *= 1.5

        # ── BIOMASS ──
        elif source == "biomass_burning":
            if profile["population_density"] == "high":
                score *= 1.2

        priors[source] = score

    return normalize(priors)


# ───────────────── CORE ENGINE ─────────────────
def detect_causes(predictions, rolling_baseline, region=None):

    if not predictions or not rolling_baseline:
        raise ValueError("Invalid inputs")

    # ── STEP 1: SPIKE RATIOS ──
    spike_ratios = {}

    for pollutant, value in predictions.items():
        baseline = rolling_baseline.get(pollutant)

        if baseline is None or baseline <= 0:
            continue

        spike_ratios[pollutant] = value / baseline

    if not spike_ratios:
        raise ValueError("No valid spike ratios")

    # ── STEP 2: REGION-AWARE PRIORS ──
    priors = compute_region_priors(region) if region else {
        k: v["prior"] for k, v in BASE_PRIORS.items()
    }

    # ── STEP 3: SOURCE SCORING ──
    scores = {}

    for source, pattern in SOURCE_PATTERNS.items():

        pollutants = pattern.get("pollutants", [])
        weights = pattern.get("weights", {})

        score = 0

        for p in pollutants:
            if p in spike_ratios:
                weight = weights.get(p, 0.5)
                ratio = spike_ratios[p]

                if ratio <= 1:
                    continue

                # amplify strong spikes
                signal = np.log(ratio)

                # non-linear amplification
                score += weight * (signal ** 2)
                

        scores[source] = np.exp(score) * priors.get(source, 0.1)

    probs = normalize(scores)

    ranked = sorted(probs.items(), key=lambda x: x[1], reverse=True)

    if len(ranked) < 2:
        raise ValueError("Insufficient sources")

    return {
        "primary_source": {
            "source": ranked[0][0],
            "confidence": float(ranked[0][1])
        },
        "secondary_source": {
            "source": ranked[1][0],
            "confidence": float(ranked[1][1])
        }
    }