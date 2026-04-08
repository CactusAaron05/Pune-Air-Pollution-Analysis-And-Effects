import json
import os
from learning_engine import get_learned_effectiveness

BASE_DIR = os.path.dirname(__file__)

# ── LOAD CONFIGS ───────────────────────────────────────────────

def load_json(path):
    with open(path) as f:
        return json.load(f)


SOLUTIONS = load_json(os.path.join(BASE_DIR, "data", "solutions_dataset.json"))
REGION_PROFILES = load_json(os.path.join(BASE_DIR, "data", "region_profiles.json"))
CONFIG = load_json(os.path.join(BASE_DIR, "data", "system_config.json"))
TIME_STRATEGY = load_json(os.path.join(BASE_DIR, "data", "time_strategy.json"))
LONG_TERM_POLICY = load_json(os.path.join(BASE_DIR, "data", "long_term_policy.json"))
POLLUTANT_WEIGHTS = load_json(os.path.join(BASE_DIR, "data", "pollutant_weights.json"))
SOURCE_PATTERNS = load_json(os.path.join(BASE_DIR, "data", "source_patterns.json"))


# ── VALIDATION LAYER (NO FALLBACKS) ─────────────────────────────

def validate_key(data, key, context):
    if key not in data:
        raise ValueError(f"{context} missing key: {key}")
    return data[key]


# ── CORE HELPERS ───────────────────────────────────────────────

def get_peak_horizon(predictions):
    return max(predictions, key=predictions.get)


def is_feasible(action, region_profile):
    constraints = action.get("constraints", {})

    for key, allowed_values in constraints.items():
        if validate_key(region_profile, key, "region_profile") not in allowed_values:
            return False

    return True

def compute_feasibility_score(action, region_profile):

    constraints = action.get("constraints", {})

    if not constraints:
        return 1.0

    total = 0
    matched = 0

    for key, allowed_values in constraints.items():

        region_value = validate_key(region_profile, key, "region_profile")

        total += 1

        if region_value in allowed_values:
            matched += 1
        else:
            matched += 0.5   # partial compatibility

    return matched / total


def compute_score(effectiveness, confidence, severity_weight, pollutant_weight, feasibility_score):

    impact = effectiveness * pollutant_weight

    context_factor = (0.6 * feasibility_score) + (0.4 * confidence)

    return impact * context_factor * severity_weight


def should_include_long_term(primary, secondary, risk_level):

    conditions = validate_key(LONG_TERM_POLICY, "conditions", "long_term_policy")

    severity_levels = validate_key(conditions, "severity_levels", "long_term_policy")
    min_conf = validate_key(conditions, "min_confidence", "long_term_policy")
    require_secondary = validate_key(conditions, "require_secondary", "long_term_policy")

    # Severity must justify structural intervention
    if risk_level not in severity_levels:
        return False

    # Use combined confidence instead of only primary
    total_confidence = primary["confidence"]
    if secondary:
        total_confidence += secondary["confidence"]

    if total_confidence < min_conf:
        return False

    if require_secondary and not secondary:
        return False

    return True


# ── MAIN ENGINE ────────────────────────────────────────────────

def generate_solutions(region, predictions, causes, health_risk):

    # ── Validate inputs ──
    if not predictions or not isinstance(predictions, dict):
        raise ValueError("Invalid predictions input")

    region_profile = validate_key(REGION_PROFILES, region, "REGION_PROFILES")

    peak = get_peak_horizon(predictions)

    cause_block = validate_key(causes, peak, "causes")

    primary = validate_key(cause_block, "primary_source", "cause_block")
    secondary = cause_block.get("secondary_source")

    risk_block = validate_key(health_risk, peak, "health_risk")

    interpretation = validate_key(risk_block, "interpretation", "health_risk")
    risk_level = validate_key(interpretation, "risk_level", "health_risk")
    severity_factor_map = {
    "Minimal": 0.5,
    "Mild": 0.8,
    "Moderate": 1.0,
    "Severe": 1.3
}

    severity_factor = severity_factor_map.get(risk_level, 1.0)
    severity_weight = validate_key(CONFIG["severity_weights"], risk_level, "severity_weights")

    relative_risks = validate_key(risk_block, "relative_risk", "health_risk")

    pollutant_weights_map = {}

    for pollutant, rr in relative_risks.items():

        weight = validate_key(
            validate_key(POLLUTANT_WEIGHTS, pollutant, "pollutant_weights"),
            "toxicity_weight",
            "pollutant_weights"
        )

        pollutant_weights_map[pollutant] = rr * weight
    

    allowed_types = validate_key(
        validate_key(TIME_STRATEGY, peak, "time_strategy"),
        "allowed_action_types",
        "time_strategy"
    )

    include_long_term = should_include_long_term(primary, secondary, risk_level)

    candidate_actions = []

    # ── PROCESS SOURCE ──

    def compute_pollutant_alignment(source, pollutant_weights_map):

        if source not in SOURCE_PATTERNS:
            return 1.0

        pollutants = validate_key(SOURCE_PATTERNS[source], "pollutants", "source_patterns")

        score = 0
        total = 0

        for p, weight in pollutant_weights_map.items():
            total += weight
            if p in pollutants:
                score += weight

        if total == 0:
            return 1.0

        return score / total

    def process_source(source_obj):

        if not source_obj:
            return

        source = validate_key(source_obj, "source", "source_obj")
        confidence = validate_key(source_obj, "confidence", "source_obj")

        pollutant_alignment = compute_pollutant_alignment(source, pollutant_weights_map)
        # Assign weight based on rank (primary vs secondary)
        source_weight = 1.0 if source_obj == primary else 0.7
        if source not in SOLUTIONS:
            raise ValueError(f"Missing solutions for source: {source}")

        source_data = SOLUTIONS[source]

        for action_type in ["short_term", "long_term"]:

            action_list = validate_key(source_data, action_type, "solutions")

            for action in action_list:

                feasibility_score = compute_feasibility_score(action, region_profile)

                # ── Learned effectiveness ──
                learned = get_learned_effectiveness(source, action["action"])

                selection_policy = validate_key(CONFIG, "learning_selection_policy", "system_config")

                use_learned = validate_key(
                    selection_policy,
                    "use_learned_when_available",
                    "learning_selection_policy"
                )

                effectiveness_value = (
                    learned if (use_learned and learned is not None)
                    else action["effectiveness"]
                )

                score = compute_score(
    effectiveness_value,
    confidence,
    severity_weight,
    1.0,
    feasibility_score
) * source_weight * pollutant_alignment * severity_factor

                candidate_actions.append({
                    "type": "short" if action_type == "short_term" else "long",
                    "action": action["action"],
                    "score": score,
                    "source": source
                })

    # ── Run for primary + secondary ──
    process_source(primary)
    process_source(secondary)

    # ── Ranking ──
    ranked = sorted(candidate_actions, key=lambda x: x["score"], reverse=True)

    selection_policy = validate_key(CONFIG, "selection_policy", "system_config")

    max_short = validate_key(selection_policy, "max_short_term", "selection_policy")
    max_long = validate_key(selection_policy, "max_long_term", "selection_policy")

    short_term = []
    long_term = []
    explanation = []

    seen_actions = set()
    seen_sources = {}

    for item in ranked:

        source = item["source"]
        action = item["action"]
        action_type = item["type"]

        source_count = seen_sources.get(source, 0)

        # ── 1. Avoid duplicate actions ──
        if action in seen_actions:
            continue

        # ── 2. Limit domination from one source ──
        if source_count >= 3:
            continue

        # ── 3. Time + policy filtering ──
        if action_type == "short":
            if "short" not in allowed_types:
                continue

        elif action_type == "long":
            if not include_long_term:
                continue

        # ── 4. Selection ──
        if action_type == "short" and len(short_term) < max_short:
            short_term.append(action)

        elif action_type == "long" and len(long_term) < max_long:
            long_term.append(action)

        else:
            continue

    # ── 5. Track usage ──
        seen_actions.add(action)
        seen_sources[source] = source_count + 1

        # ── 6. Explanation ──
        explanation.append(
            f"{action} | source={source} | score={round(item['score'], 3)}"
        )

    # ── 7. Stop condition ──
        if len(short_term) >= max_short and len(long_term) >= max_long:
            break

    return {
        "where": region,
        "when": f"next {peak}",
        "why": f"{primary['source']} + {secondary['source']}" if secondary else primary["source"],
        "short_term": short_term,
        "long_term": long_term,
        "explanation": explanation
    }