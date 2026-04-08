import json
import os

BASE_DIR = os.path.dirname(__file__)

LOG_PATH = os.path.join(BASE_DIR, "data", "action_effectiveness_log.json")
CONFIG_PATH = os.path.join(BASE_DIR, "data", "learning_config.json")


def load_json(path):
    with open(path) as f:
        return json.load(f)


def validate_key(data, key, context):
    if key not in data:
        raise ValueError(f"{context} missing key: {key}")
    return data[key]


LOG = load_json(LOG_PATH) if os.path.exists(LOG_PATH) else {}
CONFIG = load_json(CONFIG_PATH)


def compute_reduction(record):

    before_key = validate_key(CONFIG["value_source"], "before_key", "learning_config")
    after_key = validate_key(CONFIG["value_source"], "after_key", "learning_config")

    before = validate_key(record, before_key, "log_record")
    after = validate_key(record, after_key, "log_record")

    if before <= 0:
        return None

    return (before - after) / before


def aggregate(values):

    method = validate_key(CONFIG, "aggregation_method", "learning_config")

    valid = [v for v in values if v is not None]

    if not valid:
        return None

    if method == "mean":
        return sum(valid) / len(valid)

    raise ValueError(f"Unsupported aggregation method: {method}")


def get_learned_effectiveness(source, action):

    if source not in LOG:
        return None

    if action not in LOG[source]:
        return None

    records = LOG[source][action]

    min_samples = validate_key(CONFIG, "minimum_samples_required", "learning_config")

    if len(records) < min_samples:
        return None

    reductions = [compute_reduction(r) for r in records]

    return aggregate(reductions)