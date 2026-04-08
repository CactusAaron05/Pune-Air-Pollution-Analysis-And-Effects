import json
import os
import math

BASE_DIR = os.path.dirname(__file__)

CALIB_PATH = os.path.join(BASE_DIR, "data", "confidence_calibration.json")


def load_calibration():
    with open(CALIB_PATH) as f:
        return json.load(f)


CALIBRATION = load_calibration()


def validate_key(data, key, context):
    if key not in data:
        raise ValueError(f"{context} missing key: {key}")
    return data[key]


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def calibrate_confidence(source, raw_score):

    source_params = validate_key(CALIBRATION, source, "calibration")

    a = validate_key(source_params, "a", "calibration_params")
    b = validate_key(source_params, "b", "calibration_params")

    return sigmoid(a * raw_score + b)