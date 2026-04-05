import joblib
import xgboost as xgb
import os

# ── STEP 1: BASE PATH (ROBUST) ─────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")

print("MODEL DIR:", MODEL_DIR)

# ── STEP 2: FILE PATHS ─────────────────────────────────────────────
model_1h_path = os.path.join(MODEL_DIR, "aqi_model_1h.json")
model_3h_path = os.path.join(MODEL_DIR, "aqi_model_3h.json")
model_6h_path = os.path.join(MODEL_DIR, "aqi_model_6h.json")

scaler_path = os.path.join(MODEL_DIR, "feature_scaler.pkl")
features_path = os.path.join(MODEL_DIR, "feature_columns.pkl")
num_features_path = os.path.join(MODEL_DIR, "numeric_feature_columns.pkl")

# ── STEP 3: VALIDATE REQUIRED FILES ────────────────────────────────
required_files = [
    model_1h_path,
    model_3h_path,
    model_6h_path,
    scaler_path,
    features_path,
    num_features_path
]

for f in required_files:
    if not os.path.exists(f):
        raise FileNotFoundError(f"Missing file: {f}")

# ── STEP 4: LOAD AQI MODELS ────────────────────────────────────────
model_1h = xgb.XGBRegressor()
model_1h.load_model(model_1h_path)

model_3h = xgb.XGBRegressor()
model_3h.load_model(model_3h_path)

model_6h = xgb.XGBRegressor()
model_6h.load_model(model_6h_path)

# ── STEP 5: LOAD ARTIFACTS ─────────────────────────────────────────
scaler = joblib.load(scaler_path)
all_features = joblib.load(features_path)
num_features = joblib.load(num_features_path)

print("✅ AQI models and artifacts loaded successfully")

# ── STEP 6: LOAD POLLUTANT MODELS ──────────────────────────────────
pollutant_models = {}

# INTERNAL KEYS (MODEL FILE NAMING)
model_pollutants = ["PM25", "PM10", "NO2", "CO", "O3"]
horizons = ["1h", "3h", "6h"]

for p in model_pollutants:
    pollutant_models[p] = {}

    for h in horizons:
        model_path = os.path.join(MODEL_DIR, f"{p}_model_{h}.json")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Missing pollutant model: {model_path}")

        model = xgb.XGBRegressor()
        model.load_model(model_path)

        pollutant_models[p][h] = model

print("✅ Pollutant models loaded successfully")

# ── STEP 7: DISPLAY NAME MAPPING (CRITICAL) ─────────────────────────
# This ensures consistency across:
# - API
# - Health Engine
# - Cause Detection
# - Dataset

POLLUTANT_NAME_MAP = {
    "PM2.5": "PM25",
    "PM10": "PM10",
    "NO2": "NO2",
    "CO": "CO",
    "O3": "O3"
}

# Reverse mapping (optional but useful)
REVERSE_POLLUTANT_MAP = {v: k for k, v in POLLUTANT_NAME_MAP.items()}

print("✅ Pollutant name mapping ready")