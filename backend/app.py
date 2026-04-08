from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from db import get_connection

from model import (
    model_1h, model_3h, model_6h,
    pollutant_models,
    scaler, all_features, num_features,
    POLLUTANT_NAME_MAP
)

from feature_engine import (
    build_features,
    add_time_features,
    add_categorical_features
)

from cause_engine import detect_causes
from health_data import compute_health_risk
from solution_engine import generate_solutions

import pandas as pd
import numpy as np
import json
import os

# ───────────────── APP INIT ─────────────────
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(__file__)

def load_json(path):
    with open(os.path.join(BASE_DIR, path)) as f:
        return json.load(f)

ALERT_CONFIG = load_json("data/alert_config.json")


# ───────────────── ALERT ENGINE ─────────────────
def generate_alerts(pred_1h, pred_3h, pred_6h, health_risk):

    alerts = []

    # ── STEP 1: TREND DIRECTION ──
    increasing = pred_1h < pred_3h < pred_6h
    decreasing = pred_1h > pred_3h > pred_6h

    # ── STEP 2: RELATIVE CHANGE ──
    def pct_change(a, b):
        if a == 0:
            return 0
        return (b - a) / a

    change_1_3 = pct_change(pred_1h, pred_3h)
    change_3_6 = pct_change(pred_3h, pred_6h)

    sharp_increase = change_1_3 > 0.15 or change_3_6 > 0.15

    # ── STEP 3: HEALTH SEVERITY ──
    severity_map = ALERT_CONFIG["severity_levels"]

    sev_1h = severity_map[health_risk["1h"]["interpretation"]["risk_level"]]
    sev_3h = severity_map[health_risk["3h"]["interpretation"]["risk_level"]]
    sev_6h = severity_map[health_risk["6h"]["interpretation"]["risk_level"]]

    high_risk = sev_6h >= ALERT_CONFIG["thresholds"]["health_severity_threshold"]

    dominant_pollutant = health_risk["1h"]["dominant_pollutant"]

    # ── STEP 4: ALERT LOGIC ──

    # 🚨 Case 1: Worsening + high risk
    if increasing and high_risk:
        alerts.append("Severe pollution expected to worsen — immediate precautions required")

    # ⚠️ Case 2: Improving but still dangerous
    elif decreasing and high_risk:
        alerts.append("Air quality is improving but remains at hazardous levels")

    # ⚠️ Case 3: Stable high risk
    elif high_risk:
        alerts.append("Air quality remains unhealthy — limit exposure")

    # 📈 Case 4: Rapid spike (even if not severe yet)
    if sharp_increase:
        alerts.append("Rapid increase in pollution levels detected")

    # 🎯 Case 5: Pollutant-specific alert
    if dominant_pollutant in ["PM10", "PM2.5"]:
        alerts.append(f"High particulate matter levels ({dominant_pollutant}) detected")

    elif dominant_pollutant == "NO2":
        alerts.append("Elevated traffic-related pollution detected")

    elif dominant_pollutant == "O3":
        alerts.append("Elevated ozone levels detected")

    # ── STEP 5: REMOVE DUPLICATES ──
    alerts = list(dict.fromkeys(alerts))

    return alerts


# ───────────────── CORE PIPELINE ─────────────────
def run_full_pipeline(region: str):

    if not region:
        return {"error": "Region is required"}

    conn = get_connection()
    cursor = conn.cursor()

    # ── STEP 1: FETCH DATA ─────────────────
    cursor.execute("""
        SELECT *
        FROM aqi_data
        WHERE region = %s
        ORDER BY datetime DESC
        LIMIT 80
    """, (region,))

    rows = cursor.fetchall()

    if len(rows) < 30:
        cursor.close()
        conn.close()
        return {"error": "Not enough historical data"}

    rows = rows[::-1]

    # ── STEP 2: STRUCTURE DATA ─────────────────
    df_hist = pd.DataFrame([{
        "Region": r[2],
        "Datetime_IST": r[1],
        "PM2.5 (µg/m³)": r[3],
        "PM10 (µg/m³)": r[4],
        "NO2 (µg/m³)": r[5],
        "CO (mg/m³)": r[6],
        "Ozone (µg/m³)": r[7],
        "Area_Type": r[8]
    } for r in rows])

    df_hist["Datetime_IST"] = pd.to_datetime(df_hist["Datetime_IST"])

    latest_dt = df_hist.iloc[-1]["Datetime_IST"]

    # ── STEP 3: FEATURE ENGINE ─────────────────
    features = build_features(df_hist)
    features = add_time_features(features, latest_dt)
    features = add_categorical_features(features, df_hist, all_features)

    # ── STEP 4: VALIDATION ─────────────────
    missing = [col for col in all_features if col not in features]

    if missing:
        cursor.close()
        conn.close()
        return {"error": f"Missing features: {missing[:5]}"}

    df_model = pd.DataFrame([
        {col: features.get(col, 0) for col in all_features}
    ])

    # STRICT ORDER CHECK
    assert list(df_model.columns) == list(all_features)

    df_model[num_features] = scaler.transform(df_model[num_features].fillna(0))

    # ── STEP 5: AQI PREDICTION ─────────────────
    pred_1h = float(np.clip(model_1h.predict(df_model)[0], 0, 500))
    pred_3h = float(np.clip(model_3h.predict(df_model)[0], 0, 500))
    pred_6h = float(np.clip(model_6h.predict(df_model)[0], 0, 500))

    predictions = {
        "1h": pred_1h,
        "3h": pred_3h,
        "6h": pred_6h
    }

    # ── STEP 6: POLLUTANT PREDICTION ─────────────────
    pollutant_preds = {}

    for display_name, model_key in POLLUTANT_NAME_MAP.items():
        pollutant_preds[display_name] = {
            "1h": float(pollutant_models[model_key]["1h"].predict(df_model)[0]),
            "3h": float(pollutant_models[model_key]["3h"].predict(df_model)[0]),
            "6h": float(pollutant_models[model_key]["6h"].predict(df_model)[0]),
        }

    # ── STEP 7: CAUSE DETECTION ─────────────────
    causes = {}

    for horizon in ["1h", "3h", "6h"]:

        values = {p: pollutant_preds[p][horizon] for p in pollutant_preds}

        baseline = {
            "PM2.5": features.get("PM25_roll6h_mean", values["PM2.5"]),
            "PM10": features.get("PM10_roll6h_mean", values["PM10"]),
            "NO2": features.get("NO2 (µg/m³)", values["NO2"]),
            "CO": features.get("CO (mg/m³)", values["CO"]),
            "O3": features.get("Ozone (µg/m³)", values["O3"]),
        }

        causes[horizon] = detect_causes(values, baseline, region)

    # ── STEP 8: HEALTH ENGINE ─────────────────
    health_input = {h: {} for h in ["1h", "3h", "6h"]}

    for p in pollutant_preds:
        for h in health_input:
            health_input[h][p] = pollutant_preds[p][h]

    health_risk = compute_health_risk(health_input)

    # ── STEP 9: SOLUTIONS ─────────────────
    solutions = generate_solutions(region, predictions, causes, health_risk)

    # ── STEP 10: ALERTS ─────────────────
    alerts = generate_alerts(pred_1h, pred_3h, pred_6h, health_risk)

    cursor.close()
    conn.close()

    # ── FINAL RESPONSE ─────────────────
    return {
        "region": region,
        "prediction_time": str(latest_dt),
        "predicted_aqi": predictions,
        "predicted_pollutants": pollutant_preds,
        "causes": causes,
        "health_risk": health_risk,
        "alerts": alerts,
        "solutions": solutions
    }


# ───────────────── ENDPOINTS ─────────────────

@app.get("/predict")
def predict(region: str):
    return JSONResponse(content=run_full_pipeline(region))


@app.get("/dashboard")
def dashboard(region: str):
    return JSONResponse(content=run_full_pipeline(region))


@app.get("/aqi/forecast")
def get_forecast(region: str):
    result = run_full_pipeline(region)
    return {
        "region": result["region"],
        "prediction_time": result["prediction_time"],
        "predicted_aqi": result["predicted_aqi"]
    }


@app.get("/pollutants")
def get_pollutants(region: str):
    return run_full_pipeline(region)["predicted_pollutants"]


@app.get("/causes")
def get_causes(region: str):
    return run_full_pipeline(region)["causes"]


@app.get("/health")
def get_health(region: str):
    return run_full_pipeline(region)["health_risk"]


@app.get("/solutions")
def get_solutions(region: str):
    return run_full_pipeline(region)["solutions"]


@app.get("/alerts")
def get_alerts(region: str):
    return run_full_pipeline(region)["alerts"]