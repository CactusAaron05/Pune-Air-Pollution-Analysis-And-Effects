from fastapi import FastAPI
from db import get_connection
from feature_engine import build_features, add_time_features, add_categorical_features
from model import (
    model_1h, model_3h, model_6h,
    pollutant_models,
    scaler, all_features, num_features,
    POLLUTANT_NAME_MAP   
)

from health_data import compute_health_risk

import pandas as pd
import numpy as np

app = FastAPI()

# ── CONSTANTS ───────────────────────────────────────────────────────
pollutants = ["PM2.5", "PM10", "NO2", "CO", "O3"]


# ── ALERT ENGINE (TEMP — WILL IMPROVE LATER) ───────────────────────
def generate_alerts(pred_1h, pred_3h, pred_6h):
    alerts = []

    if pred_1h > 300:
        alerts.append("Severe AQI expected in 1 hour")
    elif pred_1h > 200:
        alerts.append("Poor AQI expected in 1 hour")
    elif pred_1h > 100:
        alerts.append("Moderate AQI expected in 1 hour")

    if pred_3h > 300:
        alerts.append("Severe AQI expected in 3 hours")
    elif pred_3h > 200:
        alerts.append("Poor AQI expected in 3 hours")
    elif pred_3h > 100:
        alerts.append("Moderate AQI expected in 3 hours")

    if pred_6h > 300:
        alerts.append("Severe AQI expected in 6 hours")
    elif pred_6h > 200:
        alerts.append("Poor AQI expected in 6 hours")
    elif pred_6h > 100:
        alerts.append("Moderate AQI expected in 6 hours")

    return alerts


# ── MAIN PREDICTION API ─────────────────────────────────────────────
@app.get("/predict")
def predict(region: str):

    # ── STEP 1: DB FETCH ─────────────────────────────────────────────
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT *
        FROM aqi_data
        WHERE region = %s
        ORDER BY datetime DESC
        LIMIT 24
    """, (region,))

    rows = cursor.fetchall()

    # ── STEP 2: VALIDATION ───────────────────────────────────────────
    if len(rows) < 24:
        cursor.close()
        conn.close()
        return {"error": "Not enough data yet"}

    # reverse → oldest → latest
    rows = rows[::-1]

    # ── STEP 3: STRUCTURE DATA ──────────────────────────────────────
    rows = [
        {
            "datetime": r[1],
            "pm25": r[3],
            "pm10": r[4],
            "no2": r[5],
            "co": r[6],
            "o3": r[7],
            "area_type": r[8]
        }
        for r in rows
    ]

    latest = rows[-1]

    from datetime import datetime

    dt = latest["datetime"]
    if isinstance(dt, str):
        dt = datetime.fromisoformat(dt)

    # ── STEP 4: FEATURE ENGINEERING ──────────────────────────────────
    features = build_features(rows)

    features = add_time_features(features, dt)

    features = add_categorical_features(
        features,
        region,
        dt,
        latest["area_type"],
        all_features
    )

    # ── STEP 5: FEATURE VALIDATION ───────────────────────────────────
    missing = [col for col in all_features if col not in features]
    if missing:
        cursor.close()
        conn.close()
        return {"error": f"Missing features: {missing[:5]}"}

    # ── STEP 6: PREPARE INPUT ────────────────────────────────────────
    df = pd.DataFrame([
        {col: features.get(col, 0) for col in all_features}
    ])

    df[num_features] = scaler.transform(df[num_features].fillna(0))

    # ── STEP 7: AQI PREDICTIONS ─────────────────────────────────────
    pred_1h = float(np.clip(model_1h.predict(df), 0, 500)[0])
    pred_3h = float(np.clip(model_3h.predict(df), 0, 500)[0])
    pred_6h = float(np.clip(model_6h.predict(df), 0, 500)[0])

    # ── STEP 8: POLLUTANT PREDICTIONS ───────────────────────────────
    
    # ── STEP 8: POLLUTANT PREDICTIONS ───────────────────────────────

    

    pollutant_preds = {}

    for display_name, model_key in POLLUTANT_NAME_MAP.items():
        pollutant_preds[display_name] = {
            "1h": float(pollutant_models[model_key]["1h"].predict(df)[0]),
            "3h": float(pollutant_models[model_key]["3h"].predict(df)[0]),
            "6h": float(pollutant_models[model_key]["6h"].predict(df)[0]),
        }
    # ── STEP 8.5: RESTRUCTURE FOR HEALTH ENGINE ───────────────────────

    health_input = {
        "1h": {},
        "3h": {},
        "6h": {}
    }

    for pollutant, values in pollutant_preds.items():
        health_input["1h"][pollutant] = values["1h"]
        health_input["3h"][pollutant] = values["3h"]
        health_input["6h"][pollutant] = values["6h"]


    
    # ── STEP 9: CAUSE DETECTION (DATA-DRIVEN) ─────────────────────

    causes = {}

    for horizon in ["1h", "3h", "6h"]:

        values = {
            "PM2.5": pollutant_preds["PM2.5"][horizon],
            "PM10": pollutant_preds["PM10"][horizon],
            "NO2": pollutant_preds["NO2"][horizon],
            "CO": pollutant_preds["CO"][horizon],
            "O3": pollutant_preds["O3"][horizon],
        }

        dominant = max(values, key=values.get)

        causes[horizon] = {
            "dominant_pollutant": dominant,
            "value": values[dominant]
        }

    # ── STEP 9: ALERTS ──────────────────────────────────────────────
    alerts = generate_alerts(pred_1h, pred_3h, pred_6h)

    # ── STEP 10: CLEANUP ────────────────────────────────────────────
    cursor.close()
    conn.close()

    # ── STEP 10.5: HEALTH ENGINE ─────────────────────────────────────

    health_risk = compute_health_risk(health_input)

    # ── STEP 11: RESPONSE ───────────────────────────────────────────
    return {
    "region": region,
    "prediction_time": str(dt),

    "predicted_aqi": {
        "1h": pred_1h,
        "3h": pred_3h,
        "6h": pred_6h
    },

    "predicted_pollutants": pollutant_preds,   
    "causes": causes, 

    "health_risk": health_risk,  

    "alerts": alerts
}