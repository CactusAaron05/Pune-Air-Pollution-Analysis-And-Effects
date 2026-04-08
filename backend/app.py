# from fastapi import FastAPI
# from fastapi.responses import JSONResponse

# from db import get_connection

# from model import (
#     pollutant_models,
#     scaler, all_features, num_features,
#     POLLUTANT_NAME_MAP
# )

# from uncertainty_engine import simulate_predictions, compute_uncertainty_range
# from cause_engine import detect_causes
# from health_data import compute_health_risk
# from solution_engine import generate_solutions

# import pandas as pd
# import numpy as np
# import json
# from datetime import datetime
# import os

# app = FastAPI()

# BASE_DIR = os.path.dirname(__file__)

# def load_json(path):
#     with open(os.path.join(BASE_DIR, path)) as f:
#         return json.load(f)

# ALERT_CONFIG = load_json("data/alert_config.json")


# # ───────────────── ALERT ENGINE ─────────────────
# def generate_alerts(pred_1h, pred_3h, pred_6h, health_risk):

#     alerts = []

#     delta_1_3 = pred_3h - pred_1h
#     delta_3_6 = pred_6h - pred_3h

#     severity_map = ALERT_CONFIG["severity_levels"]
#     threshold = ALERT_CONFIG["thresholds"]["rapid_change"]

#     severity_3h = severity_map[health_risk["3h"]["interpretation"]["risk_level"]]
#     severity_6h = severity_map[health_risk["6h"]["interpretation"]["risk_level"]]

#     if delta_1_3 > threshold or delta_3_6 > threshold:
#         alerts.append("Rapid deterioration in air quality expected")

#     if severity_3h >= ALERT_CONFIG["thresholds"]["health_severity_threshold"]:
#         alerts.append("Elevated health risk expected")

#     if pred_6h > pred_3h > pred_1h:
#         alerts.append("Air quality continuously worsening")

#     return alerts


# # ───────────────── MAIN API ─────────────────
# @app.get("/predict")
# def predict(region: str):

#     if not region:
#         return {"error": "Region required"}

#     conn = get_connection()
#     cursor = conn.cursor()

#     # 🔥 STEP 1: GET PRECOMPUTED AQI
#     cursor.execute("""
#         SELECT timestamp, pred_1h, pred_3h, pred_6h
#         FROM predictions
#         WHERE region = %s
#         ORDER BY timestamp DESC
#         LIMIT 1
#     """, (region,))

#     row = cursor.fetchone()

#     if not row:
#         return {"error": "No predictions available yet"}

#     timestamp, pred_1h, pred_3h, pred_6h = row

#     predictions = {
#         "1h": float(pred_1h),
#         "3h": float(pred_3h),
#         "6h": float(pred_6h)
#     }

#     # 🔥 STEP 2: FETCH RECENT DATA FOR FEATURE CONTEXT
#     cursor.execute("""
#         SELECT *
#         FROM aqi_data
#         WHERE region = %s
#         ORDER BY datetime DESC
#         LIMIT 80
#     """, (region,))

#     rows = cursor.fetchall()

#     if len(rows) < 10:
#         return {"error": "Not enough data for analysis"}

#     rows = rows[::-1]

#     df = pd.DataFrame([{
#         "Datetime_IST": r[1],
#         "PM2.5 (µg/m³)": r[3],
#         "PM10 (µg/m³)": r[4],
#         "NO2 (µg/m³)": r[5],
#         "CO (mg/m³)": r[6],
#         "Ozone (µg/m³)": r[7],
#         "Area_Type": r[8],
#         "Region": r[2]
#     } for r in rows])

#     df["Datetime_IST"] = pd.to_datetime(df["Datetime_IST"])

#     # ───── BASIC FEATURES (only what is needed) ─────
#     df["PM25_roll6h_mean"] = df["PM2.5 (µg/m³)"].rolling(6).mean()
#     df["PM10_roll6h_mean"] = df["PM10 (µg/m³)"].rolling(6).mean()

#     df = df.fillna(method="bfill")
#     latest = df.iloc[-1]

#     # ───── PREPARE MODEL INPUT ─────
#     df_model = pd.get_dummies(df, columns=["Area_Type", "Region"], drop_first=True)

#     for col in all_features:
#         if col not in df_model.columns:
#             df_model[col] = 0

#     df_model = df_model[all_features]
#     df_model[num_features] = scaler.transform(df_model[num_features])

#     df_model = df_model.tail(1)

#     # ───── POLLUTANT PREDICTIONS ─────
#     pollutant_preds = {}

#     for name, key in POLLUTANT_NAME_MAP.items():
#         pollutant_preds[name] = {
#             "1h": float(pollutant_models[key]["1h"].predict(df_model)[0]),
#             "3h": float(pollutant_models[key]["3h"].predict(df_model)[0]),
#             "6h": float(pollutant_models[key]["6h"].predict(df_model)[0]),
#         }

#     # ───── CAUSES ─────
#     causes = {}
#     for h in ["1h","3h","6h"]:
#         vals = {p: pollutant_preds[p][h] for p in pollutant_preds}

#         baseline = {
#             "PM2.5": latest["PM25_roll6h_mean"],
#             "PM10": latest["PM10_roll6h_mean"],
#             "NO2": latest["NO2 (µg/m³)"],
#             "CO": latest["CO (mg/m³)"],
#             "O3": latest["Ozone (µg/m³)"],
#         }

#         causes[h] = detect_causes(vals, baseline)

#     # ───── HEALTH ─────
#     health_input = {h:{} for h in ["1h","3h","6h"]}

#     for p in pollutant_preds:
#         for h in health_input:
#             health_input[h][p] = pollutant_preds[p][h]

#     health_risk = compute_health_risk(health_input)

#     # ───── SOLUTIONS ─────
#     solutions = generate_solutions(region, predictions, causes, health_risk)

#     alerts = generate_alerts(pred_1h, pred_3h, pred_6h, health_risk)

#     cursor.close()
#     conn.close()

#     return JSONResponse(content={
#         "region": region,
#         "prediction_time": str(timestamp),
#         "predicted_aqi": predictions,
#         "predicted_pollutants": pollutant_preds,
#         "causes": causes,
#         "health_risk": health_risk,
#         "alerts": alerts,
#         "solutions": solutions
#     })



from fastapi import FastAPI
from fastapi.responses import JSONResponse



from db import get_connection

from model import (
    pollutant_models,
    scaler, all_features, num_features,
    POLLUTANT_NAME_MAP
)

from uncertainty_engine import simulate_predictions, compute_uncertainty_range
from cause_engine import detect_causes
from health_data import compute_health_risk
from solution_engine import generate_solutions

import pandas as pd
import numpy as np
import json
from datetime import datetime
import os

app = FastAPI()

from fastapi.middleware.cors import CORSMiddleware

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

    delta_1_3 = pred_3h - pred_1h
    delta_3_6 = pred_6h - pred_3h

    severity_map = ALERT_CONFIG["severity_levels"]
    threshold = ALERT_CONFIG["thresholds"]["rapid_change"]

    severity_3h = severity_map[health_risk["3h"]["interpretation"]["risk_level"]]
    severity_6h = severity_map[health_risk["6h"]["interpretation"]["risk_level"]]

    if delta_1_3 > threshold or delta_3_6 > threshold:
        alerts.append("Rapid deterioration in air quality expected")

    if severity_3h >= ALERT_CONFIG["thresholds"]["health_severity_threshold"]:
        alerts.append("Elevated health risk expected")

    if pred_6h > pred_3h > pred_1h:
        alerts.append("Air quality continuously worsening")

    return alerts


# ───────────────── CORE PIPELINE (NO LOGIC CHANGE) ─────────────────
def run_full_pipeline(region: str):

    if not region:
        return {"error": "Region required"}

    conn = get_connection()
    cursor = conn.cursor()

    # STEP 1: GET PRECOMPUTED AQI
    cursor.execute("""
        SELECT timestamp, pred_1h, pred_3h, pred_6h
        FROM predictions
        WHERE region = %s
        ORDER BY timestamp DESC
        LIMIT 1
    """, (region,))

    row = cursor.fetchone()

    if not row:
        return {"error": "No predictions available yet"}

    timestamp, pred_1h, pred_3h, pred_6h = row

    predictions = {
        "1h": float(pred_1h),
        "3h": float(pred_3h),
        "6h": float(pred_6h)
    }

    # STEP 2: FETCH RECENT DATA
    cursor.execute("""
        SELECT *
        FROM aqi_data
        WHERE region = %s
        ORDER BY datetime DESC
        LIMIT 80
    """, (region,))

    rows = cursor.fetchall()

    if len(rows) < 10:
        return {"error": "Not enough data for analysis"}

    rows = rows[::-1]

    df = pd.DataFrame([{
        "Datetime_IST": r[1],
        "PM2.5 (µg/m³)": r[3],
        "PM10 (µg/m³)": r[4],
        "NO2 (µg/m³)": r[5],
        "CO (mg/m³)": r[6],
        "Ozone (µg/m³)": r[7],
        "Area_Type": r[8],
        "Region": r[2]
    } for r in rows])

    df["Datetime_IST"] = pd.to_datetime(df["Datetime_IST"])

    # BASIC FEATURES
    df["PM25_roll6h_mean"] = df["PM2.5 (µg/m³)"].rolling(6).mean()
    df["PM10_roll6h_mean"] = df["PM10 (µg/m³)"].rolling(6).mean()

    df = df.fillna(method="bfill")
    latest = df.iloc[-1]

    # MODEL INPUT
    df_model = pd.get_dummies(df, columns=["Area_Type", "Region"], drop_first=True)

    for col in all_features:
        if col not in df_model.columns:
            df_model[col] = 0

    df_model = df_model[all_features]
    df_model[num_features] = scaler.transform(df_model[num_features])

    df_model = df_model.tail(1)

    # POLLUTANTS
    pollutant_preds = {}

    for name, key in POLLUTANT_NAME_MAP.items():
        pollutant_preds[name] = {
            "1h": float(pollutant_models[key]["1h"].predict(df_model)[0]),
            "3h": float(pollutant_models[key]["3h"].predict(df_model)[0]),
            "6h": float(pollutant_models[key]["6h"].predict(df_model)[0]),
        }

    # CAUSES
    causes = {}
    for h in ["1h","3h","6h"]:
        vals = {p: pollutant_preds[p][h] for p in pollutant_preds}

        baseline = {
            "PM2.5": latest["PM25_roll6h_mean"],
            "PM10": latest["PM10_roll6h_mean"],
            "NO2": latest["NO2 (µg/m³)"],
            "CO": latest["CO (mg/m³)"],
            "O3": latest["Ozone (µg/m³)"],
        }

        causes[h] = detect_causes(vals, baseline)

    # HEALTH
    health_input = {h:{} for h in ["1h","3h","6h"]}

    for p in pollutant_preds:
        for h in health_input:
            health_input[h][p] = pollutant_preds[p][h]

    health_risk = compute_health_risk(health_input)

    # SOLUTIONS
    solutions = generate_solutions(region, predictions, causes, health_risk)

    alerts = generate_alerts(pred_1h, pred_3h, pred_6h, health_risk)

    cursor.close()
    conn.close()

    return {
        "region": region,
        "prediction_time": str(timestamp),
        "predicted_aqi": predictions,
        "predicted_pollutants": pollutant_preds,
        "causes": causes,
        "health_risk": health_risk,
        "alerts": alerts,
        "solutions": solutions
    }


# ───────────────── MAIN ENDPOINTS ─────────────────

@app.get("/predict")
def predict(region: str):
    result = run_full_pipeline(region)
    return JSONResponse(content=result)


@app.get("/dashboard")
def dashboard(region: str):
    result = run_full_pipeline(region)
    return JSONResponse(content=result)


# ───────────────── SPLIT ENDPOINTS ─────────────────

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
    result = run_full_pipeline(region)
    return result["predicted_pollutants"]


@app.get("/causes")
def get_causes(region: str):
    result = run_full_pipeline(region)
    return result["causes"]


@app.get("/health")
def get_health(region: str):
    result = run_full_pipeline(region)
    return result["health_risk"]


@app.get("/solutions")
def get_solutions(region: str):
    result = run_full_pipeline(region)
    return result["solutions"]


@app.get("/alerts")
def get_alerts(region: str):
    result = run_full_pipeline(region)
    return result["alerts"]