import pandas as pd
import numpy as np
import os
import joblib
import xgboost as xgb
from datetime import timedelta
from db import get_connection
from psycopg2.extras import execute_batch
from collections import deque

# ── CONFIG ─────────────────────────────────────────
CSV_PATH = "../pune_aqi_master_final.csv"
HISTORY_WINDOW = 80
BATCH_SIZE = 500

# ── LOAD MODEL ARTIFACTS ───────────────────────────
BASE_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE_DIR, "../models")

scaler = joblib.load(os.path.join(MODEL_DIR, "feature_scaler.pkl"))
all_features = joblib.load(os.path.join(MODEL_DIR, "feature_columns.pkl"))
num_features = joblib.load(os.path.join(MODEL_DIR, "numeric_feature_columns.pkl"))

model_1h = xgb.XGBRegressor(); model_1h.load_model(os.path.join(MODEL_DIR, "aqi_model_1h.json"))
model_3h = xgb.XGBRegressor(); model_3h.load_model(os.path.join(MODEL_DIR, "aqi_model_3h.json"))
model_6h = xgb.XGBRegressor(); model_6h.load_model(os.path.join(MODEL_DIR, "aqi_model_6h.json"))

# ── LOAD DATA ──────────────────────────────────────
df = pd.read_csv(CSV_PATH)

df["Region"] = df["Region"].astype(str).str.strip()
# df["Datetime_IST"] = pd.to_datetime(df["Datetime_IST"])

df["Datetime_IST"] = pd.to_datetime(df["Datetime_IST"]).dt.tz_localize(None)

df = df.sort_values(["Datetime_IST", "Region"]).reset_index(drop=True)

# ── DB ─────────────────────────────────────────────
conn = get_connection()
cur = conn.cursor()

# ── RESUME ─────────────────────────────────────────
cur.execute("SELECT MAX(datetime) FROM aqi_data")
last_timestamp = cur.fetchone()[0]

if last_timestamp:
    last_timestamp = pd.to_datetime(last_timestamp)
    start_idx = df[df["Datetime_IST"] > last_timestamp].index.min()
    if pd.isna(start_idx):
        start_idx = len(df)
else:
    start_idx = 0

print(f"🚀 Starting from index: {start_idx}")

# ── FEATURE BUILDER (UNCHANGED) ─────────────────────
def build_features(df_hist):
    df = df_hist.copy()

    df["PM25_lag1h"]  = df.groupby("Region")["PM2.5 (µg/m³)"].shift(1)
    df["PM25_lag3h"]  = df.groupby("Region")["PM2.5 (µg/m³)"].shift(3)
    df["PM25_lag6h"]  = df.groupby("Region")["PM2.5 (µg/m³)"].shift(6)
    df["PM25_lag24h"] = df.groupby("Region")["PM2.5 (µg/m³)"].shift(24)

    df["PM10_lag6h"]  = df.groupby("Region")["PM10 (µg/m³)"].shift(6)
    df["PM10_lag24h"] = df.groupby("Region")["PM10 (µg/m³)"].shift(24)

    df["PM25_roll6h_mean"] = df.groupby("Region")["PM2.5 (µg/m³)"].transform(lambda x: x.shift(1).rolling(6).mean())
    df["PM25_roll6h_std"]  = df.groupby("Region")["PM2.5 (µg/m³)"].transform(lambda x: x.shift(1).rolling(6).std())

    df["PM25_roll24h_mean"] = df.groupby("Region")["PM2.5 (µg/m³)"].transform(lambda x: x.shift(1).rolling(24).mean())
    df["PM25_roll24h_std"]  = df.groupby("Region")["PM2.5 (µg/m³)"].transform(lambda x: x.shift(1).rolling(24).std())

    df["PM10_roll6h_mean"] = df.groupby("Region")["PM10 (µg/m³)"].transform(lambda x: x.shift(1).rolling(6).mean())
    df["PM10_roll24h_mean"] = df.groupby("Region")["PM10 (µg/m³)"].transform(lambda x: x.shift(1).rolling(24).mean())
    df["PM10_roll24h_std"]  = df.groupby("Region")["PM10 (µg/m³)"].transform(lambda x: x.shift(1).rolling(24).std())

    df["Hour"] = df["Datetime_IST"].dt.hour
    df["Month"] = df["Datetime_IST"].dt.month
    df["Is_Weekend"] = df["Datetime_IST"].dt.weekday >= 5

    df["Hour_sin"] = np.sin(2 * np.pi * df["Hour"] / 24)
    df["Hour_cos"] = np.cos(2 * np.pi * df["Hour"] / 24)
    df["Month_sin"] = np.sin(2 * np.pi * df["Month"] / 12)
    df["Month_cos"] = np.cos(2 * np.pi * df["Month"] / 12)

    def get_season(m):
        if m in [12,1,2]: return "Winter"
        elif m in [3,4,5]: return "Pre-Monsoon"
        elif m in [6,7,8,9]: return "Monsoon"
        else: return "Post-Monsoon"

    df["Season"] = df["Month"].apply(get_season)

    df = pd.get_dummies(df, columns=["Area_Type", "Season", "Region"], drop_first=True)
    df = df.fillna(0)

    return df.tail(1)

# ── SYNTHETIC ──────────────────────────────────────
def generate_next_row(last_row):
    new_time = last_row["Datetime_IST"] + timedelta(hours=1)
    return {
        "Region": last_row["Region"],
        "Datetime_IST": new_time,
        "PM2.5 (µg/m³)": float(max(0, last_row["PM2.5 (µg/m³)"] + np.random.normal(0, 5))),
        "PM10 (µg/m³)": float(max(0, last_row["PM10 (µg/m³)"] + np.random.normal(0, 8))),
        "NO2 (µg/m³)": float(max(0, last_row["NO2 (µg/m³)"] + np.random.normal(0, 3))),
        "CO (mg/m³)": float(max(0, last_row["CO (mg/m³)"] + np.random.normal(0, 0.2))),
        "Ozone (µg/m³)": float(max(0, last_row["Ozone (µg/m³)"] + np.random.normal(0, 4))),
        "Area_Type": last_row.get("Area_Type", "Urban")
    }

# ── BUFFERS ────────────────────────────────────────
aqi_buffer = []
pred_buffer = []

def flush():
    if aqi_buffer:
        execute_batch(cur, """
            INSERT INTO aqi_data (
                region, datetime, pm25, pm10, no2, co, o3, area_type
            ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
            ON CONFLICT (region, datetime) DO NOTHING
        """, aqi_buffer)

    if pred_buffer:
        execute_batch(cur, """
            INSERT INTO predictions (region, timestamp, pred_1h, pred_3h, pred_6h)
            VALUES (%s,%s,%s,%s,%s)
        """, pred_buffer)

    print(f"💾 Flushing → AQI: {len(aqi_buffer)} | Predictions: {len(pred_buffer)}")

    aqi_buffer.clear()
    pred_buffer.clear()
    conn.commit()

# ── IN-MEMORY HISTORY ──────────────────────────────
history_map = {}
prediction_started = set()

def get_history(region):
    if region not in history_map:
        history_map[region] = deque(maxlen=HISTORY_WINDOW)
    return history_map[region]

# ── MAIN LOOP ──────────────────────────────────────
i = start_idx
last_row_cache = None

while True:

    if i < len(df):
        r = df.iloc[i]
    else:
        if last_row_cache is None:
            cur.execute("""
                SELECT region, datetime, pm25, pm10, no2, co, o3, area_type
                FROM aqi_data
                ORDER BY datetime DESC
                LIMIT 1
            """)
            last = cur.fetchone()

            last_row_cache = {
                "Region": last[0],
                "Datetime_IST": pd.to_datetime(last[1]).tz_localize(None),
                "PM2.5 (µg/m³)": float(last[2]),
                "PM10 (µg/m³)": float(last[3]),
                "NO2 (µg/m³)": float(last[4]),
                "CO (mg/m³)": float(last[5]),
                "Ozone (µg/m³)": float(last[6]),
                "Area_Type": last[7]
            }

        next_row = generate_next_row(last_row_cache)

        if next_row["Datetime_IST"] > pd.Timestamp.now():
            print("⛔ Reached current time. Stopping.")
            break

        last_row_cache = next_row
        r = next_row

    region = str(r["Region"]).strip()

    # 🔥 Progress tracking (SAFE)
    if i % 1000 == 0:
        print(f"📊 Processed: {i} | Region: {region} | Time: {r['Datetime_IST']}")

    hist = get_history(region)
    hist.append([
        region,
        r["Datetime_IST"],
        float(r["PM2.5 (µg/m³)"]),
        float(r["PM10 (µg/m³)"]),
        float(r["NO2 (µg/m³)"]),
        float(r["CO (mg/m³)"]),
        float(r["Ozone (µg/m³)"]),
        r.get("Area_Type", "Urban")
    ])

    aqi_buffer.append(tuple(hist[-1]))

    if len(hist) >= 60:

        if region not in prediction_started:
            print(f"🧠 Predictions started for region: {region}")
            prediction_started.add(region)

        df_hist = pd.DataFrame(list(hist), columns=[
            "Region","Datetime_IST","PM2.5 (µg/m³)","PM10 (µg/m³)",
            "NO2 (µg/m³)","CO (mg/m³)","Ozone (µg/m³)","Area_Type"
        ])

        df_hist["Datetime_IST"] = pd.to_datetime(df_hist["Datetime_IST"]).dt.tz_localize(None)

        df_feat = build_features(df_hist)

        for col in all_features:
            if col not in df_feat.columns:
                df_feat[col] = 0

        df_feat = df_feat[all_features]
        df_feat[num_features] = scaler.transform(df_feat[num_features])

        p1 = float(model_1h.predict(df_feat)[0])
        p3 = float(model_3h.predict(df_feat)[0])
        p6 = float(model_6h.predict(df_feat)[0])

        pred_buffer.append((
            region,
            pd.to_datetime(r["Datetime_IST"]).to_pydatetime(),
            p1, p3, p6
        ))

    if len(aqi_buffer) >= BATCH_SIZE or len(pred_buffer) >= BATCH_SIZE:
        flush()

    i += 1

flush()
print("✅ DONE FAST")