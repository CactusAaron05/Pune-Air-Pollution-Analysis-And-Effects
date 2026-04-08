# import pandas as pd
# import numpy as np
# import os
# import joblib
# import xgboost as xgb
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# print("=" * 60)
# print("EVALUATION PIPELINE (FINAL)")
# print("=" * 60)

# # ================= PATH SETUP =================
# BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# CSV_PATH = os.path.join(BASE_DIR, "pune_aqi_master_final.csv")
# MODEL_DIR = os.path.join(BASE_DIR, "models")

# # ================= LOAD DATA =================
# print("\nLoading dataset...")
# df = pd.read_csv(CSV_PATH, low_memory=False)

# df['Datetime_IST'] = pd.to_datetime(df['Datetime_IST'], utc=True)
# df = df.sort_values(['Region', 'Datetime_IST']).reset_index(drop=True)

# print(f"Loaded: {df.shape}")

# # ================= FEATURE ENGINEERING =================

# # One-hot encoding (same as training)
# area_dummies = pd.get_dummies(df['Area_Type'], prefix='Area', drop_first=True)
# season_dummies = pd.get_dummies(df['Season'], prefix='Season', drop_first=True)
# region_dummies = pd.get_dummies(df['Region'], prefix='Region', drop_first=True)

# df = pd.concat([df, area_dummies, season_dummies, region_dummies], axis=1)

# # PM10 features (same as training)
# pm10_col = 'PM10 (µg/m³)'

# df['PM10_lag6h'] = df.groupby('Region')[pm10_col].shift(6)
# df['PM10_lag24h'] = df.groupby('Region')[pm10_col].shift(24)

# df['PM10_roll6h_mean'] = df.groupby('Region')[pm10_col].transform(
#     lambda x: x.shift(1).rolling(6, min_periods=1).mean()
# )
# df['PM10_roll24h_mean'] = df.groupby('Region')[pm10_col].transform(
#     lambda x: x.shift(1).rolling(24, min_periods=1).mean()
# )
# df['PM10_roll24h_std'] = df.groupby('Region')[pm10_col].transform(
#     lambda x: x.shift(1).rolling(24, min_periods=2).std()
# )

# # ================= LOAD TRAINING ARTIFACTS =================

# scaler = joblib.load(os.path.join(MODEL_DIR, "feature_scaler.pkl"))
# ALL_FEATURES = joblib.load(os.path.join(MODEL_DIR, "feature_columns.pkl"))
# NUMERIC_FEATURES = joblib.load(os.path.join(MODEL_DIR, "numeric_feature_columns.pkl"))

# print(f"Expected features: {len(ALL_FEATURES)}")

# # ================= ALIGN FEATURES =================

# # Add missing columns
# for col in ALL_FEATURES:
#     if col not in df.columns:
#         df[col] = 0

# # Keep only required columns
# df = df[ALL_FEATURES]

# print("After alignment:", df.shape)

# # ================= HANDLE MISSING VALUES =================

# df = df.fillna(0)

# if df.shape[0] == 0:
#     raise Exception("❌ Dataset became empty after preprocessing")

# # ================= SCALING =================

# df[NUMERIC_FEATURES] = scaler.transform(df[NUMERIC_FEATURES])

# # ================= LOAD MODEL =================

# model = xgb.XGBRegressor()
# model.load_model(os.path.join(MODEL_DIR, "aqi_model_1h.json"))

# # ================= TRUE EVALUATION =================

# # ================= TRUE EVALUATION =================

# print("\n" + "=" * 60)
# print("EVALUATION METRICS")
# print("=" * 60)

# # Create target BEFORE feature alignment
# df_original = pd.read_csv(CSV_PATH, low_memory=False)
# df_original['Datetime_IST'] = pd.to_datetime(df_original['Datetime_IST'], utc=True)
# df_original = df_original.sort_values(['Region', 'Datetime_IST']).reset_index(drop=True)

# df_original['target_1h'] = df_original['AQI'].shift(-1)

# # Now apply SAME feature pipeline to df_original

# # One-hot
# area_dummies = pd.get_dummies(df_original['Area_Type'], prefix='Area', drop_first=True)
# season_dummies = pd.get_dummies(df_original['Season'], prefix='Season', drop_first=True)
# region_dummies = pd.get_dummies(df_original['Region'], prefix='Region', drop_first=True)

# df_original = pd.concat([df_original, area_dummies, season_dummies, region_dummies], axis=1)

# # PM10 features
# pm10_col = 'PM10 (µg/m³)'

# df_original['PM10_lag6h'] = df_original.groupby('Region')[pm10_col].shift(6)
# df_original['PM10_lag24h'] = df_original.groupby('Region')[pm10_col].shift(24)

# df_original['PM10_roll6h_mean'] = df_original.groupby('Region')[pm10_col].transform(
#     lambda x: x.shift(1).rolling(6, min_periods=1).mean()
# )
# df_original['PM10_roll24h_mean'] = df_original.groupby('Region')[pm10_col].transform(
#     lambda x: x.shift(1).rolling(24, min_periods=1).mean()
# )
# df_original['PM10_roll24h_std'] = df_original.groupby('Region')[pm10_col].transform(
#     lambda x: x.shift(1).rolling(24, min_periods=2).std()
# )

# # Drop rows where target is missing
# df_original = df_original.dropna(subset=['target_1h'])

# # Align features
# for col in ALL_FEATURES:
#     if col not in df_original.columns:
#         df_original[col] = 0

# X = df_original[ALL_FEATURES]
# y = df_original['target_1h']

# # Fill + scale
# X = X.fillna(0)
# X[NUMERIC_FEATURES] = scaler.transform(X[NUMERIC_FEATURES])

# # Predict
# preds = model.predict(X)
# preds = np.clip(preds, 0, 500)

# # Metrics
# mae = mean_absolute_error(y, preds)
# rmse = np.sqrt(mean_squared_error(y, preds))
# r2 = r2_score(y, preds)

# print(f"MAE  : {mae:.2f}")
# print(f"RMSE : {rmse:.2f}")
# print(f"R²   : {r2:.4f}")



# # ================= MULTI-HORIZON EVALUATION =================

# print("\n" + "=" * 60)
# print("MULTI-HORIZON EVALUATION (1h, 3h, 6h)")
# print("=" * 60)

# # Create additional targets
# df_original['target_3h'] = df_original['AQI'].shift(-3)
# df_original['target_6h'] = df_original['AQI'].shift(-6)

# # Drop rows where any target is missing
# df_mh = df_original.dropna(subset=['target_1h', 'target_3h', 'target_6h']).copy()

# # Align features again (safety)
# for col in ALL_FEATURES:
#     if col not in df_mh.columns:
#         df_mh[col] = 0

# X_mh = df_mh[ALL_FEATURES].fillna(0)
# X_mh[NUMERIC_FEATURES] = scaler.transform(X_mh[NUMERIC_FEATURES])

# y_1h = df_mh['target_1h']
# y_3h = df_mh['target_3h']
# y_6h = df_mh['target_6h']

# # Load models
# model_1h = xgb.XGBRegressor()
# model_1h.load_model(os.path.join(MODEL_DIR, "aqi_model_1h.json"))

# model_3h = xgb.XGBRegressor()
# model_3h.load_model(os.path.join(MODEL_DIR, "aqi_model_3h.json"))

# model_6h = xgb.XGBRegressor()
# model_6h.load_model(os.path.join(MODEL_DIR, "aqi_model_6h.json"))

# # Predict
# pred_1h = np.clip(model_1h.predict(X_mh), 0, 500)
# pred_3h = np.clip(model_3h.predict(X_mh), 0, 500)
# pred_6h = np.clip(model_6h.predict(X_mh), 0, 500)

# # Metrics function
# def evaluate(y_true, y_pred, label):
#     mae = mean_absolute_error(y_true, y_pred)
#     rmse = np.sqrt(mean_squared_error(y_true, y_pred))
#     r2 = r2_score(y_true, y_pred)

#     print(f"\n{label}")
#     print(f"MAE  : {mae:.2f}")
#     print(f"RMSE : {rmse:.2f}")
#     print(f"R²   : {r2:.4f}")

# # Evaluate all
# evaluate(y_1h, pred_1h, "1 Hour Ahead")
# evaluate(y_3h, pred_3h, "3 Hours Ahead")
# evaluate(y_6h, pred_6h, "6 Hours Ahead")


# # ================= POLLUTANT EVALUATION =================

# print("\n" + "=" * 60)
# print("POLLUTANT FORECAST EVALUATION")
# print("=" * 60)

# POLLUTANTS = {
#     "PM25": ("PM2.5 (µg/m³)", "PM25"),
#     "PM10": ("PM10 (µg/m³)", "PM10"),
#     "NO2":  ("NO2 (µg/m³)", "NO2"),
#     "CO":   ("CO (mg/m³)", "CO"),
#     "O3":   ("Ozone (µg/m³)", "O3"),
# }

# def eval_pollutant(name, col, prefix):
#     print(f"\n--- {name} ---")

#     # Targets
#     df_original[f"{prefix}_t1"] = df_original[col].shift(-1)
#     df_original[f"{prefix}_t3"] = df_original[col].shift(-3)
#     df_original[f"{prefix}_t6"] = df_original[col].shift(-6)

#     df_p = df_original.dropna(subset=[
#         f"{prefix}_t1", f"{prefix}_t3", f"{prefix}_t6"
#     ]).copy()

#     # Align features
#     for c in ALL_FEATURES:
#         if c not in df_p.columns:
#             df_p[c] = 0

#     Xp = df_p[ALL_FEATURES].fillna(0)
#     Xp[NUMERIC_FEATURES] = scaler.transform(Xp[NUMERIC_FEATURES])

#     y1 = df_p[f"{prefix}_t1"]
#     y3 = df_p[f"{prefix}_t3"]
#     y6 = df_p[f"{prefix}_t6"]

#     # Load models
#     m1 = xgb.XGBRegressor(); m1.load_model(os.path.join(MODEL_DIR, f"{prefix}_model_1h.json"))
#     m3 = xgb.XGBRegressor(); m3.load_model(os.path.join(MODEL_DIR, f"{prefix}_model_3h.json"))
#     m6 = xgb.XGBRegressor(); m6.load_model(os.path.join(MODEL_DIR, f"{prefix}_model_6h.json"))

#     p1 = m1.predict(Xp)
#     p3 = m3.predict(Xp)
#     p6 = m6.predict(Xp)

#     def show(y, p, label):
#         mae = mean_absolute_error(y, p)
#         rmse = np.sqrt(mean_squared_error(y, p))
#         r2 = r2_score(y, p)
#         print(f"{label} → MAE: {mae:.2f}, RMSE: {rmse:.2f}, R²: {r2:.4f}")

#     show(y1, p1, "1h")
#     show(y3, p3, "3h")
#     show(y6, p6, "6h")


# # Run for all pollutants
# for name, (col, prefix) in POLLUTANTS.items():
#     eval_pollutant(name, col, prefix)


import pandas as pd
import numpy as np
import os
import joblib
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

print("=" * 60)
print("STRICT TEST-ONLY EVALUATION")
print("=" * 60)

# ── PATHS ──────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(BASE_DIR, "pune_aqi_master_final.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")

# ── LOAD DATA ──────────────────────────────────────
df = pd.read_csv(CSV_PATH, low_memory=False)
df["Datetime_IST"] = pd.to_datetime(df["Datetime_IST"], utc=True)
df = df.sort_values(["Region", "Datetime_IST"]).reset_index(drop=True)

print(f"Loaded: {df.shape}")

# ── FEATURE ENGINEERING (same as training) ─────────
area_dummies = pd.get_dummies(df["Area_Type"], prefix="Area", drop_first=True)
season_dummies = pd.get_dummies(df["Season"], prefix="Season", drop_first=True)
region_dummies = pd.get_dummies(df["Region"], prefix="Region", drop_first=True)

df = pd.concat([df, area_dummies, season_dummies, region_dummies], axis=1)

pm10_col = "PM10 (µg/m³)"
df["PM10_lag6h"] = df.groupby("Region")[pm10_col].shift(6)
df["PM10_lag24h"] = df.groupby("Region")[pm10_col].shift(24)

df["PM10_roll6h_mean"] = df.groupby("Region")[pm10_col].transform(
    lambda x: x.shift(1).rolling(6).mean()
)
df["PM10_roll24h_mean"] = df.groupby("Region")[pm10_col].transform(
    lambda x: x.shift(1).rolling(24).mean()
)
df["PM10_roll24h_std"] = df.groupby("Region")[pm10_col].transform(
    lambda x: x.shift(1).rolling(24).std()
)

# ── LOAD TRAINED FEATURE LIST ──────────────────────
ALL_FEATURES = joblib.load(os.path.join(MODEL_DIR, "feature_columns.pkl"))
NUMERIC_FEATURES = joblib.load(os.path.join(MODEL_DIR, "numeric_feature_columns.pkl"))

TARGET = "AQI"

# ── TARGET CREATION (same as training) ─────────────
df["target_1h"] = df["AQI"].shift(-1)
df["target_3h"] = df["AQI"].shift(-3)
df["target_6h"] = df["AQI"].shift(-6)

# ── CLEANING (same logic as training) ──────────────
df_model = df.dropna(subset=[
    TARGET,
    *NUMERIC_FEATURES,
    "target_1h", "target_3h", "target_6h"
]).copy()

print(f"After cleaning: {df_model.shape}")

# ── TIME-BASED SPLIT (same as training) ────────────
split_idx = int(len(df_model) * 0.80)
test_df = df_model.iloc[split_idx:].copy()

print(f"Test rows: {len(test_df)}")

# ── PREPARE FEATURES ───────────────────────────────
X_test = test_df[ALL_FEATURES].copy()
y_1h = test_df["target_1h"]
y_3h = test_df["target_3h"]
y_6h = test_df["target_6h"]

# ── SCALING ───────────────────────────────────────
scaler = joblib.load(os.path.join(MODEL_DIR, "feature_scaler.pkl"))
X_test[NUMERIC_FEATURES] = scaler.transform(X_test[NUMERIC_FEATURES].fillna(0))

# ── LOAD MODELS ───────────────────────────────────
model_1h = xgb.XGBRegressor()
model_1h.load_model(os.path.join(MODEL_DIR, "aqi_model_1h.json"))

model_3h = xgb.XGBRegressor()
model_3h.load_model(os.path.join(MODEL_DIR, "aqi_model_3h.json"))

model_6h = xgb.XGBRegressor()
model_6h.load_model(os.path.join(MODEL_DIR, "aqi_model_6h.json"))

# ── PREDICT ───────────────────────────────────────
p1 = np.clip(model_1h.predict(X_test), 0, 500)
p3 = np.clip(model_3h.predict(X_test), 0, 500)
p6 = np.clip(model_6h.predict(X_test), 0, 500)

# ── METRICS ───────────────────────────────────────
def evaluate(y, p, name):
    mae = mean_absolute_error(y, p)
    rmse = np.sqrt(mean_squared_error(y, p))
    r2 = r2_score(y, p)

    print(f"\n{name}")
    print(f"MAE  : {mae:.2f}")
    print(f"RMSE : {rmse:.2f}")
    print(f"R²   : {r2:.4f}")

evaluate(y_1h, p1, "1 Hour Ahead (TEST)")
evaluate(y_3h, p3, "3 Hours Ahead (TEST)")
evaluate(y_6h, p6, "6 Hours Ahead (TEST)")