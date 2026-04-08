import numpy as np
import pandas as pd


def build_features(df_hist):
    """
    Fully aligned with training pipeline.
    Uses ONLY past data (shifted).
    """

    df = df_hist.copy()

    # ───────── SORT (CRITICAL) ─────────
    df = df.sort_values("Datetime_IST")

    # ───────── PM2.5 FEATURES ─────────
    df["PM25_lag1h"] = df["PM2.5 (µg/m³)"].shift(1)
    df["PM25_lag3h"] = df["PM2.5 (µg/m³)"].shift(3)
    df["PM25_lag6h"] = df["PM2.5 (µg/m³)"].shift(6)
    df["PM25_lag24h"] = df["PM2.5 (µg/m³)"].shift(24)

    df["PM25_roll6h_mean"] = df["PM2.5 (µg/m³)"].shift(1).rolling(6).mean()
    df["PM25_roll6h_std"] = df["PM2.5 (µg/m³)"].shift(1).rolling(6).std()

    df["PM25_roll24h_mean"] = df["PM2.5 (µg/m³)"].shift(1).rolling(24).mean()
    df["PM25_roll24h_std"] = df["PM2.5 (µg/m³)"].shift(1).rolling(24).std()

    # ───────── PM10 FEATURES ─────────
    df["PM10_lag6h"] = df["PM10 (µg/m³)"].shift(6)
    df["PM10_lag24h"] = df["PM10 (µg/m³)"].shift(24)

    df["PM10_roll6h_mean"] = df["PM10 (µg/m³)"].shift(1).rolling(6).mean()
    df["PM10_roll24h_mean"] = df["PM10 (µg/m³)"].shift(1).rolling(24).mean()
    df["PM10_roll24h_std"] = df["PM10 (µg/m³)"].shift(1).rolling(24).std()

    # ───────── RAW FEATURES ─────────
    latest = df.iloc[-1]

    features = {
        "PM10 (µg/m³)": latest["PM10 (µg/m³)"],
        "NO2 (µg/m³)": latest["NO2 (µg/m³)"],
        "CO (mg/m³)": latest["CO (mg/m³)"],
        "Ozone (µg/m³)": latest["Ozone (µg/m³)"],
    }

    # ───────── MERGE LAST ROW FEATURES ─────────
    last_row = df.iloc[-1].to_dict()

    for key in last_row:
        if key.startswith(("PM25_", "PM10_")):
            features[key] = last_row[key]

    return features


def add_time_features(features, dt):
    hour = dt.hour
    month = dt.month

    features["Hour_sin"] = np.sin(2 * np.pi * hour / 24)
    features["Hour_cos"] = np.cos(2 * np.pi * hour / 24)

    features["Month_sin"] = np.sin(2 * np.pi * month / 12)
    features["Month_cos"] = np.cos(2 * np.pi * month / 12)

    features["Is_Weekend"] = 1 if dt.weekday() >= 5 else 0

    return features


def add_categorical_features(features, df_hist, all_features):
    """
    Uses SAME encoding as training
    """

    latest = df_hist.iloc[-1]

    # Initialize all categorical columns
    for col in all_features:
        if col.startswith(("Region_", "Area_", "Season_")):
            features[col] = 0

    # Region
    region_col = f"Region_{latest['Region']}"
    if region_col in all_features:
        features[region_col] = 1

    # Area
    area_col = f"Area_{latest['Area_Type']}"
    if area_col in all_features:
        features[area_col] = 1

    # Season (MATCH TRAINING EXACTLY)
    month = latest["Datetime_IST"].month

    if month in [12, 1, 2]:
        season = "Winter"
    elif month in [3, 4, 5]:
        season = "Pre-Monsoon"
    elif month in [6, 7, 8, 9]:
        season = "Monsoon"
    else:
        season = "Post-Monsoon"

    season_col = f"Season_{season}"
    if season_col in all_features:
        features[season_col] = 1

    return features