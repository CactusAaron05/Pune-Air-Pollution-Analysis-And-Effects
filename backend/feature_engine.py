import numpy as np

def build_features(rows):
    pm25 = [r["pm25"] for r in rows]
    pm10 = [r["pm10"] for r in rows]

    features = {}

    # SAFE LAGS
    features["PM25_lag1h"] = pm25[-2] if len(pm25) >= 2 else pm25[-1]
    features["PM25_lag3h"] = pm25[-4] if len(pm25) >= 4 else pm25[-1]
    features["PM25_lag6h"] = pm25[-7] if len(pm25) >= 7 else pm25[-1]
    features["PM25_lag24h"] = pm25[-25] if len(pm25) >= 25 else pm25[-1]

    # SAFE ROLLING
    features["PM25_roll6h_mean"] = np.mean(pm25[-6:]) if len(pm25) >= 1 else 0
    features["PM25_roll6h_std"] = np.std(pm25[-6:]) if len(pm25) >= 1 else 0

    features["PM25_roll24h_mean"] = np.mean(pm25[-24:]) if len(pm25) >= 1 else 0
    features["PM25_roll24h_std"] = np.std(pm25[-24:]) if len(pm25) >= 1 else 0

    # PM10
    features["PM10_lag6h"] = pm10[-7] if len(pm10) >= 7 else pm10[-1]
    features["PM10_lag24h"] = pm10[-25] if len(pm10) >= 25 else pm10[-1]

    features["PM10_roll6h_mean"] = np.mean(pm10[-6:]) if len(pm10) >= 1 else 0
    features["PM10_roll24h_mean"] = np.mean(pm10[-24:]) if len(pm25) >= 1 else 0
    features["PM10_roll24h_std"] = np.std(pm10[-24:]) if len(pm25) >= 1 else 0

    # RAW
    latest = rows[-1]

    features["PM10 (µg/m³)"] = latest["pm10"]
    features["NO2 (µg/m³)"] = latest["no2"]
    features["CO (mg/m³)"] = latest["co"]
    features["Ozone (µg/m³)"] = latest["o3"]

    return features


# ✅ ADD THIS FUNCTION
def add_time_features(features, dt):
    hour = dt.hour
    month = dt.month

    features["Hour_sin"] = np.sin(2 * np.pi * hour / 24)
    features["Hour_cos"] = np.cos(2 * np.pi * hour / 24)

    features["Month_sin"] = np.sin(2 * np.pi * month / 12)
    features["Month_cos"] = np.cos(2 * np.pi * month / 12)

    features["Is_Weekend"] = 1 if dt.weekday() >= 5 else 0

    return features


# ✅ ADD THIS FUNCTION
def add_categorical_features(features, region, dt, area_type, all_features):

    # Initialize all categorical columns
    for col in all_features:
        if col.startswith(("Region_", "Area_", "Season_")):
            features[col] = 0

    # ── Region ─────────────────────────
    region_col = f"Region_{region}"
    if region_col in all_features:
        features[region_col] = 1

    # ── Area ───────────────────────────
    if area_type:
        if "Industrial" in area_type:
            area = "Industrial"
        elif "Residential" in area_type:
            area = "Residential"
        else:
            area = "Other"

        area_col = f"Area_{area}"
        if area_col in all_features:
            features[area_col] = 1

    # ── Season ─────────────────────────
    month = dt.month

    if month in [12, 1, 2]:
        season = "Winter"
    elif month in [3, 4, 5]:
        season = "Summer"
    elif month in [6, 7, 8, 9]:
        season = "Monsoon"
    else:
        season = "PostMonsoon"

    season_col = f"Season_{season}"
    if season_col in all_features:
        features[season_col] = 1

    return features