import pandas as pd
import numpy as np
import os
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import plotly.graph_objects as go
import plotly.express as px
import json

# ══════════════════════════════════════════════════════════════════════════════
# PUNE AQI — XGBOOST TRAINING PIPELINE (FIXED v2)
# Input : pune_aqi_master_final.csv (same folder as this script)
# Output: models/aqi_xgb_model.json
#         models/feature_scaler.pkl
#         models/feature_columns.pkl
#         models/numeric_feature_columns.pkl
#         output/xgb_feature_importance.png
#         output/xgb_actual_vs_predicted.png
#         output/xgb_metrics.json
# ══════════════════════════════════════════════════════════════════════════════

os.makedirs('models', exist_ok=True)
os.makedirs('output', exist_ok=True)

# ── STEP 1: LOAD DATA ─────────────────────────────────────────────────────────
print("=" * 60)
print("STEP 1: Loading data")
print("=" * 60)

df = pd.read_csv('pune_aqi_master_final.csv', low_memory=False)
df['Datetime_IST'] = pd.to_datetime(df['Datetime_IST'], utc=True)
df = df.sort_values(['Region', 'Datetime_IST']).reset_index(drop=True)
print(f" Loaded: {len(df):,} rows x {len(df.columns)} columns")

# ── STEP 2: FEATURE ENGINEERING ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2: Feature engineering")
print("=" * 60)

# One-hot encode Area_Type, Season, Region (drop_first avoids dummy-variable trap)
area_dummies = pd.get_dummies(df['Area_Type'], prefix='Area', drop_first=True)
season_dummies = pd.get_dummies(df['Season'], prefix='Season', drop_first=True)
region_dummies = pd.get_dummies(df['Region'], prefix='Region', drop_first=True)

df = pd.concat([df, area_dummies, season_dummies, region_dummies], axis=1)

# FIX 3: Compute PM10 lag & rolling features per region if not already present
pm10_col = 'PM10 (µg/m³)'
for col_name, shift_or_roll, kind in [
    ('PM10_lag6h', 6, 'lag'),
    ('PM10_lag24h', 24, 'lag'),
    ('PM10_roll6h_mean', 6, 'roll_mean'),
    ('PM10_roll24h_mean', 24, 'roll_mean'),
    ('PM10_roll24h_std', 24, 'roll_std'),
]:
    if col_name not in df.columns:
        if kind == 'lag':
            df[col_name] = df.groupby('Region')[pm10_col].shift(shift_or_roll)
        elif kind == 'roll_mean':
            df[col_name] = (
                df.groupby('Region')[pm10_col]
                .transform(lambda x: x.shift(1).rolling(shift_or_roll, min_periods=1).mean())
            )
        elif kind == 'roll_std':
            df[col_name] = (
                df.groupby('Region')[pm10_col]
                .transform(lambda x: x.shift(1).rolling(shift_or_roll, min_periods=2).std())
            )

# Core numeric features
NUMERIC_FEATURES = [
    # Cyclical time encoding
    'Hour_sin', 'Hour_cos',
    'Month_sin', 'Month_cos',

    # Calendar
    'Is_Weekend',

    # PM2.5 lag features
    'PM25_lag1h', 'PM25_lag3h', 'PM25_lag6h', 'PM25_lag24h',

    # PM2.5 rolling window features
    'PM25_roll6h_mean', 'PM25_roll6h_std',
    'PM25_roll24h_mean', 'PM25_roll24h_std',

    # Key co-pollutants
    'PM10 (µg/m³)', 'NO2 (µg/m³)', 'CO (mg/m³)', 'Ozone (µg/m³)',

    # FIX 3 NEW: PM10 memory features
    'PM10_lag6h', 'PM10_lag24h',
    'PM10_roll6h_mean', 'PM10_roll24h_mean', 'PM10_roll24h_std',
]

# Collect one-hot column names dynamically
OHE_FEATURES = (
    list(area_dummies.columns) +
    list(season_dummies.columns) +
    list(region_dummies.columns)
)

ALL_FEATURES = NUMERIC_FEATURES + OHE_FEATURES

# Safety check — keep only columns that actually exist
ALL_FEATURES = [f for f in ALL_FEATURES if f in df.columns]
NUMERIC_FEATURES = [f for f in NUMERIC_FEATURES if f in df.columns]
TARGET = 'AQI'

print(f" Total features: {len(ALL_FEATURES)}")
for f in ALL_FEATURES:
    print(f" {f}")

# ── STEP 3: DROP ROWS WITH NULL TARGET OR CRITICAL FEATURES ──────────────────
print("\n" + "=" * 60)
print("STEP 3: Cleaning for model input")
print("=" * 60)

keep_cols = list(dict.fromkeys(ALL_FEATURES + [TARGET, 'Datetime_IST', 'Region', 'AQI_Category']))
df_model = df[keep_cols].copy()

before = len(df_model)
df_model = df_model.dropna(subset=[TARGET] + NUMERIC_FEATURES)
after = len(df_model)

print(f" Rows before drop: {before:,}")
print(f" Rows after drop: {after:,} (removed {before - after:,} rows with nulls)")

# ── STEP 4: TIME-BASED TRAIN / TEST SPLIT ────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 4: Time-based 80/20 train-test split")
print("=" * 60)

split_idx = int(len(df_model) * 0.80)
train_df = df_model.iloc[:split_idx].copy()
test_df = df_model.iloc[split_idx:].copy()

X_test = test_df[ALL_FEATURES]
y_test = test_df[TARGET]

print(f" Train: {len(train_df):,} rows "
      f"({train_df['Datetime_IST'].min().date()} → {train_df['Datetime_IST'].max().date()})")
print(f" Test : {len(test_df):,} rows "
      f"({test_df['Datetime_IST'].min().date()} → {test_df['Datetime_IST'].max().date()})")

# ── STEP 4b: FIX 1 — OVERSAMPLE EXTREME AQI ROWS ────────────────────────────
print("\n" + "=" * 60)
print("STEP 4b: Oversampling extreme AQI categories")
print("=" * 60)

OVERSAMPLE_FACTOR = 5
OVERSAMPLE_CATS = ['Very Poor', 'Severe']

if 'AQI_Category' in train_df.columns:
    extreme_train = train_df[train_df['AQI_Category'].isin(OVERSAMPLE_CATS)].copy()
else:
    extreme_train = pd.DataFrame(columns=train_df.columns)

if len(extreme_train) > 0:
    train_df = pd.concat(
        [train_df] + [extreme_train] * (OVERSAMPLE_FACTOR - 1),
        ignore_index=True
    ).sample(frac=1, random_state=42).reset_index(drop=True)

print(f" Oversampled train rows: {len(train_df):,}")
print(f" Extreme rows duplicated from categories: {OVERSAMPLE_CATS}")

WEIGHT_MAP = {
    'Good': 1.0,
    'Satisfactory': 1.0,
    'Moderate': 1.5,
    'Poor': 2.0,
    'Very Poor': 4.0,
    'Severe': 6.0
}

if 'AQI_Category' in train_df.columns:
    sample_weight = train_df['AQI_Category'].map(WEIGHT_MAP).fillna(1.0).values
else:
    sample_weight = np.ones(len(train_df))

X_train = train_df[ALL_FEATURES]
y_train = train_df[TARGET]

# ── STEP 5: FEATURE SCALING ───────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 5: Standard scaling on numeric features")
print("=" * 60)

scaler = StandardScaler()
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()

X_train_scaled[NUMERIC_FEATURES] = scaler.fit_transform(
    X_train[NUMERIC_FEATURES].fillna(0)
)
X_test_scaled[NUMERIC_FEATURES] = scaler.transform(
    X_test[NUMERIC_FEATURES].fillna(0)
)

joblib.dump(scaler, 'models/feature_scaler.pkl')
joblib.dump(ALL_FEATURES, 'models/feature_columns.pkl')
joblib.dump(NUMERIC_FEATURES, 'models/numeric_feature_columns.pkl')

print(" Saved → models/feature_scaler.pkl")
print(" Saved → models/feature_columns.pkl")
print(" Saved → models/numeric_feature_columns.pkl")

# ── STEP 6: XGBOOST MODEL TRAINING ───────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 6: Training XGBoost model")
print("=" * 60)

model = xgb.XGBRegressor(
    n_estimators=600,
    max_depth=7,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=3,
    reg_alpha=0.1,
    reg_lambda=1.0,
    objective='reg:squarederror',
    eval_metric='rmse',
    early_stopping_rounds=30,
    random_state=42,
    n_jobs=-1,
)

model.fit(
    X_train_scaled,
    y_train,
    sample_weight=sample_weight,
    eval_set=[(X_test_scaled, y_test)],
    verbose=50,
)

model.save_model('models/aqi_xgb_model.json')
print(f"\n Best iteration : {model.best_iteration}")
print(" Model saved → models/aqi_xgb_model.json")

# ── STEP 7: EVALUATION ───────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 7: Evaluation on test set")
print("=" * 60)

y_pred = model.predict(X_test_scaled)
y_pred = np.clip(y_pred, 0, 500)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f" MAE (Mean Absolute Error) : {mae:.2f} AQI points")
print(f" RMSE (Root Mean Sq. Error): {rmse:.2f} AQI points")
print(f" R² (Explained Variance)   : {r2:.4f}")

test_results = test_df.copy()
test_results['Predicted_AQI'] = y_pred

print("\n Per-Region MAE on test set:")
for region, grp in test_results.groupby('Region'):
    r_mae = mean_absolute_error(grp[TARGET], grp['Predicted_AQI'])
    print(f" {region:<25} MAE = {r_mae:.1f}")

if 'AQI_Category' in test_results.columns:
    print("\n Per-Category MAE on test set:")
    for cat, grp in test_results.groupby('AQI_Category'):
        if len(grp) > 0:
            c_mae = mean_absolute_error(grp[TARGET], grp['Predicted_AQI'])
            print(f" {cat:<15} MAE = {c_mae:.1f}   (n={len(grp):,})")

metrics = {
    'MAE': round(mae, 2),
    'RMSE': round(rmse, 2),
    'R2': round(r2, 4),
    'best_iteration': int(model.best_iteration) if model.best_iteration is not None else None,
    'n_features': len(ALL_FEATURES),
    'train_rows': len(X_train),
    'test_rows': len(X_test),
}
with open('output/xgb_metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)
print("\n Metrics saved → output/xgb_metrics.json")

# ── STEP 8: FEATURE IMPORTANCE CHART ─────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 8: Feature importance chart")
print("=" * 60)

importance_df = pd.DataFrame({
    'Feature': ALL_FEATURES,
    'Importance': model.feature_importances_,
}).sort_values('Importance', ascending=True).tail(20)

pm10_new_features = {
    'PM10_lag6h', 'PM10_lag24h',
    'PM10_roll6h_mean', 'PM10_roll24h_mean', 'PM10_roll24h_std'
}
importance_df['Color'] = importance_df['Feature'].apply(
    lambda x: '#e74c3c' if x in pm10_new_features else '#3498db'
)

fig_imp = go.Figure(go.Bar(
    y=importance_df['Feature'],
    x=importance_df['Importance'],
    orientation='h',
    marker_color=importance_df['Color'],
    text=importance_df['Importance'].round(4),
    textposition='outside',
))
fig_imp.update_layout(
    title={"text": "Top 20 Feature Importances — XGBoost AQI Model<br>"
                   "<span style='font-size:16px;font-weight:normal;'>"
                   "Red = newly added PM10 lag/rolling features</span>"},
    margin=dict(t=120, b=60, l=180, r=60),
    width=950,
    height=580,
)
fig_imp.update_xaxes(title_text='Importance (Gain)', tickfont=dict(size=11))
fig_imp.update_yaxes(title_text='Feature', tickfont=dict(size=11))
fig_imp.update_traces(cliponaxis=False)
fig_imp.write_image('output/xgb_feature_importance.png')

with open('output/xgb_feature_importance.png.meta.json', 'w') as f:
    json.dump({
        "caption": "XGBoost top-20 feature importance (Pune AQI)",
        "description": "Gain-based feature importances from XGBoost AQI regression model; red bars are new PM10 lag and rolling features."
    }, f)

print(" Chart saved → output/xgb_feature_importance.png")

# ── STEP 9: ACTUAL vs PREDICTED SCATTER ──────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 9: Actual vs Predicted scatter plot")
print("=" * 60)

AQI_CATEGORY_COLORS = {
    'Good': '#2ecc71',
    'Satisfactory': '#a8e063',
    'Moderate': '#f7dc6f',
    'Poor': '#e67e22',
    'Very Poor': '#e74c3c',
    'Severe': '#8e44ad',
}
CAT_ORDER = ['Good', 'Satisfactory', 'Moderate', 'Poor', 'Very Poor', 'Severe']

cat_all = test_results['AQI_Category'].fillna('Unknown').values if 'AQI_Category' in test_results.columns else np.array(['Unknown'] * len(test_results))

fig_scatter = go.Figure()

for cat in CAT_ORDER:
    mask = cat_all == cat
    if mask.sum() == 0:
        continue
    fig_scatter.add_trace(go.Scatter(
        x=np.array(y_test)[mask],
        y=y_pred[mask],
        mode='markers',
        name=cat,
        marker=dict(
            color=AQI_CATEGORY_COLORS.get(cat, '#bdc3c7'),
            size=4,
            opacity=0.5
        ),
    ))

unknown_mask = cat_all == 'Unknown'
if unknown_mask.sum() > 0:
    fig_scatter.add_trace(go.Scatter(
        x=np.array(y_test)[unknown_mask],
        y=y_pred[unknown_mask],
        mode='markers',
        name='Unknown',
        marker=dict(color='#bdc3c7', size=4, opacity=0.5),
    ))

fig_scatter.add_trace(go.Scatter(
    x=[0, 500],
    y=[0, 500],
    mode='lines',
    name='Perfect Fit',
    line=dict(color='black', dash='dash', width=1.5),
))

fig_scatter.update_layout(
    title={"text": f"Actual vs Predicted AQI — XGBoost (R² = {r2:.3f})<br>"
                   "<span style='font-size:16px;font-weight:normal;'>"
                   f"All {len(y_test):,} test rows | colour = AQI category</span>"},
    legend=dict(
        orientation='h',
        yanchor='bottom',
        y=-0.22,
        xanchor='center',
        x=0.5,
        font=dict(size=12),
    ),
    margin=dict(t=120, b=130, l=70, r=20),
    width=900,
    height=640,
)

fig_scatter.update_xaxes(title_text='Actual AQI', range=[0, 520])
fig_scatter.update_yaxes(title_text='Predicted AQI', range=[0, 520])
fig_scatter.write_image('output/xgb_actual_vs_predicted.png')

with open('output/xgb_actual_vs_predicted.png.meta.json', 'w') as f:
    json.dump({
        "caption": f"XGBoost: Actual vs Predicted AQI (R²={round(r2, 3)})",
        "description": "Scatter of actual vs predicted AQI on the full test set, color-coded by AQI category."
    }, f)

print(" Chart saved → output/xgb_actual_vs_predicted.png")

# ── STEP 10: INFERENCE EXAMPLE ───────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 10: Inference example (copy into FastAPI endpoint)")
print("=" * 60)
print("""
import xgboost as xgb, joblib, numpy as np, pandas as pd

model = xgb.XGBRegressor()
model.load_model('models/aqi_xgb_model.json')
scaler = joblib.load('models/feature_scaler.pkl')
all_features = joblib.load('models/feature_columns.pkl')
num_features = joblib.load('models/numeric_feature_columns.pkl')

sample = {
    'Hour_sin': np.sin(2*np.pi*8/24), 'Hour_cos': np.cos(2*np.pi*8/24),
    'Month_sin': np.sin(2*np.pi*12/12), 'Month_cos': np.cos(2*np.pi*12/12),
    'Is_Weekend': 0,
    'PM25_lag1h': 65.0, 'PM25_lag3h': 58.0,
    'PM25_lag6h': 52.0, 'PM25_lag24h': 70.0,
    'PM25_roll6h_mean': 60.0, 'PM25_roll6h_std': 5.0,
    'PM25_roll24h_mean': 62.0, 'PM25_roll24h_std': 8.0,
    'PM10 (µg/m³)': 120.0, 'NO2 (µg/m³)': 30.0,
    'CO (mg/m³)': 1.2, 'Ozone (µg/m³)': 25.0,
    'PM10_lag6h': 110.0, 'PM10_lag24h': 135.0,
    'PM10_roll6h_mean': 118.0, 'PM10_roll24h_mean': 125.0, 'PM10_roll24h_std': 14.0,
    'Area_Industrial': 1, 'Season_Winter': 1, 'Region_Hadapsar': 1,
}

row = pd.DataFrame([{col: sample.get(col, 0) for col in all_features}])
row[num_features] = scaler.transform(row[num_features])
predicted_aqi = float(np.clip(model.predict(row), 0, 500)[0])
print(f"Predicted AQI: {predicted_aqi:.1f}")
""")

print("\n" + "=" * 60)
print("ALL STEPS COMPLETE — outputs saved to models/ and output/")
print("=" * 60)