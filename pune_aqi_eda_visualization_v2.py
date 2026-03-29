import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import json
import os

# ══════════════════════════════════════════════════════════════════════════════
#  PUNE AQI — EDA + VISUALIZATION SCRIPT  (v2 — overlap fixes applied)
#  Input : pune_aqi_master_final.csv  (same folder as this script)
#  Output: 8 chart PNGs saved to output/
# ══════════════════════════════════════════════════════════════════════════════

df = pd.read_csv('pune_aqi_master_final.csv', low_memory=False)
df['Datetime_IST'] = pd.to_datetime(df['Datetime_IST'], utc=True)
df['YearMonth']    = df['Datetime_IST'].dt.to_period('M').astype(str)

os.makedirs('output', exist_ok=True)

# ── CONSTANTS ─────────────────────────────────────────────────────────────────
REGIONS = sorted(df['Region'].unique())

SEASON_ORDER  = ['Winter', 'Pre-Monsoon', 'Monsoon', 'Post-Monsoon']
AQI_CAT_ORDER = ['Good', 'Satisfactory', 'Moderate', 'Poor', 'Very Poor', 'Severe']

AQI_COLORS = {
    'Good':         '#2ecc71',
    'Satisfactory': '#a8e063',
    'Moderate':     '#f7dc6f',
    'Poor':         '#e67e22',
    'Very Poor':    '#e74c3c',
    'Severe':       '#8e44ad',
    'Unknown':      '#bdc3c7',
}

SEASON_COLORS = {
    'Winter':       '#5b8dee',
    'Pre-Monsoon':  '#f9a825',
    'Monsoon':      '#2ecc71',
    'Post-Monsoon': '#e67e22',
}

# Short names — eliminates overlapping long labels on axes
SHORT = {
    'Alandi':                'Alandi',
    'Bhosari':               'Bhosari',
    'Bhumkar Nagar':         'Bhumkar Ngr',
    'Hadapsar':              'Hadapsar',
    'Karve Road':            'Karve Rd',
    'Katraj Dairy':          'Katraj',
    'Mhada Colony':          'Mhada Col',
    'Panchawati Pashan':     'Pashan',
    'Transport Nagar Nigdi': 'Nigdi',
}

# Y-axis order for horizontal bars: cleanest → most polluted
SHORT_ORDER = [
    'Pashan', 'Bhosari', 'Katraj', 'Nigdi',
    'Karve Rd', 'Alandi', 'Bhumkar Ngr', 'Mhada Col', 'Hadapsar'
]

df['Short'] = df['Region'].map(SHORT)


# ── PRE-COMPUTE AGGREGATES ────────────────────────────────────────────────────

# 1. Region-wise mean AQI
region_aqi = df.groupby('Short')['AQI'].mean().sort_values().reset_index()
region_aqi.columns = ['Station', 'Mean_AQI']

# 2. Monthly average AQI (show every 3rd month on x-axis to prevent overlap)
monthly     = df.groupby(['YearMonth', 'Region'])['AQI'].mean().reset_index()
monthly_piv = monthly.pivot(index='YearMonth', columns='Region', values='AQI').sort_index()
all_months  = list(monthly_piv.index)
tick_vals   = all_months[::3]

# 3. Hourly diurnal PM2.5
hourly_pm25 = df.groupby(['Hour', 'Short'])['PM2.5 (µg/m³)'].mean().reset_index()
hourly_piv  = hourly_pm25.pivot(index='Hour', columns='Short', values='PM2.5 (µg/m³)')

# 4. AQI category % per region
cat_pct = (
    df.groupby(['Short', 'AQI_Category'])
      .size()
      .unstack(fill_value=0)
      .apply(lambda r: r / r.sum() * 100, axis=1)
      .reset_index()
)

# 5. Dominant pollutant counts
dom = df['AQI_Dominant_Pollutant'].value_counts().reset_index()
dom.columns = ['Pollutant', 'Count']
dom = dom[dom['Pollutant'].notna() & (dom['Pollutant'] != 'nan')]

# 6. Correlation matrix
corr_cols   = ['PM2.5 (µg/m³)', 'PM10 (µg/m³)', 'NO2 (µg/m³)', 'NOx (ppb)',
               'CO (mg/m³)', 'Ozone (µg/m³)', 'Benzene (µg/m³)', 'AQI']
corr        = df[corr_cols].corr().round(2)
corr_labels = ['PM2.5', 'PM10', 'NO2', 'NOx', 'CO', 'Ozone', 'Benzene', 'AQI']

# 7. Weekday vs Weekend PM2.5
wk_raw = df.groupby(['Short', 'Is_Weekend'])['PM2.5 (µg/m³)'].mean().reset_index()
wk_raw['Day_Type'] = wk_raw['Is_Weekend'].map({0: 'Weekday', 1: 'Weekend'})
wk_pivot = wk_raw.pivot(index='Short', columns='Day_Type',
                         values='PM2.5 (µg/m³)').reset_index()

# 8. Seasonal AQI
season_aqi = (
    df[df['AQI_Category'] != 'Unknown']
    .groupby(['Season', 'Short'])['AQI']
    .mean().reset_index()
)
season_aqi.columns = ['Season', 'Station', 'AQI']


# ── CHART 1: Region-wise Mean AQI (horizontal bar) ───────────────────────────
print("Generating Chart 1: Region-wise Mean AQI...")

bar_colors = [
    '#2ecc71' if v <= 50 else
    '#a8e063' if v <= 100 else
    '#f7dc6f' if v <= 200 else
    '#e67e22'
    for v in region_aqi['Mean_AQI']
]

fig1 = go.Figure(go.Bar(
    y=region_aqi['Station'],
    x=region_aqi['Mean_AQI'].round(1),
    orientation='h',
    marker_color=bar_colors,
    text=region_aqi['Mean_AQI'].round(1),
    textposition='outside',
))
fig1.update_layout(
    title={"text": "Hadapsar Worst, Pashan Cleanest in Pune (2024–25)<br>"
                   "<span style='font-size:16px;font-weight:normal;'>Mean AQI by station | CPCB standard</span>"},
    margin=dict(t=110, b=60, l=100, r=60),
    width=850, height=480,
)
fig1.update_xaxes(title_text='Mean AQI', tickfont=dict(size=11), range=[0, 145])
fig1.update_yaxes(title_text='Station', tickfont=dict(size=12))
fig1.update_traces(cliponaxis=False)
fig1.write_image('output/chart1_region_aqi.png')
with open('output/chart1_region_aqi.png.meta.json', 'w') as f:
    json.dump({"caption": "Mean AQI by Pune region (2024–25)",
               "description": "Horizontal bar showing avg AQI per station, color-coded by CPCB category."}, f)


# ── CHART 2: Monthly AQI Trend (multi-line) ───────────────────────────────────
print("Generating Chart 2: Monthly AQI trend...")

fig2 = go.Figure()
for region in REGIONS:
    if region in monthly_piv.columns:
        fig2.add_trace(go.Scatter(
            x=all_months,
            y=list(monthly_piv[region]),
            mode='lines+markers',
            name=SHORT[region],
            line=dict(width=2),
            marker=dict(size=5),
        ))
fig2.update_layout(
    title={"text": "Winter Spike & Monsoon Dip Across All Regions (2024–25)<br>"
                   "<span style='font-size:16px;font-weight:normal;'>Monthly mean AQI | Jun–Sep lowest</span>"},
    legend=dict(orientation='h', yanchor='bottom', y=1.08,
                xanchor='center', x=0.5, font=dict(size=11)),
    margin=dict(t=130, b=80, l=60, r=20),
    width=1100, height=550,
)
fig2.update_xaxes(title_text='Month', tickvals=tick_vals, ticktext=tick_vals,
                  tickangle=45, tickfont=dict(size=11))
fig2.update_yaxes(title_text='Mean AQI', tickfont=dict(size=11))
fig2.write_image('output/chart2_monthly_trend.png')
with open('output/chart2_monthly_trend.png.meta.json', 'w') as f:
    json.dump({"caption": "Monthly AQI trend by region (2024–25)",
               "description": "Line chart monthly avg AQI all 9 regions. Winter peaks and monsoon dip visible."}, f)


# ── CHART 3: Diurnal PM2.5 Cycle (hourly) ────────────────────────────────────
print("Generating Chart 3: Diurnal PM2.5 cycle...")

fig3 = go.Figure()
for short in [SHORT[r] for r in REGIONS]:
    if short in hourly_piv.columns:
        fig3.add_trace(go.Scatter(
            x=list(hourly_piv.index),
            y=list(hourly_piv[short]),
            mode='lines',
            name=short,
            line=dict(width=2),
        ))
fig3.update_layout(
    title={"text": "PM2.5 Peaks Late Night & Afternoon — Not Rush Hour<br>"
                   "<span style='font-size:16px;font-weight:normal;'>Hourly mean PM2.5 | 3–4AM & 1–3PM peaks</span>"},
    legend=dict(orientation='h', yanchor='bottom', y=1.08,
                xanchor='center', x=0.5, font=dict(size=11)),
    margin=dict(t=130, b=70, l=70, r=20),
    width=1000, height=500,
)
fig3.update_xaxes(title_text='Hour of Day (IST)',
                  tickvals=list(range(0, 24, 2)), tickfont=dict(size=11))
fig3.update_yaxes(title_text='PM2.5 (µg/m³)', tickfont=dict(size=11))
fig3.write_image('output/chart3_diurnal_pm25.png')
with open('output/chart3_diurnal_pm25.png.meta.json', 'w') as f:
    json.dump({"caption": "Diurnal PM2.5 pattern by region",
               "description": "Hourly avg PM2.5 showing 3–4AM and 1–3PM peaks across all regions."}, f)


# ── CHART 4: AQI Category Stacked Bar ────────────────────────────────────────
print("Generating Chart 4: AQI category stacked bar...")

fig4 = go.Figure()
for cat in AQI_CAT_ORDER:
    if cat in cat_pct.columns:
        fig4.add_trace(go.Bar(
            name=cat,
            x=cat_pct['Short'],
            y=cat_pct[cat].round(1),
            marker_color=AQI_COLORS[cat],
        ))
fig4.update_layout(
    barmode='stack',
    title={"text": "Hadapsar Spends Most Hours in Moderate+ AQI (2024–25)<br>"
                   "<span style='font-size:16px;font-weight:normal;'>% of hours per AQI category | CPCB standard</span>"},
    legend=dict(orientation='h', yanchor='bottom', y=1.08,
                xanchor='center', x=0.5, font=dict(size=12)),
    margin=dict(t=130, b=80, l=70, r=20),
    width=1000, height=520,
)
fig4.update_xaxes(title_text='Station', tickfont=dict(size=12), tickangle=0)
fig4.update_yaxes(title_text='% of Hours', tickfont=dict(size=11))
fig4.write_image('output/chart4_aqi_cat_stack.png')
with open('output/chart4_aqi_cat_stack.png.meta.json', 'w') as f:
    json.dump({"caption": "AQI category distribution per region (% of hours)",
               "description": "Stacked bar showing time proportion in each CPCB AQI category per station."}, f)


# ── CHART 5: Dominant Pollutant Bar ──────────────────────────────────────────
print("Generating Chart 5: Dominant pollutant bar...")

fig5 = go.Figure(go.Bar(
    x=dom['Pollutant'],
    y=dom['Count'],
    text=dom['Count'],
    textposition='outside',
    marker_color='#3498db',
))
fig5.update_layout(
    title={"text": "PM10 Drives AQI Most Often Across Pune (2024–25)<br>"
                   "<span style='font-size:16px;font-weight:normal;'>Hours each pollutant was dominant | all 9 stations</span>"},
    margin=dict(t=110, b=60, l=70, r=20),
    width=800, height=460,
)
fig5.update_xaxes(title_text='Pollutant', tickfont=dict(size=12))
fig5.update_yaxes(title_text='Hours as Dominant', tickfont=dict(size=11))
fig5.update_traces(cliponaxis=False)
fig5.write_image('output/chart5_dominant_pollutant.png')
with open('output/chart5_dominant_pollutant.png.meta.json', 'w') as f:
    json.dump({"caption": "Dominant AQI pollutant (all regions, 2024–25)",
               "description": "Bar chart of hours each pollutant was dominant driver of AQI."}, f)


# ── CHART 6: Pollutant Correlation Heatmap ────────────────────────────────────
print("Generating Chart 6: Correlation heatmap...")

fig6 = go.Figure(go.Heatmap(
    z=corr.values,
    x=corr_labels,
    y=corr_labels,
    colorscale='RdBu_r',
    zmid=0, zmin=-1, zmax=1,
    text=corr.values,
    texttemplate='%{text}',
    textfont=dict(size=11),
))
fig6.update_layout(
    title={"text": "PM2.5 & PM10 Most Correlated with AQI<br>"
                   "<span style='font-size:16px;font-weight:normal;'>Pearson correlation matrix | pollutants vs AQI</span>"},
    margin=dict(t=110, b=60, l=80, r=20),
    width=700, height=600,
)
fig6.update_xaxes(title_text='Pollutant', tickfont=dict(size=11))
fig6.update_yaxes(title_text='Pollutant', tickfont=dict(size=11))
fig6.write_image('output/chart6_corr_heatmap.png')
with open('output/chart6_corr_heatmap.png.meta.json', 'w') as f:
    json.dump({"caption": "Pollutant correlation heatmap (Pearson r)",
               "description": "Pearson correlation matrix of key pollutants and computed AQI."}, f)


# ── CHART 7: Weekday vs Weekend PM2.5 (horizontal grouped) ───────────────────
print("Generating Chart 7: Weekday vs Weekend PM2.5...")

fig7 = go.Figure()
fig7.add_trace(go.Bar(
    y=wk_pivot['Short'],
    x=wk_pivot['Weekday'].round(1),
    name='Weekday',
    orientation='h',
    text=wk_pivot['Weekday'].round(1),
    textposition='auto',
))
fig7.add_trace(go.Bar(
    y=wk_pivot['Short'],
    x=wk_pivot['Weekend'].round(1),
    name='Weekend',
    orientation='h',
    text=wk_pivot['Weekend'].round(1),
    textposition='auto',
))
fig7.update_layout(
    barmode='group',
    title={"text": "Weekday vs Weekend PM2.5 by Station (2024–25)<br>"
                   "<span style='font-size:16px;font-weight:normal;'>Mean PM2.5 µg/m³ | residential zones higher on weekends</span>"},
    legend=dict(orientation='h', yanchor='bottom', y=1.08,
                xanchor='center', x=0.5, font=dict(size=13)),
    margin=dict(t=130, b=60, l=110, r=30),
    width=900, height=520,
)
fig7.update_xaxes(title_text='PM2.5 (µg/m³)', tickfont=dict(size=11))
fig7.update_yaxes(title_text='Station', tickfont=dict(size=12),
                  categoryorder='array', categoryarray=SHORT_ORDER)
fig7.update_traces(cliponaxis=False)
fig7.write_image('output/chart7_weekday_weekend.png')
with open('output/chart7_weekday_weekend.png.meta.json', 'w') as f:
    json.dump({"caption": "Weekday vs Weekend PM2.5 by region",
               "description": "Horizontal grouped bar comparing weekday vs weekend avg PM2.5 across 9 stations."}, f)


# ── CHART 8: Seasonal AQI per Region (horizontal grouped) ────────────────────
print("Generating Chart 8: Seasonal AQI comparison...")

fig8 = go.Figure()
for season in SEASON_ORDER:
    sub = season_aqi[season_aqi['Season'] == season]
    fig8.add_trace(go.Bar(
        y=sub['Station'],
        x=sub['AQI'].round(1),
        name=season,
        orientation='h',
        marker_color=SEASON_COLORS[season],
        text=sub['AQI'].round(0).astype(int),
        textposition='auto',
    ))
fig8.update_layout(
    barmode='group',
    title={"text": "Winter Worst, Monsoon Best Across Every Station (2024–25)<br>"
                   "<span style='font-size:16px;font-weight:normal;'>Mean AQI by season | 9 Pune stations</span>"},
    legend=dict(orientation='h', yanchor='bottom', y=1.08,
                xanchor='center', x=0.5, font=dict(size=13)),
    margin=dict(t=130, b=60, l=110, r=30),
    width=1000, height=580,
)
fig8.update_xaxes(title_text='Mean AQI', tickfont=dict(size=11))
fig8.update_yaxes(title_text='Station', tickfont=dict(size=12),
                  categoryorder='array', categoryarray=SHORT_ORDER)
fig8.update_traces(cliponaxis=False)
fig8.write_image('output/chart8_seasonal_aqi.png')
with open('output/chart8_seasonal_aqi.png.meta.json', 'w') as f:
    json.dump({"caption": "Seasonal mean AQI comparison by region",
               "description": "Horizontal grouped bar of seasonal AQI ordered cleanest to most polluted."}, f)


print("\nAll 8 charts saved to output/")
charts = [
    "chart1_region_aqi.png          — Region-wise Mean AQI",
    "chart2_monthly_trend.png       — Monthly AQI trend (all regions)",
    "chart3_diurnal_pm25.png        — Hourly PM2.5 diurnal cycle",
    "chart4_aqi_cat_stack.png       — AQI category % distribution",
    "chart5_dominant_pollutant.png  — Dominant pollutant bar",
    "chart6_corr_heatmap.png        — Pollutant correlation heatmap",
    "chart7_weekday_weekend.png     — Weekday vs Weekend PM2.5",
    "chart8_seasonal_aqi.png        — Seasonal AQI by region",
]
for c in charts:
    print(f"  {c}")
