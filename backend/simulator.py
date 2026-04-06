import pandas as pd
import time
from db import get_connection

# ── CONFIG ─────────────────────────────────────────
CSV_PATH = "../pune_aqi_master_final.csv"
SLEEP_SECONDS = 2  # 1–5 sec = faster/slower simulation

# ── LOAD ───────────────────────────────────────────
df = pd.read_csv(CSV_PATH)
df["Datetime_IST"] = pd.to_datetime(df["Datetime_IST"])

# keep ONLY required columns (no engineered features)
cols = [
    "Region",
    "Datetime_IST",
    "PM2.5 (µg/m³)",
    "PM10 (µg/m³)",
    "NO2 (µg/m³)",
    "CO (mg/m³)",
    "Ozone (µg/m³)",
    "AQI",
    "AQI_Category",
]
df = df[cols].sort_values(["Region", "Datetime_IST"]).reset_index(drop=True)

# ── DB ─────────────────────────────────────────────
conn = get_connection()
cur = conn.cursor()

print("Starting stream...")

for i, r in df.iterrows():
    import pandas as pd
import time
from db import get_connection

# ── LOAD CSV ─────────────────────────────────────────
df = pd.read_csv("../pune_aqi_master_final.csv")
df["Datetime_IST"] = pd.to_datetime(df["Datetime_IST"])

df = df.sort_values(["Datetime_IST", "Region"]).reset_index(drop=True)

# ── DB CONNECTION ───────────────────────────────────
conn = get_connection()
cur = conn.cursor()

print("Starting stream...")

for i, r in df.iterrows():

    # handle area_type safely (based on your dataset reality)
    area_type = None
    if "Area_Type" in df.columns:
        area_type = r["Area_Type"]
    elif "Area Type" in df.columns:
        area_type = r["Area Type"]
    else:
        area_type = "Urban"   # safe default

    cur.execute(
        """
        INSERT INTO aqi_data (
            region,
            datetime,
            pm25,
            pm10,
            no2,
            co,
            o3,
            area_type
        )
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
        """,
        (
            r["Region"],
            r["Datetime_IST"],
            r["PM2.5 (µg/m³)"],
            r["PM10 (µg/m³)"],
            r["NO2 (µg/m³)"],
            r["CO (mg/m³)"],
            r["Ozone (µg/m³)"],   # CSV → DB(o3)
            area_type
        ),
    )

    conn.commit()

    print(f"{i+1} inserted | {r['Region']}")

    # time.sleep(1)
    conn.commit()

    print(f"{i+1} inserted | {r['Region']} | {r['Datetime_IST']}")
    time.sleep(SLEEP_SECONDS)