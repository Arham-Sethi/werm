import requests
import pandas as pd
from datetime import date, timedelta
from pathlib import Path
import math

# ================= CONFIG =================
LAT = 22.783044
LON = 75.947837

DAYS = 7
T_BASE = 10.0  # °C base temperature

OUT_PATH = Path("/content/drive/MyDrive/WERM/outputs/temp/gdd.csv")
# ==========================================

POWER_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"
FILL_VALUES = {-999, -999.0, -9999, -9999.0}

def fetch_power_daily(lat, lon, start_ymd, end_ymd):
    params = {
        "parameters": "T2M_MAX,T2M_MIN,ALLSKY_SFC_SW_DWN",
        "community": "AG",
        "longitude": lon,
        "latitude": lat,
        "start": start_ymd,
        "end": end_ymd,
        "format": "JSON",
    }
    r = requests.get(POWER_URL, params=params, timeout=60)
    r.raise_for_status()
    return r.json()

def is_valid(v):
    if v is None:
        return False
    try:
        v = float(v)
    except Exception:
        return False
    if math.isnan(v):
        return False
    return v not in FILL_VALUES

def gdd_day(tmax, tmin, base):
    tavg = (tmax + tmin) / 2.0
    return max(tavg - base, 0.0)

def main():
    # Fetch a wide window
    end = date.today()
    start = end - timedelta(days=30)

    js = fetch_power_daily(
        LAT, LON,
        start.strftime("%Y%m%d"),
        end.strftime("%Y%m%d")
    )

    param = js["properties"]["parameter"]

    tmax_series = param.get("T2M_MAX", {})
    tmin_series = param.get("T2M_MIN", {})
    sol_series  = param.get("ALLSKY_SFC_SW_DWN", {})

    all_keys = sorted(set(tmax_series.keys()) & set(tmin_series.keys()))
    if len(all_keys) == 0:
        raise RuntimeError("No overlapping temperature dates returned.")

    # 🔥 FILTER VALID DAYS FIRST
    valid_keys = []
    for k in all_keys:
        if is_valid(tmax_series.get(k)) and is_valid(tmin_series.get(k)):
            valid_keys.append(k)

    if len(valid_keys) < DAYS:
        print(f"WARNING: Only {len(valid_keys)} valid days available.")

    # Take last 7 VALID days
    keys = valid_keys[-DAYS:]

    rows = []
    for k in keys:
        tmax = float(tmax_series[k])
        tmin = float(tmin_series[k])
        solar = sol_series.get(k)
        solar = float(solar) if is_valid(solar) else None

        gdd = gdd_day(tmax, tmin, T_BASE)

        rows.append({
            "date": k,
            "T2M_MAX_C": tmax,
            "T2M_MIN_C": tmin,
            "Tavg_C": (tmax + tmin) / 2.0,
            "Daily_GDD_Cd": gdd,
            "ALLSKY_SFC_SW_DWN": solar
        })

    df = pd.DataFrame(rows)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)

    total_gdd = df["Daily_GDD_Cd"].sum()

    print("Saved:", OUT_PATH)
    print("Rows:", len(df))
    print("Date range:", df["date"].min(), "to", df["date"].max())
    print("Final cumulative GDD (last 7 valid days):", total_gdd)

if __name__ == "__main__":
    main()
