"""
Script Inferensi Prediksi TAF WALS (Thunderstorm Auto-Forecaster)
==================================================================
Memuat model .joblib untuk memprediksi probabilitas TS dalam 
horizon 3 jam mendatang berdasarkan data terkini.
"""

import os
import joblib
import numpy as np
import pandas as pd
from datetime import datetime, timezone
import requests

from fetch_metar_api import fetch_latest_metar
from parse_metar_structured import parse_one_line

# Path ke file model
MODEL_PATH = "models/taf_wals_ts_model.joblib"

# Urutan fitur harus sama persis seperti saat training di Colab
FEATURE_COLS = [
    "temp_c", "dewpoint_c", "dewpoint_depression", "qnh_hpa", 
    "wind_speed_kt", "wind_u", "wind_v", "visibility_m",
    "cape_(J/kg)", "lifted_index_()", "convective_inhibition_(J/kg)",
    "sin_hour", "cos_hour", "has_cb", "has_ts"
]

def load_model():
    """Memuat model LightGBM dari file .joblib"""
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    elif os.path.exists("taf_wals_ts_model.joblib"):
        return joblib.load("taf_wals_ts_model.joblib")
    else:
        raise FileNotFoundError(f"File model tidak ditemukan di {MODEL_PATH}")

def fetch_latest_nwp_openmeteo():
    """Mengambil parameter termodinamika terkini dari Open-Meteo API (WALS / Samarinda)."""
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": -0.3866,
        "longitude": 117.2322,
        "hourly": "cape,lifted_index,convective_inhibition",
        "timezone": "UTC",
        "forecast_days": 1
    }
    resp = requests.get(url, params=params, timeout=15)
    resp.raise_for_status()
    data = resp.json()["hourly"]
    
    return pd.DataFrame({
        "time": pd.to_datetime(data["time"], utc=True),
        "cape_(J/kg)": data["cape"],
        "lifted_index_()": data["lifted_index"],
        "convective_inhibition_(J/kg)": data["convective_inhibition"]
    })

def predict_current_ts_risk():
    """Fungsi utama inferensi real-time."""
    print("[INFERENSI] Memuat model Machine Learning...")
    model = load_model()
    
    # 1. Ambil METAR WALS Terkini
    print("[INFERENSI] Menarik METAR WALS terkini...")
    raw_lines = fetch_latest_metar("WALS")
    if not raw_lines:
        print("[ERROR] Tidak ada data METAR yang diterima.")
        return
        
    latest_raw = raw_lines[0]
    now_utc = datetime.now(timezone.utc)
    parsed = parse_one_line(latest_raw, now_utc)
    
    if not parsed:
        print("[ERROR] Gagal memparsing METAR.")
        return
        
    df_obs = pd.DataFrame([parsed])
    df_obs["valid_time_utc"] = pd.to_datetime(df_obs["valid_time_utc"])
    
    # 2. Ambil data NWP Open-Meteo
    print("[INFERENSI] Menarik data termodinamika NWP...")
    df_nwp = fetch_latest_nwp_openmeteo()
    
    # 3. Merging (Asof Join)
    df_merged = pd.merge_asof(
        df_obs.sort_values("valid_time_utc"),
        df_nwp.sort_values("time"),
        left_on="valid_time_utc",
        right_on="time",
        direction="nearest"
    )
    
    # 4. Feature Engineering
    df_merged["dewpoint_depression"] = df_merged["temp_c"] - df_merged["dewpoint_c"]
    rad = np.radians(df_merged["wind_dir_deg"].fillna(0))
    df_merged["wind_u"] = -df_merged["wind_speed_kt"].fillna(0) * np.sin(rad)
    df_merged["wind_v"] = -df_merged["wind_speed_kt"].fillna(0) * np.cos(rad)
    
    hour = df_merged["valid_time_utc"].dt.hour.values[0]
    df_merged["sin_hour"] = np.sin(2 * np.pi * hour / 24.0)
    df_merged["cos_hour"] = np.cos(2 * np.pi * hour / 24.0)
    
    X = df_merged[FEATURE_COLS].fillna(0)
    
    # 5. Prediksi Risk TS
    prob_ts = model.predict_proba(X)[0][1]
    prob_percent = prob_ts * 100
    
    print("\n" + "="*55)
    print(f"📌 METAR Terkini       : {latest_raw}")
    print(f"🕒 Waktu Observasi     : {df_merged['valid_time_utc'].values[0]}")
    print(f"🌩️ Probabilitas TS (+3j): {prob_percent:.2f}%")
    print("="*55)
    
    if prob_percent >= 40.0:
        taf_code = "PROB40 TSRA"
    elif prob_percent >= 30.0:
        taf_code = "PROB30 TSRA"
    else:
        taf_code = "NSW (No Significant Weather)"
        
    print(f"💡 Rekomendasi TAF     : {taf_code}\n")

if __name__ == "__main__":
    predict_current_ts_risk()