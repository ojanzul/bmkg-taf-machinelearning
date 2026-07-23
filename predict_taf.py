"""
Script Inferensi Auto-Forecaster TAF WALS
=========================================
Menarik data METAR & NWP terkini, memprediksi risiko TS (+3 jam),
dan mengirimkan hasilnya ke tabel 'taf_predictions' di Supabase.
"""

import os
import joblib
import numpy as np
import pandas as pd
from datetime import datetime, timezone
import requests
from supabase import create_client

from fetch_metar_api import fetch_latest_metar
from parse_metar_structured import parse_one_line

MODEL_PATH = "models/taf_wals_ts_model.joblib"

FEATURE_COLS = [
    "temp_c", "dewpoint_c", "dewpoint_depression", "qnh_hpa", 
    "wind_speed_kt", "wind_u", "wind_v", "visibility_m",
    "cape_(J/kg)", "lifted_index_()", "convective_inhibition_(J/kg)",
    "sin_hour", "cos_hour", "has_cb", "has_ts"
]

def get_supabase_client():
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_KEY")
    if not url or not key:
        print("[WARN] Key Supabase tidak ditemukan, hasil tidak disimpann ke DB.")
        return None
    return create_client(url, key)

def load_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    elif os.path.exists("taf_wals_ts_model.joblib"):
        return joblib.load("taf_wals_ts_model.joblib")
    else:
        raise FileNotFoundError(f"Model tidak ditemukan di {MODEL_PATH}")

def fetch_latest_nwp_openmeteo():
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

def predict_and_save():
    print("[INFERENSI] Memuat model Machine Learning...")
    model = load_model()
    
    print("[INFERENSI] Menarik METAR WALS terkini...")
    raw_lines = fetch_latest_metar("WALS")
    if not raw_lines:
        print("[ERROR] Tidak ada data METAR.")
        return
        
    latest_raw = raw_lines[0]
    now_utc = datetime.now(timezone.utc)
    parsed = parse_one_line(latest_raw, now_utc)
    if not parsed:
        print("[ERROR] Gagal memparsing METAR.")
        return
        
    df_obs = pd.DataFrame([parsed])
    df_obs["valid_time_utc"] = pd.to_datetime(df_obs["valid_time_utc"], utc=True)
    
    print("[INFERENSI] Menarik data NWP Open-Meteo...")
    df_nwp = fetch_latest_nwp_openmeteo()
    
    df_merged = pd.merge_asof(
        df_obs.sort_values("valid_time_utc"),
        df_nwp.sort_values("time"),
        left_on="valid_time_utc",
        right_on="time",
        direction="nearest"
    )
    
    df_merged["dewpoint_depression"] = df_merged["temp_c"] - df_merged["dewpoint_c"]
    rad = np.radians(df_merged["wind_dir_deg"].fillna(0))
    df_merged["wind_u"] = -df_merged["wind_speed_kt"].fillna(0) * np.sin(rad)
    df_merged["wind_v"] = -df_merged["wind_speed_kt"].fillna(0) * np.cos(rad)
    
    hour = df_merged["valid_time_utc"].dt.hour.values[0]
    df_merged["sin_hour"] = np.sin(2 * np.pi * hour / 24.0)
    df_merged["cos_hour"] = np.cos(2 * np.pi * hour / 24.0)
    
    X = df_merged[FEATURE_COLS].fillna(0)
    prob_ts = float(model.predict_proba(X)[0][1])
    prob_percent = prob_ts * 100
    
    if prob_percent >= 40.0:
        taf_code = "PROB40 TSRA"
    elif prob_percent >= 30.0:
        taf_code = "PROB30 TSRA"
    else:
        taf_code = "NSW (No Significant Weather)"
        
    valid_time_str = df_merged["valid_time_utc"].dt.strftime("%Y-%m-%dT%H:%M:%SZ").values[0]
    
    print("\n" + "="*55)
    print(f"📌 METAR Terkini       : {latest_raw}")
    print(f"🕒 Waktu Observasi     : {valid_time_str}")
    print(f"🌩️ Probabilitas TS (+3j): {prob_percent:.2f}%")
    print(f"💡 Rekomendasi TAF     : {taf_code}")
    print("="*55)
    
    # Simpan ke Supabase
    supabase = get_supabase_client()
    if supabase:
        record = {
            "valid_time_utc": valid_time_str,
            "metar_raw": latest_raw,
            "prob_ts": prob_ts,
            "taf_suggestion": taf_code
        }
        try:
            supabase.table("taf_predictions").upsert(record, on_conflict="valid_time_utc").execute()
            print("[SUPABASE] Berhasil menyimpan prediksi ke database!")
        except Exception as e:
            print(f"[SUPABASE ERROR] {e}")

if __name__ == "__main__":
    predict_and_save()
