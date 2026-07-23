"""
daily_inference.py
Dijalankan tiap TAF issuance time (00/06/12/18Z) via GitHub Actions cron
terpisah dari update_metar_dataset.yml.

Alur:
1. Ambil rolling window observasi METAR terakhir (12+ jam) dari Supabase
2. Ambil forecast NWP ASLI (bukan reanalysis) dari Open-Meteo Forecast API --
   past_days=1 dipakai supaya field yang sama juga tersedia utk beberapa jam
   ke belakang (jaga-jaga kalau butuh overlap dgn window METAR)
3. Hitung fitur pakai taf_features.py (modul SAMA yang dipakai training)
4. Load model_bundle_v1.joblib, prediksi 15 kombinasi (3 fenomena x 5 horizon)
5. Simpan hasil ke Supabase (tabel taf_predictions) untuk monitoring performa
   asli vs prediksi -- PENTING karena model ini dilatih pakai ERA5
   reanalysis (hindsight), performa asli di produksi (pakai forecast asli)
   perlu diukur, bukan diasumsikan sama dengan AUC hasil eksperimen offline.

ENV VARS yang dibutuhkan (GitHub Actions secrets):
  SUPABASE_URL, SUPABASE_KEY   -- sama seperti run_pipeline.py
  MODEL_BUNDLE_PATH (opsional) -- default "model_bundle_v1.joblib"
"""
import os
import sys
from datetime import datetime, timedelta, timezone

import joblib
import pandas as pd
import requests

from parse_metar_structured import parse_one_line, FIELDNAMES
from taf_features import compute_past_features, compute_future_features, HORIZONS, WX_TARGETS

ICAO = "WALS"
STATION_LAT, STATION_LON = -0.374, 117.255
MODEL_BUNDLE_PATH = os.environ.get("MODEL_BUNDLE_PATH", "model_bundle_v1.joblib")

NWP_HOURLY_VARS = (
    "temperature_2m,relative_humidity_2m,dew_point_2m,precipitation,"
    "surface_pressure,pressure_msl,cloud_cover,cloud_cover_low,cloud_cover_mid,"
    "cloud_cover_high,wind_speed_10m,wind_direction_10m,wind_gusts_10m,"
    "vapour_pressure_deficit,shortwave_radiation"
)


def fetch_recent_metar_from_supabase(hours_back: int = 18) -> pd.DataFrame:
    from supabase import create_client

    url, key = os.environ.get("SUPABASE_URL"), os.environ.get("SUPABASE_KEY")
    if not (url and key):
        raise RuntimeError("SUPABASE_URL/SUPABASE_KEY belum di-set -- tidak bisa ambil data observasi.")
    client = create_client(url, key)

    since = (datetime.now(timezone.utc) - timedelta(hours=hours_back)).isoformat()
    resp = (
        client.table("metar_observations")
        .select("station,kind,obs_datetime,raw_text")
        .eq("station", ICAO)
        .gte("obs_datetime", since)
        .order("obs_datetime")
        .execute()
    )
    if not resp.data:
        raise RuntimeError(
            f"Tidak ada observasi METAR {ICAO} dalam {hours_back} jam terakhir di Supabase. "
            f"Cek apakah update_metar_dataset.yml jalan normal."
        )

    rows = []
    for rec in resp.data:
        obs_dt = pd.Timestamp(rec["obs_datetime"]).to_pydatetime()
        parsed = parse_one_line(rec["raw_text"], obs_dt)
        if parsed:
            rows.append(parsed)

    df = pd.DataFrame(rows, columns=FIELDNAMES)
    df["valid_time_utc"] = pd.to_datetime(df["valid_time_utc"])
    return df.sort_values("valid_time_utc").reset_index(drop=True).set_index("valid_time_utc")


def fetch_forecast_nwp() -> pd.DataFrame:
    """Forecast NWP ASLI dari Open-Meteo -- past_days utk nowcast state, forecast_days utk fitur masa depan."""
    params = {
        "latitude": STATION_LAT,
        "longitude": STATION_LON,
        "hourly": NWP_HOURLY_VARS,
        "past_days": 1,
        "forecast_days": 2,
        "timezone": "UTC",
        "wind_speed_unit": "kn",
    }
    resp = requests.get("https://api.open-meteo.com/v1/forecast", params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()["hourly"]

    df = pd.DataFrame(data)
    df.columns = [c.split(" (")[0] for c in df.columns]  # jaga-jaga kalau API kirim unit di nama kolom
    df["time"] = pd.to_datetime(df["time"])
    return df.set_index("time").sort_index()


def run_inference(issue_time: pd.Timestamp, metar_df: pd.DataFrame, nwp_df: pd.DataFrame,
                   bundle: dict) -> list[dict]:
    past_feat = compute_past_features(issue_time, metar_df, nwp_df)
    if past_feat is None:
        raise RuntimeError(
            f"Data observasi/NWP terlalu sedikit di sekitar {issue_time} untuk hitung fitur "
            f"'past window' -- cek kelengkapan data 12 jam terakhir."
        )

    predictions = []
    for wx in WX_TARGETS:
        for h in HORIZONS:
            key = f"{wx}_h{h}"
            fut_feat = compute_future_features(issue_time, h, nwp_df)
            if fut_feat is None:
                print(f"[WARN] fitur future utk {key} tidak lengkap, dilewati", file=sys.stderr)
                continue

            feat_cols = bundle["features"][key]
            row = {**past_feat, **fut_feat}
            X = pd.DataFrame([[row.get(c) for c in feat_cols]], columns=feat_cols)

            model = bundle["models"][key]
            proba = float(model.predict_proba(X)[0, 1])

            predictions.append({
                "station": ICAO,
                "issue_time": issue_time.isoformat(),
                "target": wx,
                "horizon_hours": h,
                "probability": round(proba, 4),
                "model_generation": bundle["trained_on"]["generation"],
            })
    return predictions


def save_predictions_to_supabase(predictions: list[dict]) -> int:
    from supabase import create_client

    url, key = os.environ.get("SUPABASE_URL"), os.environ.get("SUPABASE_KEY")
    if not (url and key) or not predictions:
        return 0
    client = create_client(url, key)
    resp = (
        client.table("taf_predictions")
        .upsert(predictions, on_conflict="station,issue_time,target,horizon_hours")
        .execute()
    )
    return len(resp.data) if resp.data else 0


def main():
    issue_time = pd.Timestamp(datetime.now(timezone.utc)).floor("h")
    print(f"[INFO] Inferensi untuk issue_time={issue_time.isoformat()}")

    bundle = joblib.load(MODEL_BUNDLE_PATH)
    print(f"[INFO] Model bundle generasi {bundle['trained_on']['generation']}, "
          f"dilatih dari {bundle['trained_on']['n_rows']} baris "
          f"({bundle['trained_on']['date_range'][0]} s/d {bundle['trained_on']['date_range'][1]})")
    print(f"[INFO] {bundle['trained_on']['note']}")

    metar_df = fetch_recent_metar_from_supabase()
    print(f"[INFO] {len(metar_df)} observasi METAR diambil dari Supabase")

    nwp_df = fetch_forecast_nwp()
    print(f"[INFO] {len(nwp_df)} baris NWP forecast diambil dari Open-Meteo")

    predictions = run_inference(issue_time, metar_df, nwp_df, bundle)

    print("\n=== HASIL PREDIKSI ===")
    for p in predictions:
        print(f"  {p['target']:6s} h+{p['horizon_hours']:>2}j : {p['probability']*100:5.1f}%")

    n_saved = save_predictions_to_supabase(predictions)
    print(f"\n[SUPABASE] {n_saved} prediksi disimpan ke tabel taf_predictions")


if __name__ == "__main__":
    main()
