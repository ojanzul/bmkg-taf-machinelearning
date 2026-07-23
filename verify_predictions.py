"""
verify_predictions.py
=========================
Dijalankan berkala (disarankan tiap jam atau tiap beberapa jam) via
GitHub Actions, terpisah dari update_metar_dataset.yml dan
daily_inference.py.

Alur:
1. Ambil prediksi dari tabel taf_predictions yang:
   - Horizonnya SUDAH LEWAT (issue_time + horizon_hours <= sekarang - buffer)
   - Belum diverifikasi (actual_outcome masih NULL)
2. Untuk tiap prediksi, ambil observasi METAR aktual di window
   [issue_time, issue_time + horizon_hours] dari metar_observations
   (inklusif di kedua ujung -- PERSIS sama dengan definisi
   label_for_horizon() saat training, supaya tidak ada mismatch)
3. Cek apakah fenomena yang diprediksi (TS/RA/BR_FG) BENAR terjadi --
   pakai definisi PERSIS SAMA dengan label_for_horizon() di
   taf_features.py (supaya konsisten dengan definisi label saat training)
4. Tulis actual_outcome + verified_at balik ke Supabase

PENTING soal buffer waktu: kita kasih jeda 1 jam setelah horizon lewat
sebelum verifikasi, supaya observasi METAR yang relevan sudah sempat
ter-ingest oleh update_metar_dataset.yml (yang jalan tiap 30 menit,
tapi bisa saja telat beberapa menit).

PENTING soal data hilang: kalau observasi METAR di window itu kosong
atau terlalu sedikit (< 2 baris, sama seperti syarat di
label_for_horizon), prediksi itu DILEWATI (bukan ditandai gagal) --
akan dicoba lagi di run berikutnya. Ini mencegah "actual_outcome=0"
palsu yang sebenarnya cuma karena data belum lengkap, bukan karena
memang tidak terjadi apa-apa.

ENV VARS: SUPABASE_URL, SUPABASE_KEY (sama seperti script lain)
"""

import os
import sys
from datetime import datetime, timedelta, timezone

import pandas as pd

from parse_metar_structured import parse_one_line, FIELDNAMES
from taf_features import label_for_horizon

ICAO = "WALS"
VERIFY_BUFFER_HOURS = 1  # jeda setelah horizon lewat, sebelum dicoba verifikasi
BATCH_SIZE = 200  # jaga-jaga kalau backlog prediksi belum-terverifikasi besar


def get_supabase_client():
    from supabase import create_client

    url, key = os.environ.get("SUPABASE_URL"), os.environ.get("SUPABASE_KEY")
    if not (url and key):
        raise RuntimeError("SUPABASE_URL/SUPABASE_KEY belum di-set.")
    return create_client(url, key)


def fetch_verifiable_predictions(client) -> list[dict]:
    """
    Ambil prediksi yang horizonnya sudah lewat + belum ada actual_outcome.
    Filter "horizon sudah lewat" dihitung di Python (bukan di query SQL)
    karena horizon_hours beda-beda tiap baris -- lebih gampang dan aman
    difilter di sisi klien untuk volume data sekelas ini.
    """
    resp = (
        client.table("taf_predictions")
        .select("id,station,issue_time,target,horizon_hours,probability")
        .eq("station", ICAO)
        .is_("actual_outcome", "null")
        .order("issue_time")
        .limit(BATCH_SIZE)
        .execute()
    )
    if not resp.data:
        return []

    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(hours=VERIFY_BUFFER_HOURS)

    verifiable = []
    for row in resp.data:
        issue_time = pd.Timestamp(row["issue_time"])
        if issue_time.tzinfo is None:
            issue_time = issue_time.tz_localize("UTC")
        deadline = issue_time + timedelta(hours=row["horizon_hours"])
        if deadline <= cutoff:
            verifiable.append(row)
    return verifiable


def fetch_metar_window(client, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Ambil & parse observasi METAR di window (start, end] dari Supabase."""
    resp = (
        client.table("metar_observations")
        .select("obs_datetime,raw_text")
        .eq("station", ICAO)
        .gte("obs_datetime", start.isoformat())
        .lte("obs_datetime", end.isoformat())
        .order("obs_datetime")
        .execute()
    )
    rows = []
    for rec in resp.data or []:
        obs_dt = pd.Timestamp(rec["obs_datetime"]).to_pydatetime()
        parsed = parse_one_line(rec["raw_text"], obs_dt)
        if parsed:
            rows.append(parsed)

    if not rows:
        return pd.DataFrame(columns=FIELDNAMES).set_index(pd.DatetimeIndex([]))

    df = pd.DataFrame(rows, columns=FIELDNAMES)
    df["valid_time_utc"] = pd.to_datetime(df["valid_time_utc"])
    df = df.sort_values("valid_time_utc").set_index("valid_time_utc")
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    return df


def main():
    client = get_supabase_client()

    predictions = fetch_verifiable_predictions(client)
    print(f"[INFO] {len(predictions)} prediksi siap diverifikasi (horizon sudah lewat)")

    n_verified, n_skipped_no_data = 0, 0

    for pred in predictions:
        issue_time = pd.Timestamp(pred["issue_time"])
        if issue_time.tzinfo is not None:
            issue_time = issue_time.tz_localize(None)
        h = pred["horizon_hours"]

        # Ambil metar window dengan sedikit buffer di awal (butuh state
        # sebelum issue_time juga kalau nanti label_for_horizon perlu --
        # untuk definisi saat ini cukup window (issue_time, issue_time+h])
        window_start = issue_time
        window_end = issue_time + timedelta(hours=h)
        metar_df = fetch_metar_window(
            client,
            pd.Timestamp(window_start).tz_localize("UTC"),
            pd.Timestamp(window_end).tz_localize("UTC"),
        )

        outcome = label_for_horizon(issue_time, h, metar_df, pred["target"])
        if outcome is None:
            n_skipped_no_data += 1
            continue

        client.table("taf_predictions").update({
            "actual_outcome": outcome,
            "verified_at": datetime.now(timezone.utc).isoformat(),
        }).eq("id", pred["id"]).execute()
        n_verified += 1

    print(f"[SELESAI] {n_verified} diverifikasi, {n_skipped_no_data} dilewati (data METAR belum lengkap, dicoba lagi nanti)")


if __name__ == "__main__":
    main()
