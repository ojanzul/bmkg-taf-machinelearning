"""
generate_taf_draft.py
=========================
Script MANUAL (tidak dijadwalkan otomatis) -- jalankan kapan saja Anda
mau lihat draft TAF terkini:

    python generate_taf_draft.py

CARA LIHAT HASILNYA:
1. Langsung tercetak di terminal saat script selesai jalan.
2. Juga otomatis tersimpan ke tabel `taf_drafts` di Supabase -- buka
   Supabase Dashboard > Table Editor > taf_drafts kapan saja untuk
   lihat draft-draft yang pernah di-generate, tanpa perlu jalankan
   script lagi.

Alur:
1. Ambil OBSERVASI METAR TERBARU dari metar_observations -> jadi
   baseline (wind/visibility/cloud kondisi SAAT INI)
2. Ambil PREDIKSI TERBARU dari taf_predictions (issue_time paling baru
   yang tersedia) -> dikelompokkan per target jadi {horizon: probability}
3. Panggil taf_encoder.build_taf_draft() -> teks sandi TAF
4. Cetak + simpan ke Supabase

ENV VARS: SUPABASE_URL, SUPABASE_KEY
"""

import os
import sys
from datetime import datetime, timezone

import pandas as pd

from parse_metar_structured import parse_one_line
from taf_encoder import BaselineConditions, build_taf_draft

ICAO = "WALS"
VALID_HOURS = 24


def get_supabase_client():
    from supabase import create_client

    url, key = os.environ.get("SUPABASE_URL"), os.environ.get("SUPABASE_KEY")
    if not (url and key):
        raise RuntimeError("SUPABASE_URL/SUPABASE_KEY belum di-set.")
    return create_client(url, key)


def fetch_latest_baseline(client) -> BaselineConditions:
    resp = (
        client.table("metar_observations")
        .select("obs_datetime,raw_text")
        .eq("station", ICAO)
        .order("obs_datetime", desc=True)
        .limit(1)
        .execute()
    )
    if not resp.data:
        raise RuntimeError(f"Tidak ada observasi METAR untuk {ICAO} di Supabase.")

    rec = resp.data[0]
    obs_dt = pd.Timestamp(rec["obs_datetime"]).to_pydatetime()
    parsed = parse_one_line(rec["raw_text"], obs_dt)
    if parsed is None:
        raise RuntimeError(f"Gagal parsing METAR terbaru: {rec['raw_text']}")

    print(f"[INFO] Baseline dari METAR: {parsed['raw_metar']}")

    clouds = []
    for i in (1, 2, 3):
        cover = parsed.get(f"sky_layer{i}_cover")
        height = parsed.get(f"sky_layer{i}_height_ft")
        if cover and pd.notna(height):
            clouds.append((cover, int(height)))

    weather = parsed.get("weather_phenomena") or None
    if weather == "":
        weather = None

    return BaselineConditions(
        wind_dir_deg=parsed.get("wind_dir_deg"),
        wind_speed_kt=parsed.get("wind_speed_kt") or 0,
        wind_gust_kt=parsed.get("wind_gust_kt"),
        visibility_m=int(parsed.get("visibility_m") or 9999),
        weather=weather,
        clouds=clouds,
    ), obs_dt


def fetch_latest_predictions(client) -> tuple[datetime, dict[str, dict[int, float]]]:
    # Cari issue_time PALING BARU yang tersedia
    resp_latest = (
        client.table("taf_predictions")
        .select("issue_time")
        .eq("station", ICAO)
        .order("issue_time", desc=True)
        .limit(1)
        .execute()
    )
    if not resp_latest.data:
        raise RuntimeError("Belum ada prediksi di taf_predictions -- jalankan daily_inference.py dulu.")

    latest_issue_time = resp_latest.data[0]["issue_time"]

    resp = (
        client.table("taf_predictions")
        .select("target,horizon_hours,probability")
        .eq("station", ICAO)
        .eq("issue_time", latest_issue_time)
        .execute()
    )

    predictions_by_target: dict[str, dict[int, float]] = {}
    for row in resp.data:
        predictions_by_target.setdefault(row["target"], {})[row["horizon_hours"]] = row["probability"]

    issue_dt = pd.Timestamp(latest_issue_time).to_pydatetime().replace(tzinfo=None)
    print(f"[INFO] Prediksi dari issue_time: {issue_dt.isoformat()}")
    for target, probs in predictions_by_target.items():
        probs_str = ", ".join(f"h+{h}j={p*100:.0f}%" for h, p in sorted(probs.items()))
        print(f"         {target}: {probs_str}")

    return issue_dt, predictions_by_target


def save_draft(client, icao: str, issue_time: datetime, draft_text: str):
    client.table("taf_drafts").insert({
        "icao": icao,
        "issue_time": issue_time.isoformat(),
        "draft_text": draft_text,
    }).execute()


def main():
    client = get_supabase_client()

    baseline, baseline_obs_time = fetch_latest_baseline(client)
    issue_time, predictions_by_target = fetch_latest_predictions(client)

    draft = build_taf_draft(ICAO, issue_time, VALID_HOURS, baseline, predictions_by_target)

    print("\n" + "=" * 60)
    print("DRAFT SANDI TAF (untuk direview forecaster, BUKAN siap terbit)")
    print("=" * 60)
    print(draft)
    print("=" * 60)

    save_draft(client, ICAO, issue_time, draft)
    print("\n[INFO] Draft tersimpan ke tabel taf_drafts di Supabase.")

    # Simpan juga salinan lokal, siapa tahu dijalankan bukan lewat Actions
    with open("taf_draft_latest.txt", "w") as f:
        f.write(draft)
    print("[INFO] Salinan lokal: taf_draft_latest.txt")


if __name__ == "__main__":
    main()
