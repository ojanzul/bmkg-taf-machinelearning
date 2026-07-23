"""
Orchestrator pipeline harian: fetch -> parse -> label time-shifted
=====================================================================

Alur tiap kali dijalankan (idempotent, aman dijalankan berulang):

1. FETCH   : tarik METAR WALS terkini dari API resmi BMKG (2 laporan
             terakhir)
2. GABUNG  : gabungkan hasil fetch baru ke arsip mentah akumulatif
             (data/wals_metar_raw_archive.csv). Tiap baris disimpan
             berpasangan dengan `first_seen_utc` -- waktu SAAT baris itu
             PERTAMA KALI berhasil di-fetch. Ini krusial: METAR mentah
             cuma menyimpan DDHHMM (tanpa bulan/tahun), jadi first_seen_utc
             inilah yang dipakai buat menentukan bulan/tahun yang benar
             saat parsing -- BUKAN tanggal sistem saat parsing dijalankan
             (itu bug lama: re-parse arsip lama di bulan berjalan akan
             salah kasih bulan/tahun ke semua baris historis).
             Baris yang SUDAH ada di arsip TIDAK ditimpa first_seen_utc-nya
             (supaya nilai aslinya -- yang paling akurat -- tetap terjaga).
3. PARSE   : parse SELURUH arsip mentah -> tabel terstruktur, pakai
             first_seen_utc masing-masing baris sebagai hint bulan/tahun
             (overwrite penuh tiap run, murah karena datanya masih kecil;
             kalau nanti sudah jutaan baris, ini bisa dioptimasi jadi
             incremental juga)
4. LABEL   : bangun ulang label time-shifted dari seluruh data terstruktur

Semua file disimpan di folder data/ supaya gampang di-commit balik oleh
GitHub Actions.
5. SUPABASE (opsional): kalau env var SUPABASE_URL & SUPABASE_KEY di-set,
             baris METAR yang sama juga di-upsert ke Supabase (tabel
             metar_observations, lihat supabase_schema.sql). Ini melengkapi
             arsip git-CSV: Supabase jadi sumber query rolling-window yang
             lebih murah/cepat untuk dipakai script inferensi harian nanti,
             tanpa harus checkout seluruh repo git. Kalau secret belum
             di-set, langkah ini dilewati (tidak bikin pipeline gagal --
             supaya orang yang belum setup Supabase tetap bisa jalan normal
             pakai arsip git-CSV saja).
"""

import csv
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from fetch_metar_api import fetch_latest_metar
from parse_metar_structured import parse_one_line, resolve_month_year, FIELDNAMES
from build_time_shifted_labels import build_labels

DATA_DIR = Path("data")
RAW_ARCHIVE_FILE = DATA_DIR / "wals_metar_raw_archive.csv"
LEGACY_RAW_ARCHIVE_TXT = DATA_DIR / "wals_metar_raw_archive.txt"
STRUCTURED_FILE = DATA_DIR / "wals_metar_structured.csv"
TRAINING_FILE = DATA_DIR / "wals_training_dataset.csv"

ICAO = "WALS"


def step_scrape() -> list[str]:
    print(f"[FETCH] menarik observasi terkini {ICAO} dari API resmi BMKG")

    try:
        new_lines = fetch_latest_metar(ICAO)
    except Exception as e:
        print(f"[FETCH] GAGAL: {e}", file=sys.stderr)
        # Kalau fetch gagal (token invalid, API down, dll), hentikan
        # pipeline di sini -- jangan lanjut ke parse/label pakai data lama
        # seolah-olah berhasil.
        raise

    print(f"[FETCH] dapat {len(new_lines)} baris mentah")
    return new_lines


def _load_archive() -> dict[str, str]:
    """
    Baca arsip -> dict {raw_metar: first_seen_utc_iso}.
    Migrasi otomatis dari format lama (.txt polos, tanpa first_seen_utc)
    kalau arsip CSV baru belum ada tapi arsip lama masih ada -- baris lama
    dikasih first_seen_utc = sekarang (best effort; baris-baris itu memang
    baru saja mulai dikumpulkan jadi risikonya kecil).
    """
    archive: dict[str, str] = {}

    if RAW_ARCHIVE_FILE.exists():
        with RAW_ARCHIVE_FILE.open("r", newline="") as f:
            reader = csv.DictReader(f)
            for rec in reader:
                archive[rec["raw_metar"]] = rec["first_seen_utc"]
        return archive

    if LEGACY_RAW_ARCHIVE_TXT.exists():
        print(
            f"[GABUNG] arsip lama format .txt ditemukan ({LEGACY_RAW_ARCHIVE_TXT}), "
            f"migrasi ke format CSV baru dengan first_seen_utc=sekarang",
            file=sys.stderr,
        )
        now_iso = datetime.now(timezone.utc).isoformat()
        for line in LEGACY_RAW_ARCHIVE_TXT.read_text().splitlines():
            line = line.strip()
            if line:
                archive[line] = now_iso

    return archive


def step_merge_archive(new_lines: list[str], now_utc: datetime) -> dict[str, str]:
    DATA_DIR.mkdir(exist_ok=True)

    archive = _load_archive()
    n_before = len(archive)

    now_iso = now_utc.isoformat()
    for line in new_lines:
        line = line.strip()
        if line and line not in archive:
            archive[line] = now_iso  # first_seen_utc HANYA diset saat baris baru muncul

    n_added = len(archive) - n_before
    print(f"[GABUNG] {n_added} baris baru ditambahkan ke arsip (total: {len(archive)})")

    # Urutkan berdasarkan first_seen_utc (bukan sort teks -- sort teks DDHHMM
    # akan berulang tiap bulan dan tidak benar-benar kronologis lintas bulan)
    rows = sorted(archive.items(), key=lambda kv: kv[1])
    with RAW_ARCHIVE_FILE.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["raw_metar", "first_seen_utc"])
        writer.writerows(rows)

    # Bersihkan arsip lama supaya tidak ada 2 sumber kebenaran yang beda
    if LEGACY_RAW_ARCHIVE_TXT.exists():
        LEGACY_RAW_ARCHIVE_TXT.unlink()

    return archive


def step_upsert_supabase(new_lines: list[str], now_utc: datetime) -> int:
    """
    Upsert baris METAR baru ke Supabase (tabel metar_observations).
    Dilewati (bukan error) kalau SUPABASE_URL/SUPABASE_KEY belum di-set --
    supaya orang yang belum setup Supabase tetap bisa jalan normal.
    """
    if not (os.environ.get("SUPABASE_URL") and os.environ.get("SUPABASE_KEY")):
        print("[SUPABASE] SUPABASE_URL/SUPABASE_KEY belum di-set, langkah ini dilewati")
        return 0

    import re
    from supabase import create_client

    line_re = re.compile(r"^(METAR|SPECI)\s+(\w{4})\s+(\d{2})(\d{2})(\d{2})Z")
    records = []
    for line in new_lines:
        line = line.strip().rstrip("=").strip()
        m = line_re.match(line)
        if not m:
            continue
        kind, station, dd, hh, mi = m.groups()
        year, month = resolve_month_year(now_utc, int(dd))
        obs_dt = datetime(year, month, int(dd), int(hh), int(mi), tzinfo=timezone.utc)
        records.append({
            "station": station,
            "kind": kind,
            "obs_datetime": obs_dt.isoformat(),
            "raw_text": line,
        })

    if not records:
        print("[SUPABASE] tidak ada baris baru untuk di-upsert")
        return 0

    client = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])
    resp = (
        client.table("metar_observations")
        .upsert(records, on_conflict="station,obs_datetime,kind", ignore_duplicates=True)
        .execute()
    )
    n = len(resp.data) if resp.data else 0
    print(f"[SUPABASE] {n} baris di-upsert (duplikat otomatis dilewati)")
    return n


def step_parse(archive: dict[str, str]) -> pd.DataFrame:
    rows = []
    for raw_metar, first_seen_iso in archive.items():
        first_seen = datetime.fromisoformat(first_seen_iso)
        parsed = parse_one_line(raw_metar, first_seen)
        if parsed:
            rows.append(parsed)

    df = pd.DataFrame(rows, columns=FIELDNAMES)
    df = df.sort_values("valid_time_utc").reset_index(drop=True)
    df.to_csv(STRUCTURED_FILE, index=False)
    print(f"[PARSE] {len(df)} baris berhasil diparse -> {STRUCTURED_FILE}")
    return df


def step_label(df: pd.DataFrame) -> None:
    if df.empty:
        print("[LABEL] dilewati -- tidak ada data terstruktur")
        return

    df["valid_time_utc"] = pd.to_datetime(df["valid_time_utc"])
    result = build_labels(df)
    result.to_csv(TRAINING_FILE, index=False)
    print(f"[LABEL] {len(result)} baris -> {TRAINING_FILE}")


def main():
    now_utc = datetime.now(timezone.utc)
    new_lines = step_scrape()
    archive = step_merge_archive(new_lines, now_utc)
    step_upsert_supabase(new_lines, now_utc)
    df = step_parse(archive)
    step_label(df)
    print("\nPipeline selesai.")


if __name__ == "__main__":
    main()
