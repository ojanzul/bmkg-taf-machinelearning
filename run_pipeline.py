"""
Orchestrator pipeline harian: scrape -> parse -> label time-shifted
=====================================================================

Alur tiap kali dijalankan (idempotent, aman dijalankan berulang):

1. SCRAPE  : tarik METAR WALS untuk window waktu terbaru (default: 3 hari
             terakhir -- ada overlap sengaja supaya tidak ada jam yang
             kelewat kalau ada keterlambatan run/network hiccup)
2. GABUNG  : gabungkan hasil scrape baru ke arsip mentah akumulatif,
             dedup berdasarkan teks METAR persis sama (biar aman dari
             duplikat akibat overlap window)
3. PARSE   : parse SELURUH arsip mentah -> tabel terstruktur
             (overwrite penuh tiap run, murah karena datanya masih kecil;
             kalau nanti sudah jutaan baris, ini bisa dioptimasi jadi
             incremental juga)
4. LABEL   : bangun ulang label time-shifted dari seluruh data terstruktur

Semua file disimpan di folder data/ supaya gampang di-commit balik oleh
GitHub Actions.
"""

import sys
from pathlib import Path

import pandas as pd

from fetch_metar_api import fetch_latest_metar
from parse_metar_structured import parse_one_line, FIELDNAMES
from build_time_shifted_labels import build_labels

DATA_DIR = Path("data")
RAW_ARCHIVE_FILE = DATA_DIR / "wals_metar_raw_archive.txt"
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


def step_merge_archive(new_lines: list[str]) -> list[str]:
    DATA_DIR.mkdir(exist_ok=True)

    existing_lines: set[str] = set()
    if RAW_ARCHIVE_FILE.exists():
        existing_lines = {
            line.strip() for line in RAW_ARCHIVE_FILE.read_text().splitlines() if line.strip()
        }

    combined = existing_lines.union(line.strip() for line in new_lines if line.strip())
    n_added = len(combined) - len(existing_lines)
    print(f"[GABUNG] {n_added} baris baru ditambahkan ke arsip (total: {len(combined)})")

    # Urutkan biar arsipnya rapi & gampang diaudit manual
    sorted_lines = sorted(combined)
    RAW_ARCHIVE_FILE.write_text("\n".join(sorted_lines) + "\n")
    return sorted_lines


def step_parse(all_lines: list[str]) -> pd.DataFrame:
    rows = []
    for line in all_lines:
        parsed = parse_one_line(line)
        if parsed:
            rows.append(parsed)

    df = pd.DataFrame(rows, columns=FIELDNAMES)
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
    new_lines = step_scrape()
    all_lines = step_merge_archive(new_lines)
    df = step_parse(all_lines)
    step_label(df)
    print("\nPipeline selesai.")


if __name__ == "__main__":
    main()
