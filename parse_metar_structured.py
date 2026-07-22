"""
Parser METAR mentah -> data terstruktur (CSV)
===============================================

Input : file teks berisi baris METAR mentah, satu laporan per baris
        (hasil dari scrape_metar_bmkg.py, misal wals_metar_raw.txt)
Output: CSV dengan kolom siap pakai untuk feature engineering /
        pelabelan time-shifted (lihat pembahasan sebelumnya)

Pakai library python-metar (pip install metar) untuk parsing yang
robust -- lebih aman daripada regex manual karena sudah menangani
banyak edge case format ICAO.
"""

import csv
import sys
from datetime import datetime

from metar import Metar

INPUT_FILE = "wals_metar_raw.txt"
OUTPUT_FILE = "wals_metar_structured.csv"

FIELDNAMES = [
    "raw_metar",
    "station",
    "valid_time_utc",
    "is_auto",
    "wind_dir_deg",
    "wind_speed_kt",
    "wind_gust_kt",
    "visibility_m",
    "weather_phenomena",
    "sky_layer1_cover",
    "sky_layer1_height_ft",
    "sky_layer2_cover",
    "sky_layer2_height_ft",
    "sky_layer3_cover",
    "sky_layer3_height_ft",
    "temp_c",
    "dewpoint_c",
    "qnh_hpa",
    "has_ts",  # flag biner: TS terjadi DI lokasi (bukan di remarks/vicinity)
    "ts_in_vicinity",  # flag biner: TS terdeteksi di SEKITAR lokasi (kode VC)
    "ts_intensity",  # '-' ringan, '' sedang, '+' berat, 'VC' di sekitar, None kalau tidak ada
]


def sky_layer_value(obs, index):
    """Ambil (cover, height_ft) untuk layer awan ke-`index`, atau (None, None)."""
    if obs.sky and len(obs.sky) > index:
        cover, height, _ = obs.sky[index]
        height_ft = height.value("FT") if height is not None else None
        return cover, height_ft
    return None, None


def detect_thunderstorm(obs) -> tuple[int, int, str | None]:
    """
    Cek grup cuaca YANG SUDAH DIPARSING ICAO (bukan cari teks 'TS' mentah
    di seluruh baris -- itu bisa salah tangkap kata 'TS' yang muncul di
    bagian remarks/RMK, yang bukan bagian dari laporan cuaca resmi).

    Mengembalikan:
      has_ts         -> 1 kalau TS terjadi DI lokasi stasiun
      ts_in_vicinity -> 1 kalau TS terdeteksi di SEKITAR lokasi (kode VC),
                         bukan langsung di atas stasiun
      ts_intensity   -> '-', '', '+', atau 'VC' (None kalau tidak ada TS)
    """
    for intensity, descriptor, phenomenon, _, _ in obs.weather:
        if descriptor == "TS":
            if intensity == "VC":
                return 0, 1, "VC"
            return 1, 0, intensity
    return 0, 0, None


def parse_one_line(raw_line: str) -> dict | None:
    raw_line = raw_line.strip()
    if not raw_line:
        return None
    # Buang tanda '=' di akhir laporan kalau masih ada
    raw_line = raw_line.rstrip("=").strip()

    try:
        obs = Metar.Metar(raw_line)
    except Exception as e:
        print(f"[SKIP] gagal parsing: {raw_line[:60]}... -> {e}", file=sys.stderr)
        return None

    l1_cover, l1_h = sky_layer_value(obs, 0)
    l2_cover, l2_h = sky_layer_value(obs, 1)
    l3_cover, l3_h = sky_layer_value(obs, 2)

    weather_str = " ".join(str(w) for w in obs.weather) if obs.weather else ""
    has_ts, ts_in_vicinity, ts_intensity = detect_thunderstorm(obs)

    return {
        "raw_metar": raw_line,
        "station": obs.station_id,
        "valid_time_utc": obs.time.isoformat() if obs.time else None,
        "is_auto": 1 if "AUTO" in raw_line else 0,
        "wind_dir_deg": obs.wind_dir.value() if obs.wind_dir else None,
        "wind_speed_kt": obs.wind_speed.value("KT") if obs.wind_speed else None,
        "wind_gust_kt": obs.wind_gust.value("KT") if obs.wind_gust else None,
        "visibility_m": obs.vis.value("M") if obs.vis else None,
        "weather_phenomena": weather_str,
        "sky_layer1_cover": l1_cover,
        "sky_layer1_height_ft": l1_h,
        "sky_layer2_cover": l2_cover,
        "sky_layer2_height_ft": l2_h,
        "sky_layer3_cover": l3_cover,
        "sky_layer3_height_ft": l3_h,
        "temp_c": obs.temp.value("C") if obs.temp else None,
        "dewpoint_c": obs.dewpt.value("C") if obs.dewpt else None,
        "qnh_hpa": obs.press.value("HPA") if obs.press else None,
        "has_ts": has_ts,
        "ts_in_vicinity": ts_in_vicinity,
        "ts_intensity": ts_intensity,
    }


def main(input_file: str = INPUT_FILE, output_file: str = OUTPUT_FILE):
    rows = []
    with open(input_file, "r") as f:
        for line in f:
            parsed = parse_one_line(line)
            if parsed:
                rows.append(parsed)

    with open(output_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Berhasil parsing {len(rows)} baris -> {output_file}")


if __name__ == "__main__":
    main()
