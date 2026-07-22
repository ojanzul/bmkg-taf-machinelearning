"""
Bikin label time-shifted dari METAR terstruktur
=================================================

Input : wals_metar_structured.csv (hasil parse_metar_structured.py)
Output: wals_training_dataset.csv -- siap dipakai untuk melatih model

IDE DASAR
---------
METAR historis mentah cuma catatan "apa yang terjadi jam sekian" --
itu FITUR, bukan label. Supaya jadi data supervised learning, tiap
observasi jam t dipasangkan dengan observasi AKTUAL di jam t+h
(h = horizon prediksi, misal +1, +3, +6, +12, +24 jam). Observasi
masa depan itulah yang jadi "kunci jawaban" (label) untuk horizon
tersebut.

Karena METAR tidak selalu terbit tepat tiap jam (kadang half-hourly,
kadang ada SPECI di antara), pencarian observasi masa depan pakai
"nearest match" dengan toleransi window (default 30 menit) -- kalau
tidak ada observasi yang cukup dekat dengan jam target, baris itu
dapat label kosong (NaN) untuk horizon tersebut dan otomatis dibuang
saat training horizon itu.
"""

import pandas as pd

INPUT_FILE = "wals_metar_structured.csv"
OUTPUT_FILE = "wals_training_dataset.csv"

# Horizon prediksi dalam jam -- sesuaikan sesuai kebutuhan TAF (1-24 jam)
HORIZONS_HOURS = [1, 3, 6, 12, 24]

# Toleransi pencarian observasi terdekat di sekitar jam target.
# METAR reguler tiap 30 menit -> toleransi 30 menit cukup aman.
MATCH_TOLERANCE = pd.Timedelta(minutes=30)

# Kolom dari observasi masa depan yang mau dijadikan label
LABEL_SOURCE_COLUMNS = [
    "has_ts",
    "ts_in_vicinity",
    "visibility_m",
    "wind_speed_kt",
    "wind_gust_kt",
]


def compute_ceiling_ft(row: pd.Series) -> float:
    """
    Ceiling versi aviation: ketinggian lapisan awan BKN/OVC PALING RENDAH.
    FEW/SCT tidak dihitung sebagai ceiling karena tidak menutup langit
    cukup rapat (ceiling = tutupan awan >4/8).
    """
    heights = []
    for i in (1, 2, 3):
        cover = row.get(f"sky_layer{i}_cover")
        height = row.get(f"sky_layer{i}_height_ft")
        if cover in ("BKN", "OVC") and pd.notna(height):
            heights.append(height)
    return min(heights) if heights else float("nan")


def build_labels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("valid_time_utc").reset_index(drop=True)
    df["ceiling_ft"] = df.apply(compute_ceiling_ft, axis=1)

    label_cols = LABEL_SOURCE_COLUMNS + ["ceiling_ft"]
    future_lookup = df[["valid_time_utc"] + label_cols].copy()

    result = df.copy()

    for h in HORIZONS_HOURS:
        target_time_col = f"_target_time_h{h}"
        result[target_time_col] = result["valid_time_utc"] + pd.Timedelta(hours=h)

        merged = pd.merge_asof(
            result[[target_time_col]].sort_values(target_time_col),
            future_lookup.sort_values("valid_time_utc"),
            left_on=target_time_col,
            right_on="valid_time_utc",
            direction="nearest",
            tolerance=MATCH_TOLERANCE,
        )
        # merge_asof mengurutkan ulang -- kembalikan ke urutan index asli
        merged.index = result[[target_time_col]].sort_values(target_time_col).index
        merged = merged.sort_index()

        for col in label_cols:
            result[f"label_{col}_h{h}"] = merged[col]

        result = result.drop(columns=[target_time_col])

    return result


def main(input_file: str = INPUT_FILE, output_file: str = OUTPUT_FILE):
    df = pd.read_csv(input_file, parse_dates=["valid_time_utc"])
    result = build_labels(df)
    result.to_csv(output_file, index=False)

    print(f"Total baris: {len(result)}")
    for h in HORIZONS_HOURS:
        n_valid = result[f"label_has_ts_h{h}"].notna().sum()
        print(f"  Horizon +{h}j : {n_valid} baris punya label valid")
    print(f"\nDisimpan ke {output_file}")


if __name__ == "__main__":
    main()
