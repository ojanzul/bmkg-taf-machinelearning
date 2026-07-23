"""
taf_encoder.py
==================
Ubah baseline kondisi (dari METAR terkini) + probabilitas prediksi
(dari model_bundle_v1.joblib) jadi SANDI TAF format ICAO Annex 3.

*** PENTING -- BACA SEBELUM PAKAI DI OPERASIONAL ***
Output modul ini adalah DRAFT/SARAN untuk forecaster, BUKAN TAF siap
terbit. TAF adalah produk keselamatan penerbangan -- keputusan akhir
(termasuk nilai visibility/cloud spesifik saat kondisi signifikan,
pemilihan fenomena cuaca, dan validitas keseluruhan) WAJIB direview
dan disetujui forecaster berwenang sesuai SOP/regulasi BMKG yang
berlaku, bukan diterbitkan otomatis dari output model.

============================================================
STRUKTUR SANDI TAF (ICAO Annex 3 / WMO No. 49)
============================================================
TAF {ICAO} {DDHHMM}Z {DDHH}/{DDHH}
     {wind} {visibility} {weather} {cloud}
     [BECMG|TEMPO|PROB30|PROB40] {DDHH}/{DDHH} {perubahan}
     ...

============================================================
CATATAN SOAL KONVERSI PROBABILITAS -> GRUP PERUBAHAN
============================================================
Model kita punya probabilitas KUMULATIF per horizon (mis. "P(TS
terjadi kapan saja dalam 6 jam ke depan)"), sementara TAF butuh
probabilitas per BLOK WAKTU TERPISAH (mis. "P(TS terjadi ANTARA jam
ke-3 dan ke-6)"). Untuk itu, probabilitas marginal per blok
diestimasi dari selisih probabilitas kumulatif, dengan ASUMSI
SEDERHANA proses kejadian antar-blok kurang lebih independen:

    P(blok_i) ~= 1 - (1 - P_kumulatif_atas) / (1 - P_kumulatif_bawah)

Ini PENDEKATAN, bukan hasil statistik yang presisi -- untuk kejadian
langka seperti TS, pendekatan ini cukup wajar, tapi forecaster tetap
perlu menilai dengan penalaran meteorologis sendiri, terutama kalau
probabilitas antar-horizon berubah drastis/tidak monoton (indikasi
model tidak yakin, bukan sinyal cuaca yang kuat).
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta


HORIZON_BLOCKS = [(0, 1), (1, 3), (3, 6), (6, 12), (12, 24)]

# Kondisi "khas" tiap fenomena saat dipakai di grup perubahan.
# INI NILAI DEFAULT/PLACEHOLDER -- forecaster HARUS sesuaikan dengan
# kondisi aktual/pola klimatologi Samarinda, bukan dipakai mentah-mentah.
DEFAULT_ADVERSE_CONDITIONS = {
    "TS": {"weather": "TSRA", "visibility_m": 3000, "cloud": "BKN015CB"},
    "RA": {"weather": "RA", "visibility_m": 6000, "cloud": None},
    "BR_FG": {"weather": "BR", "visibility_m": 4000, "cloud": None},
}


@dataclass
class BaselineConditions:
    wind_dir_deg: float | None
    wind_speed_kt: float
    wind_gust_kt: float | None
    visibility_m: int
    weather: str | None  # None kalau tidak ada fenomena signifikan
    clouds: list[tuple[str, int]] = field(default_factory=list)  # [(cover, height_ft), ...]


def format_ddhh(dt: datetime) -> str:
    """Format DDHH -- jam 24 dipakai untuk tengah malam AKHIR periode (konvensi ICAO)."""
    return dt.strftime("%d%H")


def format_wind(dir_deg, speed_kt, gust_kt) -> str:
    if dir_deg is None:
        dir_str = "VRB"
    else:
        dir_str = f"{round(dir_deg):03d}"
    speed_str = f"{round(speed_kt):02d}"
    gust_str = f"G{round(gust_kt):02d}" if gust_kt and gust_kt > speed_kt else ""
    return f"{dir_str}{speed_str}{gust_str}KT"


def format_visibility(vis_m: int) -> str:
    if vis_m >= 9999:
        return "9999"
    return f"{int(vis_m):04d}"


def format_clouds(clouds: list[tuple[str, int]]) -> str:
    if not clouds:
        return "NSC"  # No Significant Cloud
    parts = []
    for cover, height_ft in clouds:
        height_code = f"{round(height_ft / 100):03d}" if height_ft is not None else "///"
        parts.append(f"{cover}{height_code}")
    return " ".join(parts)


def format_baseline_group(baseline: BaselineConditions) -> str:
    parts = [
        format_wind(baseline.wind_dir_deg, baseline.wind_speed_kt, baseline.wind_gust_kt),
        format_visibility(baseline.visibility_m),
    ]
    if baseline.weather:
        parts.append(baseline.weather)
    parts.append(format_clouds(baseline.clouds))
    return " ".join(parts)


def estimate_block_marginal_probabilities(cumulative_probs: dict[int, float]) -> dict[tuple[int, int], float]:
    """
    cumulative_probs: {horizon_jam: probabilitas_kumulatif}, mis. {1:0.05, 3:0.12, 6:0.18, 12:0.30, 24:0.45}
    Return: {(jam_mulai, jam_akhir): probabilitas_marginal_blok}
    """
    result = {}
    prev_cum = 0.0
    for lo, hi in HORIZON_BLOCKS:
        cum_hi = cumulative_probs.get(hi)
        if cum_hi is None:
            continue
        if prev_cum >= 1.0:
            marginal = 0.0
        else:
            marginal = 1 - (1 - cum_hi) / (1 - prev_cum)
        marginal = max(0.0, min(1.0, marginal))
        result[(lo, hi)] = marginal
        prev_cum = cum_hi
    return result


def classify_group_type(prob: float) -> str | None:
    """Aturan sama persis dengan yang sudah dipakai di aplikasi TAF awal."""
    if prob >= 0.50:
        return "TEMPO"
    elif prob >= 0.40:
        return "PROB40 TEMPO"
    elif prob >= 0.30:
        return "PROB30 TEMPO"
    return None


def build_change_groups(issue_time: datetime, target: str, cumulative_probs: dict[int, float]) -> list[str]:
    """Bangun daftar baris grup perubahan (TEMPO/PROB30/PROB40) untuk satu fenomena."""
    marginals = estimate_block_marginal_probabilities(cumulative_probs)
    adverse = DEFAULT_ADVERSE_CONDITIONS.get(target)
    if adverse is None:
        raise ValueError(f"Belum ada kondisi default untuk target: {target}")

    lines = []
    for (lo, hi), prob in marginals.items():
        group_type = classify_group_type(prob)
        if group_type is None:
            continue

        start = issue_time + timedelta(hours=lo)
        end = issue_time + timedelta(hours=hi)
        time_range = f"{format_ddhh(start)}/{format_ddhh(end)}"

        cond_parts = [format_visibility(adverse["visibility_m"]), adverse["weather"]]
        if adverse["cloud"]:
            cond_parts.append(adverse["cloud"])

        lines.append(
            f"     {group_type} {time_range} {' '.join(cond_parts)}"
            f"   [P={prob*100:.0f}%]"  # info probabilitas -- HAPUS baris ini kalau mau format ICAO murni
        )
    return lines


def build_taf_draft(
    icao: str,
    issue_time: datetime,
    valid_hours: int,
    baseline: BaselineConditions,
    predictions_by_target: dict[str, dict[int, float]],
) -> str:
    """
    predictions_by_target: {"TS": {1:0.05, 3:0.12, ...}, "RA": {...}, "BR_FG": {...}}
    (format cumulative probability per horizon per target, sesuai output daily_inference.py)
    """
    valid_start = issue_time
    valid_end = issue_time + timedelta(hours=valid_hours)

    header = (
        f"TAF {icao} {issue_time.strftime('%d%H%M')}Z "
        f"{format_ddhh(valid_start)}/{format_ddhh(valid_end)}"
    )
    baseline_line = f"     {format_baseline_group(baseline)}"

    all_change_lines = []
    for target, cum_probs in predictions_by_target.items():
        all_change_lines.extend(build_change_groups(issue_time, target, cum_probs))

    lines = [header, baseline_line] + all_change_lines
    lines.append("=")
    return "\n".join(lines)
