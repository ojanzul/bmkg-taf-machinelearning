"""
taf_features.py
Modul fitur BERSAMA -- dipakai baik oleh training (03_train_multihorizon.py /
02_build_multihorizon_dataset.py) maupun script inferensi harian
(daily_inference.py). SATU sumber logika, supaya fitur yang dipakai model
saat training dan saat prediksi produksi selalu identik.

Kalau perlu ubah definisi fitur, ubah di SINI SAJA, lalu retrain model.
"""
import numpy as np
import pandas as pd

LOOKBACK_HOURS = 12
HORIZONS = [1, 3, 6, 12, 24]
WX_TARGETS = ["TS", "RA", "BR_FG"]

BASE_FEATURES = [
    "hour_utc", "month", "doy_sin", "doy_cos",
    "cur_wind_speed_kt", "cur_visibility_m", "cur_temp_c", "cur_has_cb",
    "freq_TS_12h", "freq_RA_12h", "freq_BR_FG_12h", "freq_vcTS_12h", "freq_cb_12h",
    "qnh_trend_12h_nwp", "cloud_high_mean_12h", "vpd_mean_12h",
]


def future_feature_names(h: int) -> list[str]:
    return [
        f"fut_precip_sum_h{h}", f"fut_cloud_high_max_h{h}", f"fut_vpd_max_h{h}",
        f"fut_pressure_range_h{h}", f"fut_gust_max_h{h}", f"fut_rh_mean_h{h}",
    ]


def compute_past_features(t: pd.Timestamp, metar_df: pd.DataFrame, nwp_df: pd.DataFrame,
                           lookback_hours: int = LOOKBACK_HOURS) -> dict | None:
    """
    metar_df: index=valid_time_utc, kolom hasil parse_metar_structured.py
              (wind_speed_kt, visibility_m, temp_c, has_cb, weather_phenomena,
              has_ts, ts_in_vicinity)
    nwp_df:   index=time, kolom Open-Meteo (pressure_msl, cloud_cover_high,
              vapour_pressure_deficit, dst.)
    Return None kalau data di window terlalu sedikit (gagal hitung dengan andal).
    """
    past_metar = metar_df.loc[t - pd.Timedelta(hours=lookback_hours): t]
    past_nwp = nwp_df.loc[t - pd.Timedelta(hours=lookback_hours): t]
    if len(past_metar) < 3 or len(past_nwp) < 3:
        return None

    wx_ts = past_metar["has_ts"].astype(int)
    wx_ra = past_metar["weather_phenomena"].fillna("").str.contains(r"\bRA\b", regex=True).astype(int)
    wx_brfg = past_metar["weather_phenomena"].fillna("").str.contains(r"\bBR\b|\bFG\b", regex=True).astype(int)

    last = past_metar.iloc[-1]
    feat = {
        "hour_utc": t.hour,
        "month": t.month,
        "doy_sin": np.sin(2 * np.pi * t.dayofyear / 365.25),
        "doy_cos": np.cos(2 * np.pi * t.dayofyear / 365.25),
        "cur_wind_speed_kt": last.get("wind_speed_kt", np.nan),
        "cur_visibility_m": last.get("visibility_m", np.nan),
        "cur_temp_c": last.get("temp_c", np.nan),
        "cur_has_cb": last.get("has_cb", 0),
        "freq_TS_12h": wx_ts.mean(),
        "freq_RA_12h": wx_ra.mean(),
        "freq_BR_FG_12h": wx_brfg.mean(),
        "freq_vcTS_12h": (past_metar["ts_in_vicinity"] == 1).mean(),
        "freq_cb_12h": past_metar["has_cb"].mean(),
        "qnh_trend_12h_nwp": past_nwp["pressure_msl"].iloc[-1] - past_nwp["pressure_msl"].iloc[0],
        "cloud_high_mean_12h": past_nwp["cloud_cover_high"].mean(),
        "vpd_mean_12h": past_nwp["vapour_pressure_deficit"].mean(),
    }
    return feat


def compute_future_features(t: pd.Timestamp, h: int, nwp_df: pd.DataFrame) -> dict | None:
    """
    nwp_df di sini HARUS berisi data FORECAST (Open-Meteo Forecast API) saat
    dipakai untuk inferensi produksi -- bukan reanalysis. Saat training,
    reanalysis ERA5 dipakai sebagai upper-bound proxy (lihat catatan
    di model_bundle['trained_on']).
    """
    future_nwp = nwp_df.loc[t: t + pd.Timedelta(hours=h)]
    if len(future_nwp) < 2:
        return None
    return {
        f"fut_precip_sum_h{h}": future_nwp["precipitation"].sum(),
        f"fut_cloud_high_max_h{h}": future_nwp["cloud_cover_high"].max(),
        f"fut_vpd_max_h{h}": future_nwp["vapour_pressure_deficit"].max(),
        f"fut_pressure_range_h{h}": future_nwp["pressure_msl"].max() - future_nwp["pressure_msl"].min(),
        f"fut_gust_max_h{h}": future_nwp["wind_gusts_10m"].max(),
        f"fut_rh_mean_h{h}": future_nwp["relative_humidity_2m"].mean(),
    }


def label_for_horizon(t: pd.Timestamp, h: int, metar_df: pd.DataFrame, wx: str) -> int | None:
    """Dipakai training: apakah fenomena wx terjadi kapan saja dalam (t, t+h]."""
    future_metar = metar_df.loc[t: t + pd.Timedelta(hours=h)]
    if len(future_metar) < 2:
        return None
    if wx == "TS":
        col = future_metar["has_ts"].astype(int)
    elif wx == "RA":
        col = future_metar["weather_phenomena"].fillna("").str.contains(r"\bRA\b", regex=True).astype(int)
    elif wx == "BR_FG":
        col = future_metar["weather_phenomena"].fillna("").str.contains(r"\bBR\b|\bFG\b", regex=True).astype(int)
    else:
        raise ValueError(f"target tidak dikenal: {wx}")
    return int(col.sum() > 0)
