"""
02_build_multihorizon_dataset.py
Anchor grid: tiap 6 jam (00/06/12/18Z, selaras jadwal terbit TAF).
Untuk tiap anchor time t dan tiap horizon h dalam HORIZONS:
  - fitur "past" (sama utk semua horizon): state METAR saat t + rolling 12h
    persistence + rolling 12h NWP nowcast
  - fitur "future_h": ringkasan NWP UNTUK JENDELA (t, t+h] -- meniru forecast
    NWP asli yang akan dipakai saat inferensi produksi (Open-Meteo Forecast API)
  - label_h: apakah fenomena X terjadi KAPAN SAJA dalam jendela (t, t+h]
    (bukan "nilai TEPAT di jam +h" seperti build_time_shifted_labels.py di
    repo -- untuk TS/hujan/kabut yang relevan buat TAF adalah "apakah
    terjadi", bukan nilai titik tunggal)
"""
import pandas as pd
import numpy as np

metar = pd.read_pickle("metar_structured_v2.pkl").sort_values("valid_time_utc").reset_index(drop=True)
metar = metar.set_index("valid_time_utc")
metar["wx_BR_FG"] = 0  # placeholder, diisi dari weather_phenomena di bawah
metar["wx_BR_FG"] = metar["weather_phenomena"].fillna("").str.contains(r"\bBR\b|\bFG\b", regex=True).astype(int)
metar["wx_RA"] = metar["weather_phenomena"].fillna("").str.contains(r"\bRA\b", regex=True).astype(int)
metar["wx_TS"] = metar["has_ts"].astype(int)

nwp = pd.read_csv("nwp_raw.csv", skiprows=3)
nwp.columns = [c.split(" (")[0] for c in nwp.columns]
nwp["time"] = pd.to_datetime(nwp["time"])
nwp = nwp.set_index("time").sort_index()

HORIZONS = [1, 3, 6, 12, 24]
WX_TARGETS = ["TS", "RA", "BR_FG"]
LOOKBACK_HOURS = 12

full_start = max(metar.index.min(), nwp.index.min())
full_end = min(metar.index.max(), nwp.index.max())
issue_times = pd.date_range(full_start.ceil("6h"), full_end - pd.Timedelta(hours=max(HORIZONS)), freq="6h")
print(f"Candidate issuance points: {len(issue_times)}")

rows = []
for t in issue_times:
    past_metar = metar.loc[t - pd.Timedelta(hours=LOOKBACK_HOURS): t]
    past_nwp = nwp.loc[t - pd.Timedelta(hours=LOOKBACK_HOURS): t]
    if len(past_metar) < 3 or len(past_nwp) < 3:
        continue

    feat = {"issue_time": t}
    feat["hour_utc"] = t.hour
    feat["month"] = t.month
    feat["doy_sin"] = np.sin(2 * np.pi * t.dayofyear / 365.25)
    feat["doy_cos"] = np.cos(2 * np.pi * t.dayofyear / 365.25)

    last = past_metar.iloc[-1]
    feat["cur_wind_speed_kt"] = last.get("wind_speed_kt", np.nan)
    feat["cur_visibility_m"] = last.get("visibility_m", np.nan)
    feat["cur_temp_c"] = last.get("temp_c", np.nan)
    feat["cur_has_cb"] = last.get("has_cb", 0)
    for wx in ["TS", "RA", "BR_FG"]:
        feat[f"freq_{wx}_{LOOKBACK_HOURS}h"] = past_metar[f"wx_{wx}"].mean()
    feat["freq_vcTS_12h"] = (past_metar["ts_in_vicinity"] == 1).mean()
    feat["freq_cb_12h"] = past_metar["has_cb"].mean()

    feat["qnh_trend_12h_nwp"] = past_nwp["pressure_msl"].iloc[-1] - past_nwp["pressure_msl"].iloc[0]
    feat["cloud_high_mean_12h"] = past_nwp["cloud_cover_high"].mean()
    feat["vpd_mean_12h"] = past_nwp["vapour_pressure_deficit"].mean()

    ok_row = True
    for h in HORIZONS:
        future_metar = metar.loc[t: t + pd.Timedelta(hours=h)]
        future_nwp = nwp.loc[t: t + pd.Timedelta(hours=h)]
        if len(future_metar) < 2 or len(future_nwp) < 2:
            ok_row = False
            break

        feat[f"fut_precip_sum_h{h}"] = future_nwp["precipitation"].sum()
        feat[f"fut_cloud_high_max_h{h}"] = future_nwp["cloud_cover_high"].max()
        feat[f"fut_vpd_max_h{h}"] = future_nwp["vapour_pressure_deficit"].max()
        feat[f"fut_pressure_range_h{h}"] = future_nwp["pressure_msl"].max() - future_nwp["pressure_msl"].min()
        feat[f"fut_gust_max_h{h}"] = future_nwp["wind_gusts_10m"].max()
        feat[f"fut_rh_mean_h{h}"] = future_nwp["relative_humidity_2m"].mean()

        for wx in WX_TARGETS:
            feat[f"label_{wx}_h{h}"] = int(future_metar[f"wx_{wx}"].sum() > 0)

    if ok_row:
        rows.append(feat)

ds = pd.DataFrame(rows)
ds.to_pickle("multihorizon_dataset.pkl")
ds.to_csv("multihorizon_dataset.csv", index=False)

print(f"\nFinal dataset rows: {len(ds)}")
print(f"Date range: {ds.issue_time.min()} -> {ds.issue_time.max()}")
for h in HORIZONS:
    rates = {wx: ds[f"label_{wx}_h{h}"].mean() for wx in WX_TARGETS}
    print(f"  h+{h:>2}j positive rate:", {k: round(v, 3) for k, v in rates.items()})
