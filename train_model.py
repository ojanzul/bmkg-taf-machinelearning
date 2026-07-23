"""
03_train_multihorizon.py
Latih 1 model per (fenomena x horizon) = 3 x 5 = 15 model.
Evaluasi pakai TimeSeriesSplit CV (sama seperti eksperimen sebelumnya).
Model final (dilatih di SELURUH data) disimpan jadi 1 bundle joblib untuk
dipakai script inferensi harian.
"""
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score, brier_score_loss

ds = pd.read_pickle("multihorizon_dataset.pkl").sort_values("issue_time").reset_index(drop=True)

BASE_FEATURES = [
    "hour_utc", "month", "doy_sin", "doy_cos",
    "cur_wind_speed_kt", "cur_visibility_m", "cur_temp_c", "cur_has_cb",
    "freq_TS_12h", "freq_RA_12h", "freq_BR_FG_12h", "freq_vcTS_12h", "freq_cb_12h",
    "qnh_trend_12h_nwp", "cloud_high_mean_12h", "vpd_mean_12h",
]
HORIZONS = [1, 3, 6, 12, 24]
WX_TARGETS = ["TS", "RA", "BR_FG"]

tscv = TimeSeriesSplit(n_splits=5, test_size=100)

model_bundle = {"models": {}, "features": {}, "metrics": {}}

for wx in WX_TARGETS:
    for h in HORIZONS:
        future_cols = [
            f"fut_precip_sum_h{h}", f"fut_cloud_high_max_h{h}", f"fut_vpd_max_h{h}",
            f"fut_pressure_range_h{h}", f"fut_gust_max_h{h}", f"fut_rh_mean_h{h}",
        ]
        feat_cols = BASE_FEATURES + future_cols
        X = ds[feat_cols]
        y = ds[f"label_{wx}_h{h}"]

        aucs, briers = [], []
        for tr_idx, te_idx in tscv.split(X):
            y_tr, y_te = y.iloc[tr_idx], y.iloc[te_idx]
            if y_tr.nunique() < 2 or y_te.nunique() < 2:
                continue
            clf = HistGradientBoostingClassifier(max_depth=4, learning_rate=0.06, max_iter=200, random_state=42)
            clf.fit(X.iloc[tr_idx], y_tr)
            proba = clf.predict_proba(X.iloc[te_idx])[:, 1]
            aucs.append(roc_auc_score(y_te, proba))
            briers.append(brier_score_loss(y_te, proba))

        mean_auc, mean_brier = np.mean(aucs), np.mean(briers)
        clim_brier = np.mean([brier_score_loss(y.iloc[te], np.full(len(te), y.iloc[tr].mean()))
                               for tr, te in tscv.split(X) if y.iloc[tr].nunique() > 1 and y.iloc[te].nunique() > 1])
        print(f"{wx:6s} h+{h:>2}j  AUC={mean_auc:.3f}  Brier(model)={mean_brier:.3f}  Brier(clim)={clim_brier:.3f}")

        # fit final model on ALL data for deployment
        final_clf = HistGradientBoostingClassifier(max_depth=4, learning_rate=0.06, max_iter=200, random_state=42)
        final_clf.fit(X, y)

        key = f"{wx}_h{h}"
        model_bundle["models"][key] = final_clf
        model_bundle["features"][key] = feat_cols
        model_bundle["metrics"][key] = {"auc": mean_auc, "brier": mean_brier, "brier_climatology": clim_brier,
                                          "base_rate": y.mean()}

model_bundle["base_features"] = BASE_FEATURES
model_bundle["horizons"] = HORIZONS
model_bundle["targets"] = WX_TARGETS
model_bundle["trained_on"] = {
    "n_rows": len(ds),
    "date_range": [str(ds.issue_time.min()), str(ds.issue_time.max())],
    "generation": 1,
    "note": "Fitur future NWP pakai ERA5 reanalysis (hindsight), BUKAN forecast asli -- "
            "AUC di sini adalah upper bound. Perlu monitoring performa asli di produksi.",
}

joblib.dump(model_bundle, "model_bundle_v1.joblib")
print(f"\nModel bundle disimpan: model_bundle_v1.joblib ({len(model_bundle['models'])} model)")
