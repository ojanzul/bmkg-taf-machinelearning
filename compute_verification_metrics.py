"""
compute_verification_metrics.py
===================================
Dijalankan berkala (disarankan 1x/hari via GitHub Actions), TERPISAH
dari verify_predictions.py -- script itu yang menandai prediksi mana
yang benar/salah (hit/miss), script INI yang merangkumnya jadi angka
performa yang bisa langsung dijawabkan ke Kepala Stasiun.

Alur: Issue time -> Prediksi -> METAR aktual -> Hit/Miss (verify_predictions.py)
      -> lalu di sini: agregasi 7 hari terakhir -> Accuracy/Brier/ROC-AUC/Reliability

METRIK YANG DIHITUNG (per fenomena: TS, RA, BR_FG):
- accuracy    : proporsi prediksi benar di threshold 0.5 (TEMPO/tidak)
- precision/recall : sama seperti sebelumnya, tapi sekarang WINDOWED 7 hari
- brier_score : rata-rata (probabilitas - kejadian_aktual)^2, 0=sempurna
- roc_auc     : seberapa baik model membedakan kejadian vs tidak,
                di semua kemungkinan threshold sekaligus (bukan cuma 0.5)
- reliability_ece : Expected Calibration Error -- kalau model bilang
                "70%", apakah kejadian itu BENAR terjadi ~70% dari
                waktu? Makin kecil ECE, makin bisa dipercaya angka
                probabilitasnya (bukan cuma urutan rankingnya benar).

CATATAN JUJUR: kalau n_verified di suatu window kecil (<~20), semua
angka ini akan sangat fluktuatif dari hari ke hari -- jangan
diinterpretasi sebagai "model memburuk/membaik" sampai n cukup besar.
ROC-AUC juga akan NULL kalau di window itu SEMUA kejadian sama
(semua 0 atau semua 1) -- itu bukan cara Python menyerah, ROC-AUC
memang tidak terdefinisi tanpa kedua kelas.

ENV VARS: SUPABASE_URL, SUPABASE_KEY
"""

import os
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

WINDOW_DAYS = 7
TARGETS = ["TS", "RA", "BR_FG"]
MIN_N_FOR_METRICS = 3  # di bawah ini, jangan hitung apa-apa (terlalu bising)
ECE_BINS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]  # 5 bin, cukup kasar untuk sampel kecil


def get_supabase_client():
    from supabase import create_client

    url, key = os.environ.get("SUPABASE_URL"), os.environ.get("SUPABASE_KEY")
    if not (url and key):
        raise RuntimeError("SUPABASE_URL/SUPABASE_KEY belum di-set.")
    return create_client(url, key)


def fetch_verified_predictions(client, target: str, since: datetime) -> pd.DataFrame:
    resp = (
        client.table("taf_predictions")
        .select("issue_time,probability,actual_outcome")
        .eq("target", target)
        .not_.is_("actual_outcome", "null")
        .gte("issue_time", since.isoformat())
        .execute()
    )
    if not resp.data:
        return pd.DataFrame(columns=["issue_time", "probability", "actual_outcome"])
    return pd.DataFrame(resp.data)


def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, bins: list[float]) -> float | None:
    """
    Expected Calibration Error: rata-rata selisih |probabilitas rata-rata
    yang diprediksi - frekuensi aktual| per bin, dibobot jumlah sampel per bin.
    """
    if len(y_true) == 0:
        return None
    ece = 0.0
    n_total = len(y_true)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (y_prob >= lo) & (y_prob < hi if hi < 1.0 else y_prob <= hi)
        n_bin = mask.sum()
        if n_bin == 0:
            continue
        mean_pred = y_prob[mask].mean()
        observed_freq = y_true[mask].mean()
        ece += (n_bin / n_total) * abs(mean_pred - observed_freq)
    return round(ece, 4)


def compute_metrics_for_target(df: pd.DataFrame) -> dict | None:
    n = len(df)
    if n < MIN_N_FOR_METRICS:
        return None

    y_true = df["actual_outcome"].astype(int).values
    y_prob = df["probability"].astype(float).values
    y_pred = (y_prob >= 0.5).astype(int)

    n_positive = int(y_true.sum())
    accuracy = float((y_pred == y_true).mean())

    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    precision = tp / (tp + fp) if (tp + fp) > 0 else None
    recall = tp / (tp + fn) if (tp + fn) > 0 else None

    brier = float(np.mean((y_prob - y_true) ** 2))

    roc_auc = None
    if len(set(y_true)) == 2:  # ROC-AUC butuh kedua kelas ada di window ini
        roc_auc = float(roc_auc_score(y_true, y_prob))

    ece = compute_ece(y_true, y_prob, ECE_BINS)

    return {
        "n_verified": n,
        "n_positive": n_positive,
        "accuracy": round(accuracy, 4),
        "precision_score": round(precision, 4) if precision is not None else None,
        "recall_score": round(recall, 4) if recall is not None else None,
        "brier_score": round(brier, 4),
        "roc_auc": round(roc_auc, 4) if roc_auc is not None else None,
        "reliability_ece": ece,
    }


def main():
    client = get_supabase_client()
    since = datetime.now(timezone.utc) - timedelta(days=WINDOW_DAYS)

    print(f"[INFO] Menghitung metrik verifikasi -- window {WINDOW_DAYS} hari terakhir (sejak {since.date()})")

    for target in TARGETS:
        df = fetch_verified_predictions(client, target, since)
        metrics = compute_metrics_for_target(df)

        if metrics is None:
            print(f"[{target}] dilewati -- cuma {len(df)} prediksi terverifikasi (min {MIN_N_FOR_METRICS})")
            continue

        row = {
            "target": target,
            "window_days": WINDOW_DAYS,
            "computed_at": datetime.now(timezone.utc).isoformat(),
            **metrics,
        }
        client.table("taf_verification_metrics").upsert(
            row, on_conflict="target,window_days"
        ).execute()

        roc_str = f"{metrics['roc_auc']:.3f}" if metrics["roc_auc"] is not None else "N/A (1 kelas saja)"
        print(f"[{target}] n={metrics['n_verified']} akurasi={metrics['accuracy']:.3f} "
              f"brier={metrics['brier_score']:.3f} roc_auc={roc_str} ece={metrics['reliability_ece']:.3f}")

    print("\n[SELESAI] Metrik tersimpan ke taf_verification_metrics.")


if __name__ == "__main__":
    main()
