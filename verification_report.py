"""
verification_report.py
==========================
Tampilkan ringkasan performa PRODUKSI (bukan eksperimen offline) --
baca dari view taf_prediction_performance (lihat supabase_verification_schema.sql)
yang otomatis terisi seiring verify_predictions.py berjalan.

Jalankan kapan saja secara manual untuk cek performa terkini:
  python verification_report.py
"""

import os

import pandas as pd


def main():
    from supabase import create_client

    url, key = os.environ.get("SUPABASE_URL"), os.environ.get("SUPABASE_KEY")
    if not (url and key):
        raise RuntimeError("SUPABASE_URL/SUPABASE_KEY belum di-set.")
    client = create_client(url, key)

    resp = client.table("taf_prediction_performance").select("*").execute()
    if not resp.data:
        print("Belum ada prediksi yang terverifikasi sama sekali.")
        return

    df = pd.DataFrame(resp.data)

    print("=== PERFORMA PRODUKSI (prediksi vs kejadian aktual) ===\n")
    for _, row in df.iterrows():
        n = row["n_verified"]
        if n == 0:
            continue
        tp, fp, fn = row["true_positive"], row["false_positive"], row["false_negative"]
        precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
        recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")

        print(f"{row['target']:7s} h+{row['horizon_hours']:>2}j  "
              f"(n={n:>4}, {row['actual_positive_rate']*100:5.1f}% kejadian aktual)")
        print(f"  Brier score : {row['brier_score']:.4f}  (makin kecil makin baik, 0=sempurna)")
        print(f"  Precision   : {precision:.3f}" if precision == precision else "  Precision   : - (tidak ada prediksi positif)")
        print(f"  Recall      : {recall:.3f}" if recall == recall else "  Recall      : - (tidak ada kejadian aktual)")
        print()

    print("\nCatatan: dengan sampel kecil (n < ~50), angka-angka ini masih")
    print("berfluktuasi besar -- perlakukan sebagai indikasi arah, bukan")
    print("angka final, terutama untuk target/horizon yang jarang terjadi.")


if __name__ == "__main__":
    main()
