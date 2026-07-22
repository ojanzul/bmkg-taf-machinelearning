# Pipeline Otomatis METAR WALS -> Dataset Siap Latih

Pipeline harian yang menarik METAR WALS dari sumber resmi BMKG,
mem-parsing jadi data terstruktur, dan membangun label time-shifted
untuk keperluan model ML prakiraan TAF.

## Alur

```
scrape_metar_bmkg.py  -> tarik METAR mentah dari web-aviation.bmkg.go.id
parse_metar_structured.py -> ubah jadi kolom (angin, visibility, awan, TS, dll)
build_time_shifted_labels.py -> bikin label prediksi h+1/h+3/h+6/h+12/h+24 jam
run_pipeline.py -> jalankan ketiganya berurutan, idempotent & incremental
```

Semua output tersimpan di folder `data/`:
- `wals_metar_raw_archive.txt` -- arsip mentah akumulatif (dedup otomatis)
- `wals_metar_structured.csv` -- data terstruktur, kolom siap pakai
- `wals_training_dataset.csv` -- dataset final dengan label time-shifted

## Setup di GitHub

1. Buat repo baru (bisa private), upload semua file di folder ini
   (termasuk folder `.github/workflows/`).
2. Push ke GitHub -- workflow otomatis aktif, jadwal jalan tiap hari
   jam 01:00 UTC (09:00 WITA).
3. Untuk trigger manual (tanpa nunggu jadwal): buka tab **Actions** di
   repo GitHub -> pilih workflow "Update METAR WALS dataset" ->
   klik **Run workflow**.
4. Tidak perlu setup secret/API key apa pun -- sumber datanya publik.

## PENTING sebelum dipakai produksi

- **Informasikan ke tim IT/data BMKG** dulu soal rencana penarikan
  data otomatis ini, sesuai etika penggunaan sistem internal.
- Jadwal defaultnya 1x/hari -- jangan dipercepat tanpa alasan kuat,
  supaya tidak membebani server BMKG.
- Cek log run pertama di tab Actions untuk pastikan scraping berhasil
  (situs bisa berubah struktur sewaktu-waktu, terutama token CSRF-nya).
- Dataset masih perlu digabung dengan data historis lama (kalau ada
  arsip lebih panjang dari BMKG) sebelum benar-benar layak untuk
  training model produksi -- lihat catatan soal jumlah data minimal
  di pembahasan sebelumnya.

## Menjalankan manual di lokal (opsional, untuk testing)

```bash
pip install -r requirements.txt
python run_pipeline.py
```
