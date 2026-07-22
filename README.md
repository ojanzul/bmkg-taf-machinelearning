# Pipeline Otomatis METAR WALS -> Dataset Siap Latih

Pipeline yang menarik METAR WALS dari **API resmi BMKG** (`cuaca.bmkg.go.id`),
mem-parsing jadi data terstruktur, dan membangun label time-shifted untuk
keperluan model ML prakiraan TAF.

## Alur

```
fetch_metar_api.py         -> tarik METAR terkini dari API resmi (token-based)
parse_metar_structured.py  -> ubah jadi kolom (angin, visibility, awan, TS, CB, dll)
build_time_shifted_labels.py -> bikin label prediksi h+1/h+3/h+6/h+12/h+24 jam
run_pipeline.py             -> jalankan ketiganya berurutan, idempotent & incremental
```

Semua output tersimpan di folder `data/`:
- `wals_metar_raw_archive.txt` -- arsip mentah akumulatif (dedup otomatis)
- `wals_metar_structured.csv` -- data terstruktur, kolom siap pakai
- `wals_training_dataset.csv` -- dataset final dengan label time-shifted

## Keterbatasan penting: data historis vs data ke depan

API resmi BMKG yang dipakai di sini **hanya mengembalikan observasi
TERKINI** (2 laporan terakhir), **tidak mendukung query rentang tanggal
historis**. Konsekuensinya:

- **Data KE DEPAN (mulai sekarang):** pipeline ini otomatis mengumpulkan
  terus-menerus, tapi jadwalnya **wajib sering (tiap 30 menit)** --
  kalau kejarang, ada jam yang terlewat dan datanya hilang permanen
  (API ini tidak bisa "mundur" ambil yang terlewat).
- **Data HISTORIS (sebelum pipeline ini mulai jalan):** API ini TIDAK
  BISA dipakai. Opsi:
  1. **Paling disarankan:** minta arsip historis langsung ke tim data
     BMKG (paling lengkap & akurat).
  2. Alternatif: scraping `web-aviation.bmkg.go.id` pakai
     `scrape_metar_bmkg.py` (disertakan di repo ini) -- tapi situs ini
     memblokir request dari cloud/data center (termasuk GitHub Actions
     cloud runner), jadi harus dijalankan dari **GitHub self-hosted
     runner** di jaringan kantor, bukan runner cloud biasa.

## Setup di GitHub

1. **Set API token sebagai secret** (JANGAN taruh di kode):
   Settings -> Secrets and variables -> Actions -> New repository
   secret -> nama `BMKG_API_TOKEN`, isi dengan token Anda.
2. Push semua file di folder ini ke repo (termasuk folder
   `.github/workflows/`, pastikan path-nya benar persis
   `.github/workflows/update_metar_dataset.yml`, bukan di root repo).
3. Workflow otomatis aktif, jadwal jalan tiap 30 menit.
4. Untuk trigger manual: tab **Actions** -> workflow "Update METAR WALS
   dataset" -> **Run workflow**.

## PENTING sebelum dipakai produksi

- Informasikan ke tim IT/data BMKG soal rencana penarikan data otomatis
  ini, sesuai etika penggunaan sistem internal.
- Cek log tiap run di tab Actions, terutama minggu pertama -- pastikan
  API token masih valid dan formatnya tidak berubah.
- Gabungkan dengan data historis (dari tim data BMKG) sebelum dataset
  ini benar-benar layak untuk training model produksi.

## Menjalankan manual di lokal (opsional, untuk testing)

```bash
pip install -r requirements.txt

# Linux/Mac
export BMKG_API_TOKEN="isi_token_anda"
# Windows (cmd)
set BMKG_API_TOKEN=isi_token_anda

python run_pipeline.py
```
