"""
Ambil METAR terkini dari API resmi BMKG (cuaca.bmkg.go.id)
=============================================================

Menggantikan scrape_metar_bmkg.py (scraping HTML web-aviation.bmkg.go.id)
untuk pengumpulan data KE DEPAN, karena:
- Resmi, token-based, tidak perlu handling CSRF/session
- Tidak terblokir WAF seperti percobaan scraping sebelumnya (perlu
  dikonfirmasi sekali lagi setelah dites dari GitHub Actions)

KETERBATASAN PENTING:
- API ini HANYA mengembalikan observasi TERKINI (2 laporan terakhir),
  TIDAK mendukung query rentang tanggal historis (sudah dites beberapa
  kombinasi parameter, semua hasilnya sama).
- Karena itu, pipeline yang pakai modul ini WAJIB dijadwalkan sering
  (disarankan tiap 30 menit, mengikuti interval terbit METAR reguler),
  bukan 1x/hari seperti rencana scraping sebelumnya -- kalau jadwalnya
  kejarang, ada jam yang terlewat dan datanya hilang permanen (API ini
  tidak punya cara "mundur" ambil yang terlewat).
- Untuk backfill data historis (sebelum pipeline ini mulai jalan),
  modul ini TIDAK BISA dipakai -- perlu minta arsip ke tim data BMKG,
  atau scraping web-aviation.bmkg.go.id lewat self-hosted runner.
"""

import os
import sys

import requests

ENDPOINT_TEMPLATE = "https://cuaca.bmkg.go.id/api/v1/aviation/metar/{icao}"


def get_api_token() -> str:
    token = os.environ.get("BMKG_API_TOKEN")
    if not token:
        raise RuntimeError(
            "Environment variable BMKG_API_TOKEN belum di-set. "
            "Di GitHub Actions, ini harus diisi lewat repository secret "
            "bernama BMKG_API_TOKEN, JANGAN di-hardcode di kode."
        )
    return token


def fetch_latest_metar(icao: str) -> list[str]:
    """
    Ambil laporan METAR terkini (biasanya 2 laporan terakhir) untuk
    satu stasiun ICAO. Mengembalikan list raw METAR string (field
    'data_text' dari API), format sama seperti sebelumnya sehingga
    bisa langsung dipakai parse_metar_structured.py tanpa perubahan.
    """
    token = get_api_token()
    url = ENDPOINT_TEMPLATE.format(icao=icao)

    resp = requests.get(url, params={"api_token": token}, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    records = data.get(icao, [])
    raw_lines = [r["data_text"].strip() for r in records if r.get("data_text")]
    return raw_lines


if __name__ == "__main__":
    # Tes cepat manual
    try:
        lines = fetch_latest_metar("WALS")
        print(f"Dapat {len(lines)} baris:")
        for line in lines:
            print(" ", line)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
