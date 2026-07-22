"""
Scraper METAR/SPECI historis dari web-aviation.bmkg.go.id
==========================================================

Sumber: https://web-aviation.bmkg.go.id/web/metar_speci.php

Situs ini pakai proteksi CSRF (ala Laravel), jadi alurnya:
1. GET halaman dulu -> ambil cookie session + token CSRF
2. POST data pencarian (stasiun, rentang tanggal, dll) pakai token itu
3. Parse HTML respons -> ambil baris METAR/SPECI mentah

CATATAN ETIKA & KESOPANAN SCRAPING:
- Ini domain resmi BMKG tempat Anda bekerja. Sebelum menjadwalkan ini
  otomatis (misal via GitHub Actions harian), informasikan dulu ke tim
  IT/data BMKG soal rencana ini.
- Jangan set jadwal terlalu sering (cukup 1x/hari cukup untuk arsip
  historis). Beri jeda antar request kalau menarik banyak stasiun.
- Script ini HANYA untuk keperluan internal/riset BMKG oleh pegawai
  yang berwenang mengakses data tersebut.
"""

import re
import time
from datetime import datetime

import requests
from bs4 import BeautifulSoup

BASE_URL = "https://web-aviation.bmkg.go.id/web/metar_speci.php"

HEADERS_COMMON = {
    "User-Agent": "Mozilla/5.0 (BMKG-internal-research-script)",
}


def get_csrf_token(session: requests.Session) -> str:
    """Ambil token CSRF dari halaman awal (meta tag atau hidden input)."""
    resp = session.get(BASE_URL, headers=HEADERS_COMMON, timeout=30)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    meta = soup.find("meta", attrs={"name": "csrf-token"})
    if meta and meta.get("content"):
        return meta["content"]

    token_input = soup.find("input", attrs={"name": "_token"})
    if token_input and token_input.get("value"):
        return token_input["value"]

    raise RuntimeError(
        "Token CSRF tidak ditemukan di halaman. "
        "Struktur halaman mungkin berubah -- cek ulang lewat DevTools."
    )


def fetch_metar_raw_html(
    session: requests.Session,
    token: str,
    stasiun: str,
    start_dt: datetime,
    end_dt: datetime,
) -> str:
    """Kirim POST pencarian dan kembalikan HTML mentah hasil respons."""
    payload = {
        "stasiun": stasiun,
        "from": start_dt.strftime("%Y-%m-%dT%H:%M"),
        "to": end_dt.strftime("%Y-%m-%dT%H:%M"),
        "metar": "SA",
        "speci": "SP",
        "_token": token,
    }
    headers = {
        **HEADERS_COMMON,
        "X-Requested-With": "XMLHttpRequest",
        "Referer": BASE_URL,
    }
    resp = session.post(BASE_URL, data=payload, headers=headers, timeout=30)
    resp.raise_for_status()
    return resp.text


def parse_metar_lines(html_text: str) -> list[str]:
    """
    Ekstrak baris METAR/SPECI mentah dari HTML respons.
    Pola: 'METAR WALS 220230Z ... =' atau 'SPECI WALS ... ='

    PENTING: kalau hasilnya kosong, kemungkinan struktur HTML respons
    berbeda dari asumsi di sini -- cek tab 'Response' di DevTools
    (bukan cuma 'Payload') untuk lihat format aslinya, lalu sesuaikan
    regex/parsing di bawah.
    """
    pattern = re.compile(r"((?:METAR|SPECI)\s+\w{4}\s+\d{6}Z.*?=)")
    return pattern.findall(html_text)


def scrape_wals(start_dt: datetime, end_dt: datetime) -> list[str]:
    session = requests.Session()
    token = get_csrf_token(session)
    html = fetch_metar_raw_html(session, token, "WALS", start_dt, end_dt)
    return parse_metar_lines(html)


if __name__ == "__main__":
    # Contoh: tarik data 5 hari terakhir (sesuaikan rentang sesuai kebutuhan)
    start = datetime(2026, 7, 17, 20, 46)
    end = datetime(2026, 7, 22, 8, 46)

    lines = scrape_wals(start, end)

    print(f"Ditemukan {len(lines)} baris METAR/SPECI untuk WALS:\n")
    for line in lines:
        print(line)

    # Simpan ke file mentah -- nanti tinggal disambung ke tahap parsing
    # terstruktur (lihat pembahasan sebelumnya soal parsing METAR jadi kolom)
    with open("wals_metar_raw.txt", "w") as f:
        f.write("\n".join(lines))

    time.sleep(1)  # jeda sopan kalau nanti loop banyak stasiun/tanggal
