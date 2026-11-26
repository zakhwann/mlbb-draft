"""
Ambil data draft (pick, ban, sisi, pemenang) pertandingan MLBB dari Liquipedia.

Cara kerja:
  1. Cari halaman turnamen + semua subhalamannya (mis. .../Regular_Season, .../Playoffs)
  2. Ambil WIKITEXT tiap halaman lewat MediaWiki API resmi (bukan halaman HTML,
     karena scraping HTML dilarang oleh API Terms of Use Liquipedia)
  3. Parse template {{Match}} dan {{Map}} -> satu baris per game
  4. Simpan ke CSV

Aturan Liquipedia yang dipatuhi skrip ini:
  - User-Agent wajib menyebut nama aplikasi + kontak  -> isi USER_AGENT di bawah
  - Maksimal 1 request / 2 detik untuk API biasa      -> REQUEST_DELAY = 2.1
  - Pakai gzip                                         -> otomatis oleh requests
  - Wikitext disimpan di cache supaya tidak fetch ulang
  - Data berlisensi CC-BY-SA: cantumkan "Data: Liquipedia" saat dipublikasikan

Pemakaian:
  python fetch_liquipedia.py "MPL/Indonesia/Season_16"
  python fetch_liquipedia.py "MPL/Indonesia/Season_16" --inspect   # lihat isi 1 template Map mentah
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

import mwparserfromhell
import pandas as pd
import requests

# ===== GANTI dengan nama project + email kamu (wajib menurut aturan Liquipedia) =====
USER_AGENT = "MLBBDraftAnalytics/0.1 zakhwan779@gmail.com"
# =====================================================================================

API_URL = "https://liquipedia.net/mobilelegends/api.php"
REQUEST_DELAY = 2.1  # detik antar request
CACHE_DIR = Path(__file__).parent / "cache"
OUTPUT_DIR = Path(__file__).parent / "data"

# Liquipedia kadang memakai singkatan nama hero / tim. Semua dinormalkan ke satu nama.
# Tambahkan di sini kalau menemukan alias baru (cek dengan fungsi check_names di analisis).
HERO_ALIASES = {
    "arlot": "arlott", "ceci": "cecilion", "esme": "esmeralda", "gatot": "gatotkaca",
    "haya": "hayabusa", "lance": "lancelot", "luoyi": "luo yi", "x.borg": "xborg",
    "lapulapu": "lapu-lapu", "lapu lapu": "lapu-lapu", "lapu": "lapu-lapu",
    "yss": "yi sun-shin", "yi sun shin": "yi sun-shin", "yi": "yi sun-shin",
    "yuzhong": "yu zhong", "valen": "valentina", "vale": "valentina",
    "fred": "fredrinn", "phove": "phoveus", "bea": "beatrix", "bene": "benedetta",
    "fara": "faramis", "mathil": "mathilda", "teri": "terizla", "wan": "wanwan",
    "popol & kupa": "popol and kupa", "popol": "popol and kupa",
    "guin": "guinevere", "yz": "yu zhong", "mino": "minotaur", "zx": "zhuxin",
    "paq": "paquito", "paqu": "paquito", "khu": "khufra", "leo": "leomord",
    "luo": "luo yi", "nova": "novaria", "obsi": "obsidia", "pove": "phoveus",
    "rafa": "rafaela", "xb": "xborg",
    # "val" bisa Valir atau Valentina; satu-satunya kemunculan (MPL ID S18) ada di game
    # yang juga berisi Valir, jadi di data ini artinya Valentina.
    "val": "valentina",
}
# Kunci huruf kecil -> nama resmi. Nama yang tidak ada di sini tetap dipakai apa adanya.
TEAM_ALIASES = {
    "natus vincere": "NAVI", "navi": "NAVI",
    "rrq hoshi": "RRQ Hoshi", "rrq": "RRQ Hoshi",
    "onic": "ONIC", "onic esports": "ONIC",
    "onic ph": "ONIC PH", "onic philippines": "ONIC PH",
    "bigetron esports": "Bigetron", "bigetron by vitality": "Bigetron",
    "alter ego": "Alter Ego", "evos": "EVOS", "geek fam id": "Geek Fam ID",
    "dewa united esports": "Dewa United Esports", "team liquid id": "Team Liquid ID",
    "ap.bren": "AP.Bren", "aurora gaming ph": "Aurora Gaming PH",
    "omega esports": "Omega Esports", "team falcons ph": "Team Falcons PH",
    "team liquid ph": "Team Liquid PH", "tnc pro team": "TNC Pro Team",
    "twisted minds ph": "Twisted Minds PH",
    "onic id": "ONIC", "evos esports": "EVOS", "geek fam indonesia": "Geek Fam ID",
    "aurora ph": "Aurora Gaming PH", "aurora gaming": "Aurora Gaming PH",
    "fut esports (male team)": "FUT Esports", "gzg": "Guangzhou Gaming",
    "kog": "King of Gamers Club",
    # catatan: "Team Falcons" (MENA) dan "Team Falcons PH" memang tim berbeda
}


def norm_hero(name) -> str:
    if not isinstance(name, str):
        return name
    n = " ".join(name.strip().lower().split())
    return HERO_ALIASES.get(n, n)


def norm_team(name) -> str:
    if not isinstance(name, str):
        return name
    n = " ".join(name.strip().split())
    return TEAM_ALIASES.get(n.lower(), n)


_session = requests.Session()
_session.headers.update({"User-Agent": USER_AGENT, "Accept-Encoding": "gzip"})
_last_request = 0.0


# ---------------------------------------------------------------- fetching
def api_get(params: dict) -> dict:
    """Satu request ke MediaWiki API, dengan jeda wajib."""
    global _last_request
    wait = REQUEST_DELAY - (time.time() - _last_request)
    if wait > 0:
        time.sleep(wait)
    params = {**params, "format": "json", "formatversion": "2"}
    resp = _session.get(API_URL, params=params, timeout=30)
    _last_request = time.time()
    resp.raise_for_status()
    return resp.json()


def list_pages(root: str) -> list[str]:
    """Halaman turnamen + semua subhalamannya (Regular_Season, Playoffs, dst)."""
    pages = [root]
    cont = {}
    while True:
        data = api_get({"action": "query", "list": "allpages",
                        "apprefix": root + "/", "aplimit": "50", **cont})
        pages += [p["title"] for p in data["query"]["allpages"]]
        if "continue" not in data:
            break
        cont = data["continue"]
    return pages


def get_wikitext(title: str) -> str:
    """Wikitext satu halaman, pakai cache lokal."""
    CACHE_DIR.mkdir(exist_ok=True)
    cache_file = CACHE_DIR / (re.sub(r"[^\w.-]", "_", title) + ".wiki")
    if cache_file.exists():
        return cache_file.read_text(encoding="utf-8")
    data = api_get({"action": "query", "prop": "revisions", "titles": title,
                    "rvprop": "content", "rvslots": "main"})
    page = data["query"]["pages"][0]
    if page.get("missing"):
        return ""
    text = page["revisions"][0]["slots"]["main"]["content"]
    cache_file.write_text(text, encoding="utf-8")
    return text


# ---------------------------------------------------------------- parsing
def _val(tpl, name: str) -> str:
    """Nilai parameter template sebagai teks bersih ('' kalau tidak ada)."""
    if not tpl.has(name):
        return ""
    return tpl.get(name).value.strip_code().strip()


def _opponent_name(match_tpl, idx: int) -> str:
    """|opponent1={{TeamOpponent|onic}} -> 'onic'."""
    if not match_tpl.has(f"opponent{idx}"):
        return ""
    value = match_tpl.get(f"opponent{idx}").value
    inner = value.filter_templates(recursive=False)
    if inner:
        t = inner[0]
        if t.has("1"):
            return t.get("1").value.strip_code().strip()
        if t.has("template"):
            return t.get("template").value.strip_code().strip()
    return value.strip_code().strip()


def _numbered(tpl, prefix: str) -> list[str]:
    """Semua parameter seperti t1h1, t1h2, ... urut berdasarkan angka."""
    found = []
    for p in tpl.params:
        m = re.fullmatch(prefix + r"(\d+)", str(p.name).strip())
        if m:
            found.append((int(m.group(1)), p.value.strip_code().strip()))
    return [v for _, v in sorted(found) if v]


def get_patch(wikitext: str) -> str:
    """Ambil |patch= dari infobox turnamen (kalau ada)."""
    for tpl in mwparserfromhell.parse(wikitext).filter_templates():
        if str(tpl.name).strip().lower().startswith("infobox") and tpl.has("patch"):
            return _val(tpl, "patch")
    return ""


def parse_games(wikitext: str, page_title: str, patch: str = "") -> list[dict]:
    """Satu dict per game dari semua template {{Match}} di halaman."""
    rows = []
    code = mwparserfromhell.parse(wikitext)
    for m_idx, match in enumerate(code.filter_templates(recursive=True)):
        if str(match.name).strip().lower() != "match":
            continue
        team1 = norm_team(_opponent_name(match, 1))
        team2 = norm_team(_opponent_name(match, 2))
        date = _val(match, "date")

        for p in match.params:
            pname = str(p.name).strip()
            if not re.fullmatch(r"map\d+", pname):
                continue
            maps = [t for t in p.value.filter_templates(recursive=False)
                    if str(t.name).strip().lower() == "map"]
            if not maps:
                continue
            g = maps[0]
            winner = _val(g, "winner")
            if winner not in ("1", "2"):
                continue  # game belum dimainkan / tidak dimainkan

            row = {
                "page": page_title,
                "match_idx": m_idx,
                "date": date,
                "patch": patch,
                "team1": team1,
                "team2": team2,
                "game": int(pname[3:]),
                "team1_side": _val(g, "team1side"),
                "team2_side": _val(g, "team2side"),
                "winner": int(winner),
                "length": _val(g, "length"),
            }
            for team in (1, 2):
                for i, hero in enumerate(_numbered(g, f"t{team}h")[:5], 1):
                    row[f"t{team}_pick{i}"] = norm_hero(hero)
                for i, hero in enumerate(_numbered(g, f"t{team}b")[:5], 1):
                    row[f"t{team}_ban{i}"] = norm_hero(hero)
            rows.append(row)
    return rows


def inspect_first_map(wikitext: str) -> None:
    """Cetak 1 template Map mentah, untuk cek nama-nama parameternya."""
    for tpl in mwparserfromhell.parse(wikitext).filter_templates(recursive=True):
        if str(tpl.name).strip().lower() == "map":
            print(str(tpl)[:2000])
            return
    print("Tidak ada template {{Map}} di halaman ini.")


# ---------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tournament", help='Judul halaman, mis. "MPL/Indonesia/Season_16"')
    ap.add_argument("--inspect", action="store_true",
                    help="Cetak 1 template Map mentah lalu berhenti")
    args = ap.parse_args()

    if "example.com" in USER_AGENT:
        sys.exit("Isi dulu USER_AGENT dengan nama project + email kamu.")

    root = args.tournament.replace(" ", "_")
    pages = list_pages(root)
    print(f"{len(pages)} halaman ditemukan:", *pages, sep="\n  ")

    patch = get_patch(get_wikitext(root))
    all_rows = []
    for title in pages:
        text = get_wikitext(title)
        if args.inspect and "{{Map" in text:
            print(f"\n--- contoh Map dari {title} ---")
            inspect_first_map(text)
            return
        rows = parse_games(text, title, patch)
        print(f"  {title}: {len(rows)} game")
        all_rows += rows

    if not all_rows:
        print("\nTidak ada game yang terbaca. Jalankan dengan --inspect untuk cek format.")
        return

    df = pd.DataFrame(all_rows)
    OUTPUT_DIR.mkdir(exist_ok=True)
    out = OUTPUT_DIR / (re.sub(r"[^\w.-]", "_", root) + ".csv")
    df.to_csv(out, index=False)
    print(f"\n{len(df)} game disimpan ke {out}")
    print(f"Kolom pick lengkap (5 v 5): {df.filter(like='_pick5').notna().all(axis=1).sum()} game")
    print("Data: Liquipedia (CC-BY-SA 3.0) — cantumkan atribusi saat dipublikasikan.")


if __name__ == "__main__":
    main()
