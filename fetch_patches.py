"""
Ambil daftar patch MLBB + tanggal rilisnya dari Liquipedia, lalu simpan ke
data/patches/patches.csv. Dipakai untuk memetakan tiap game ke patch yang
berlaku saat game itu dimainkan.

  python fetch_patches.py --inspect      # lihat wikitext 1 patch terbaru (cek format)
  python fetch_patches.py --since 2025   # ambil semua patch sejak tahun 2025

Catatan: turnamen pro kadang memakai patch yang sedikit tertinggal dari server
publik. Jadi pemetaan berdasarkan tanggal adalah PERKIRAAN yang cukup baik.
"""

import argparse
import re
from pathlib import Path

import mwparserfromhell
import pandas as pd

from fetch_liquipedia import api_get, get_wikitext, norm_hero

OUT_DIR = Path(__file__).parent / "data" / "patches"


def list_patch_pages() -> list[str]:
    titles, cont = [], {}
    while True:
        data = api_get({"action": "query", "list": "allpages", "apprefix": "Patch ",
                        "aplimit": "500", **cont})
        titles += [p["title"] for p in data["query"]["allpages"]]
        if "continue" not in data:
            break
        cont = data["continue"]
    # hanya halaman versi, mis. "Patch 2.2.16" atau "Patch 2.1.18b"
    return [t for t in titles if re.fullmatch(r"Patch \d+\.\d+\.\d+\w*", t)]


def release_date(wikitext: str) -> str:
    """Cari parameter tanggal rilis di infobox patch."""
    for tpl in mwparserfromhell.parse(wikitext).filter_templates():
        if "infobox" not in str(tpl.name).lower():
            continue
        for p in tpl.params:
            name = str(p.name).strip().lower()
            if "release" in name or name in ("date", "edate", "sdate"):
                value = p.value.strip_code().strip()
                m = re.search(r"\d{4}-\d{2}-\d{2}", value)
                if m:
                    return m.group(0)
    return ""


def _change_kind(raw: str) -> str:
    raw = raw.lower()
    for kind in ("buff", "nerf", "revamp", "new"):
        if kind in raw:
            return kind
    return "adjust"


def hero_changes(wikitext: str) -> list[tuple[str, str]]:
    """
    Tiap hero ditulis sebagai template {{Herobc/...|hero=Nama|type=Buff/Nerf/Revamp}}.
    Satu hero bisa muncul di beberapa template; diambil satu baris per hero.
    """
    changes = {}
    for tpl in mwparserfromhell.parse(wikitext).filter_templates(recursive=True):
        if not str(tpl.name).strip().lower().startswith("herobc") or not tpl.has("hero"):
            continue
        hero = norm_hero(tpl.get("hero").value.strip_code())
        raw_type = tpl.get("type").value.strip_code() if tpl.has("type") else ""
        kind = _change_kind(raw_type or str(tpl.name))
        # kalau hero muncul lebih dari sekali, buff/nerf/revamp lebih informatif dari adjust
        if changes.get(hero, "adjust") == "adjust":
            changes[hero] = kind
    return sorted(changes.items())


def inspect_hero_templates(wikitext: str) -> None:
    """Tampilkan nama-nama template Herobc + contoh 1 template yang bukan revamp."""
    from collections import Counter
    tpls = [t for t in mwparserfromhell.parse(wikitext).filter_templates(recursive=True)
            if str(t.name).strip().lower().startswith("herobc")]
    print("Template Herobc:", dict(Counter(str(t.name).strip() for t in tpls)))
    for t in tpls:
        if "revamp" not in str(t).lower()[:400]:
            print("\nContoh template penyesuaian hero:\n" + str(t)[:1200])
            break


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inspect", action="store_true")
    ap.add_argument("--since", type=int, default=2025, help="ambil patch sejak tahun ini")
    args = ap.parse_args()

    titles = list_patch_pages()
    print(f"{len(titles)} halaman patch ditemukan.")

    def version_key(t):
        return [int(x) if x.isdigit() else 0 for x in re.findall(r"\d+", t)]

    titles.sort(key=version_key, reverse=True)

    if args.inspect:
        text = get_wikitext(titles[0])
        print(f"--- {titles[0]} ---")
        print(f"Tanggal terbaca: {release_date(text) or '(tidak ketemu)'}")
        inspect_hero_templates(text)
        print(f"\nPerubahan hero terbaca: {hero_changes(text)}")
        return

    patch_rows, change_rows = [], []
    for title in titles:
        text = get_wikitext(title)
        date = release_date(text)
        if date and int(date[:4]) < args.since:
            break  # sudah urut dari terbaru, sisanya lebih lama
        version = title.replace("Patch ", "")
        patch_rows.append({"patch": version, "release_date": date})
        change_rows += [{"patch": version, "hero": h, "change": c}
                        for h, c in hero_changes(text)]
        print(f"  {version}: {date or '?'}, {len(hero_changes(text))} perubahan hero")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(patch_rows).sort_values("release_date").to_csv(
        OUT_DIR / "patches.csv", index=False)
    pd.DataFrame(change_rows).to_csv(OUT_DIR / "hero_changes.csv", index=False)
    print(f"\nTersimpan di {OUT_DIR}")


if __name__ == "__main__":
    main()
