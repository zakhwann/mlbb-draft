"""
Ekspor statistik ke docs/data/meta.json untuk web Draft Assistant (folder docs/).

  python export_web_data.py                 # pakai game 100 hari terakhir
  python export_web_data.py --days 150

Lalu lihat hasilnya di browser:
  python -m http.server 8000 -d docs        # buka http://localhost:8000

Folder docs/ bisa langsung di-hosting gratis dengan GitHub Pages.
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from draft_pairs import PairModel
from train_baseline import load

ROOT = Path(__file__).parent
PATCH_DIR = ROOT / "data" / "patches"
OUT = ROOT / "docs" / "data" / "meta.json"
MIN_PAIR_GAMES = 5


def hero_meta(games: pd.DataFrame) -> list[dict]:
    n = len(games)
    picks, bans, wins = {}, {}, {}
    for t in (1, 2):
        won = (games["winner"] == t).to_numpy()
        for i in range(1, 6):
            for h, w in zip(games[f"t{t}_pick{i}"], won):
                picks[h] = picks.get(h, 0) + 1
                wins[h] = wins.get(h, 0) + int(w)
            for h in games[f"t{t}_ban{i}"]:
                bans[h] = bans.get(h, 0) + 1
    out = []
    for h in sorted(set(picks) | set(bans)):
        p, b, w = picks.get(h, 0), bans.get(h, 0), wins.get(h, 0)
        out.append({
            "hero": h, "picks": p, "bans": b, "wins": w,
            "presence": round((p + b) / n, 4),
            "banRate": round(b / n, 4),
            # smoothing: 10 game netral, supaya hero yang jarang dipakai tidak tampak ekstrem
            "winRate": round((w + 5) / (p + 10), 4),
        })
    return out


def pairs(table: dict) -> dict:
    return {f"{a}|{b}": [round(float(s), 3), round(float(n), 1)]
            for (a, b), (s, n, _, _) in table.items() if n >= MIN_PAIR_GAMES}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=100, help="jendela data meta (hari)")
    args = ap.parse_args()

    games_all = load()
    last = games_all["game_date"].max()
    start = last - pd.Timedelta(days=args.days)
    games = games_all[games_all["game_date"] >= start]
    pm = PairModel().fit(games, np.ones(len(games)))

    all_heroes = set(np.concatenate([
        games_all[[f"t{t}_{k}{i}" for t in (1, 2) for k in ("pick", "ban") for i in range(1, 6)]]
        .stack().unique()]))

    flags, newer = {}, []
    changes_file, patches_file = PATCH_DIR / "hero_changes.csv", PATCH_DIR / "patches.csv"
    if changes_file.exists() and patches_file.exists():
        changes = pd.read_csv(changes_file, dtype={"patch": str})
        patches = pd.read_csv(patches_file, dtype={"patch": str}, parse_dates=["release_date"])
        all_heroes |= set(changes["hero"])
        newer_df = patches[patches["release_date"] > last].sort_values("release_date")
        newer = [{"patch": p, "date": f"{d:%Y-%m-%d}"}
                 for p, d in zip(newer_df["patch"], newer_df["release_date"])]
        for _, r in changes[changes["patch"].isin(newer_df["patch"])].iterrows():
            flags[r["hero"]] = f"{r['change']} ({r['patch']})"

    data = {
        "generated": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "window": {"start": f"{start:%Y-%m-%d}", "end": f"{last:%Y-%m-%d}", "games": len(games),
                   "tournaments": sorted(games["tournament"].unique().tolist())},
        "heroes": sorted(all_heroes),
        "meta": hero_meta(games),
        "counter": pairs(pm.ctr),
        "synergy": pairs(pm.syn),
        "newerPatches": newer,
        "patchFlags": flags,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"{len(games)} game ({data['window']['start']} s/d {data['window']['end']}), "
          f"{len(data['meta'])} hero, {len(data['counter'])} counter, {len(data['synergy'])} sinergi")
    print(f"Tersimpan: {OUT} ({OUT.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
