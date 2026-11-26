"""
Gabungkan semua CSV turnamen di folder data/, bersihkan, validasi, lalu simpan
ke data/combined/all_games.csv.

  python prepare_data.py

Yang dilakukan:
  - Normalisasi nama hero & tim (alias dari fetch_liquipedia.py), jadi CSV lama
    tidak perlu di-fetch ulang setiap kali daftar alias bertambah
  - Tambah kolom: tournament, game_id, blue_team, red_team, blue_win, minutes
  - Validasi tiap game: 5 pick + 5 ban per tim, tidak ada hero muncul dua kali
    dalam satu game. Game yang bermasalah ditandai di kolom `issue` (tidak dibuang)
  - Cetak daftar hero yang jarang muncul (kandidat alias / salah ketik baru)
"""

from collections import Counter
from pathlib import Path

import pandas as pd

from fetch_liquipedia import norm_hero, norm_team

DATA_DIR = Path(__file__).parent / "data"
OUT_DIR = DATA_DIR / "combined"
PICKS = [f"t{t}_pick{i}" for t in (1, 2) for i in range(1, 6)]
BANS = [f"t{t}_ban{i}" for t in (1, 2) for i in range(1, 6)]


def to_minutes(x):
    try:
        m, s = str(x).split(":")
        return int(m) + int(s) / 60
    except ValueError:
        return None


def check_game(row) -> str:
    problems = []
    heroes = [row[c] for c in PICKS + BANS if isinstance(row.get(c), str)]
    if sum(isinstance(row.get(c), str) for c in PICKS) < 10:
        problems.append("pick<10")
    if sum(isinstance(row.get(c), str) for c in BANS) < 10:
        problems.append("ban<10")
    dup = [h for h, n in Counter(heroes).items() if n > 1]
    if dup:
        problems.append("duplikat:" + "/".join(dup))
    if {row["team1_side"], row["team2_side"]} != {"blue", "red"}:
        problems.append("sisi?")
    return ";".join(problems)


PATCH_FILE = DATA_DIR / "patches" / "patches.csv"


def add_patch(df: pd.DataFrame) -> pd.DataFrame:
    """
    Isi kolom `patch` per game = patch terakhir yang rilis pada/sebelum tanggal game.
    Kolom patch lama (dari infobox turnamen) disimpan sebagai `patch_infobox`.
    Juga menambah `days_since_patch`: game di awal patch biasanya meta belum stabil.
    """
    df = df.rename(columns={"patch": "patch_infobox"})
    # "October 29, 2025 - 13:00" -> 2025-10-29 (juga menangani "October 21,2026")
    day = (df["date"].astype(str).str.split(" - ").str[0]
           .str.replace(r"\s*,\s*", ", ", regex=True).str.strip())
    df["game_date"] = pd.to_datetime(day, format="mixed", errors="coerce")
    if df["game_date"].isna().any():
        bad = df.loc[df["game_date"].isna(), "date"].unique()[:5]
        print(f"  Tanggal tidak terbaca ({df['game_date'].isna().sum()} game), contoh: {list(bad)}")
    if not PATCH_FILE.exists():
        print("  (data/patches/patches.csv belum ada, kolom patch dilewati)")
        return df

    patches = pd.read_csv(PATCH_FILE, dtype={"patch": str})
    patches["release_date"] = pd.to_datetime(patches["release_date"], errors="coerce")
    patches = patches.dropna(subset=["release_date"]).sort_values("release_date")

    has_date = df["game_date"].notna()
    dated = df[has_date].sort_values("game_date")
    merged = pd.merge_asof(dated, patches, left_on="game_date",
                           right_on="release_date", direction="backward")
    merged["days_since_patch"] = (merged["game_date"] - merged["release_date"]).dt.days
    merged = merged.drop(columns="release_date")
    out = pd.concat([merged, df[~has_date]], ignore_index=True)
    print(f"  Patch terpetakan: {out['patch'].notna().sum()}/{len(out)} game, "
          f"{out['patch'].nunique()} patch berbeda")
    return out


def main() -> None:
    files = sorted(DATA_DIR.glob("*.csv"))
    if not files:
        raise SystemExit("Tidak ada CSV di folder data/.")
    frames = []
    for f in files:
        df = pd.read_csv(f)
        df["tournament"] = f.stem
        frames.append(df)
        print(f"  {f.name}: {len(df)} game")
    df = pd.concat(frames, ignore_index=True)

    for c in PICKS + BANS:
        df[c] = df[c].map(norm_hero)
    df["team1"] = df["team1"].map(norm_team)
    df["team2"] = df["team2"].map(norm_team)
    # nama yang hanya beda huruf besar/kecil ("the mongolz" vs "The MongolZ")
    # disatukan ke ejaan yang paling sering muncul (utamakan yang memakai huruf besar)
    all_names = pd.concat([df["team1"], df["team2"]])
    canonical = (all_names.groupby(all_names.str.lower())
                 .agg(lambda s: max(s.value_counts().items(),
                                    key=lambda kv: (kv[0] != kv[0].lower(), kv[1]))[0]))
    df["team1"] = df["team1"].str.lower().map(canonical)
    df["team2"] = df["team2"].str.lower().map(canonical)

    df["game_id"] = (df["tournament"] + "_" + df["match_idx"].astype(str)
                     + "_g" + df["game"].astype(str))
    t1_blue = df["team1_side"].str.lower() == "blue"
    df["blue_team"] = df["team1"].where(t1_blue, df["team2"])
    df["red_team"] = df["team2"].where(t1_blue, df["team1"])
    df["blue_win"] = ((df["winner"] == 1) == t1_blue).astype(int)
    df["minutes"] = df["length"].map(to_minutes)
    df["issue"] = df.apply(check_game, axis=1)
    df = add_patch(df)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "all_games.csv"
    df.to_csv(out, index=False)

    bad = df[df["issue"] != ""]
    print(f"\nTotal {len(df)} game, {len(bad)} bermasalah -> {out}")
    if len(bad):
        print(bad[["game_id", "issue"]].to_string(index=False))

    counts = Counter(h for c in PICKS + BANS for h in df[c].dropna())
    rare = sorted(h for h, n in counts.items() if n <= 3)
    print(f"\n{len(counts)} hero unik. Muncul <=3 kali (cek: hero asli atau alias?):")
    print("  " + ", ".join(rare))
    print("\nTim:", ", ".join(sorted(set(df["team1"]) | set(df["team2"]))))


if __name__ == "__main__":
    main()
