"""
Sinergi & counter antar hero, dan uji apakah keduanya menambah daya prediksi.

  python draft_pairs.py                 # uji walk-forward + tabel sinergi/counter terkini
  python draft_pairs.py --since 2026-03-01 --min-games 12

Definisi (semua dalam skala logit, 0 = netral):
  - Sinergi(A,B): seberapa sering tim yang memakai A DAN B menang, DIBANDING yang
    diharapkan dari kekuatan A dan B masing-masing. Positif = lebih dari jumlahnya.
  - Counter(A,B): seberapa sering tim dengan A menang melawan tim dengan B,
    dibanding yang diharapkan dari kekuatan A dan B masing-masing.
Pasangan dengan sedikit game "ditarik" ke 0 (shrinkage), jadi angka besar hanya
muncul kalau datanya cukup banyak.
"""

import argparse
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from train_baseline import (FOLDS, add_elo, blue_red, bootstrap_acc_ci, evaluate,
                            load, time_weights)

HERO_SMOOTH = 10   # hero dengan sedikit game dianggap mendekati 50%
PAIR_SMOOTH = 15   # pasangan dengan sedikit game dianggap netral


def logit(p):
    p = np.clip(p, 1e-3, 1 - 1e-3)
    return np.log(p / (1 - p))


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class PairModel:
    """Menghitung kekuatan hero, sinergi, dan counter dari sekumpulan game."""

    def fit(self, df: pd.DataFrame, w: np.ndarray):
        blue, red = blue_red(df)
        y = df["blue_win"].to_numpy()
        h_n, h_w = defaultdict(float), defaultdict(float)
        for team, won in ((blue, y), (red, 1 - y)):
            for row, win, wt in zip(team, won, w):
                for h in row:
                    h_n[h] += wt
                    h_w[h] += wt * win
        self.hero = {h: logit((h_w[h] + HERO_SMOOTH * 0.5) / (h_n[h] + HERO_SMOOTH))
                     for h in h_n}

        # sinergi: pasangan dalam satu tim
        s_n, s_obs, s_exp = defaultdict(float), defaultdict(float), defaultdict(float)
        # counter: (hero kita, hero lawan)
        c_n, c_obs, c_exp = defaultdict(float), defaultdict(float), defaultdict(float)
        for b_row, r_row, win, wt in zip(blue, red, y, w):
            for team, opp, won in ((b_row, r_row, win), (r_row, b_row, 1 - win)):
                for a, b in combinations(sorted(team), 2):
                    s_n[(a, b)] += wt
                    s_obs[(a, b)] += wt * won
                    s_exp[(a, b)] += wt * sigmoid(self.hero[a] + self.hero[b])
                for a in team:
                    for b in opp:
                        c_n[(a, b)] += wt
                        c_obs[(a, b)] += wt * won
                        c_exp[(a, b)] += wt * sigmoid(self.hero[a] - self.hero[b])

        def shrunk(n, obs, exp):
            out = {}
            for k in n:
                e = exp[k] / n[k]
                p = (obs[k] + PAIR_SMOOTH * e) / (n[k] + PAIR_SMOOTH)
                out[k] = (logit(p) - logit(e), n[k], obs[k] / n[k], e)
            return out

        self.syn = shrunk(s_n, s_obs, s_exp)
        self.ctr = shrunk(c_n, c_obs, c_exp)
        return self

    def features(self, df: pd.DataFrame) -> np.ndarray:
        """Per game: [selisih kekuatan hero, selisih total sinergi, total counter biru vs merah]."""
        blue, red = blue_red(df)
        rows = []
        for b_row, r_row in zip(blue, red):
            hero = (sum(self.hero.get(h, 0) for h in b_row)
                    - sum(self.hero.get(h, 0) for h in r_row))
            syn = (sum(self.syn.get(p, (0,))[0] for p in combinations(sorted(b_row), 2))
                   - sum(self.syn.get(p, (0,))[0] for p in combinations(sorted(r_row), 2)))
            ctr = sum(self.ctr.get((a, b), (0,))[0] for a in b_row for b in r_row)
            rows.append((hero, syn, ctr))
        return np.array(rows)

    def table(self, which: str, min_games: float) -> pd.DataFrame:
        src = self.syn if which == "syn" else self.ctr
        cols = ["hero_a", "hero_b"]
        df = pd.DataFrame([(a, b, s, n, wr, e) for (a, b), (s, n, wr, e) in src.items()],
                          columns=cols + ["score", "games", "win_rate", "expected"])
        return df[df["games"] >= min_games].sort_values("score", ascending=False)


def crossfit_features(train: pd.DataFrame, w: np.ndarray, k: int = 5) -> np.ndarray:
    folds = np.arange(len(train)) % k
    feats = np.zeros((len(train), 3))
    for f in range(k):
        mask = folds == f
        pm = PairModel().fit(train[~mask], w[~mask])
        feats[mask] = pm.features(train[mask])
    return feats


def walk_forward(df):
    print("=== Apakah sinergi/counter menambah daya prediksi? (walk-forward) ===")
    sets = {
        "Elo saja": ["elo"],
        "Elo + kekuatan hero": ["elo", "hero"],
        "Elo + hero + sinergi": ["elo", "hero", "syn"],
        "Elo + hero + sinergi + counter": ["elo", "hero", "syn", "ctr"],
    }
    col = {"elo": 0, "hero": 1, "syn": 2, "ctr": 3}
    totals = defaultdict(lambda: [0.0, 0.0, 0])
    for fold, start, end in FOLDS:
        train = df[df["game_date"] < start]
        test = df[(df["game_date"] >= start) & (df["game_date"] <= end)]
        w = time_weights(train["game_date"], pd.Timestamp(start))
        pm = PairModel().fit(train, w)
        # Fitur untuk game LATIH dihitung secara cross-fit: tiap game dinilai oleh
        # PairModel yang dilatih tanpa game itu. Tanpa ini, model "menghafal" dan
        # terlalu percaya pada sinergi (overfitting).
        X_tr = np.column_stack([train["elo_diff"] * 10, crossfit_features(train, w)])
        X_te = np.column_stack([test["elo_diff"] * 10, pm.features(test)])
        print(f"\n{fold} (uji {len(test)} game)")
        for name, feats in sets.items():
            idx = [col[f] for f in feats]
            m = LogisticRegression(C=0.1, max_iter=2000)
            m.fit(X_tr[:, idx], train["blue_win"], sample_weight=w)
            p = m.predict_proba(X_te[:, idx])[:, 1]
            r = evaluate(name, test["blue_win"].to_numpy(), p)
            t = totals[name]
            t[0] += r["accuracy"] * len(test)
            t[1] += r["log_loss"] * len(test)
            t[2] += len(test)
            print(f"  {name:32s} acc {r['accuracy']:.3f}  logloss {r['log_loss']:.3f}")
    print("\nGabungan:")
    for name, (acc, ll, n) in totals.items():
        print(f"  {name:32s} acc {acc / n:.3f}  logloss {ll / n:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", default="2026-03-01", help="periode untuk tabel sinergi/counter")
    ap.add_argument("--min-games", type=float, default=12)
    args = ap.parse_args()

    df = add_elo(load())
    walk_forward(df)

    recent = df[df["game_date"] >= args.since]
    pm = PairModel().fit(recent, np.ones(len(recent)))
    out = Path(__file__).parent / "data" / "combined"
    syn, ctr = pm.table("syn", args.min_games), pm.table("ctr", args.min_games)
    syn.to_csv(out / "synergy.csv", index=False)
    ctr.to_csv(out / "counter.csv", index=False)
    pd.set_option("display.width", 120)
    print(f"\n=== Sinergi terkuat sejak {args.since} (min {args.min_games:g} game bersama) ===")
    print(syn.head(10).round(3).to_string(index=False))
    print(f"\n=== Counter terkuat (hero_a unggul melawan hero_b) ===")
    print(ctr.head(10).round(3).to_string(index=False))
    print(f"\nTabel lengkap: {out / 'synergy.csv'} dan {out / 'counter.csv'}")


if __name__ == "__main__":
    main()
