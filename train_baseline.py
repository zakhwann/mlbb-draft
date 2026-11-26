"""
Fase 2, model draft pertama: seberapa baik draft (5 hero vs 5 hero) memprediksi pemenang?

  python train_baseline.py

Evaluasi WALK-FORWARD (jujur terhadap waktu): model selalu dilatih HANYA dengan game
sebelum periode uji, lalu diuji di turnamen berikutnya. Tidak ada data masa depan bocor.

Model yang dibandingkan:
  0. "Selalu sisi biru"   -> baseline paling bodoh
  1. Rata-rata win rate hero (dengan smoothing) -> baseline yang masuk akal
  2. Logistic regression: tiap hero +1 kalau di tim biru, -1 kalau di tim merah,
     game lama diberi bobot lebih kecil (time decay)
  3. Elo tim: hanya kekuatan tim, tanpa melihat draft
  4. Elo + draft: apakah draft menambah akurasi di atas kekuatan tim?

Metrik:
  - accuracy  : % tebakan pemenang yang benar
  - log loss  : seberapa baik PROBABILITAS-nya (lebih kecil = lebih baik; 0.693 = tebak 50:50)
  - brier     : mirip log loss, lebih mudah dibaca (0.25 = tebak 50:50)
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss

DATA = Path(__file__).parent / "data" / "combined" / "all_games.csv"

# (nama periode uji, tanggal mulai, tanggal akhir)
FOLDS = [
    ("MPL S17 (ID+PH)", "2026-03-01", "2026-06-30"),
    ("MSC 2026", "2026-07-01", "2026-08-10"),
    ("MPL S18 (ID+PH)", "2026-08-11", "2026-12-31"),
]
HALF_LIFE_DAYS = 120  # bobot game turun setengah tiap 120 hari
SMOOTH = 10           # model 1: anggap tiap hero punya 10 game "netral" (50%)
C_REG = 0.3           # model 2: kekuatan regularisasi (kecil = lebih hati-hati)


def load() -> pd.DataFrame:
    df = pd.read_csv(DATA, parse_dates=["game_date"])
    df = df[df["issue"].isna()].copy()
    t1_blue = df["team1_side"] == "blue"
    for i in range(1, 6):
        df[f"blue_pick{i}"] = df[f"t1_pick{i}"].where(t1_blue, df[f"t2_pick{i}"])
        df[f"red_pick{i}"] = df[f"t2_pick{i}"].where(t1_blue, df[f"t1_pick{i}"])
    return df.sort_values("game_date").reset_index(drop=True)


def time_weights(dates: pd.Series, ref: pd.Timestamp) -> np.ndarray:
    age = (ref - dates).dt.days.clip(lower=0).to_numpy()
    return 0.5 ** (age / HALF_LIFE_DAYS)


def blue_red(df):
    blue = df[[f"blue_pick{i}" for i in range(1, 6)]].to_numpy()
    red = df[[f"red_pick{i}" for i in range(1, 6)]].to_numpy()
    return blue, red


# ---------------------------------------------------------------- model 1
def hero_winrates(train: pd.DataFrame, w: np.ndarray) -> dict:
    blue, red = blue_red(train)
    y = train["blue_win"].to_numpy()
    wins, games = {}, {}
    for side, won in ((blue, y), (red, 1 - y)):
        for j in range(5):
            for hero, win, wt in zip(side[:, j], won, w):
                games[hero] = games.get(hero, 0) + wt
                wins[hero] = wins.get(hero, 0) + wt * win
    return {h: (wins[h] + SMOOTH * 0.5) / (games[h] + SMOOTH) for h in games}


def predict_winrate(test, wr, blue_rate):
    blue, red = blue_red(test)
    b = np.vectorize(lambda h: wr.get(h, 0.5))(blue).mean(axis=1)
    r = np.vectorize(lambda h: wr.get(h, 0.5))(red).mean(axis=1)
    # selisih rata-rata win rate -> probabilitas, digeser oleh keuntungan sisi biru
    logit = np.log(blue_rate / (1 - blue_rate)) + 8 * (b - r)
    return 1 / (1 + np.exp(-logit))


# ---------------------------------------------------------------- model 2
def encode(df, heroes: list[str]) -> np.ndarray:
    idx = {h: k for k, h in enumerate(heroes)}
    X = np.zeros((len(df), len(heroes)))
    blue, red = blue_red(df)
    for r in range(len(df)):
        for h in blue[r]:
            if h in idx:
                X[r, idx[h]] += 1
        for h in red[r]:
            if h in idx:
                X[r, idx[h]] -= 1
    return X


# ---------------------------------------------------------------- model 3 & 4
ELO_K = 24


def add_elo(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rating Elo tiap tim, diperbarui game demi game sesuai urutan waktu.
    Kolom elo_diff = rating biru - rating merah SEBELUM game dimainkan (tanpa bocoran).
    """
    rating, diffs = {}, []
    for blue, red, won in zip(df["blue_team"], df["red_team"], df["blue_win"]):
        rb, rr = rating.get(blue, 1500.0), rating.get(red, 1500.0)
        diffs.append(rb - rr)
        expected = 1 / (1 + 10 ** (-(rb - rr) / 400))
        rating[blue] = rb + ELO_K * (won - expected)
        rating[red] = rr - ELO_K * (won - expected)
    df["elo_diff"] = np.array(diffs) / 400
    return df


def evaluate(name, y, p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return {"model": name, "accuracy": accuracy_score(y, p > 0.5),
            "log_loss": log_loss(y, p, labels=[0, 1]), "brier": brier_score_loss(y, p)}


def bootstrap_acc_ci(y, pred, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    correct = (y == pred).astype(float)
    accs = [correct[rng.integers(0, len(y), len(y))].mean() for _ in range(n)]
    return np.percentile(accs, [2.5, 97.5])


def main() -> None:
    df = add_elo(load())
    print(f"{len(df)} game valid, {df['game_date'].min():%Y-%m-%d} s/d {df['game_date'].max():%Y-%m-%d}\n")
    results = []
    for fold, start, end in FOLDS:
        train = df[df["game_date"] < start]
        test = df[(df["game_date"] >= start) & (df["game_date"] <= end)]
        if len(test) == 0:
            continue
        ref = pd.Timestamp(start)
        w = time_weights(train["game_date"], ref)
        y_tr, y_te = train["blue_win"].to_numpy(), test["blue_win"].to_numpy()
        blue_rate = np.average(y_tr, weights=w)

        # model 0
        p0 = np.full(len(test), blue_rate)
        # model 1
        p1 = predict_winrate(test, hero_winrates(train, w), blue_rate)
        # model 2
        heroes = sorted(set(np.concatenate(blue_red(train)).ravel()))
        lr = LogisticRegression(C=C_REG, max_iter=2000)
        lr.fit(encode(train, heroes), y_tr, sample_weight=w)
        p2 = lr.predict_proba(encode(test, heroes))[:, 1]
        # model 3: kekuatan tim saja (Elo) + keuntungan sisi biru
        side = np.log(blue_rate / (1 - blue_rate))
        p3 = 1 / (1 + np.exp(-(side + np.log(10) * test["elo_diff"].to_numpy())))
        # model 4: Elo + draft -> apakah draft MENAMBAH informasi di atas kekuatan tim?
        X_tr = np.hstack([encode(train, heroes), train[["elo_diff"]].to_numpy() * 10])
        X_te = np.hstack([encode(test, heroes), test[["elo_diff"]].to_numpy() * 10])
        lr4 = LogisticRegression(C=0.05, max_iter=3000).fit(X_tr, y_tr, sample_weight=w)
        p4 = lr4.predict_proba(X_te)[:, 1]

        print(f"=== Uji: {fold} | latih {len(train)} game, uji {len(test)} game ===")
        for name, p in (("0 selalu biru", p0), ("1 win rate hero", p1), ("2 logistic reg", p2),
                        ("3 Elo tim", p3), ("4 Elo + draft", p4)):
            r = evaluate(name, y_te, p)
            lo, hi = bootstrap_acc_ci(y_te, (p > 0.5).astype(int))
            r.update(fold=fold, n_test=len(test), acc_lo=lo, acc_hi=hi)
            results.append(r)
            print(f"  {name:16s} acc {r['accuracy']:.3f} (95% CI {lo:.3f}-{hi:.3f})  "
                  f"logloss {r['log_loss']:.3f}  brier {r['brier']:.3f}")
        print()

    res = pd.DataFrame(results)
    summary = (res.assign(correct=res["accuracy"] * res["n_test"])
               .groupby("model")[["correct", "n_test"]].sum())
    summary["accuracy"] = summary["correct"] / summary["n_test"]
    print("=== Gabungan semua periode uji ===")
    print(summary["accuracy"].round(3).to_string())
    out = DATA.parent / "baseline_results.csv"
    res.to_csv(out, index=False)
    print(f"\nHasil tersimpan di {out}")


if __name__ == "__main__":
    main()
