# MLBB Draft Analytics

Analisis data draft pertandingan profesional **Mobile Legends: Bang Bang** untuk menjawab satu pertanyaan:
**seberapa besar pengaruh draft (pick & ban) terhadap hasil pertandingan di level pro?**

Proyek ini mencakup pipeline data dari Liquipedia, pembersihan dan validasi data, pemetaan setiap game ke patch yang berlaku,
serta evaluasi model secara *walk-forward*, di mana model hanya boleh belajar dari masa lalu untuk menebak masa depan.

---

## Data

| | |
|---|---|
| Jumlah game | **1.462** (1.454 valid setelah validasi) |
| Turnamen | MPL ID S15–S18, MPL PH S16–S18, MSC 2026 |
| Periode | Maret 2025 – November 2026 |
| Hero unik | 118 |
| Tim | 36 |
| Patch | 18 patch berbeda |

Setiap baris adalah satu game: tim, sisi (biru/merah), pemenang, durasi, 5 pick dan 5 ban per tim, serta patch yang berlaku.

**Kualitas data.** Liquipedia diisi oleh komunitas, jadi datanya tidak selalu konsisten. Yang ditangani pipeline:
- Nama hero ditulis dengan banyak alias (`yss`, `yi sun shin`, `yi` → `yi sun-shin`; `zx` → `zhuxin`; `lapu`, `lapu lapu` → `lapu-lapu`). Normalisasinya menurunkan jumlah hero unik dari 133 menjadi 118.
- Nama tim yang sama ditulis berbeda (`onic ph`, `ONIC PH`, `onic philippines`), lalu disatukan.
- Format tanggal yang tidak seragam (`October 21,2026`, `Sep 11, 2026`, `August 23 , 2026`).
- 8 game (0,5%) berisi hero yang sama dua kali dalam satu draft. Game ini ditandai dan tidak dipakai dalam analisis.
- Patch di infobox turnamen hanya satu nilai untuk seluruh musim. Sebagai gantinya, patch per game ditentukan dari tanggal rilis patch.

---

## Temuan

### 1. Patch menggeser meta secara drastis

Perubahan dari developer langsung terlihat di draft tim pro:

| Hero | Perubahan | Dampak di pertandingan pro |
|---|---|---|
| **Suyou** | Nerf di patch 1.9.64B | Kehadiran di draft (pick + ban) turun dari 91% → 69% → 49%, lalu hilang (3%). Win rate saat di-pick turun dari 73% (22/30) ke 31% (5/16). |
| **Lancelot** | Buff di patch 1.9.90 | Dari hampir tidak dipakai (0–8%) menjadi hadir di **100%** draft pada patch berikutnya. |
| **Kalea** | Hero baru di patch 1.9.64 | Dari 0% ke 49%, lalu 98% draft dalam dua patch. |

Implikasinya: statistik hero harus selalu dibaca per patch, dan data dari patch lama perlu diberi bobot lebih kecil.

### 2. Di level pro, kekuatan tim jauh lebih menentukan daripada draft

Lima model dibandingkan dengan evaluasi *walk-forward* pada **872 game uji** di tiga periode (MPL S17, MSC 2026, MPL S18).
Setiap model hanya dilatih dengan game sebelum periode uji.

| Model | Akurasi menebak pemenang |
|---|---|
| Selalu menebak sisi biru | 53,8% |
| Rata-rata win rate hero | 52,3% |
| Logistic regression dari draft | 52,9% |
| **Kekuatan tim (Elo rating)** | **61,6%** |
| Elo + draft | 61,7% |

Draft saja tidak lebih baik dari selalu menebak sisi biru. Rating Elo sederhana, tanpa melihat draft sama sekali, mencapai sekitar 62%,
dan menambahkan draft di atasnya hampir tidak mengubah hasil.

Penjelasan yang paling masuk akal: tim pro sama-sama mengambil hero meta dan mem-ban hero terkuat, sehingga draft kedua tim cenderung seimbang.
Yang membedakan adalah eksekusi.

### 3. Skor sinergi dan counter dari data pro sebagian besar adalah noise

Sinergi (pasangan hero dalam satu tim) dan counter (hero A melawan hero B) dihitung relatif terhadap kekuatan masing-masing hero,
dengan *shrinkage* agar pasangan dengan sedikit game tidak terlihat ekstrem.

- Sebagai fitur tambahan, sinergi dan counter hanya mengubah akurasi dari 61,1% menjadi 61,2–61,4%.
- Skor yang dihitung dari Maret–Agustus 2026 hampir tidak berkorelasi dengan skor dari MPL S18:
  korelasi sinergi −0,08 sampai 0,24, korelasi counter 0,03 sampai 0,18. Bahkan kekuatan hero individual hanya berkorelasi 0,12 antar periode.

Artinya, pasangan yang terlihat "bersinergi" di satu periode umumnya tidak bertahan di periode berikutnya:
sampelnya terlalu kecil dan meta keburu berubah karena patch.

### Catatan tambahan: keuntungan sisi biru

Pada musim 2025, sisi biru menang 49–55% tergantung turnamen. Pada MPL S18 (2026), sisi biru menang **56,6%**,
kemungkinan terkait perubahan map yang diacak di setiap game pada musim tersebut.

---

## Metodologi singkat

- **Walk-forward evaluation**: dilatih dengan masa lalu, diuji di turnamen berikutnya. Tidak ada data masa depan yang bocor ke pelatihan.
- **Time decay**: bobot game lama turun setengah setiap 120 hari.
- **Elo rating**: diperbarui game demi game (K = 24). Yang dipakai adalah rating *sebelum* game dimainkan.
- **Shrinkage**: win rate hero dan skor pasangan ditarik ke nilai netral kalau jumlah game-nya sedikit.
- **Cross-fitting**: fitur sinergi untuk data latih dihitung dari game *lain* (5-fold). Tanpa ini, model "menghafal"
  dan akurasinya justru turun ke sekitar 56%. Ini kesalahan yang sempat terjadi di versi awal, lalu diperbaiki.
- **Interval kepercayaan**: akurasi dilaporkan dengan bootstrap 95% CI. Dengan sekitar 380 game uji per periode,
  rentangnya kira-kira ±5 poin persen, jadi selisih kecil antar model tidak bisa dianggap bermakna.

---

## Struktur proyek

```
mlbb-draft/
├── fetch_liquipedia.py   # ambil data match per turnamen (MediaWiki API, patuh rate limit)
├── fetch_patches.py      # ambil daftar patch, tanggal rilis, dan buff/nerf per hero
├── prepare_data.py       # gabung, normalisasi alias, validasi, petakan patch per game
├── train_baseline.py     # evaluasi walk-forward: draft vs Elo
├── draft_pairs.py        # sinergi & counter + uji apakah menambah daya prediksi
├── requirements.txt
└── data/
    ├── <turnamen>.csv
    ├── patches/          # patches.csv, hero_changes.csv
    └── combined/         # all_games.csv, synergy.csv, counter.csv, baseline_results.csv
```

## Cara menjalankan

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
```

Isi `USER_AGENT` di `fetch_liquipedia.py` dengan nama proyek dan email kamu (diwajibkan oleh aturan API Liquipedia), lalu:

```bash
python fetch_patches.py --since 2024
python fetch_liquipedia.py "MPL/Indonesia/Season_18"   # ulangi untuk turnamen lain
python prepare_data.py
python train_baseline.py
python draft_pairs.py
```

---

## Keterbatasan

- Patch per game ditentukan dari tanggal rilis. Turnamen pro kadang memakai patch yang sedikit tertinggal dari server publik, jadi pemetaan ini adalah perkiraan.
- Urutan pick/ban (draft order) dan data per pemain belum dipakai.
- Data hanya dari tiga jenis turnamen. Liga lain (MY, SG, MENA, dan sebagainya) belum masuk.
- Hasil berlaku untuk level pro. Di level ranked, pengaruh draft kemungkinan berbeda.

## Rencana berikutnya

Prototipe **draft assistant** untuk dipakai tim saat draft, yang menggabungkan tiga sumber:
1. Meta patch terkini dari data pro, selalu dengan jumlah game dan label keyakinan.
2. Data tim sendiri: hero pool dan win rate tiap pemain dari scrim dan turnamen.
3. Basis pengetahuan sinergi dan counter yang bisa diedit coach, dengan data sebagai validasi.

---

## Atribusi

Data pertandingan dan patch berasal dari [Liquipedia](https://liquipedia.net/mobilelegends/), dilisensikan di bawah
[CC-BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/). Data diambil melalui MediaWiki API sesuai
[API Terms of Use](https://liquipedia.net/api-terms-of-use) Liquipedia.

Proyek ini tidak berafiliasi dengan Moonton maupun Liquipedia.
