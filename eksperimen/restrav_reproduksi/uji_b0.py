"""Uji kebenaran protokol B0 (split, normalisasi, bobot kelas, AUROC, determinisme, label acak).

Pemakaian: python uji_b0.py D:/Skripsi/data/fitur/comgenvid_asli.h5
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

import train_b0 as b0

HERE = Path(__file__).parent
h5 = sys.argv[1]
dev = "cuda:0" if torch.cuda.is_available() else "cpu"
ok = True


def cek(nama, kondisi, info=""):
    global ok
    ok &= bool(kondisi)
    print(f"[{'LULUS' if kondisi else 'GAGAL'}] {nama} {info}")


split = pd.read_csv(HERE / "split" / "comgenvid_split_v1.csv")
acak = pd.read_csv(HERE / "split" / "comgenvid_split_acak_v1.csv")

# 1. split
cek("split: setiap grup hanya di satu split", (split.groupby("group").split.nunique() == 1).all())
cek("split: semua 5.100 video punya split", len(split) == 5100 and split.split.notna().all())
yt = split.assign(yt=split.group)
tr_g = set(yt[(yt.split == "train") & (yt.source_model == "MSVD")].yt)
te_g = set(yt[(yt.split == "test") & (yt.source_model == "MSVD")].yt)
cek("split: tidak ada YouTube ID MSVD yang sama di train dan test", not (tr_g & te_g))
# seberapa bocor split acak (tanpa grup)? pakai kunci grup dari split utama
m = acak.merge(split[["file_name", "group"]].rename(columns={"group": "yt"}), on="file_name")
tr_a = set(m[(m.split == "train") & (m.source_model == "MSVD")].yt)
te_a = m[(m.split == "test") & (m.source_model == "MSVD")]
bocor = te_a.yt.isin(tr_a).mean()
print(f"       info: di split ACAK, {bocor:.1%} klip MSVD test punya saudara (YouTube ID sama) di train")

# 2. fitur + normalisasi
X, meta = b0.baca_fitur(h5, split)
s = meta.split.values
mean = X[s == "train"].mean(0, keepdims=True)
std = X[s == "train"].std(0, keepdims=True) + 1e-8
Xn = (X - mean) / std
cek("normalisasi: mean train ≈ 0", np.abs(Xn[s == "train"].mean(0)).max() < 1e-4)
cek("normalisasi: std train ≈ 1", np.abs(Xn[s == "train"].std(0) - 1).max() < 1e-3)
cek("normalisasi: statistik test TIDAK dipakai (mean test ≠ 0)", np.abs(Xn[s == "test"].mean(0)).max() > 1e-3)
nol = int((X[:, 14] <= 0).sum())  # indeks 14 = min jarak (mn_d)
print(f"       info: {nol} video punya jarak antar-frame = 0 (frame kembar)")

# 3. bobot kelas
y = meta.label.values[s == "train"]
n, n1 = len(y), y.sum()
w = np.where(y == 1, n / (2 * n1), n / (2 * (n - n1)))
cek("bobot: total bobot real = total bobot fake", abs(w[y == 1].sum() - w[y == 0].sum()) < 1e-6,
    f"({w[y == 1].sum():.1f} vs {w[y == 0].sum():.1f})")

# 4. AUROC: sklearn vs hitung manual (Mann-Whitney: P(skor real > skor fake), seri = 0,5)
rng = np.random.default_rng(0)
yy = rng.integers(0, 2, 500)
pp = np.round(rng.random(500) + 0.3 * yy, 2)  # dibulatkan agar ada nilai seri
pos, neg = pp[yy == 1], pp[yy == 0]
manual = ((pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()) / (len(pos) * len(neg))
cek("AUROC: sklearn = Mann-Whitney manual", abs(manual - roc_auc_score(yy, pp)) < 1e-12)

# 5. determinisme
tr, te = s == "train", s == "test"
Xn = Xn.astype(np.float32)
a1 = roc_auc_score(meta.label.values[te], b0.skor(b0.latih(Xn[tr], meta.label.values[tr], 7, dev)[0], Xn[te], dev))
a2 = roc_auc_score(meta.label.values[te], b0.skor(b0.latih(Xn[tr], meta.label.values[tr], 7, dev)[0], Xn[te], dev))
cek("determinisme: seed sama -> AUROC sama", abs(a1 - a2) < 1e-3, f"({a1:.4f} vs {a2:.4f})")

# 6. label train diacak -> AUROC test harus ≈ 0,5 (kalau tidak, ada kebocoran di pipeline)
ya = rng.permutation(meta.label.values[tr])
a0 = roc_auc_score(meta.label.values[te], b0.skor(b0.latih(Xn[tr], ya, 0, dev)[0], Xn[te], dev))
cek("label acak: AUROC test ≈ 0,5", abs(a0 - 0.5) < 0.06, f"({a0:.4f})")

print("\nSEMUA LULUS" if ok else "\nADA YANG GAGAL")
sys.exit(0 if ok else 1)
