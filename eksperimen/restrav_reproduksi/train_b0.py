"""B0: ReStraV (21 fitur -> MLP) dengan protokol bersih.

Yang SAMA dengan train.py upstream (commit c51ee52):
  - 21 fitur yang sama, MLP 21-64-32-1 (ReLU), BCEWithLogits, Adam lr 1e-3, batch 32, 10 epoch.
  - Label 1 = real, 0 = fake. Skor = P(real).

Yang DIUBAH (dan alasannya):
  1. Split tetap dari CSV (buat_split.py): train/val/test 60/20/20, dikelompokkan per video
     sumber YouTube untuk MSVD. Upstream: split acak 50/50 tanpa kelompok.
  2. Normalisasi mean/std di-fit pada TRAIN saja. Upstream: di seluruh data sebelum split (bocor).
  3. Semua data dipakai; ketidakseimbangan kelas (1 real : 2 fake) diatasi dengan bobot per sampel
     di loss sehingga kedua kelas berbobot total sama (setara prior 50/50). Upstream: membuang
     sebagian fake secara acak.
  4. Threshold tau dipilih di VAL dengan memaksimalkan balanced accuracy. Upstream: F1 di train.
     (F1 bergantung pada proporsi kelas, sedangkan test di sini tidak 50/50.)
  5. Beberapa seed (inisialisasi MLP + urutan batch); hasil dilaporkan rata-rata dan simpangan baku.
     Split TIDAK berubah antar seed. Upstream: satu run, seed torch tidak di-set.

Metrik utama: AUROC (tidak butuh threshold). Juga AUROC per pasangan sumber (MSVD vs Sora,
MSVD vs VEO3), balanced accuracy, accuracy, dan F1 kelas real pada tau dari val.

Evaluasi kondisi lain (mis. CRF lain) dengan --eval NAMA=fitur.h5: model yang dilatih pada --h5
dievaluasi pada baris test dari file itu (split sama, dicocokkan lewat path). Normalisasi tetap
memakai mean/std train dari --h5.

Contoh:
    python train_b0.py --h5 D:/Skripsi/data/fitur/comgenvid_asli.h5 --out D:/Skripsi/data/hasil/b0_asli
"""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score

HERE = Path(__file__).parent


class MLP(nn.Module):  # sama dengan upstream train.py
    def __init__(self, in_dim=21, h1=64, h2=32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, h1), nn.ReLU(), nn.Linear(h1, h2), nn.ReLU(), nn.Linear(h2, 1))

    def forward(self, x):
        return self.net(x)


def baca_fitur(h5_path, split_df):
    """Gabungkan fitur HDF5 dengan split (kunci = path). Gagal keras kalau ada yang tidak cocok."""
    with h5py.File(h5_path, "r") as f:
        paths = [p.decode() if isinstance(p, bytes) else p for p in f["path"][:]]
        X = f["features"][:].astype(np.float32)
        y = f["label"][:].astype(np.int64)
    d = pd.DataFrame({"file_name": paths, "label_h5": y})
    d["i"] = np.arange(len(d))
    m = d.merge(split_df, on="file_name", how="inner", validate="one_to_one")
    if len(m) != len(d):
        raise ValueError(f"{len(d) - len(m)} video di {h5_path} tidak ada di split")
    if (m.label_h5 != m.label).any():
        raise ValueError("label di HDF5 tidak cocok dengan label di split")
    if not np.isfinite(X).all():
        raise ValueError("ada NaN/inf di fitur")
    return X[m.i.values], m.drop(columns=["i", "label_h5"]).reset_index(drop=True)


def skor(model, X, device):
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(torch.from_numpy(X).to(device))).cpu().numpy().ravel()


def pilih_tau(y, p):
    """tau yang memaksimalkan balanced accuracy (= Youden J) di val. Grid sama dengan upstream."""
    taus = np.linspace(0.1, 0.9, 81)
    ba = [balanced_accuracy_score(y, (p >= t).astype(int)) for t in taus]
    return float(taus[int(np.argmax(ba))])


def metrik(meta, p, tau):
    y = meta.label.values
    pred = (p >= tau).astype(int)
    out = {
        "auroc": roc_auc_score(y, p),
        "bal_acc": balanced_accuracy_score(y, pred),
        "acc": accuracy_score(y, pred),
        "f1_real": f1_score(y, pred),
        "tpr_real": pred[y == 1].mean(),   # real terdeteksi real
        "tnr_fake": 1 - pred[y == 0].mean(),  # fake terdeteksi fake
    }
    real = meta.source_model.values == "MSVD"
    for src in sorted(set(meta.source_model) - {"MSVD"}):
        sel = real | (meta.source_model.values == src)
        out[f"auroc_MSVD_vs_{src}"] = roc_auc_score(y[sel], p[sel])
        out[f"tnr_{src}"] = 1 - pred[meta.source_model.values == src].mean()
    return out


def latih(Xtr, ytr, seed, device, epochs=10, batch=32, lr=1e-3):
    torch.manual_seed(seed)
    np.random.seed(seed)
    gen = torch.Generator().manual_seed(seed)
    n, n1 = len(ytr), int(ytr.sum())
    # bobot per sampel: total bobot kelas real = total bobot kelas fake = n/2
    w = np.where(ytr == 1, n / (2 * n1), n / (2 * (n - n1))).astype(np.float32)
    ds = torch.utils.data.TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr).float(), torch.from_numpy(w))
    dl = torch.utils.data.DataLoader(ds, batch_size=batch, shuffle=True, generator=gen)
    model = MLP().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.BCEWithLogitsLoss(reduction="none")
    hist = []
    for _ in range(epochs):
        model.train()
        tot = 0.0
        for xb, yb, wb in dl:
            xb, yb, wb = xb.to(device), yb.to(device).unsqueeze(1), wb.to(device).unsqueeze(1)
            opt.zero_grad()
            loss = (crit(model(xb), yb) * wb).mean()
            loss.backward()
            opt.step()
            tot += loss.item() * len(xb)
        hist.append(tot / n)
    return model, hist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", required=True, help="fitur untuk training (dan evaluasi kondisi yang sama)")
    ap.add_argument("--split", default=str(HERE / "split" / "comgenvid_split_v1.csv"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--eval", nargs="*", default=[], help="NAMA=path.h5, kondisi lain untuk evaluasi")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    split_df = pd.read_csv(args.split)
    X, meta = baca_fitur(args.h5, split_df)
    s = meta.split.values
    print(f"{len(meta)} video. Per split:\n{pd.crosstab(meta.source_model, meta.split)}")

    # normalisasi: fit di train saja
    mean = X[s == "train"].mean(axis=0, keepdims=True)
    std = X[s == "train"].std(axis=0, keepdims=True) + 1e-8
    Xn = ((X - mean) / std).astype(np.float32)
    np.save(out / "mean.npy", mean)
    np.save(out / "std.npy", std)

    kondisi = {"sama": (Xn, meta)}
    for item in args.eval:
        nama, p = item.split("=", 1)
        Xe, me = baca_fitur(p, split_df)
        kondisi[nama] = (((Xe - mean) / std).astype(np.float32), me)

    tr, va = s == "train", s == "val"
    baris, pred_rows = [], []
    for seed in args.seeds:
        model, hist = latih(Xn[tr], meta.label.values[tr], seed, args.device, epochs=args.epochs)
        tau = pilih_tau(meta.label.values[va], skor(model, Xn[va], args.device))
        for nama, (Xk, mk) in kondisi.items():
            te = mk.split.values == "test"
            p = skor(model, Xk[te], args.device)
            m = metrik(mk[te].reset_index(drop=True), p, tau)
            baris.append({"seed": seed, "kondisi": nama, "tau": tau, "loss_akhir": hist[-1], **m})
            pred_rows.append(pd.DataFrame({"seed": seed, "kondisi": nama, "file_name": mk.file_name.values[te],
                                           "source_model": mk.source_model.values[te],
                                           "label": mk.label.values[te], "p_real": p}))
        torch.save(model.state_dict(), out / f"model_seed{seed}.pt")
        print(f"seed {seed}: tau={tau:.2f} loss={hist[-1]:.4f} AUROC(sama)={baris[-len(kondisi)]['auroc']:.4f}")

    hasil = pd.DataFrame(baris)
    hasil.to_csv(out / "metrik_per_seed.csv", index=False)
    pd.concat(pred_rows).to_csv(out / "prediksi_test.csv", index=False)
    kol = [c for c in hasil.columns if c not in ("seed", "kondisi")]
    ring = hasil.groupby("kondisi")[kol].agg(["mean", "std"])
    ring.to_csv(out / "ringkasan.csv")
    with open(out / "konfigurasi.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)
    pd.set_option("display.width", 200)
    for nama in kondisi:
        r = hasil[hasil.kondisi == nama]
        print(f"\n=== Test, kondisi '{nama}' ({len(args.seeds)} seed, rata-rata ± simpangan baku) ===")
        for c in kol:
            print(f"  {c:22s} {r[c].mean():.4f} ± {r[c].std(ddof=1) if len(r) > 1 else 0:.4f}")


if __name__ == "__main__":
    main()
