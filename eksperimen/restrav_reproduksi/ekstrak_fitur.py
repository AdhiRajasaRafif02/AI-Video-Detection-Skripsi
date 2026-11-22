"""Ekstraksi 21 fitur ReStraV (+ CLS per frame) untuk ComGenVid.

Output HDF5 kompatibel dengan train.py upstream (dataset `path`, `label`, `features`),
ditambah `source_model` dan `cls` [N, 24, 384] float16 untuk eksperimen lanjutan.
Label mengikuti upstream: 1 = real, 0 = fake.

Bisa dilanjutkan: video yang sudah ada di file output dilewati.

Contoh:
    # uji coba 10 video per sumber
    python ekstrak_fitur.py --per-sumber 10 --out D:/Skripsi/data/fitur/uji_kecil.h5
    # seluruh dataset (video asli, tanpa re-encode)
    python ekstrak_fitur.py --out D:/Skripsi/data/fitur/comgenvid_asli.h5
"""
import argparse
import json
import random
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from tqdm import tqdm

import restrav_lokal as rl

T = 24
D_CLS = 384


def baca_metadata(root):
    rows = []
    with open(root / "metadata.jsonl", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def pilih(rows, per_sumber, seed):
    if not per_sumber:
        return rows
    rng = random.Random(seed)
    out = []
    for src in sorted({r["source_model"] for r in rows}):
        grup = [r for r in rows if r["source_model"] == src]
        out += rng.sample(grup, min(per_sumber, len(grup)))
    return out


def buka_h5(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    f = h5py.File(path, "a")
    if "path" not in f:
        s = h5py.string_dtype()
        f.create_dataset("path", (0,), maxshape=(None,), dtype=s)
        f.create_dataset("source_model", (0,), maxshape=(None,), dtype=s)
        f.create_dataset("label", (0,), maxshape=(None,), dtype="i")
        f.create_dataset("features", (0, 21), maxshape=(None, 21), dtype="f")
        f.create_dataset("cls", (0, T, D_CLS), maxshape=(None, T, D_CLS), dtype="f2",
                         chunks=(64, T, D_CLS))
    return f


def tambah(f, path, src, label, feat, cls):
    n = f["path"].shape[0]
    for k in ("path", "source_model", "label", "features", "cls"):
        f[k].resize(n + 1, axis=0)
    f["path"][n] = path
    f["source_model"][n] = src
    f["label"][n] = label
    f["features"][n] = feat
    f["cls"][n] = cls


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=r"D:\Skripsi\data\comgenvid")
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-sumber", type=int, default=0, help="ambil N video acak per sumber (0 = semua)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    root = Path(args.data)
    rows = pilih(baca_metadata(root), args.per_sumber, args.seed)
    out = Path(args.out)
    f = buka_h5(out)
    sudah = set(p.decode() if isinstance(p, bytes) else p for p in f["path"][:])
    gagal_log = out.with_suffix(".gagal.txt")

    todo = [r for r in rows if r["file_name"] not in sudah]
    print(f"{len(rows)} video dipilih, {len(sudah)} sudah ada, {len(todo)} akan diproses. Device: {args.device}")
    rl.load_dinov2(args.device)

    t0, ok, hilang, gagal = time.time(), 0, 0, 0
    for r in tqdm(todo, desc="Ekstraksi"):
        vp = root / r["file_name"]
        if not vp.exists():
            hilang += 1
            continue
        try:
            Z, cls = rl.embed_video(vp, args.device, T=T)
            feat = rl.features_from_Z(Z).cpu().numpy()[0]
        except Exception as e:  # catat lalu lanjut
            gagal += 1
            with open(gagal_log, "a", encoding="utf-8") as g:
                g.write(f"{r['file_name']}\t{e!r}\n")
            continue
        label = 1 if r["subset"] == "real" or r["source_model"] == "MSVD" else 0
        tambah(f, r["file_name"], r["source_model"], label, feat, cls.cpu().numpy().astype(np.float16))
        ok += 1
        if ok % 50 == 0:
            f.flush()
    f.close()
    dt = time.time() - t0
    print(f"Selesai: {ok} berhasil, {gagal} gagal (lihat {gagal_log.name}), {hilang} file belum ada. "
          f"{dt:.0f} s ({dt / max(ok, 1):.2f} s/video)")


if __name__ == "__main__":
    main()
