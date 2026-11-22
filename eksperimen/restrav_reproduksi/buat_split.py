"""Buat split train/val/test yang TETAP dan anti-bocor untuk ComGenVid.

Kenapa perlu dikelompokkan (grouped split):
  Klip MSVD diberi nama <YouTubeID>_<detik-awal>_<detik-akhir>.mp4. Satu video YouTube bisa
  menghasilkan banyak klip (sampai 21 klip), kadang saling tumpang-tindih waktunya. Kalau klip dari
  video YouTube yang sama masuk ke train DAN test, model bisa "mengenali video sumbernya", bukan
  membedakan real vs AI. Karena itu semua klip dengan YouTube ID yang sama dimasukkan ke split yang sama.
  Sora: kunci grup = username pembuat (di data ini setiap username hanya punya 1 video).
  VEO3: nama file berupa UUID, tidak ada info pengelompokan, jadi setiap video = 1 grup.

Pembagian per sumber (MSVD, Sora, VEO3) supaya proporsi tiap sumber sama di train/val/test.
Target 60/20/20 dihitung dari jumlah klip.

Split disimpan ke CSV dan dipakai SAMA untuk semua kondisi CRF dan semua model (kunci = file_name).

Contoh:
    python buat_split.py                      # split utama (grouped)
    python buat_split.py --tanpa-grup --out split/comgenvid_split_acak_v1.csv   # diagnostik kebocoran
"""
import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path

import pandas as pd

TARGET = (("train", 0.6), ("val", 0.2), ("test", 0.2))


def kunci_grup(r):
    nama = r["filename"].rsplit(".", 1)[0]
    if r["source_model"] == "MSVD":
        return "MSVD:" + re.sub(r"_\d+_\d+$", "", nama)
    if r["source_model"] == "Sora":
        return "Sora:" + re.sub(r"_\d+%2Fmd$", "", nama)
    return f"{r['source_model']}:{nama}"


def bagi(groups_sizes, rng):
    """groups_sizes: dict grup -> jumlah klip. Kembalikan dict grup -> split."""
    grup = sorted(groups_sizes)
    rng.shuffle(grup)
    total = sum(groups_sizes.values())
    batas, kum = [], 0.0
    for nama, p in TARGET:
        kum += p
        batas.append((nama, kum * total))
    hasil, n = {}, 0
    for g in grup:
        # tempatkan grup di split pertama yang belum penuh (dilihat dari titik tengah grup)
        tengah = n + groups_sizes[g] / 2
        for nama, b in batas:
            if tengah <= b:
                hasil[g] = nama
                break
        n += groups_sizes[g]
    return hasil


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=r"D:\Skripsi\data\comgenvid")
    ap.add_argument("--out", default=str(Path(__file__).parent / "split" / "comgenvid_split_v1.csv"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tanpa-grup", action="store_true", help="setiap klip = grupnya sendiri (diagnostik)")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(Path(args.data) / "metadata.jsonl", encoding="utf-8") if l.strip()]
    df = pd.DataFrame({
        "file_name": [r["file_name"] for r in rows],
        "source_model": [r["source_model"] for r in rows],
        "label": [1 if r["source_model"] == "MSVD" else 0 for r in rows],
        "group": [r["file_name"] if args.tanpa_grup else kunci_grup(r) for r in rows],
    })
    assert df.file_name.is_unique

    rng = random.Random(args.seed)
    split = {}
    for src in sorted(df.source_model.unique()):
        sizes = Counter(df.loc[df.source_model == src, "group"])
        split.update(bagi(dict(sizes), rng))
    df["split"] = df.group.map(split)
    assert df.split.notna().all()

    # cek: tidak ada grup yang muncul di lebih dari satu split
    assert (df.groupby("group").split.nunique() == 1).all()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.sort_values("file_name").to_csv(out, index=False)
    print(f"Disimpan: {out}")
    print(pd.crosstab(df.source_model, df.split, margins=True))
    print("Jumlah grup per split:\n", df.groupby(["source_model", "split"]).group.nunique().unstack())


if __name__ == "__main__":
    main()
