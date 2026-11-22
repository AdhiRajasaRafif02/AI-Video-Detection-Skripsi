"""Unduh dataset ComGenVid (video + metadata) dari Hugging Face.

Tidak mengunduh embeddings.parquet (DINOv3, tidak dipakai ReStraV).
Bisa dijalankan ulang: file yang sudah ada dilewati.

Pemakaian:
    .venv\\Scripts\\python.exe eksperimen\\persiapan_data\\unduh_comgenvid.py [folder_tujuan]
"""
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

REPO_ID = "OmerXYZ/comgenvid"
TUJUAN = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(r"D:\Skripsi\data\comgenvid")

POLA = [
    "README.md",
    "metadata.jsonl",
    "sora_sampled_videos.csv",
    "videos/**",
]


def main():
    TUJUAN.mkdir(parents=True, exist_ok=True)
    print(f"Mengunduh {REPO_ID} -> {TUJUAN}", flush=True)
    snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        allow_patterns=POLA,
        local_dir=TUJUAN,
        max_workers=8,
    )
    n = sum(1 for _ in (TUJUAN / "videos").rglob("*.mp4"))
    print(f"Selesai. Jumlah file .mp4: {n} (diharapkan 5100)", flush=True)


if __name__ == "__main__":
    main()
