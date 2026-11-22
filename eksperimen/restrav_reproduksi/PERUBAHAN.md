# Reproduksi ReStraV: sumber dan perubahan

## Sumber

- Repo resmi: https://github.com/ChristianInterno/ReStraV
- Commit yang dikunci: `c51ee5205002c345f59546f97c3c71747a4bfa14` (13 Okt 2026)
- Lisensi: Apache-2.0 (lihat `upstream/LICENSE`)
- `upstream/` adalah salinan **tanpa perubahan** dari skrip-skrip di commit tersebut. Gambar `assets/` dan `DATA/ref_file_paths.txt` (1,5 MB) tidak disalin.

## File lokal dan perbedaannya dengan upstream

| File lokal | Asal upstream | Perubahan | Alasan | Berpengaruh ke hasil? |
|---|---|---|---|---|
| `restrav_lokal.py` | `dinov2_features.py` | Decoder PyAV menggantikan torchcodec, dengan aturan sampling yang sama (24 frame, jendela 2 detik di tengah video, `repeat_last`) | torchcodec di Windows butuh FFmpeg *shared*; FFmpeg di laptop versi static | Mungkin sedikit: pemilihan frame dibuat sama, tetapi decoder berbeda dan bisa berbeda di tepi (frame pertama, video variabel FPS). Belum dibandingkan langsung dengan torchcodec |
| `restrav_lokal.py` | `dinov2_features.py` | DINOv2 dijalankan per video (24 frame), bukan 128 video sekaligus | GTX 1650 hanya 4 GB VRAM | Tidak (hanya ukuran batch inferensi) |
| `restrav_lokal.py` | `dinov2_features.py` | CLS token per frame juga dikembalikan dan disimpan | Untuk varian CLS-only dan projector nanti | Tidak untuk 21 fitur |
| `restrav_lokal.py` | `dinov2_features.py` | Fungsi `compute_temporal_geometry`, `moment4`, `features_from_Z` | — | Disalin apa adanya |
| `ekstrak_fitur.py` | `DATA/extract_training_features.py` | Membaca ComGenVid (metadata.jsonl), bukan data latih VSC + VidProM; bisa dilanjutkan; mencatat video gagal | Dataset skripsi berbeda | Ya, datanya berbeda (sengaja) |
| `train_lokal.py` | `train.py` | Path HDF5 dan folder output dari argumen; `cuda:0` | Struktur folder dan GPU lokal | Tidak |
| `buat_split.py` | — (baru) | Split tetap 60/20/20 per sumber, dikelompokkan per YouTube ID untuk MSVD | Anti-bocor; split sama untuk semua CRF | Ya (protokol B0) |
| `train_b0.py` | `train.py` | Model dan hyperparameter sama; normalisasi dari train, semua data dengan bobot kelas, τ dari val, 5 seed, bisa evaluasi kondisi lain lewat `--eval` | Protokol B0 bersih. Rincian di docstring dan `HASIL.md` Run 2 | Ya (sengaja) |
| `uji_b0.py` | — (baru) | Uji kebenaran split, normalisasi, bobot, AUROC, determinisme, label acak | Memastikan bagian wajib benar | — |

## Catatan yang perlu dibawa ke skripsi

1. **Dimensi embedding: teks paper tidak konsisten, kode resmi = 98.688-D.**
   - Paper (arXiv 2507.00583v4, Bag. 4, paragraf "To implement this…", sekitar hlm. 5) menulis frame di-resize ke 224×224, lalu "384 CLS tokens and 196 patch embedding (16∗16 patches of the 224∗224 inputs)", dan z ∈ ℝ^75648. Teks ini bertentangan dengan dirinya sendiri: 224/14 = 16, jadi 16×16 = **256** patch, bukan 196. Angka 75.648 = (196+1) × 384, cocok dengan 196 patch, bukan 256.
   - Kode resmi (`upstream/dinov2_features.py`) me-resize ke 224×224 dan mengambil CLS + semua patch: (256+1) × 384 = **98.688-D**. Versi lokal mengikuti kode.
   - Sikap untuk skripsi: tulis bahwa reproduksi mengikuti kode resmi (98.688-D), dan sebut ketidakkonsistenan di teks paper sebagai catatan. Nomor halaman pasti perlu dicek di PDF versi NeurIPS.
2. **Protokol `train.py` resmi punya kebocoran kecil:** mean/std normalisasi dihitung dari seluruh data sebelum split train/test. Threshold τ* dipilih di data train. Hanya satu seed (42), split 50/50, 10 epoch. Untuk reproduksi dijalankan apa adanya. Untuk eksperimen skripsi (B0 dst.), normalisasi harus di-fit pada train saja, dan split harus tetap sama antar-CRF.
3. **Label:** 1 = real, 0 = fake (mengikuti upstream). Probabilitas keluaran = P(real).
4. **Dataset ComGenVid punya perbedaan sistematis antar kelas** (bitrate, FPS, resolusi, 29 video Sora ber-watermark). Hasil pada video "asli" bisa memanfaatkan jalan pintas itu. Lihat `docs/04_log_bimbingan.md` entri 22 Nov 2026.

## Cara menjalankan

```powershell
# dari folder repo
.\.venv\Scripts\python.exe eksperimen\persiapan_data\unduh_comgenvid.py
cd eksperimen\restrav_reproduksi
..\..\.venv\Scripts\python.exe ekstrak_fitur.py --per-sumber 10 --out D:\Skripsi\data\fitur\uji_kecil.h5
..\..\.venv\Scripts\python.exe train_lokal.py D:\Skripsi\data\fitur\uji_kecil.h5 D:\Skripsi\data\hasil\uji_kecil
```

Output fitur dan model disimpan di `D:\Skripsi\data\` (di luar repo, tidak di-commit).

## Verifikasi setup (22 Nov 2026)

- Lingkungan: `.venv` di root repo, versi paket di `requirements_lokal.txt`. PyTorch 2.5.1+cu121, GPU GTX 1650 terdeteksi.
- Dataset: 5.100/5.100 video ComGenVid terunduh ke `D:\Skripsi\data\comgenvid`.
- Uji ekstraksi 30 video (10 per sumber, seed 42): 30 berhasil, 0 gagal, ±1,04 detik/video. Perkiraan seluruh dataset: ±1,5 jam per kondisi.
- Tidak ada NaN dan tidak ada jarak nol (frame kembar) di 7 langkah awal.
- Decoder: untuk 14 video (4 per sumber + 2 video berdurasi < 2 detik), frame hasil seek identik piksel-demi-piksel dengan decode penuh tanpa seek. Ini memastikan seek benar, **bukan** bukti identik dengan torchcodec.
- `train_lokal.py` berjalan sampai selesai pada 30 video uji. Angka performanya (10 video test) tidak bermakna dan tidak boleh dikutip.
- Ekstraksi penuh 5.100 video dan reproduksi apa adanya: lihat `HASIL.md` (Run 1).
