# Eksperimen

> **Peringatan:** semua isi folder ini adalah **simulasi dengan data sintetis**, bukan eksperimen skripsi dengan DINOv2 dan video asli. Hasilnya hanya ilustrasi mekanisme dan tidak boleh dikutip sebagai hasil penelitian.

## `simulasi_noise_curvature/`

`sim_noise_curvature.py`: lintasan buatan 384-D, 24 frame. Menunjukkan bahwa noise per-frame menarik curvature ke ±120° untuk kedua kelas, sehingga gap real–AI menyempit. Hasil (gap 40° → 25° → 10° → 4° seiring noise naik) dipakai di `docs/01_ringkasan_bimbingan_2026-11-13.md` bagian 4.

```bash
python3 sim_noise_curvature.py
```

## `simulasi_projector/`

Simulasi model B0, B1, P2, P3, P4 (lihat `docs/02_riset_projector_2026-11-21.md`, Q3).

- `sim_projector.py`: data sintetis 64-D, noise anisotropik per tingkat kompresi, fitur 21-D ala ReStraV, MLP 21→64→32→1, projector 64→64 tanpa bias dengan inisialisasi identitas, early stopping, λ dipilih lewat validation. Menulis `results.json` dan bobot `p3_seed*.pt` (bobot tidak di-commit).
- `analyze.py`: rata-rata AUROC per model, lalu analisis projector P3 (panjang kolom W, drift fitur, gap curvature, mean curvature video asli).
- `analyze2.py`: pembanding P2 serta uji "konsistensi saja" (tanpa loss klasifikasi).
- `results.json`: hasil 3 seed (21 Nov 2026).

```bash
pip install torch scikit-learn numpy
python3 sim_projector.py   # ±3–5 menit di CPU
python3 analyze.py
python3 analyze2.py
```

## Rencana eksperimen sungguhan (belum ada)

Saran struktur saat mulai:

- `eksperimen/restrav_reproduksi/`: reproduksi kode ReStraV resmi.
- `eksperimen/ekstraksi_embedding/`: skrip re-encode H.264 (ffmpeg, CRF 0/10/20/30/40, FPS disamakan) dan ekstraksi CLS DINOv2 → simpan `.npy` (jangan di-commit, ukurannya besar).
- `eksperimen/model_b0_p4/`: training dan evaluasi B0–P4 dari embedding yang disimpan.
