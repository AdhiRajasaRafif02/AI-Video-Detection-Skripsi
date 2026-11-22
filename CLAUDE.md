# CLAUDE.md — Skripsi: Deteksi Video AI pada Kondisi Kompresi H.264

File ini dibaca otomatis oleh Claude Code setiap sesi. Isinya ringkasan status terkini dan aturan kerja. Detail lengkap ada di folder `docs/`.

## Tentang proyek

- **Mahasiswa:** Adhi Rajasa Rafif
- **Bahasa kerja:** Bahasa Indonesia. Istilah teknis bahasa Inggris boleh dipakai, tapi jelaskan dengan bahasa sederhana. Mahasiswa ingin memahami *proses dan alasannya*, bukan hanya istilah.
- **Topik:** apakah detektor video AI masih bisa membedakan video asli dari video buatan AI ketika **kedua kelas** sama-sama dikompresi H.264 (beberapa tingkat CRF). Metode dasar: **ReStraV** (Internò dkk., NeurIPS 2025), yaitu DINOv2 → lintasan embedding → distance & curvature → 21 fitur → MLP.
- **Arah skripsi:** (1) eksperimen ReStraV + DINOv2 pada video terkompresi, lalu (2) mengembangkan peningkatan metode dan membuktikannya lewat eksperimen ulang.

## Status terkini (perbarui bagian ini setiap ada perubahan)

**Terakhir diperbarui:** 22 November 2026

| Hal | Status |
|---|---|
| Topik dan fokus ReStraV | Disepakati mahasiswa. Fokus dipersempit dari "ReStraV vs MISLnet" menjadi ReStraV saja |
| Kritik dosen: "hanya eksperimen, bukan membuat sesuatu" | Dijawab dengan usulan projector konsisten-kompresi (belum disetujui) |
| Usulan metode | **Projector** (lapisan linear terlatih) di antara DINOv2 dan perhitungan geometri, dilatih dengan BCE + loss konsistensi berpasangan (video asli vs versi H.264). Model: B0, B1, P2, P3 (utama), P4 (varian kontrastif). **Usulan, belum disetujui dosen** |
| Bimbingan 21 Nov 2026 | Dosen menanyakan: kenapa namanya projector, berapa node, efek sebelum/sesudah, works tanpa contrastive loss?, apakah contrastive learning?, referensi matematisnya. Jawaban lengkap di `docs/02_riset_projector_2026-11-21.md` |
| Tugas dosen: studi literatur metode berbasis vektor untuk Bab 1 | **Draf pertama selesai (21 Nov 2026)** di `docs/05_studi_literatur_bab1_2026-11-21.md`. Keberadaan referensi terverifikasi, tetapi isi dan lokasi (halaman/persamaan) paper selain ReStraV belum dicek di PDF (ditandai `[cek PDF]`). Interpretasi "berbasis vektor" belum dikonfirmasi dosen |
| Dataset | ComGenVid dicek 22 Nov: 5.100 video (MSVD real 1.700, Sora 1.700, VEO3 1.700), 28,18 GB, diunduh ke `D:\Skripsi\data\comgenvid` (di luar repo). Ada perbedaan sistematis antar kelas (bitrate, FPS, resolusi, 29 Sora ber-watermark): risiko jalan pintas |
| Setup ReStraV | Kode resmi commit `c51ee52` disalin ke `eksperimen/restrav_reproduksi/upstream/`; versi lokal (PyAV, batch per video, cuda:0) dan catatan perubahan di `PERUBAHAN.md`. Lingkungan `.venv` (torch 2.5.1 cu121). Kode resmi memakai CLS+patch **98.688-D** (257×384 pada 224×224), sedangkan teks paper menulis 75.648-D dan tidak konsisten secara internal (lihat PERUBAHAN.md) |
| Eksperimen nyata | Fitur 5.100/5.100 video asli terekstrak. **Run 1** (reproduksi `train.py` apa adanya): AUROC 0,834. **Run 2 = B0 protokol bersih** (split tetap berkelompok per YouTube ID 60/20/20, normalisasi dari train, bobot kelas, τ dari val, 5 seed): **AUROC 0,834 ± 0,004** (MSVD vs Sora 0,824, MSVD vs VEO3 0,845), balanced acc 0,748 ± 0,010. Uji kebenaran `uji_b0.py` lulus. Jalan pintas codec belum dikontrol; protokol belum dikonfirmasi dosen. Rincian di `eksperimen/restrav_reproduksi/HASIL.md`. Simulasi sintetis di `eksperimen/` bukan hasil skripsi |
| Judul, CRF final, persetujuan metode | Belum final |

## Peta repo

| Path | Isi |
|---|---|
| `docs/00_KONTEKS_DAN_PROGRES.md` | **Dokumen konteks utama dan terlengkap.** Kronologi, arahan dosen, penjelasan ReStraV, usulan metode, hipotesis, rancangan eksperimen, literatur, risiko. Baca ini dulu untuk tugas yang menyangkut substansi |
| `docs/01_ringkasan_bimbingan_2026-11-13.md` | Bahan bimbingan 13 Nov: inti penelitian, hipotesis H1–H3, rancangan model, pertanyaan untuk dosen |
| `docs/02_riset_projector_2026-11-21.md` | Jawaban 6 pertanyaan dosen soal projector, lengkap dengan lokasi di referensi (bagian/persamaan/halaman), hasil simulasi, dan revisi desain |
| `docs/03_penjelasan_konsep.md` | Penjelasan sederhana konsep-konsep yang pernah ditanyakan mahasiswa (vektor, geometri, projector, loss, backprop, training, AUROC) |
| `docs/04_log_bimbingan.md` | Log kronologis bimbingan dan keputusan |
| `docs/05_studi_literatur_bab1_2026-11-21.md` | Draf studi literatur metode berbasis vektor (UnivFD, Cozzolino, ReStraV, D3, SPLIT, TRACE), celah kompresi, daftar `[cek PDF]` |
| `referensi/daftar_referensi.md` | Daftar paper dengan tautan dan catatan relevansi |
| `eksperimen/restrav_reproduksi/` | Kode ReStraV resmi (`upstream/`) + versi lokal untuk ComGenVid: `restrav_lokal.py`, `ekstrak_fitur.py`, `train_lokal.py`, protokol B0 (`buat_split.py`, `train_b0.py`, `uji_b0.py`, `split/`), `PERUBAHAN.md`, dan `HASIL.md` (catatan setiap run) |
| `eksperimen/persiapan_data/` | `unduh_comgenvid.py` |
| `eksperimen/simulasi_projector/` | Simulasi sintetis B0–P4 (PyTorch). Ilustrasi mekanisme saja |
| `eksperimen/simulasi_noise_curvature/` | Simulasi: noise per-frame menarik curvature ke ~120° |

## Aturan kerja untuk Claude

1. **Bedakan status dengan tegas:** hasil yang sudah dilakukan, fakta dari literatur, usulan asisten, arahan dosen (menurut cerita mahasiswa), dan hal yang belum diketahui. Jangan ubah usulan menjadi "keputusan" tanpa konfirmasi mahasiswa.
2. **Jangan mengarang** hasil eksperimen, persetujuan dosen, metadata dataset, atau klaim novelty "pertama". Angka simulasi sintetis tidak boleh disebut hasil skripsi.
3. **Setiap argumen ilmiah wajib punya referensi dengan lokasinya** (bagian, persamaan, tabel, halaman). Mahasiswa meminta tidak ada celah logika. Kalau lokasi belum diverifikasi dari sumber, katakan demikian.
4. **Hal yang sudah dikoreksi, jangan diulang:**
   - ReStraV asli memakai CLS + patch token (teks paper menulis 75.648-D tetapi tidak konsisten dengan 224×224; kode resmi menghasilkan 98.688-D dan reproduksi mengikuti kode; lihat `eksperimen/restrav_reproduksi/PERUBAHAN.md`). CLS-only (384-D) adalah varian yang harus disebut sebagai varian.
   - Projector tidak memakai bias (bias hilang di Δz). Versi 384→128 = 49.152 bobot, bukan 49.280.
   - Desain P3 **bukan** contrastive learning.
   - MTOR (2026) sudah memakai ReStraV sebagai baseline di bawah gangguan, termasuk H.264 dengan QP (bukan CRF). Jangan klaim sebagai yang pertama menguji ReStraV pada kompresi.
5. **Prioritas ringan dan realistis untuk skripsi**, tetapi tetap jujur soal biaya ekstraksi DINOv2 dan storage.
6. Penelitian ini **computer vision / video forensics**, bukan NLP.

## Aturan memperbarui konteks (penting)

Setiap kali ada informasi baru dari mahasiswa (hasil bimbingan, keputusan, perubahan rencana, hasil eksperimen):

1. Perbarui tabel **"Status terkini"** di file ini, termasuk tanggalnya.
2. Tambahkan entri di `docs/04_log_bimbingan.md`.
3. Perbarui bagian yang relevan di `docs/00_KONTEKS_DAN_PROGRES.md`. Tandai bagian yang direvisi, jangan hapus jejak keputusan lama.
4. Kalau ada riset atau dokumen baru yang panjang, buat file baru di `docs/` dengan nomor urut dan tanggal, lalu daftarkan di "Peta repo".
5. Commit dengan pesan berbahasa Indonesia yang menjelaskan apa yang berubah, misalnya `docs: catat hasil bimbingan 28 Nov`.

Jangan commit PDF paper (hak cipta), dataset video, atau file embedding besar. Simpan tautannya di `referensi/daftar_referensi.md`.

## Langkah berikutnya yang disarankan

1. Studi literatur metode berbasis vektor/embedding untuk Bab 1 (tugas dosen).
2. Konfirmasi ke dosen: arah projector (P3) dan varian P4, hipotesis H1–H3, dataset, grid CRF.
3. ~~Ekstraksi penuh, reproduksi apa adanya, dan protokol B0 bersih~~ (selesai 22 Nov, lihat `HASIL.md`). Split tetap: `eksperimen/restrav_reproduksi/split/comgenvid_split_v1.csv`. Berikutnya: re-encode seragam (codec, FPS, resolusi sama untuk semua kelas) untuk menetralkan perbedaan codec antar kelas, lalu jalankan `train_b0.py` lagi.
4. Pilot kecil: ±100 video × CRF 0/20/40 untuk menguji H1 (apakah gap curvature real–AI menyempit).
