# CLAUDE.md — Skripsi: Deteksi Video AI pada Kondisi Kompresi H.264

File ini dibaca otomatis oleh Claude Code setiap sesi. Isinya ringkasan status terkini dan aturan kerja. Detail lengkap ada di folder `docs/`.

## Tentang proyek

- **Mahasiswa:** Adhi Rajasa Rafif
- **Bahasa kerja:** Bahasa Indonesia. Istilah teknis bahasa Inggris boleh dipakai, tapi jelaskan dengan bahasa sederhana. Mahasiswa ingin memahami *proses dan alasannya*, bukan hanya istilah.
- **Topik:** apakah detektor video AI masih bisa membedakan video asli dari video buatan AI ketika **kedua kelas** sama-sama dikompresi H.264 (beberapa tingkat CRF). Metode dasar: **ReStraV** (Internò dkk., NeurIPS 2025), yaitu DINOv2 → lintasan embedding → distance & curvature → 21 fitur → MLP.
- **Arah skripsi:** (1) eksperimen ReStraV + DINOv2 pada video terkompresi, lalu (2) mengembangkan peningkatan metode dan membuktikannya lewat eksperimen ulang.

## Status terkini (perbarui bagian ini setiap ada perubahan)

**Terakhir diperbarui:** 21 November 2026

| Hal | Status |
|---|---|
| Topik dan fokus ReStraV | Disepakati mahasiswa. Fokus dipersempit dari "ReStraV vs MISLnet" menjadi ReStraV saja |
| Kritik dosen: "hanya eksperimen, bukan membuat sesuatu" | Dijawab dengan usulan projector konsisten-kompresi (belum disetujui) |
| Usulan metode | **Projector** (lapisan linear terlatih) di antara DINOv2 dan perhitungan geometri, dilatih dengan BCE + loss konsistensi berpasangan (video asli vs versi H.264). Model: B0, B1, P2, P3 (utama), P4 (varian kontrastif). **Usulan, belum disetujui dosen** |
| Bimbingan 21 Nov 2026 | Dosen menanyakan: kenapa namanya projector, berapa node, efek sebelum/sesudah, works tanpa contrastive loss?, apakah contrastive learning?, referensi matematisnya. Jawaban lengkap di `docs/02_riset_projector_2026-11-21.md` |
| Tugas dari dosen yang belum dikerjakan | **Studi literatur metode-metode berbasis vektor (dengan referensi jelas) untuk mulai menulis Bab 1** |
| Dataset | Kandidat ComGenVid. Belum diunduh atau diverifikasi |
| Eksperimen nyata | **Belum ada.** Yang ada hanya simulasi sintetis (`eksperimen/`), yang bukan hasil skripsi |
| Judul, CRF final, persetujuan metode | Belum final |

## Peta repo

| Path | Isi |
|---|---|
| `docs/00_KONTEKS_DAN_PROGRES.md` | **Dokumen konteks utama dan terlengkap.** Kronologi, arahan dosen, penjelasan ReStraV, usulan metode, hipotesis, rancangan eksperimen, literatur, risiko. Baca ini dulu untuk tugas yang menyangkut substansi |
| `docs/01_ringkasan_bimbingan_2026-11-13.md` | Bahan bimbingan 13 Nov: inti penelitian, hipotesis H1–H3, rancangan model, pertanyaan untuk dosen |
| `docs/02_riset_projector_2026-11-21.md` | Jawaban 6 pertanyaan dosen soal projector, lengkap dengan lokasi di referensi (bagian/persamaan/halaman), hasil simulasi, dan revisi desain |
| `docs/03_penjelasan_konsep.md` | Penjelasan sederhana konsep-konsep yang pernah ditanyakan mahasiswa (vektor, geometri, projector, loss, backprop, training, AUROC) |
| `docs/04_log_bimbingan.md` | Log kronologis bimbingan dan keputusan |
| `referensi/daftar_referensi.md` | Daftar paper dengan tautan dan catatan relevansi |
| `eksperimen/simulasi_projector/` | Simulasi sintetis B0–P4 (PyTorch). Ilustrasi mekanisme saja |
| `eksperimen/simulasi_noise_curvature/` | Simulasi: noise per-frame menarik curvature ke ~120° |

## Aturan kerja untuk Claude

1. **Bedakan status dengan tegas:** hasil yang sudah dilakukan, fakta dari literatur, usulan asisten, arahan dosen (menurut cerita mahasiswa), dan hal yang belum diketahui. Jangan ubah usulan menjadi "keputusan" tanpa konfirmasi mahasiswa.
2. **Jangan mengarang** hasil eksperimen, persetujuan dosen, metadata dataset, atau klaim novelty "pertama". Angka simulasi sintetis tidak boleh disebut hasil skripsi.
3. **Setiap argumen ilmiah wajib punya referensi dengan lokasinya** (bagian, persamaan, tabel, halaman). Mahasiswa meminta tidak ada celah logika. Kalau lokasi belum diverifikasi dari sumber, katakan demikian.
4. **Hal yang sudah dikoreksi, jangan diulang:**
   - ReStraV asli memakai CLS + patch token (75.648-D). CLS-only (384-D) adalah varian yang harus disebut sebagai varian.
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
3. Reproduksi kode ReStraV resmi (https://github.com/ChristianInterno/ReStraV); cek CLS vs CLS+patch dan cara menghitung 21 fitur.
4. Pilot kecil: ±100 video × CRF 0/20/40 untuk menguji H1 (apakah gap curvature real–AI menyempit).
