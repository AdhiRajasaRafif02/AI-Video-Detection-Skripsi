# Log Bimbingan dan Keputusan

Format setiap entri: tanggal, sumber, apa yang terjadi, status. Entri terbaru di atas. "Arahan dosen" = yang diceritakan mahasiswa, bukan notulen langsung.

---

## 22 November 2026 — Sesi Claude Code: setup ReStraV dan dataset

- **Keputusan mahasiswa:** mulai eksperimen baseline (real vs AI, tanpa projector, tanpa kompresi tambahan) sambil menunggu persetujuan projector.
- Dataset ComGenVid dicek: 5.100 video (MSVD/Sora/VEO3 masing-masing 1.700), 28,18 GB video; `embeddings.parquet` berisi DINOv3 (tidak dipakai). Diunduh ke `D:\Skripsi\data\comgenvid`.
- **Risiko kebocoran:** MSVD bitrate jauh lebih tinggi (±9 vs ±1,4 Mbps Sora, perkiraan dari ukuran/durasi), FPS MSVD bervariasi sedangkan Sora 30 dan VEO3 24, resolusi VEO3 seragam 1280×720, 29 video Sora ber-watermark. Hasil pada video asli harus dibaca dengan catatan ini.
- Kode ReStraV resmi (commit `c51ee52`) disalin dan disesuaikan (`eksperimen/restrav_reproduksi/PERUBAHAN.md`).
- **Temuan:** kode resmi memakai CLS+patch = 257×384 = 98.688-D pada 224×224, sedangkan teks paper (Bag. 4) menulis 75.648-D dan bertentangan sendiri (224×224 dengan ViT-S/14 menghasilkan 256 patch, bukan 196). Reproduksi mengikuti kode. `train.py` resmi menormalisasi sebelum split.
- **Hasil nyata pertama (Run 1):** ekstraksi 5.100/5.100 video berhasil (1 jam 42 menit). Reproduksi `train.py` apa adanya: AUROC 0,834, accuracy 0,758 (test 1.700 video, satu seed). Protokol belum bersih, jadi belum menjadi B0 final. Rincian di `eksperimen/restrav_reproduksi/HASIL.md`.
- **B0 protokol bersih (Run 2):** split tetap berkelompok per YouTube ID (MSVD: 1.306 video sumber untuk 1.700 klip), normalisasi dari train, bobot kelas, τ dari val, 5 seed. AUROC 0,834 ± 0,004. Pada split acak tanpa kelompok, AUROC 0,842 ± 0,002 (26,5% klip MSVD test punya saudara di train). **Usulan asisten, belum dikonfirmasi dosen:** protokol ini dipakai untuk semua CRF, dengan AUROC sebagai metrik utama.

---

## 21 November 2026 — Sesi Claude Code: draf studi literatur Bab 1

Dibuat `docs/05_studi_literatur_bab1_2026-11-21.md` untuk tugas dosen. Referensi baru terverifikasi eksistensinya (DINOv2, Cozzolino, D3, dsb.). Isi paper selain ReStraV baru dari ringkasan alphaXiv, **belum dibaca di PDF**. Interpretasi "berbasis vektor" belum dikonfirmasi dosen.

---

## 21 November 2026 — Bimbingan (hasil dilaporkan mahasiswa)

**Masukan dosen tentang projector:**
1. Kenapa dinamakan projector padahal neuron biasa?
2. Bentuknya berapa node?
3. Bagaimana efek sebelum dan sesudahnya?
4. Apakah works kalau tanpa contrastive loss?
5. Apakah ini contrastive learning?
6. Referensi apa yang dipakai untuk matematisnya?

**Tugas dari dosen:** membuat studi literatur metode-metode berbasis vektor dengan referensi jelas, untuk mulai menulis Bab 1. **Status: belum dikerjakan** (ditunda, mahasiswa memprioritaskan memahami projector).

**Tindak lanjut:** riset lengkap di `docs/02_riset_projector_2026-11-21.md`. Revisi desain yang diusulkan: projector tanpa bias, 384→384 dengan inisialisasi identitas, konsistensi pada fitur terstandardisasi, tambah model P4 (contrastive) sebagai ablasi. **Status: usulan, belum disetujui dosen.**

**Permintaan mahasiswa:** setiap argumen harus punya lokasi di referensi dan tanpa celah logika.

---

## 13 November 2026 — Persiapan bimbingan (sesi Claude)

- Mahasiswa melaporkan kritik dosen sebelumnya: menganalisis ReStraV + DINOv2 pada video terkompresi "hanya eksperimen, bukan membuat sesuatu". Dosen juga bertanya "hipotesisnya apa".
- Yang pasti menurut mahasiswa: (1) eksperimen ReStraV + DINOv2 pada video terkompresi, (2) peningkatan metode lalu eksperimen ulang.
- Disusun: projector konsisten-kompresi sebelum geometri, hipotesis H1–H3, model B0/B1/P1/P2/P3. Lihat `docs/01_ringkasan_bimbingan_2026-11-13.md`.

---

## 29 Oktober 2026 — "Studi Literatur 5" (ChatGPT)

- Fokus dipersempit: ReStraV vs MISLnet → ReStraV saja.
- Kritik dosen: ReStraV memang sudah memakai DINOv2, jadi pasangan itu bukan kontribusi baru. Pilihan yang dibahas: solusi degradasi kompresi atau perbandingan encoder.
- Usulan saat itu: paired geometry consistency (adapter setelah 21 fitur). Direvisi 13 Nov.

---

## Sebelumnya (Juli – Oktober 2026)

Lihat tabel kronologi di `docs/00_KONTEKS_DAN_PROGRES.md` Bagian 4. Ringkasnya: eksplorasi topik (Juli) → fokus video AI terkompresi (Agustus) → ReStraV sebagai landasan (3 Sep) → arahan dosen: metode ringan, 2–3 paper 10 tahun terakhir termasuk review (24 Sep) → arah analisis + perbaikan, real dan AI dikompresi setara (1 Okt) → ReStraV vs MISLnet (8 Okt).
