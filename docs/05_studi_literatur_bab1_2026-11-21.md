# Studi Literatur Bab 1: Deteksi Video AI Berbasis Vektor (Embedding)

Draf kerja, 21 November 2026. Menjawab tugas dosen: "studi literatur metode-metode berbasis vektor dengan referensi jelas, untuk mulai menulis Bab 1".

## 0. Status dan batas dokumen ini (baca dulu)

- **Interpretasi tugas (perlu dikonfirmasi ke dosen):** "berbasis vektor" saya artikan sebagai metode yang mengubah gambar/frame menjadi vektor fitur dari model pralatih beku (CLIP, DINOv2, XCLIP), lalu mendeteksi lewat classifier atau statistik pada vektor itu. Metode berbasis artefak piksel/frekuensi tidak dibahas rinci.
- **Yang sudah diverifikasi (21 Nov 2026):** keberadaan, judul, penulis pertama, dan tahun semua paper di Bagian 6 (lewat OpenAlex/arXiv).
- **Yang BELUM diverifikasi:** isi klaim di Bagian 2–3 untuk paper selain ReStraV. Ringkasan isi diambil dari laporan otomatis alphaXiv (bukan PDF yang dibaca langsung), jadi **nomor halaman, persamaan, dan tabel belum ada**. Sebelum dikutip di Bab 1, buka PDF dan isi kolom "Lokasi" (ditandai `[cek PDF]`). Lokasi ReStraV, UnivFD, CLIP berasal dari `docs/02_riset_projector_2026-11-21.md` dan `referensi/daftar_referensi.md`.
- Angka performa di bawah adalah angka yang dilaporkan paper masing-masing pada protokol mereka sendiri. Jangan dibandingkan langsung antar paper.
- Ini tinjauan ringkas, bukan systematic review. Belum ada pencarian terstruktur (kata kunci, basis data, kriteria inklusi).

## 1. Kerangka: tiga pertanyaan yang membedakan metode

Semua metode di sini memakai pola yang sama: **frame → encoder beku → vektor**. Bedanya ada di dua hal:

1. **Vektor tunggal atau lintasan vektor?** Gambar: satu vektor per gambar. Video: deretan vektor per frame, sehingga bisa diukur "bentuk" lintasannya.
2. **Bagaimana vektor dipakai?** Dilatih sebagai input classifier (supervised), atau dihitung statistiknya tanpa pelatihan (training-free).

| Kelompok | Vektor | Cara pakai | Contoh |
|---|---|---|---|
| A. Fitur beku + classifier ringan (gambar) | 1 vektor/gambar | Linear/SVM dilatih | UnivFD, Cozzolino dkk. |
| B. Lintasan embedding, classifier dilatih (video) | N vektor/video | Fitur geometri → MLP | ReStraV |
| C. Lintasan embedding, tanpa pelatihan (video) | N vektor/video (CLS atau patch) | Statistik turunan ke-2 / roughness | D3, SPLIT |
| D. Probe model generatif (pembeda) | Respons velocity DiT | Representasi + consistency | TRACE |

## 2. Landasan: dari mana vektornya

| Referensi | Peran | Lokasi |
|---|---|---|
| Radford dkk. (2021). CLIP. ICML | Encoder gambar–teks kontrastif; sumber fitur UnivFD dan Cozzolino | Bag. 2.3 hlm. 4 (dari daftar_referensi) |
| Oquab dkk. (2023). DINOv2 | Encoder visual self-supervised; dipakai ReStraV, dan salah satu backbone di D3 dan SPLIT | `[cek PDF]` |
| Hénaff, Goris & Simoncelli (2019). Nature Neurosci. | Asal gagasan *perceptual straightening*: lintasan representasi video alami lebih lurus | `[cek PDF]` |

## 3. Metode per kelompok

### A. Fitur beku + classifier ringan (gambar)

**UnivFD (Ojha dkk., CVPR 2023).** Fitur CLIP beku, lalu satu lapisan linear dilatih dengan BCE (Bag. 4 hlm. 5, dari daftar_referensi). Gagasan kuncinya: classifier di ruang fitur CLIP yang tidak dilatih untuk deteksi generalisasinya lebih baik daripada classifier yang dilatih end-to-end. Relevansi ke skripsi: ini cetak biru "encoder beku + kepala kecil", yang juga dipakai projector P3.

**Cozzolino dkk. (CVPRW 2024).** Fitur CLIP ViT-L/14 (layer kedua dari terakhir) + SVM linear, dilatih dari sedikit contoh. Laporan alphaXiv: dengan 10–100 pasang real/fake sudah baik, optimum di 1.000–10.000 pasang; lebih tahan terhadap crop, resize, JPEG/WebP dibanding metode berbasis jejak low-level; skor CLIP-detector dan detektor low-level hampir ortogonal; penggabungan keduanya menaikkan performa. Pengujian degradasi: JPEG kualitas 65–100 diterapkan ke citra uji. `[cek PDF: nomor bagian dan tabel]`. Catatan penting untuk skripsi: ini bukti bahwa fitur semantik beku relatif tahan kompresi **pada gambar**; belum ada bukti setara untuk lintasan video dan H.264.

### B. Lintasan embedding dengan classifier dilatih

**ReStraV (Internò dkk., NeurIPS 2025)**, metode dasar skripsi. DINOv2 menghasilkan vektor per frame; lintasan antar-frame diukur dengan jarak (distance) dan sudut belok (curvature) (Pers. 1, Bag. 3 hlm. 4); DINOv2 CLS+patch (Bag. 4 hlm. 5); 21 fitur statistik lalu MLP 64→32 (Bag. 6 hlm. 6). Paper aslinya tidak menguji kompresi. Hubungan ke Hénaff dkk.: hipotesis bahwa video asli punya lintasan lebih lurus.

### C. Lintasan embedding tanpa pelatihan

**D3 (Zheng dkk., ICCV 2025).** Vektor per frame dari encoder beku (XCLIP-B/16, CLIP, DINOv2, bahkan MobileNet-v3). Turunan pertama = jarak L2 antar-frame; turunan kedua = selisih jarak berurutan; skor = simpangan baku turunan kedua. Real lebih "bergejolak" daripada AI. Laporan alphaXiv: L2 lebih baik daripada cosine; fitur orde-2 lebih baik daripada orde-1; ketahanan diuji pada blur (σ 0–4) dan JPEG (q 100–60) dengan penurunan kecil; tidak ada uji H.264/CRF yang saya temukan di ringkasan. `[cek PDF]`. **Relevansi:** D3 dan ReStraV sama-sama memakai statistik lintasan vektor. Perbedaannya, ReStraV melatih classifier di atas 21 fitur, D3 tidak melatih apa pun.

**SPLIT (Hyun & Kim, 2026).** Memakai patch token (bukan hanya CLS) dari encoder beku; Two-step Temporal Roughness (rasio-log jarak 1-langkah vs 2-langkah per patch) dan Local Spatial Motion Incoherence. Dievaluasi pada Fake Recall di FPR 0,1/1/5%, dengan ReStraV dan D3 sebagai baseline. Laporan alphaXiv: ReStraV adalah baseline terkuat yang dilatih tetapi hanya 33,9–45,5% recall pada FPR 0,1%; blur σ=2 paling merusak (SPLIT 74,45→55,42, ReStraV 54,73), JPEG lebih ringan. `[cek PDF]`. **Relevansi:** menunjukkan (i) ReStraV lemah pada rezim FPR rendah, (ii) protokol evaluasi alternatif selain AUROC. Catatan: baseline supervised di sana dilatih pada GenVideo-100K, bukan protokol skripsi.

### D. Pembeda: probe model generatif

**TRACE (Cao dkk., 2026).** Bukan embedding encoder visual, melainkan respons velocity dari video DiT beku (Wan2.2) di beberapa waktu-aliran, plus konsistensi antar-frame dan objective Real-Centered. Dievaluasi pada AIGVDBench; uji robustness JPEG, blur, dan resize ada di lampiran. `[cek PDF: bagian dan tabel]`. Relevansi: "konsistensi" di TRACE adalah selisih velocity antar-frame, **bukan** pasangan bersih–terkompresi seperti pada usulan projector. Ini yang membedakannya dari usulan skripsi.

## 4. Apa yang sudah ada untuk kompresi, dan celahnya

| Temuan literatur | Sumber | Implikasi |
|---|---|---|
| ReStraV asli tidak menguji kompresi | Internò dkk. | Celah awal |
| ReStraV dipakai sebagai baseline di 12 gangguan termasuk H.264 QP 23/30/38; rata-rata AP 87,91%, rentang 16,38 pp | MTOR (Wang dkk., 2026) | **Sudah ada** uji ReStraV pada H.264 (QP, bukan CRF). Jangan klaim "pertama" |
| Uji kompresi FFmpeg multi-CRF disebut sebagai pekerjaan lanjutan | Papougiannakis & Tous (2026) | Celah yang diakui penulis lain |
| Peringatan kebocoran panjang klip/codec; angka ReStraV 0,85→0,60 hanya dari cuplikan pencarian | Cakiroglu dkk. (2026) | **Verifikasi di PDF** sebelum dikutip |
| D3, SPLIT, Cozzolino diuji pada JPEG/blur, bukan H.264 multi-CRF (menurut ringkasan, belum dicek PDF) | lihat Bagian 3 | Kompresi video antar-frame belum diteliti sistematis untuk metode lintasan |

**Celah yang tersisa (dengan syarat verifikasi di atas):** analisis sistematis metode lintasan embedding (ReStraV, dan sebagai pembanding D3) pada H.264 multi-CRF dengan **kedua kelas dikompresi setara**, dan peningkatan yang dirancang untuk itu. Ini selaras dengan arah skripsi, tetapi kalimat "belum ada" harus dibuktikan dengan pencarian terstruktur, bukan klaim.

## 5. Yang perlu dikerjakan berikutnya

1. Konfirmasi ke dosen interpretasi "berbasis vektor" (Bagian 0).
2. Baca PDF: ReStraV (cek ulang), D3, SPLIT, Cozzolino, TRACE, MTOR, Cakiroglu; isi semua `[cek PDF]`.
3. Satu review sebagai referensi (arahan dosen 24 Sep): kandidat Raza dkk. (2026) di daftar_referensi, belum dibaca.
4. Pastikan rentang 10 tahun (2016–2026) terpenuhi; saat ini hampir semua rujukan 2019–2026.
5. Pencarian terstruktur untuk mendukung klaim celah (kata kunci, basis data, tanggal).
6. Ubah tabel Bagian 1 menjadi bagan Bab 1 (landasan teori → kerja terkait → celah).

## 6. Referensi baru dan statusnya

Semua diverifikasi eksistensinya 21 Nov 2026 (OpenAlex/arXiv, judul + penulis pertama + tahun cocok). Isi belum diverifikasi.

| Referensi | Pengenal |
|---|---|
| Oquab dkk. (2023). DINOv2: Learning Robust Visual Features without Supervision | arXiv 2304.07193 |
| Radford dkk. (2021). CLIP | arXiv 2103.00020 |
| Ojha, Li & Lee (2023). Towards Universal Fake Image Detectors… CVPR | arXiv 2302.10174 |
| Cozzolino, Poggi, Corvi, Nießner & Verdoliva (2024). Raising the Bar of AI-generated Image Detection with CLIP. CVPRW | doi 10.1109/cvprw63382.2024.00439 · arXiv 2312.00195 |
| Internò dkk. (2025). AI-Generated Video Detection via Perceptual Straightening. NeurIPS | arXiv 2507.00583 |
| Zheng dkk. (2025). D3: Training-Free AI-Generated Video Detection Using Second-Order Features. ICCV | doi 10.1109/iccv51701.2025.01194 · arXiv 2508.00701 |
| Hyun & Kim (2026). SPLIT | arXiv 2607.02886 (verifikasi lewat alphaXiv, belum lewat OpenAlex) |
| Cao dkk. (2026). TRACE | arXiv 2609.25775 (idem) |
