# Hasil eksperimen ReStraV (lokal)

Setiap run dicatat di sini: tanggal, data, protokol, angka, dan batasannya. File output (fitur, model, prediksi) ada di `D:\Skripsi\data\` dan tidak di-commit.

---

## Run 1 — Reproduksi apa adanya pada ComGenVid asli (22 Nov 2026)

**Status: hasil nyata, tetapi protokolnya belum bersih. Belum boleh dipakai sebagai angka B0 final.**

### Data dan ekstraksi
- ComGenVid, video asli tanpa re-encode: 5.100 video (MSVD 1.700 real, Sora 1.700 fake, VEO3 1.700 fake).
- `ekstrak_fitur.py`: 5.100 berhasil, 0 gagal, 0 NaN, 1,20 s/video (total 1 jam 42 menit di GTX 1650).
- Output: `D:\Skripsi\data\fitur\comgenvid_asli.h5` (21 fitur + CLS per frame).

### Protokol (`train_lokal.py`, logika sama dengan `train.py` upstream)
- Balancing: 1.700 real + 1.700 fake diambil acak (seed 42) dari 3.400 fake, sehingga 1.700 video fake tidak dipakai.
- Split 50/50 stratified: train 1.700, test 1.700 (850 real, 424 Sora, 426 VEO3).
- MLP 21→64→32→1, 10 epoch, satu seed (42).
- Normalisasi mean/std dihitung dari seluruh data sebelum split (kebocoran kecil, mengikuti upstream).
- Threshold τ* = 0,480 dipilih dari F1 di train (F1 train = 0,805).

### Angka (test, n = 1.700)

| Metrik | Nilai |
|---|---|
| Accuracy | 0,758 |
| Precision (kelas real) | 0,741 |
| Recall (kelas real) | 0,793 |
| F1 | 0,766 |
| AUROC | 0,834 |

Confusion matrix (baris = label benar [fake, real], kolom = prediksi [fake, real]): `[[614, 236], [176, 674]]`.

Per sumber (dihitung dari `test_predictions_all.csv`):

| Sumber | n test | Accuracy pada τ* | Rata-rata P(real) | AUROC vs MSVD |
|---|---|---|---|---|
| MSVD (real) | 850 | 0,793 | 0,676 | — |
| Sora | 424 | 0,715 | 0,348 | 0,820 |
| VEO3 | 426 | 0,730 | 0,301 | 0,847 |

Log lengkap: `D:\Skripsi\data\hasil\comgenvid_asli_train.log`.

### Cara membaca hasil ini
- **Fakta:** pada data dan protokol ini, ReStraV mencapai AUROC 0,834. Ini satu run dengan satu seed, jadi belum ada ukuran variasi.
- **Belum dibandingkan** dengan angka di paper ReStraV. Datanya berbeda (paper memakai benchmark lain), jadi selisihnya belum bisa ditafsirkan.
- **Interpretasi asisten (belum diuji):** ComGenVid punya perbedaan sistematis antar kelas (bitrate, FPS, resolusi, watermark Sora). Kalau 21 fitur memanfaatkan jalan pintas itu, AUROC seharusnya mendekati 1. Angka 0,83 tidak mendukung dugaan jalan pintas yang kuat, tetapi juga tidak menyingkirkannya. Ini perlu diuji dengan re-encode seragam.
- Kebocoran normalisasi kemungkinan kecil pengaruhnya, tetapi tetap harus dihilangkan di protokol B0.

### Yang perlu dilakukan sebelum angka ini dipakai sebagai B0
1. Normalisasi di-fit pada train saja. → dikerjakan di Run 2
2. Split tetap (disimpan ke file), dipakai sama untuk semua CRF dan semua model. → Run 2
3. Beberapa seed (misalnya 5), laporkan rata-rata ± simpangan baku. → Run 2
4. Pertimbangkan memakai semua 3.400 video fake dengan class weight, bukan membuang separuhnya. → Run 2
5. Re-encode seragam (codec, FPS, resolusi yang sama untuk semua kelas) sebagai kontrol jalan pintas. → **belum**

---

## Run 2 — B0 protokol bersih pada ComGenVid asli (22 Nov 2026)

**Status: hasil nyata. Ini kandidat angka B0 untuk kondisi video asli.** Kontrol jalan pintas codec (re-encode seragam) belum dilakukan, dan protokolnya belum dikonfirmasi dosen.

### Protokol (`train_b0.py`, split dari `buat_split.py`)

| Aspek | Upstream `train.py` (Run 1) | B0 (Run 2) | Alasan |
|---|---|---|---|
| Fitur, MLP, loss, optimizer, epoch | 21 fitur, 21-64-32-1, BCEWithLogits, Adam 1e-3, batch 32, 10 epoch | **sama** | Supaya B0 tetap ReStraV |
| Split | acak 50/50, tanpa kelompok | **tetap 60/20/20 per sumber, dikelompokkan per YouTube ID (MSVD)**, disimpan di `split/comgenvid_split_v1.csv` | MSVD punya 1.306 video YouTube untuk 1.700 klip (satu video sampai 21 klip). Split sama dipakai untuk semua CRF |
| Normalisasi | mean/std dari seluruh data | **dari train saja** | Menghilangkan kebocoran |
| Ketidakseimbangan kelas | buang 1.700 fake secara acak | **semua data dipakai, bobot per sampel** (total bobot real = fake) | Tidak membuang data; prior tetap 50/50 |
| Threshold τ | maks F1 di train | **maks balanced accuracy di val** | Tidak memilih di data latih; F1 bergantung proporsi kelas |
| Seed | 1 (seed torch tidak di-set) | **5 seed (0–4)**, split tetap | Mengukur variasi |

Ukuran: train 3.060 (1.020 per sumber), val 1.020, test 1.020 (340 MSVD, 340 Sora, 340 VEO3).

### Uji kebenaran (`uji_b0.py`): semua lulus
- Tidak ada grup (YouTube ID) yang muncul di lebih dari satu split.
- Setelah normalisasi, mean train ≈ 0 dan std train ≈ 1, sedangkan mean test ≠ 0 (statistik test tidak dipakai).
- Total bobot kelas real = fake (1.530 vs 1.530).
- AUROC sklearn = AUROC yang dihitung manual dengan rumus Mann-Whitney (termasuk nilai seri).
- Seed sama menghasilkan AUROC sama (0,8381 vs 0,8381).
- Kalau label train diacak, AUROC test = 0,48 (≈ 0,5). Artinya tidak ada jalur bocor label di pipeline.
- Info: 12 video punya jarak antar-frame = 0 (frame kembar). 10 MSVD (5 di antaranya ber-FPS ≤ 10,45, sehingga 24 frame dalam 2 detik pasti ada yang berulang; 2 berdurasi < 2 detik) dan 2 Sora. Jumlahnya kecil, tetapi frame kembar menghasilkan sudut yang tidak terdefinisi dengan baik. Perlu dicatat sebagai batasan ReStraV pada video FPS rendah.

### Angka (test, n = 1.020, 5 seed, rata-rata ± simpangan baku)

| Metrik | B0 (split berkelompok) |
|---|---|
| **AUROC** | **0,834 ± 0,004** |
| AUROC MSVD vs Sora | 0,824 ± 0,005 |
| AUROC MSVD vs VEO3 | 0,845 ± 0,006 |
| Balanced accuracy (τ dari val) | 0,748 ± 0,010 |
| Accuracy | 0,763 ± 0,009 |
| F1 kelas real | 0,663 ± 0,013 |
| Real terdeteksi real (TPR) | 0,703 ± 0,063 |
| Sora terdeteksi fake | 0,782 ± 0,051 |
| VEO3 terdeteksi fake | 0,804 ± 0,038 |
| τ terpilih | 0,56 ± 0,07 |

AUROC per seed: 0,8366; 0,8281; 0,8344; 0,8377; 0,8352.

### Diagnostik: split acak tanpa kelompok (`split/comgenvid_split_acak_v1.csv`)
Protokol sama, hanya splitnya acak per klip. Di split ini 26,5% klip MSVD test punya klip "saudara" (YouTube ID sama) di train.

| | Split berkelompok | Split acak |
|---|---|---|
| AUROC | 0,834 ± 0,004 | 0,842 ± 0,002 |
| Balanced accuracy | 0,748 ± 0,010 | 0,770 ± 0,006 |

Selisihnya kecil (±0,007 AUROC). Isi test kedua split berbeda, jadi perbandingan ini kasar. **Interpretasi asisten:** kebocoran klip saudara MSVD hanya sedikit menaikkan angka. Split berkelompok tetap dipakai karena lebih jujur.

### Cara membaca
- **Fakta:** pada ComGenVid asli dengan protokol bersih, ReStraV mencapai AUROC 0,834 ± 0,004. Ini sangat dekat dengan Run 1 (0,834), jadi kebocoran normalisasi dan pembuangan data di upstream tidak mengubah AUROC secara berarti pada data ini.
- Variasi antar seed kecil untuk AUROC (±0,004), tetapi τ berubah cukup besar (0,45–0,61). Metrik yang memakai threshold (accuracy, TPR) lebih tidak stabil. **Usulan asisten:** AUROC dijadikan metrik utama.
- VEO3 sedikit lebih mudah dibedakan dari MSVD daripada Sora.
- Jalan pintas codec/resolusi/FPS **belum** dikontrol. Itu langkah berikutnya (re-encode seragam).

### Reproduksi
```powershell
cd eksperimen\restrav_reproduksi
..\..\.venv\Scripts\python.exe buat_split.py
..\..\.venv\Scripts\python.exe uji_b0.py D:\Skripsi\data\fitur\comgenvid_asli.h5
..\..\.venv\Scripts\python.exe train_b0.py --h5 D:\Skripsi\data\fitur\comgenvid_asli.h5 --out D:\Skripsi\data\hasil\b0_asli
```
Output (`metrik_per_seed.csv`, `ringkasan.csv`, `prediksi_test.csv`, model per seed) ada di `D:\Skripsi\data\hasil\b0_asli\`.
