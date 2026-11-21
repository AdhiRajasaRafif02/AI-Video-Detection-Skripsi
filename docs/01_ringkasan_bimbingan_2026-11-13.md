# Ringkasan Bimbingan Skripsi – 13 November 2026

> Disalin dari Claude Doc "Ringkasan Bimbingan Skripsi – 13 November 2026". Sebagian isi sudah direvisi pada 21 Nov 2026: tidak ada bias pada projector, ada tambahan model P4, dan ukuran utama 384→384. Lihat `02_riset_projector_2026-11-21.md`.

Topik: deteksi video buatan AI pada video yang sudah dikompresi H.264, dengan ReStraV + DINOv2 sebagai dasar, lalu dikembangkan agar lebih tahan kompresi.

> Status: semua yang ada di sini masih **rancangan untuk didiskusikan**. Belum ada eksperimen yang dijalankan dan belum ada hasil.

## 1. Inti penelitian dalam 3 kalimat

1. Video di dunia nyata hampir selalu dikompresi (diunggah, dikirim, di-encode ulang), dan kompresi bisa merusak petunjuk yang dipakai detektor video AI.
2. Saya akan mengukur seberapa besar kompresi H.264 merusak ReStraV, dan **kenapa** itu terjadi.
3. Berdasarkan penyebab itu, saya merancang komponen tambahan kecil supaya ReStraV lebih tahan kompresi, lalu membuktikan apakah komponen itu benar-benar membantu.

## 2. Jawaban untuk kritik "ini hanya eksperimen"

Penelitian saya tidak berhenti di "mencoba ReStraV pada video terkompresi". Alurnya tiga tahap:

1. **Analisis mekanisme**: mencari tahu kenapa kompresi merusak geometri lintasan ReStraV.
2. **Merancang komponen baru**: sebuah *projector* yang dilatih agar geometri lintasan tetap konsisten sebelum dan sesudah kompresi.
3. **Pembuktian**: membandingkan komponen baru dengan ReStraV asli dan dengan cara perbaikan biasa (augmentasi kompresi).

Jadi yang saya buat adalah **komponen + cara melatihnya**, dan eksperimennya dipakai untuk membuktikan komponen itu.

## 3. Cara kerja ReStraV (versi sederhana)

1. Ambil 24 frame dari 2 detik video.
2. Setiap frame diubah DINOv2 menjadi vektor angka (embedding). Urutan vektor ini membentuk sebuah **lintasan**.
3. Dari lintasan dihitung dua hal: **distance** (seberapa jauh berpindah antar-frame) dan **curvature** (seberapa tajam lintasan berbelok).
4. Temuan paper: lintasan video asli cenderung **lebih lurus**, video AI **lebih berbelok-belok**.
5. Fitur ini (21 angka) masuk ke classifier kecil (MLP), lalu keluar prediksi real atau AI.

Hal penting dari paper ReStraV:

- Fitur paling penting adalah **rata-rata curvature**.
- Paper ReStraV **tidak pernah menguji kompresi**. Paper lanjutan (Papougiannakis & Tous, 2026) bahkan menyebut uji kompresi dengan beberapa nilai CRF sebagai pekerjaan yang belum dilakukan. Ini celah yang saya isi.
- Paper MTOR (2026) menunjukkan performa ReStraV naik-turun cukup jauh di bawah berbagai gangguan video (rentang sekitar 16 poin AP).

## 4. Masalahnya: kenapa kompresi diduga merusak ReStraV

**Analogi:** bayangkan memotret orang berjalan lurus dengan kamera yang goyang. Jalannya lurus, tapi di foto-foto posisinya tampak bergeser acak, sehingga lintasannya terlihat berbelok-belok.

Kompresi berperan seperti "kamera goyang". Ia membuat embedding setiap frame sedikit bergetar, dan curvature sangat peka terhadap getaran ini karena curvature mengukur **sudut** antar-langkah.

Secara matematis, kalau getarannya dominan, curvature akan tertarik ke sekitar **120°** untuk video asli maupun video AI. Akibatnya:

- Selisih curvature video asli dan AI **menyempit**.
- Video asli paling terdampak, karena awalnya lurus sehingga "naik" paling banyak. Video asli jadi terlihat mirip AI, dan detektor lebih sering salah menuduh video asli sebagai AI.
- Video dengan gerakan kecil paling terdampak, karena langkahnya pendek dibanding getarannya.

Simulasi sederhana (lintasan buatan, **bukan data asli**, hanya ilustrasi; kode di `eksperimen/simulasi_noise_curvature/`):

| Besar getaran | Curvature "asli" | Curvature "AI" | Selisih |
| --- | --- | --- | --- |
| Tidak ada | 30° | 70° | 40° |
| Kecil | 57° | 82° | 25° |
| Sedang | 92° | 102° | 10° |
| Besar | 110° | 114° | 4° |

Sumber getaran tambahan yang khas H.264: kualitas frame tidak sama (frame I lebih bagus, frame B lebih jelek), sehingga kualitas "naik-turun" berulang di sepanjang video.

## 5. Usulan perbaikan: projector konsisten-kompresi

**Ide:** menambahkan satu lapisan kecil yang bisa dilatih (*projector*) di antara DINOv2 dan perhitungan distance/curvature.

Frame → DINOv2 (tidak dilatih ulang) → **projector** → hitung distance & curvature → classifier → real / AI

**Analogi:** projector seperti "kacamata penyaring goyangan". Ia belajar membuang bagian embedding yang paling diganggu kompresi, dan mempertahankan bagian yang membedakan video asli dan AI.

**Cara melatihnya:** satu video dipakai dua kali, versi asli dan versi terkompresinya. Projector dilatih dengan dua tujuan sekaligus:

1. Tetap bisa menebak real/AI dengan benar (classification loss).
2. Geometri lintasan versi asli dan versi terkompresi harus **mirip** (consistency loss).

Rumus: L = L_klasifikasi(asli) + L_klasifikasi(kompresi) + λ × selisih geometri(asli, kompresi)

**Kenapa projector dipasang sebelum geometri, bukan sesudahnya?** Kalau curvature sudah rusak saat dihitung, apa pun yang dipasang sesudahnya tidak bisa memperbaikinya. Jadi gangguannya harus disaring sebelum curvature dihitung. (Ini revisi dari usulan sebelumnya, yaitu adapter yang dipasang setelah 21 fitur.)

Tambahan sederhana tanpa training (juga sebagai pembanding):

- **Curvature multi-skala**: hitung belokan dengan jarak 2–3 frame, supaya langkahnya lebih panjang dan getaran jadi relatif kecil.
- **Statistik yang lebih tahan**: median/persentil sebagai pengganti min/max yang sangat peka terhadap lonjakan sesaat.

## 6. Hipotesis

Hipotesis adalah dugaan yang **bisa terbukti salah** lewat eksperimen.

- **H1 (dampak kompresi):** semakin tinggi CRF (kompresi makin berat), AUROC ReStraV turun dan selisih curvature video asli–AI menyempit, dengan kenaikan curvature terbesar pada video asli. H1 ditolak kalau performa dan selisih curvature tetap stabil.
- **H2 (perbaikan):** pada kompresi berat, ReStraV + projector konsisten-kompresi menghasilkan AUROC lebih tinggi daripada ReStraV dengan augmentasi kompresi biasa, tanpa menurunkan performa pada video asli secara berarti.
- **H3 (generalisasi):** keunggulan itu tetap ada pada tingkat kompresi yang tidak dipakai saat training (misalnya CRF 40).

Kalimat untuk dosen: *"Saya menduga kompresi menambah getaran pada embedding frame sehingga lintasan video asli jadi berbelok seperti video AI. Karena itu saya merancang projector yang dilatih agar geometri lintasan tetap konsisten antara video asli dan versi terkompresinya."*

## 7. Rancangan eksperimen

**Model yang dibandingkan** (versi 13 Nov; versi terbaru menambah P4, lihat dokumen 02):

| Kode | Isi | Pertanyaan yang dijawab |
| --- | --- | --- |
| B0 | ReStraV asli, dilatih video asli saja | Seberapa parah kerusakan akibat kompresi? |
| B1 | B0 + augmentasi kompresi saat training | Apakah cara perbaikan biasa sudah cukup? |
| P1 | B1 + curvature multi-skala + statistik tahan | Apakah perbaikan tanpa training sudah membantu? |
| P2 | Projector tanpa consistency loss | Apakah tambahan lapisan saja sudah membantu? |
| P3 | Projector + consistency loss | **Usulan utama** |

Kontribusi dinilai dari **P3 dibanding B1 dan P2**, bukan hanya dibanding B0.

**Data dan kompresi (rancangan):**

- Kandidat dataset: ComGenVid (1.700 video asli, 1.700 Sora, 1.700 Veo3). Kelayakannya masih perlu dicek.
- Video asli dan video AI dikompresi dengan **pengaturan yang sama**. Tingkat kompresi adalah kondisi uji, bukan kelas ketiga.
- Training: video asli + CRF 10, 20, 30. Testing: asli + CRF 0, 10, 20, 30, 40. CRF 40 sengaja tidak dipakai saat training.
- Pembagian data per video (70/15/15). Semua versi kompresi dari satu video berada di bagian yang sama supaya tidak bocor.

**Ukuran keberhasilan:** AUROC, F1, akurasi, false positive pada video asli, selisih curvature asli–AI per CRF, dan perubahan fitur sebelum/sesudah kompresi.

**Komputasi ringan:** DINOv2 cukup dijalankan sekali per video per tingkat kompresi, lalu hasilnya disimpan (sekitar 37 KB per video). Semua model B0–P3 dilatih ulang dari data simpanan itu dalam hitungan menit.

## 8. Kenapa usulan ini layak (dukungan literatur)

- **Celahnya nyata:** ReStraV belum pernah diuji pada beberapa tingkat CRF; paper lanjutannya menyebut ini sebagai pekerjaan yang belum dilakukan (Papougiannakis & Tous, 2026).
- **Konsistensi antara versi bersih dan versi terganggu sudah terbukti meningkatkan ketahanan** di bidang lain: Stability Training (Zheng dkk., CVPR 2016) dan AugMix (Hendrycks dkk., ICLR 2020).
- **Di deteksi deepfake terkompresi**, pendekatan sejenis juga dipakai: ADD (AAAI 2022), QAD (ICCV 2023), ODDN (AAAI 2025).
- **Hubungan kelurusan lintasan dan ketahanan:** representasi yang dilatih supaya lurus terbukti lebih tahan noise (Niu dkk., NeurIPS 2024), dan model yang tahan gangguan cenderung punya lintasan lebih lurus (Harrington dkk., 2022/2023).
- **Pembanding augmentasi kompresi** juga punya dasar: Wang dkk. (CVPR 2020) dan MISLnet (Vahdati dkk., CVPRW 2024).

## 9. Batasan yang disampaikan dengan jujur

- **Tidak** mengklaim sebagai yang pertama menguji ReStraV pada kompresi (MTOR sudah pernah memakai ReStraV sebagai pembanding di bawah kompresi), dan **tidak** mengklaim menemukan consistency learning. Yang baru adalah penerapan konsistensi berpasangan pada **geometri lintasan ReStraV** untuk kompresi H.264. Sejauh penelusuran, ini belum ada.
- Hasilnya bisa saja tidak sesuai hipotesis. Kalau P3 tidak lebih baik dari B1, itu tetap hasil yang sah dan akan dilaporkan.
- Dataset hanya punya dua generator (Sora dan Veo3), jadi hasilnya belum tentu berlaku untuk semua generator video AI.
- Simulasi di bagian 4 hanya ilustrasi, bukan hasil eksperimen.

## 10. Pertanyaan untuk dosen pembimbing

**Arah dan kontribusi**

1. Apakah alur "analisis mekanisme → projector konsisten-kompresi → pembuktian" sudah cukup sebagai kontribusi skripsi, bukan "hanya eksperimen"?
2. Apakah hipotesis H1–H3 sudah tepat, atau perlu disederhanakan/ditambah?
3. Apakah cukup fokus ke ReStraV saja, atau dosen ingin ada pembanding metode/encoder lain (misalnya CLIP atau DINOv3)?

**Ruang lingkup**

4. Apakah kelima model perlu semua, atau boleh dikurangi kalau waktu terbatas? (Minimal: B0, B1, P2, P3.)
5. Apakah cukup memakai kompresi H.264 saja, atau perlu juga H.265 / simulasi media sosial?
6. Tingkat kompresi CRF 0, 10, 20, 30, 40 apakah sudah sesuai?

**Data dan teknis**

7. Apakah dataset ComGenVid (1.700 asli, 3.400 AI) bisa diterima, atau ada dataset lain yang disarankan?
8. Bolehkah memakai CLS token saja (lebih ringan) walaupun paper asli memakai CLS + patch, asalkan dijelaskan sebagai varian?
9. Apakah ada akses GPU/laboratorium?

**Penulisan dan administrasi**

10. Apakah judul "Analisis dan Peningkatan Ketahanan Deteksi Video AI Berbasis ReStraV terhadap Degradasi Kompresi H.264" sudah sesuai?
11. Apa target berikutnya: menulis Bab 1–3 dulu, atau uji coba kecil dulu? Kapan tenggatnya?

## 11. Jawaban siap kalau ditanya

**"Apa bedanya dengan augmentasi kompresi biasa?"** Augmentasi hanya memperlihatkan video terkompresi ke classifier. Usulan ini secara eksplisit memaksa geometri lintasan video yang **sama** tetap mirip sebelum dan sesudah kompresi, dan menyaringnya sebelum curvature dihitung. Itu sebabnya augmentasi (B1) dijadikan pembanding wajib.

**"Kenapa tidak melatih ulang DINOv2 saja?"** Terlalu berat untuk skripsi dan bisa merusak kemampuan DINOv2 yang menjadi dasar ReStraV. Projector kecil jauh lebih ringan dan DINOv2 tetap utuh.

**"Kalau ternyata kompresi tidak merusak ReStraV?"** H1 ditolak, dan itu temuan yang sah. Fokus kemudian bergeser ke kondisi mana yang masih merusak (misalnya CRF sangat tinggi atau video dengan gerakan kecil), dan seberapa stabil fitur geometrinya.

**"Apa itu CRF?"** Pengaturan kualitas pada encoder H.264. Semakin besar angkanya, kompresinya makin kuat dan kualitasnya makin turun. CRF bukan persen dan bukan bitrate.

**"Apa langkah pertama?"** Uji coba kecil (sekitar 100 video × CRF 0/20/40) untuk mengecek H1: apakah selisih curvature asli–AI benar-benar menyempit.
