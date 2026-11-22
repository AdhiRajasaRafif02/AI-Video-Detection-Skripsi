# Konteks dan progres skripsi: deteksi video AI pada kondisi kompresi

**Dokumen handoff untuk Claude atau asisten riset lainnya**  
**Disusun:** 10 November 2026 · **Diperbarui:** 13 November 2026 (Bagian 8A) dan 21 November 2026 (Bagian 8B: masukan dosen tentang projector dan revisi desain)  
**Mahasiswa:** Adhi Rajasa Rafif  
**Bahasa kerja:** Bahasa Indonesia; istilah teknis bahasa Inggris boleh digunakan dan dijelaskan dengan bahasa sederhana.

> **Cara membaca status dalam dokumen ini:** “dibahas/diminta mahasiswa” adalah sesuatu yang tercatat dalam percakapan. “Arahan dosen” adalah arahan yang diceritakan mahasiswa, bukan notulen langsung dari dosen. “Usulan” berarti rekomendasi asisten yang belum terbukti telah disetujui dosen atau ditetapkan mahasiswa. Jangan menyamakan usulan terakhir dengan hasil eksperimen atau keputusan final.

## 1. Ringkasan eksekutif

Skripsi ini membahas **deteksi video hasil generative AI ketika video mengalami degradasi akibat kompresi**. Pertanyaan intinya: apakah sebuah detektor masih dapat membedakan video nyata dari video hasil generative AI ketika **kedua kelas** sama-sama mengalami kompresi, khususnya H.264 dengan beberapa tingkat CRF?

Perjalanan diskusi mengarah ke **ReStraV** sebagai metode utama. ReStraV memakai visual encoder pretrained, terutama **DINOv2**, untuk menghasilkan embedding frame, lalu menganalisis **stepwise distance** dan **temporal curvature** dari lintasan embedding antar-frame. Fitur geometri tersebut digunakan oleh classifier untuk memprediksi video real atau AI-generated. Kompresi berpotensi mengubah embedding, geometri, dan akhirnya performa klasifikasi; arah dan besarnya dampak harus dibuktikan melalui eksperimen.

Mahasiswa sebelumnya mempertimbangkan perbandingan **ReStraV vs MISLnet**, kemudian meminta alasan untuk mempersempit fokus menjadi **ReStraV saja**. Dosen mempertanyakan apakah ReStraV memang selalu menggunakan DINOv2; jika demikian, menggabungkan ReStraV dan DINOv2 saja belum menunjukkan solusi baru, dan menurut mahasiswa dosen menilai pendekatan “hanya eksperimen, bukan membuat sesuatu”. Dosen juga menanyakan **“hipotesisnya apa”**, dan mahasiswa belum punya jawaban yang mantap.

Yang sudah pasti menurut mahasiswa (13 November 2026): (1) melakukan eksperimen ReStraV + DINOv2 pada video terkompresi, dan (2) mengembangkan peningkatan metode tersebut lalu bereksperimen kembali untuk membuktikan hasilnya lebih baik. Bentuk peningkatannya belum diputuskan.

**Usulan terbaru (13 November 2026, lihat Bagian 8A):** *compression-consistent trajectory projector* — projector kecil yang dapat dilatih, dipasang **di antara DINOv2 dan perhitungan geometri**, dilatih dengan classification loss plus **paired consistency loss** antara geometri lintasan video source dan versi H.264 dari video yang sama. Ini **merevisi** usulan 29 Oktober (adapter setelah 21 fitur) karena adapter setelah fitur tidak dapat memperbaiki curvature yang sudah rusak oleh noise. Usulan ini disertai hipotesis H1–H3 dan susunan model B0, B1, P1, P2, P3. **Ini baru rekomendasi riset; belum disetujui dosen dan belum ada hasil eksperimen.**

Kondisi yang terakhir diusulkan untuk pengujian adalah **source, H.264 CRF 0, 10, 20, 30, 40**; pelatihan model perbaikan diusulkan menggunakan source dan CRF 10, 20, 30, dengan CRF 40 serta opsional 25 dan 35 sebagai kondisi tingkat kompresi yang tidak dilihat saat training. Ini **rancangan, bukan protokol yang sudah disahkan atau dijalankan**.

## 2. Apa yang diketahui, dan batas dokumen ini

Dokumen ini disusun dari percakapan proyek ChatGPT **“Skripsi”**, beberapa percakapan skripsi terkait di luar proyek, serta pemeriksaan langsung terhadap beberapa sumber primer. Percakapan skripsi terbaru yang ditemukan dan dibaca saat penyusunan awal adalah **“Studi Literatur 5” pada 29 Oktober 2026**. Pembaruan 13 November 2026 berasal dari sesi Claude di Project “Skripsi”, setelah mahasiswa mengunggah PDF paper ReStraV (arXiv 2507.00583v4, 27 halaman, termasuk lampiran) yang dibaca penuh. Folder kerja lokal skripsi yang tersedia saat penyusunan tidak berisi naskah atau hasil eksperimen untuk diverifikasi.

Beberapa lampiran dan dokumen dari percakapan lama pernah dibahas atau dibuat, tetapi isinya tidak semuanya tersedia langsung dalam konteks saat ini. Oleh karena itu:

- Dokumen ini berusaha selengkap mungkin **berdasarkan sumber yang dapat diakses**, bukan mengklaim mencakup setiap diskusi yang mungkin ada di luar sumber tersebut.
- Tanggal pada kronologi mengacu pada tanggal percakapan yang dapat ditemukan; tanggal bimbingan langsung tidak selalu tercatat.
- Tidak ditemukan bukti yang cukup untuk menyatakan judul akhir, dataset final, jumlah sampel, spesifikasi GPU, hasil pengujian, model yang sudah dilatih, maupun persetujuan final dosen.
- Nama label file/percakapan dan rujukan tercantum agar mudah dilacak. Selalu minta dokumen asli kepada mahasiswa jika memerlukan detail yang tidak tercantum.

## 3. Latar belakang dan konteks penelitian

Kemampuan generative AI membuat video semakin realistis, sehingga penonton bisa kesulitan membedakan video nyata dari video sintetis. Dalam penggunaan nyata, video sering mengalami encoding, kompresi ulang, atau pemrosesan oleh platform. Perubahan tersebut berpotensi menghilangkan atau mengubah petunjuk visual maupun temporal yang digunakan detektor.

Penelitian ini berada dalam ranah **computer vision, deep learning, video forensics, synthetic-media detection, dan robustness**. Ini bukan penelitian NLP; DINOv2 menghasilkan representasi visual numerik, bukan mengubah objek menjadi kata-kata secara eksplisit.

Objek yang terutama dimaksud adalah **video yang dihasilkan AI secara generatif**, bukan secara otomatis seluruh jenis deepfake wajah atau video manipulasi parsial. Literatur deepfake masih berguna sebagai dasar, tetapi perbedaan objek penelitian harus dijelaskan dan hasilnya tidak boleh langsung digeneralisasi.

**Klasifikasi tetap dua kelas:**

1. Video nyata / natural / real.
2. Video AI-generated / synthetic / fake.

Tingkat kompresi adalah **kondisi perlakuan/pengujian**, bukan kelas ketiga. Pada setiap kondisi yang dibandingkan, video real dan AI sebaiknya diberi perlakuan kompresi yang setara. Bila hanya kelas AI yang dikompresi, model bisa membedakan kelas berdasarkan jejak kompresi, bukan karakteristik generatif.

Contoh struktur logis:

```text
Real source          vs AI source
Real H.264 CRF 10    vs AI H.264 CRF 10
Real H.264 CRF 20    vs AI H.264 CRF 20
Real H.264 CRF 30    vs AI H.264 CRF 30
Real H.264 CRF 40    vs AI H.264 CRF 40
```

Tidak perlu mengklaim bahwa kompresi pasti menurunkan performa. Stabil, meningkat, atau turun sama-sama mungkin dan harus dilaporkan sesuai hasil.

## 4. Kronologi perkembangan dan keputusan yang terlihat

| Waktu/percakapan | Perkembangan | Status |
|---|---|---|
| Juli 2026, pencarian topik ML | Mahasiswa menjajaki topik computer vision dan tertarik pada deteksi gambar AI yang telah diunggah/diedit di media sosial. | Tahap eksplorasi awal; bukan fokus final. |
| Agustus 2026, persiapan bimbingan | Fokus beralih ke deteksi video AI pada degradasi kompresi. Dibahas latar belakang, kandidat judul, eksperimen beberapa tingkat kompresi, dataset, metrik, dan bahan bimbingan. | Kerangka awal; sebagian angka CRF kala itu hanya contoh. |
| 3 September 2026, “Persiapan Skripsi” | ReStraV dibahas sebagai kandidat landasan dan tersedia implementasi resmi. Mahasiswa memperjelas bahwa metode ini computer vision dan memakai DINOv2, geometri trajectory, serta classifier. | Landasan yang dipelajari, belum bukti implementasi skripsi. |
| 15 September 2026, “Menulis Judul Seminar Skripsi” | Disarankan judul Indonesia tentang pengaruh degradasi kompresi terhadap deteksi video AI. | Usulan untuk pengisian seminar; tidak terbukti sebagai judul resmi final. |
| 24 September 2026, studi literatur 1–4 | Dibahas ReStraV, AIGVDBench, sebuah review deepfake, dan VideoVeritas/Qwen. Mahasiswa menyampaikan arahan dosen agar mencari pendekatan lebih ringan, paper terkait dalam rentang 10 tahun, dan review; dosen menyoroti biaya jika semua frame diubah menjadi vektor. | Arahan dosen menurut penuturan mahasiswa; literatur dibahas untuk memperjelas pilihan. |
| 1 Oktober 2026, “Arah Riset Deteksi Video AI” | Mahasiswa menanyakan arah antara perbandingan metode atau peningkatan metode. Asisten merekomendasikan analisis dampak kompresi lalu mencoba perbaikan ringan. Dijelaskan bahwa real dan AI sama-sama harus dikompresi pada kondisi setara. | Arah yang dibahas dan disarankan. |
| 8 Oktober 2026, “Cari Paper Deteksi Video AI Kompresi” | Dibahas ReStraV/DINOv2 + distance/curvature sebagai fokus, sementara MISLnet sempat diusulkan sebagai pembanding. Dibahas bahwa hasil yang lebih buruk dari pembanding tetap sah jika eksperimen adil. | Rencana historis; fokus kemudian dipersempit. |
| 29 Oktober 2026, “Studi Literatur 5” | Mahasiswa meminta alasan perubahan **ReStraV vs MISLnet → ReStraV saja**. Lalu menyampaikan kritik dosen terkait ketergantungan pada DINOv2 dan kontribusi penelitian. Ditelaah alternatif encoder; sesudah itu mahasiswa meminta pertimbangan antara solusi degradasi dan perbandingan encoder. | Jelas merupakan pergeseran fokus dalam diskusi; rincian persetujuan formal belum diketahui. |
| 29 Oktober 2026, akhir “Studi Literatur 5” | Mahasiswa meminta riset mendalam strategi perbaikan yang belum ada pada paper yang dilampirkan. Asisten merekomendasikan paired geometry consistency learning dengan compression-aware training sebagai pembanding. | Usulan; kemudian direvisi pada 13 November. |
| 13 November 2026, sesi Claude (Project “Skripsi”) | Mahasiswa menyatakan dilema: dikritik “hanya eksperimen”, arah metode belum pasti, belum bisa menjawab “hipotesisnya apa”. Yang pasti: eksperimen ReStraV + DINOv2 pada video terkompresi, lalu peningkatan metode dan eksperimen ulang. Paper ReStraV dibaca penuh; dilakukan penelusuran literatur. Asisten mengusulkan **projector konsisten-kompresi sebelum geometri**, hipotesis H1–H3, dan model B0/B1/P1/P2/P3 (Bagian 8A). Mahasiswa meminta usulan ini dicatat di dokumen. | **Usulan terbaru**, belum dibawa ke dosen, belum diimplementasikan. |

### Mengapa beralih dari ReStraV vs MISLnet ke ReStraV saja?

Alasan yang pernah disiapkan untuk bimbingan:

- Pertanyaan utama menjadi lebih spesifik: bagaimana kompresi memengaruhi **geometri temporal** dan kemampuan deteksi pada ReStraV.
- ReStraV memiliki fitur distance dan curvature yang langsung dapat dianalisis; MISLnet lebih menekankan jejak forensik tingkat citra/piksel dan bukan secara bawaan mengukur geometri temporal yang sama.
- Dengan fokus ReStraV, analisis dapat diperdalam melalui distance-only, curvature-only, dan gabungan, serta pengujian strategi mitigasi dalam kerangka yang sama.
- Ini adalah keputusan pembatasan lingkup, **bukan klaim bahwa MISLnet tidak cocok atau bahwa ReStraV pasti lebih unggul**. MISLnet tetap merupakan referensi penting tentang deteksi video sintetis dan kompresi.

## 5. Arahan dan pertanyaan dari dosen yang dilaporkan mahasiswa

1. **Cari metode yang lebih ringan.** Kekhawatirannya: apabila semua frame diproses menjadi embedding/vektor, biaya komputasi dapat besar. Penggunaan pretrained model dan classifier kecil membantu, tetapi ekstraksi frame dengan DINOv2 tetap perlu dihitung biayanya.
2. **Cari 2–3 paper terkait dalam rentang 10 tahun terakhir**, termasuk review, menggunakan kata kunci misalnya melalui Scite.ai. Ini dicatat dari percakapan 24 September 2026.
3. **Apakah ReStraV selalu berpasangan dengan DINOv2?** Jika ReStraV asli memang sudah memakai DINOv2, hanya memakai pasangan itu bukan otomatis pengembangan solusi baru.
4. Pembimbing menginginkan kontribusi yang lebih jelas daripada sekadar menggabungkan hal yang sudah dipasangkan. Menurut mahasiswa, dosen menilai menganalisis ReStraV + DINOv2 pada video terkompresi saja sebagai “hanya eksperimen, bukan membuat sesuatu”. Pilihan yang sempat dibahas meliputi **mencari solusi ketika kompresi memperburuk hasil** atau **membandingkan encoder/metode** untuk menguji peran DINOv2.
5. **“Ini hipotesisnya apa?”** Dosen meminta hipotesis yang jelas. Usulan rumusan hipotesis ada di Bagian 8A.4.

Ini bukan bukti bahwa dosen telah menyetujui suatu arsitektur, dataset, judul, ataupun rencana eksperimen tertentu. Langkah berikutnya adalah membawa rancangan konkret untuk didiskusikan.

## 6. ReStraV dan DINOv2: penjelasan yang harus konsisten

### 6.1 Gambaran proses

```text
Video real / AI
    ↓
Pilih potongan waktu dan sejumlah frame
    ↓
Encoder visual pretrained (paper utama: DINOv2 ViT-S/14, frozen)
    ↓
Embedding z₁, z₂, ..., zT untuk frame-frame berurutan
    ↓
Hitung distance dan curvature trajectory
    ↓
Susun sinyal dan statistik geometri per video
    ↓
Classifier ringan, misalnya MLP
    ↓
Prediksi real atau AI-generated
```

**DINOv2** adalah feature extractor yang mengubah frame menjadi representasi angka. **ReStraV** adalah keseluruhan pendekatan yang menggunakan geometri lintasan representasi untuk deteksi; DINOv2 tidak secara otomatis memberikan label real atau AI.

Intuisi perceptual straightening dalam paper: pada ruang representasi yang dipilih, lintasan video natural dalam eksperimen mereka cenderung lebih lurus dibanding beberapa video generatif. Ini merupakan temuan/hipotesis yang bergantung pada encoder dan data, bukan aturan universal bahwa semua video AI pasti melengkung atau semua video real pasti lurus. Lihat [paper asli ReStraV](https://arxiv.org/html/2507.00583).

Misalkan `zᵢ` adalah embedding frame ke-i:

```text
Δzᵢ = zᵢ₊₁ − zᵢ
distance dᵢ = ||Δzᵢ||₂
curvature θᵢ = arccos( (Δzᵢ · Δzᵢ₊₁) / (||Δzᵢ||₂ ||Δzᵢ₊₁||₂) )
```

Distance menunjukkan besar perubahan representasi; curvature mengukur perubahan arah antarperpindahan. Implementasi perlu menangani norma mendekati nol dan pembulatan numerik secara aman, serta mengikuti konvensi normalisasi dan satuan sudut yang dipilih.

### 6.2 Detail yang sempat berbeda dalam percakapan: CLS dan patch

**Paper ReStraV asli** menyatakan 24 frame diambil dari jendela sekitar 2 detik, diubah ukurannya menjadi 224 × 224, lalu diproses DINOv2 ViT-S/14. Paper menjelaskan penggunaan **CLS token dan seluruh patch embeddings** dari blok terakhir (block.11) yang digabung menjadi representasi `zᵢ` berdimensi **75.648** untuk setiap frame. Representasi ini bukan 75.648 kata atau label objek; setiap dimensi tidak diberi arti semantik spesifik satu per satu. [Sumber: paper asli, Bagian 4](https://arxiv.org/html/2507.00583).

Sebelumnya juga pernah disarankan menggunakan **CLS saja dengan 384 dimensi** dan bahkan 8 frame untuk menekan biaya. Selain itu, paper lanjutan Papougiannakis & Tous menjelaskan konfigurasi ReStraV berbasis CLS 384-D untuk 24 frame, dengan MLP 21→64→32→1, Adam lr 10⁻³, 10 epoch. Itu adalah **konfigurasi/varian yang perlu dibedakan** dari deskripsi CLS-plus-patch dalam paper asli, dan keputusan memakai CLS saja atau 8 frame tidak boleh ditulis seolah identik dengan reproduksi persis. Lihat [paper lanjutan, bagian 3.2](https://www.mdpi.com/2079-9292/15/15/3245).

Paper asli menjelaskan descriptor klasifikasi **21 fitur**: tujuh nilai distance, enam nilai curvature, dan delapan statistik agregat (mean, variance, minimum, maksimum untuk masing-masing kelompok). Paper yang sama juga membahas 24 frame untuk lintasan umum; karena deskripsi konfigurasi dan pengukuran biayanya memiliki detail yang perlu dicocokkan, **periksa versi PDF dan kode resmi sebelum menetapkan persis bagaimana 21 fitur dan sampling dihitung**. Jangan menyimpulkan bahwa 21 fitur berarti encoder hanya menghitung 21 angka atau semua embedding tidak lagi memerlukan biaya. [Sumber paper asli, Bagian 5–6](https://arxiv.org/html/2507.00583); [kode ReStraV](https://github.com/ChristianInterno/ReStraV).

### 6.3 Temuan paper ReStraV yang relevan untuk kompresi (dibaca penuh 13 November 2026)

- **Mean curvature adalah fitur terpenting** dalam permutation feature importance (Lampiran A.3), lebih dari dua kali fitur berikutnya (variance curvature). Statistik agregat jauh lebih penting daripada nilai per-langkah.
- Video natural: **mean curvature lebih rendah tetapi variance curvature lebih tinggi** (Bagian 5).
- Mekanismenya **relatif**: DINOv2 meluruskan video natural tetapi tidak meluruskan video AI (gap ≈45,46°, Bagian 4). Korelasi antara kemampuan straightening absolut dan kemampuan deteksi dapat diabaikan.
- **Paper tidak menguji kompresi atau post-processing apa pun.** Ablasi hanya mencakup panjang video, interval sampling, jumlah frame (optimal sekitar T≈20), posisi jendela, dan frekuensi scene cut (Lampiran B–C).
- Threshold dioptimalkan pada training set untuk F1, lalu dipakai tanpa diubah pada test set.
- Latensi ±48 ms per video (DINOv2 ±43,6 ms) pada RTX-2080 8 GB.
- Paper menyebut kurtosis curvature sebagai arah lanjutan, serta mengakui keterbatasan berupa Goodhart’s law dan generator masa depan.

### 6.4 Apakah ReStraV harus DINOv2?

**Tidak secara konsep**, karena fitur geometri dapat dihitung pada representasi encoder visual lain. Namun:

- Implementasi utama ReStraV asli memakai DINOv2. Paper menganalisis gap curvature 14 encoder. Perbandingan gap curvature bukan bukti bahwa semua encoder sudah diuji sebagai classifier lengkap dengan performa setara.
- Papougiannakis & Tous melaporkan percobaan mengganti DINOv2 ViT-S/14 dengan DINOv3 ViT-S/16 dalam konteks geometrinya, dengan perubahan AUC +0,004 pada VidProM. Ini bukan pembuktian komprehensif lintas semua dataset atau CRF. [Sumber, bagian 6.4](https://www.mdpi.com/2079-9292/15/15/3245).
- CLIP pernah muncul sebagai **usulan** encoder pembanding. Ini menambah ekstraksi dan masalah kesetaraan dimensi, token, normalisasi, dan protokol. Jangan menjanjikan DINOv2 pasti yang terbaik.
- Membuktikan encoder dapat diganti **tidak otomatis** menjawab tuntutan kontribusi/novelty dosen. Itulah mengapa alternatif yang lebih akhir disarankan adalah menguji mekanisme mitigasi kompresi.

## 7. Fokus dan rumusan masalah yang dapat dibawa ke bimbingan

**Fokus kerja saat ini:** analisis ketahanan deteksi AI-generated video berbasis geometri temporal ReStraV terhadap kompresi H.264, lalu merancang dan mengevaluasi komponen peningkatan yang ringan (usulan terbaru: Bagian 8A).

Rumusan berikut merupakan **usulan kerja**, bukan rumusan resmi yang sudah disetujui:

1. Bagaimana variasi tingkat kompresi H.264 memengaruhi performa ReStraV dalam membedakan video real dan AI-generated?
2. Bagaimana fitur temporal distance dan curvature berubah antara video source dan versi terkompresi? Apakah pola perubahan berbeda antara kelas atau generator?
3. Seberapa besar kontribusi fitur distance-only, curvature-only, dan gabungannya pada kondisi kompresi yang diuji?
4. Apakah projector konsisten-kompresi (Bagian 8A) dapat mengurangi penurunan performa dibanding ReStraV standar **dan** compression-aware training biasa?
5. Apakah metode yang dilatih pada sebagian tingkat kompresi dapat bekerja pada tingkat CRF yang tidak dipakai ketika training, tanpa penurunan yang tidak dapat diterima pada source?

Tujuan dapat ditulis netral: **mengukur, menganalisis, merancang, dan mengevaluasi**, bukan “membuktikan metode pasti lebih unggul”. Perubahan fitur yang berjalan bersamaan dengan perubahan performa adalah **hubungan/asosiasi**, bukan otomatis pembuktian sebab tunggal. Ablasi dan kontrol membantu memperkuat interpretasi.

### Usulan judul yang pernah muncul

1. **“Analisis Pengaruh Degradasi Kompresi terhadap Kinerja Deteksi Video Hasil Generasi Kecerdasan Buatan.”** Usulan lebih awal, belum mengunci metode.
2. **“Analisis dan Peningkatan Ketahanan Deteksi Video AI Berbasis ReStraV terhadap Degradasi Kompresi H.264.”** Usulan lebih akhir jika memang ada tahap pengembangan dan evaluasi solusi.

Keduanya belum terverifikasi sebagai judul resmi. Kata “peningkatan” dalam judul kedua sebaiknya dipahami sebagai **upaya pengembangan dan pengujian**, bukan janji hasil pasti meningkat; redaksi final perlu disepakati pembimbing.

## 8. Usulan solusi 29 Oktober 2026: paired geometry consistency learning (adapter setelah fitur)

> **Status:** usulan historis, **direvisi** pada 13 November 2026 menjadi projector sebelum geometri (Bagian 8A). Bagian ini dipertahankan untuk jejak alasan dan karena preseden literaturnya tetap berlaku.

### 8.1 Gagasan

Untuk satu video, tersedia dua versi dengan konten dan label yang sama:

- Video **source**: file dataset yang diterima, sebelum kompresi tambahan oleh peneliti.
- Video yang sama setelah proses **H.264** pada tingkat CRF tertentu.

Ekstrak fitur ReStraV dari setiap versi. Tambahkan **adapter kecil yang dapat dilatih** setelah fitur geometri. Adapter dan classifier dilatih agar tetap mampu mengklasifikasi sekaligus membuat representasi kedua versi lebih konsisten. DINOv2 tetap frozen.

```text
uᵢˢ = hφ(xᵢˢ),   uᵢᑫ = hφ(xᵢᑫ)
Lconsistency = (1/N) Σᵢ ||uᵢˢ − uᵢᑫ||²₂
Ltotal = Lclassification + λ Lconsistency
```

**Kelemahan yang menyebabkan revisi:** jika curvature sudah rusak oleh noise kompresi pada saat dihitung, adapter yang bekerja pada 21 fitur sesudahnya tidak dapat memulihkan informasi arah lintasan yang hilang. Komponen yang dapat dilatih lebih masuk akal ditempatkan **sebelum** perhitungan geometri.

### 8.2 Posisi ilmiah dan batas novelty (tetap berlaku)

- **Compression-aware training** dan **consistency learning** sudah dikenal dalam literatur; tidak layak diklaim pertama secara global.
- Ketiadaan mekanisme ini pada satu paper **bukan bukti tidak pernah ada di seluruh literatur**; perlu penelusuran novelty ulang sebelum klaim final.
- Contoh preseden konsep: [FreqDebias, CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Kashiani_FreqDebias_Towards_Generalizable_Deepfake_Detection_via_Consistency-Driven_Frequency_Debiasing_CVPR_2025_paper.html), [contrastive spatiotemporal distillation untuk deepfake terkompresi](https://www.sciencedirect.com/science/article/pii/S0925231223009955), dan [ODDN, AAAI 2025](https://ojs.aaai.org/index.php/AAAI/article/view/32063). Preseden tambahan ada di Bagian 8A.5.

### 8.3 Alternatif yang pernah dibahas

| Opsi | Ide | Posisi saat ini |
|---|---|---|
| Compression-aware training | Latih classifier dengan fitur video source dan berbagai CRF. | Pembanding wajib (B1 di Bagian 8A). |
| Paired consistency setelah fitur (adapter) | Selaraskan 21 fitur source–compressed melalui adapter. | Direvisi menjadi projector sebelum geometri. |
| Geometry feature restoration | Latih modul memetakan fitur compressed agar mendekati fitur source. | Alternatif; informasi hilang tidak selalu bisa dipulihkan. |
| Stability-aware selection/weighting | Pilih atau bobot fitur yang stabil sekaligus diskriminatif. | Alternatif/pengembangan tambahan. |
| Perbandingan encoder (DINOv2 vs CLIP/DINOv3) | Gunakan geometri pada representasi encoder yang berbeda. | Eksperimen pendukung bila diminta pembimbing. |

Jangan mengandalkan metadata CRF pada saat inferensi kecuali aplikasi memang diasumsikan memperoleh metadata tersebut.

## 8A. Usulan terbaru (13 November 2026): projector konsisten-kompresi sebelum geometri

> **Status:** rekomendasi asisten setelah membaca penuh paper ReStraV dan menelusuri literatur. **Belum dibawa ke dosen, belum disetujui, belum diimplementasikan, belum ada hasil.** Nama “compression-consistent trajectory projector” / “ReStraV-CR” adalah label kerja.

### 8A.1 Mengapa kompresi diduga merusak ReStraV (mekanisme yang diusulkan)

**(a) Noise embedding menarik curvature ke ~120°.** Kompresi membuat embedding tiap frame “bergetar”. Jika `zᵢ' = zᵢ + εᵢ` dengan noise acak independen, dua displacement berurutan berbagi satu `ε` dengan tanda berlawanan: `E[Δεᵢ·Δεᵢ₊₁] = −Dσ²` dan `E||Δε||² = 2Dσ²`. Akibatnya cos ≈ −½ dan curvature cenderung ke **120°** untuk kedua kelas. Distance juga membesar (`E||Δz'||² = ||Δz||² + 2Dσ²`).

Simulasi mainan (lintasan sintetis 384-D, 24 frame, 200 sampel per kelas; **bukan data nyata**):

| Noise relatif terhadap langkah | Curvature “real” (asli 30°) | Curvature “AI” (asli 70°) | Gap |
|---|---|---|---|
| 0 | 30,0° | 70,0° | 40,0° |
| 0,39 | 56,9° | 81,7° | 24,8° |
| 0,98 | 91,8° | 102,3° | 10,5° |
| 1,96 | 110,0° | 113,8° | 3,8° |

Prediksi yang dapat diuji: gap real–AI menyempit; **curvature video real naik lebih cepat** sehingga tampak “mirip AI” (false positive real naik); video dengan gerak kecil (langkah pendek) paling terdampak. Noise kompresi nyata berkorelasi antar-frame, jadi efek nyata bisa lebih kecil; itulah yang perlu diukur.

**(b) Fluktuasi kualitas antar jenis frame H.264.** Default x264: ipratio 1,40 dan pbratio 1,30; frame I berkualitas lebih tinggi dari P, B lebih rendah dari P; pola frame contoh IBBBPBBBPBPI. Fluktuasi berulang ini diduga menambah jitter lintasan. Catatan: pbratio tidak dipakai bila mb-tree aktif (default), sehingga besarnya efek perlu diukur, bukan diasumsikan.

### 8A.2 Desain metode

```text
Frame → DINOv2 (frozen) → zᵢ → projector g_φ (trainable) → uᵢ → distance & curvature → fitur geometri → MLP → real/AI
```

- `g_φ`: projector kecil, misalnya linear 384→128 atau MLP kecil dengan residual; bekerja per frame pada embedding yang di-cache.
- Geometri dihitung dari `uᵢ` secara differentiable (pakai cos dengan clamp/epsilon).
- Pelatihan berpasangan (video yang sama, versi source dan versi CRF acak dari {10, 20, 30}):

```text
L = L_cls(source) + L_cls(kompresi) + λ · ‖f(source) − f(kompresi)‖²
```

  `f` = vektor fitur geometri (dinormalisasi dengan statistik training). λ dipilih dengan validation set.
- **Tambahan tanpa training (juga sebagai ablasi):** curvature multi-skala (Δz dengan jarak 2–3 frame sehingga langkah lebih panjang dan rasio sinyal/noise naik) dan statistik robust (median/persentil menggantikan min/max).
- **Risiko collapse:** projector dapat memetakan semua ke konstan agar consistency nol; classification loss mencegahnya, opsional ditambah regularisasi variansi. Pantau.
- **Pilihan token:** usulan memakai CLS 384-D demi efisiensi (atau CLS + mean patch); laporkan sebagai varian terhadap paper asli (CLS + patch 75.648-D). Cek kode resmi.

### 8A.3 Susunan model dan ablasi

| Kode | Isi | Pertanyaan |
|---|---|---|
| B0 | ReStraV asli, dilatih source saja | Seberapa besar penurunan akibat kompresi? |
| B1 | B0 + augmentasi kompresi saat training | Pembanding wajib: cukupkah augmentasi biasa? |
| P1 | B1 + multi-skala + statistik robust | Apakah perbaikan tanpa training sudah membantu? |
| P2 | Projector, tanpa loss konsistensi | Apakah sekadar tambahan parameter membantu? |
| P3 | Projector + paired consistency loss | **Usulan utama** |

Kontribusi consistency dinilai dari **P3 vs P2 dan P3 vs B1**, bukan hanya P3 vs B0. Bila P3 tidak unggul, hasil itu tetap sah dilaporkan. Ablasi distance-only / curvature-only / gabungan tetap relevan untuk menjawab RM3.

### 8A.4 Hipotesis yang diusulkan untuk dijawab ke dosen

- **H1 (dampak):** Semakin tinggi CRF, AUROC ReStraV baseline (B0) turun dan gap mean curvature real–AI menyempit, dengan kenaikan curvature lebih besar pada video real. *H1 ditolak jika AUROC dan gap stabil.*
- **H2 (perbaikan):** Pada CRF tinggi, P3 menghasilkan AUROC lebih tinggi daripada B1 dan P2, tanpa penurunan berarti pada video source.
- **H3 (generalisasi):** Keunggulan P3 bertahan pada tingkat CRF yang tidak dipakai saat training (mis. CRF 40, opsional 25/35).
- Sub-hipotesis analisis: degradasi lebih besar pada video dengan gerak kecil (distance rendah).

Kalimat siap pakai: *“Saya menduga kompresi menambah noise pada embedding frame sehingga curvature video real naik mendekati video AI. Karena itu saya merancang projector yang dilatih agar geometri lintasan konsisten antara versi asli dan versi terkompresinya, lalu membuktikannya terhadap ReStraV standar dan augmentasi kompresi biasa.”*

Jawaban atas kritik “hanya eksperimen”: alurnya **analisis mekanisme → rancang komponen baru berdasarkan mekanisme → buktikan dengan pembanding dan ablasi yang adil**.

### 8A.5 Dukungan literatur (ditelusuri 13 November 2026)

- **Celah eksplisit:** Papougiannakis & Tous (2026) menyebut evaluasi AUC di bawah kompresi FFmpeg pada beberapa nilai CRF sebagai pekerjaan lanjutan, dan tidak menguji post-processing apa pun.
- **ReStraV goyah di bawah gangguan:** MTOR (2026) melaporkan rata-rata AP ReStraV 87,91% atas 12 gangguan dengan rentang 16,38 poin persentase; tidak memecah per QP dan tidak menawarkan perbaikan untuk ReStraV.
- **Peringatan protokol:** Cakiroglu dkk., “Where the Real Clips Came From” (arXiv 2606.31004, 2026): ringkasan pencarian menyebut re-encode menurunkan ReStraV di GenVidBench dari 0,85 ke 0,60 karena 8 fps mengulang frame pada klip real; paper juga menunjukkan panjang klip / jumlah frame dapat membocorkan label dan menyarankan re-encode kanonik, penyamaan panjang klip, probe real-vs-real, seed ganda, dan bootstrap CI. (Angka 0,85→0,60 berasal dari cuplikan hasil pencarian; teks HTML yang terbaca tidak memuatnya. Verifikasi di PDF.)
- **Consistency untuk ketahanan:** Stability training (Zheng dkk., CVPR 2016) melatih keluaran konstan terhadap salinan terdistorsi termasuk kompresi; AugMix (Hendrycks dkk., ICLR 2020) memakai JS-divergence consistency antar-augmentasi.
- **Deepfake terkompresi:** ADD (Le & Woo, AAAI 2022) distillation untuk gambar terkompresi; QAD (Le & Woo, ICCV 2023) quality-agnostic detection; ODDN (Tao dkk., AAAI 2025) data berpasangan/tidak berpasangan untuk OSN.
- **Jembatan straightening ↔ robustness:** Niu, Savin & Simoncelli (NeurIPS 2024): objektif straightening menghasilkan representasi lebih tahan noise dan adversarial; Harrington dkk. (2022/ICLR 2023): adversarial training menghasilkan representasi lebih lurus.
- **Augmentasi sebagai baseline:** Wang dkk. (CVPR 2020) blur + JPEG augmentation; Vahdati dkk. (CVPRW 2024) MISLnet diuji H.264 CRF 0–40.
- **Sensitivitas DINOv2:** “Probing Perturbation Invariance in DINOv2” (ICML 2026): gambar AI lebih sensitif terhadap noise Gaussian daripada foto asli, diukur lewat patch token. Mendukung analisis per kelas dan pilihan token.
- **Pembeda dari karya serupa:** TRACE (Cao dkk., arXiv 2609.25775) memakai “consistency” antar-frame bersebelahan pada velocity field DiT, bukan pasangan bersih–terkompresi, dan tidak menguji codec video. Dalam penelusuran ini belum ditemukan paired consistency source–terkompresi pada geometri lintasan ReStraV; **klaim tetap harus dibatasi pada penelusuran yang dilakukan.**

### 8A.6 Kelayakan

DINOv2 dijalankan **sekali** per video per kondisi; embedding CLS (24 × 384 float ≈ 37 KB per video) disimpan. Semua varian B0–P3 dilatih ulang dari cache, dan projector kecil berlatih dalam hitungan menit. Ini selaras dengan arahan dosen soal metode ringan. Embedding CLS + patch penuh (≈7 MB per video per kondisi) jauh lebih berat; perlu diputuskan.

## 8B. Masukan dosen 21 November 2026 dan revisi desain projector

> **Status:** masukan dosen dilaporkan mahasiswa. Revisi di bawah adalah usulan asisten, belum disetujui dosen, belum ada hasil eksperimen nyata. Riset lengkapnya ada di Claude Doc "Riset Projector untuk Menjawab Dosen" (21 Nov 2026).

**Pertanyaan dosen tentang projector:** (1) kenapa dinamakan projector padahal neuron biasa; (2) berapa node; (3) bagaimana efek sebelum dan sesudahnya; (4) apakah works tanpa contrastive loss; (5) apakah ini contrastive learning; (6) referensi matematisnya. Dosen juga meminta studi literatur metode-metode berbasis vektor dengan referensi jelas, untuk mulai menulis Bab 1 (ditunda dulu, diprioritaskan memahami projector). Mahasiswa meminta setiap argumen disertai lokasi di referensi dan tanpa celah logika.

**Jawaban inti yang disusun:**
- Projector = satu lapisan linear. Nama mengikuti SimCLR (projection head, Bag. 2.1), BYOL (projector, Bag. 3.1; Lamp. F.1 Tabel 14: kedalaman 1 = linear), dan CLIP (linear projection, Bag. 2.3). Karena ReStraV hanya memakai selisih antar-frame, Δu = WΔz (bias hilang), sehingga projector setara dengan mempelajari **metrik Mahalanobis M = WᵀW** untuk distance dan curvature (Kulis 2013, Bag. 2). W ortogonal tidak mengubah geometri.
- Ukuran: utama 384→384 tanpa bias, inisialisasi identitas (titik awal = ReStraV, 147.456 bobot); varian k ∈ {64, 128, 256} dipilih lewat validation. Koreksi: angka 49.280 sebelumnya salah karena memasukkan bias; versi 128 = 49.152.
- Bukan contrastive learning (tidak ada negatif); ini supervised + paired consistency regularization, arsitektur Siamese. Preseden tanpa negatif: Stability Training (Zheng dkk. 2016, Pers. 2 dan 8, distorsi JPEG), Π-model (Laine & Aila 2017, Alg. 1; membantu juga saat semua data berlabel), AugMix (Pers. 1, Tabel 4). Fungsi negatif = anti-collapse (Hadsell dkk. 2006 setelah Pers. 4; Wang & Isola 2020 Teorema 1); di desain ini anti-collapse dipegang loss klasifikasi.
- Simulasi sintetis (64-D, 3 seed, BUKAN hasil skripsi): projector meregangkan dimensi bersih (1,17) dan menyusutkan dimensi bising (0,85), drift fitur turun ±41%. Tetapi P2 (tanpa konsistensi) belajar pola mirip; keunggulan P3 atas B1/P2 hanya ±0,02–0,03 AUROC di level terlatih. Varian kontrastif P4 lebih tahan di kompresi berat pada 2 dari 3 seed.

**Revisi desain yang diusulkan:** projector tanpa bias; inisialisasi identitas; loss konsistensi pada fitur terstandardisasi (statistik versi asli, di-detach) untuk mencegah "curang lewat skala" (distance ikut skala W, curvature tidak); **tambah model P4** (projector + InfoNCE berpasangan) untuk menguji perlu tidaknya contrastive; laporkan 5 ukuran sebelum/sesudah (drift, gap curvature real–AI, mean curvature video asli, spektrum M, AUROC per CRF); pantau simpangan baku fitur. Susunan model: B0, B1, P2, P3 (utama), P4.

## 9. Rancangan eksperimen yang terakhir diusulkan

### 9.1 Kandidat dataset: ComGenVid

ComGenVid muncul dalam rekomendasi sebagai **kandidat** dataset utama. Dataset card menyatakan total 5.100 video: 1.700 video real dari MSVD, 1.700 video Sora, dan 1.700 video Veo3. [Dataset card](https://huggingface.co/datasets/OmerXYZ/comgenvid); [paper STALL yang memperkenalkannya](https://arxiv.org/abs/2603.15026).

Hal yang sangat penting:

- Angka yang seimbang adalah **tiga sumber**. Jika Sora dan Veo3 digabung sebagai kelas AI, perbandingan mentahnya adalah **1.700 real : 3.400 AI**, bukan 50:50. Pilih strategi yang sesuai—misalnya subset, pembobotan, atau melaporkan evaluasi per kelas—tanpa melakukan balancing berdasarkan test outcome.
- Real dan AI berasal dari sumber yang berbeda; durasi, resolusi, FPS, gaya konten, kualitas, distribusi gerak, atau pipeline asal dapat menjadi shortcut. Dataset card melaporkan, misalnya, Veo3 memiliki durasi dan resolusi yang jauh lebih seragam dibanding MSVD. Analisis dan/atau kontrol harus mempertimbangkan confound tersebut; menyamakan codec tambahan tidak otomatis menghapus perbedaan sumber. Lihat juga peringatan protokol di Bagian 8A.5 (panjang klip, frame duplikat, probe real-vs-real).
- Dataset card menyebut embedding **DINOv3 ViT-L/16** yang sudah dihitung. Itu **tidak dapat langsung diperlakukan sebagai embedding DINOv2 ReStraV**, apalagi sebagai representasi versi H.264 yang baru dibuat.
- Ketersediaan video, lisensi, ukuran unduhan, jumlah sampel yang benar-benar dapat dipakai, dan kelayakan komputasi harus diverifikasi sebelum dataset difinalkan.
- Dataset lain/eksternal dan evaluasi lintas generator bisa memperkuat validitas, tetapi kemampuan dan ketersediaannya belum dipastikan. Dengan hanya Sora dan Veo3, klaim umum untuk semua generator tidak kuat.

### 9.2 Split dan kontrol kebocoran

Usulan awal: **70% training, 15% validation, 15% testing**, namun proporsi final belum disahkan. Split dilakukan berdasarkan **identitas video dan, jika relevan, kelompok konten/asal yang saling terkait** sebelum membuat versi kompresi. Seluruh versi dari satu video harus berada pada split yang sama. Cegah duplikasi, potongan video yang sama, atau pasangan konten yang berkaitan bocor antar-split.

Stratifikasi kelas dan sumber/generator bila memungkinkan. Tentukan kebijakan balancing pada training dan prosedur evaluasi di depan; simpan manifest, random seed, identitas sumber, label, dan kondisi. Normalisasi, pemilihan fitur, pemilihan hiperparameter (termasuk λ) dan threshold hanya menggunakan training/validation sesuai protokol, **tidak menggunakan test**.

### 9.3 Kondisi source dan kompresi

| Fungsi | Rancangan terakhir yang disarankan |
|---|---|
| B0 training | Source saja. |
| B1, P1–P3 training | Source dan contoh H.264 CRF 10, 20, 30. |
| Validation | Video validation yang terpisah; kondisi dan tujuan tuning ditetapkan sebelum test. |
| Test utama | Source serta CRF 0, 10, 20, 30, 40. |
| Test tambahan, jika layak | CRF 25 dan 35 untuk interpolasi; CRF 40 adalah tingkat berat yang tidak dipakai training dalam rancangan ini. |

Yang unseen di sini adalah **tingkat kompresi**, bukan otomatis generator.

**Source bukan sinonim video mentah tak terkompresi.** Video dataset bisa saja sudah pernah dikompresi. Source artinya sebelum rekode tambahan penelitian. **CRF 0 juga bukan source**: ia tetap file hasil re-encoding. Pada konfigurasi libx264 yang mendukung lossless, CRF 0 dapat lossless terhadap frame masukan yang diterima encoder, tetapi tidak memulihkan informasi yang hilang sebelumnya; pixel format atau konversi tambahan dapat memengaruhi sifat lossless. Periksa konfigurasi aktual.

Secara umum pada konfigurasi encoder yang sama, CRF lebih besar memberi kompresi lebih agresif dan kualitas lebih rendah, tetapi CRF bukan persentase kompresi maupun bitrate tetap. Gunakan software/versi encoder dan pengaturan yang dicatat konsisten; perhatikan resolusi, **FPS (samakan; hindari frame duplikat yang membuat Δz = 0 dan curvature tak terdefinisi)**, durasi, pixel format, audio, crop, timestamp, struktur GOP/B-frame, dan frame sampling. Hindari menyebut hasil rekode terkontrol sebagai bukti semua jalur kompresi WhatsApp/TikTok identik.

Rancangan CRF yang lebih lama seperti 18, 23, 28, 33, 38 pernah diberikan sebagai contoh. Itu **bukan bersamaan wajib** dengan rancangan terbaru dan belum terbukti final.

### 9.4 Evaluasi yang disarankan

- **AUROC** sebagai ukuran pemisahan skor; definisikan kelas positif dengan jelas.
- **Accuracy, precision, recall AI, F1-score**, confusion matrix, dan **false positive pada real** (penting untuk H1). Threshold ditentukan dengan validation dan tidak dioptimalkan ulang pada test di setiap CRF.
- **Perubahan kinerja per CRF terhadap source**: `ΔAUC(q) = AUC(q) − AUC(source)`; nilai negatif menunjukkan penurunan.
- **Perbandingan P3 terhadap B0, B1, P1, P2** pada kondisi identik.
- **Gap mean curvature real–AI per CRF** dan **feature drift berpasangan** source vs CRF, per kelompok fitur, kelas, generator, dan kelompok gerak (distance rendah vs tinggi).
- Grafik CRF vs AUROC/F1, pergeseran fitur, performa source, dan biaya komputasi.
- Bila memadai: paired bootstrap berdasarkan identitas video, beberapa seed, dan probe real-vs-real sebagai lantai pembanding.

## 10. Posisi literatur yang telah dipelajari

### ReStraV — referensi metode utama

Internò dkk., **“AI-Generated Video Detection via Perceptual Straightening”**, NeurIPS 2025 (arXiv v4, 3 Feb 2026, dibaca penuh 13 Nov 2026). [Paper](https://arxiv.org/abs/2507.00583) · [Kode](https://github.com/ChristianInterno/ReStraV). Angka paper tidak boleh dilaporkan sebagai hasil skripsi mahasiswa.

### AIGVDBench — referensi benchmark dan generalisasi

Ma dkk., **“Your One-Stop Solution for AI-Generated Video Detection”**. [Paper](https://arxiv.org/abs/2601.11035) · [Kode](https://github.com/LongMa-2025/AIGVDBench).

### Review deepfake — peta metode dan keterbatasan

Raza dkk. (2026). [Paper](https://www.mdpi.com/2673-2688/7/2/68).

### VideoVeritas — pendekatan multimodal yang lebih berat

Tan dkk. (2026). [Paper](https://arxiv.org/abs/2602.08828).

### MISLnet dan kompresi

Vahdati dkk., CVPR Workshops 2024. Diuji terhadap H.264 recompression CRF 0–40. [Paper](https://openaccess.thecvf.com/content/CVPR2024W/WMF/html/Vahdati_Beyond_Deepfake_Images_Detecting_AI-Generated_Videos_CVPRW_2024_paper.html).

### Paper lanjutan perceptual straightening

Papougiannakis & Tous, Electronics 2026. Rich384 (21-D + mean CLS 384-D) dan GurAI (fusi dengan DeMamba). **Tidak menguji post-processing; menyebut AUC di bawah kompresi FFmpeg multi-CRF sebagai pekerjaan lanjutan.** Jangan klaim penggabungan mean embedding dan geometri sebagai ide baru. [Paper](https://www.mdpi.com/2079-9292/15/15/3245).

### MTOR

Wang dkk. (2026). ReStraV sebagai baseline pada 12 gangguan termasuk H.264 QP 23/30/38; rata-rata AP ReStraV 87,91% (rentang 16,38 pp) atas 12 gangguan, bukan khusus H.264. **QP ≠ CRF.** Jangan klaim menjadi yang pertama menguji ReStraV pada video terkompresi. [Paper](https://arxiv.org/html/2610.06378).

### Referensi baru 13 November 2026

- Cakiroglu dkk., **“Where the Real Clips Came From”** (2026) — audit kebocoran panjang klip/codec, efek re-encode pada ReStraV. [arXiv](https://arxiv.org/html/2606.31004)
- Cao dkk., **TRACE** (2026) — consistency antar-frame via DiT; pembeda dari usulan. [arXiv](https://arxiv.org/pdf/2609.25775)
- Zheng dkk., **Stability Training**, CVPR 2016. [arXiv](https://arxiv.org/pdf/1604.04326)
- Hendrycks dkk., **AugMix**, ICLR 2020. [arXiv](https://arxiv.org/pdf/1912.02781)
- Le & Woo, **ADD**, AAAI 2022. [AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/19886)
- Le & Woo, **QAD**, ICCV 2023. [arXiv](https://arxiv.org/pdf/2309.05911)
- Tao dkk., **ODDN**, AAAI 2025. [AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/32063)
- Niu, Savin & Simoncelli, NeurIPS 2024. [Proceedings](https://proceedings.neurips.cc/paper_files/paper/2024/hash/473c578564ed9fe8041abfef772d37db-Abstract.html)
- Harrington dkk., NeurIPS Workshop 2022 / ICLR 2023. [NeurIPS](https://nips.cc/virtual/2022/62203)
- Wang dkk., **CNN-generated images are surprisingly easy to spot… for now**, CVPR 2020. [CVF](https://openaccess.thecvf.com/content_CVPR_2020/html/Wang_CNN-Generated_Images_Are_Surprisingly_Easy_to_Spot..._for_Now_CVPR_2020_paper.html)
- **Probing Perturbation Invariance in DINOv2**, ICML 2026. [ICML](https://icml.cc/virtual/2026/79536)
- Hénaff, Goris & Simoncelli, **Perceptual straightening of natural videos**, Nature Neuroscience 2019 (dasar teori). [Nature](https://www.nature.com/articles/s41593-019-0377-4)
- Referensi x264 (ipratio/pbratio, pola frame): [Fora Soft](https://www.forasoft.com/learn/video-encoding/articles/bit-allocation-gop), [MeGUI x264 Settings](https://en.wikibooks.org/wiki/MeGUI/x264_Settings)

## 11. Hal yang sudah dikerjakan, belum, dan belum diketahui

### Sudah tampak dalam percakapan

- Menentukan minat pada deteksi video AI terkait degradasi kompresi sebagai konteks utama.
- Membahas konsep ReStraV, embedding visual DINOv2, distance, curvature, classifier, dan efisiensi.
- Menyusun ringkasan beberapa paper serta bahan penjelasan/bimbingan dan rancangan judul sementara.
- Memahami bahwa eksperimen idealnya membandingkan real dan AI dalam kondisi kompresi yang sama.
- Mempertimbangkan MISLnet, lalu meminta argumen untuk fokus pada ReStraV.
- Mengidentifikasi kritik dosen tentang kontribusi, “hanya eksperimen”, dan pertanyaan hipotesis.
- (13 Nov) Paper ReStraV dibaca penuh; mekanisme degradasi diusulkan dan diilustrasikan dengan simulasi mainan; usulan projector konsisten-kompresi, hipotesis H1–H3, dan model B0–P3 disusun dengan dukungan literatur.
- **(Revisi 22 Nov)** ComGenVid diunduh (5.100 video). Kode ReStraV resmi berjalan di laptop mahasiswa (versi lokal, lihat `eksperimen/restrav_reproduksi/PERUBAHAN.md`). Fitur 5.100 video asli (tanpa re-encode) sudah diekstrak. Reproduksi `train.py` apa adanya: AUROC 0,834, accuracy 0,758 pada 1.700 video test, satu seed. Protokolnya belum bersih (normalisasi sebelum split, separuh fake dibuang), jadi ini **belum** angka B0 final. Rincian di `eksperimen/restrav_reproduksi/HASIL.md`.
- **(Revisi 22 Nov)** B0 dengan protokol bersih (split tetap berkelompok per YouTube ID, normalisasi dari train, 5 seed): AUROC 0,834 ± 0,004 pada video asli. Protokolnya usulan asisten dan belum dikonfirmasi dosen. Jalan pintas codec belum dikontrol. Ini melengkapi Bagian 9.2 (split dan kontrol kebocoran).

### Belum ada bukti yang dapat diverifikasi dalam sumber yang tersedia

- ~~Dataset final telah dipilih, diunduh, dibersihkan, dan displit.~~ (22 Nov: diunduh, belum diputuskan final, split tetap belum dibuat)
- ~~Reproduksi kode ReStraV berhasil pada perangkat mahasiswa.~~ (22 Nov: berhasil, lihat di atas)
- Video telah dire-encode atau model telah dilatih pada kondisi kompresi. (22 Nov: fitur video asli sudah diekstrak, re-encode belum)
- Nilai AUROC, F1, accuracy, grafik, atau angka penurunan performa akibat kompresi. (22 Nov: baru ada angka reproduksi pada video asli.) Angka simulasi di Bagian 8A.1 adalah ilustrasi sintetis, bukan hasil skripsi.
- Projector/consistency loss telah dibuat atau menunjukkan perbaikan.
- Dosen telah menyetujui judul, metode, hipotesis, CRF final, dataset, ukuran sampel, atau jadwal.
- Ketersediaan GPU, kapasitas storage, deadline, format kampus, dan target kelulusan.

## 12. Risiko dan keputusan yang perlu dibahas dengan pembimbing

1. **Kontribusi yang diterima:** apakah dosen menerima projector konsisten-kompresi (P3) sebagai kontribusi, dan hipotesis H1–H3?
2. **Spesifikasi baseline:** CLS + patch (paper asli) atau CLS-only (efisien, dipakai paper lanjutan). Cocokkan dengan kode resmi.
3. **Anggaran komputasi:** pilot DINOv2 pada sejumlah kecil video dan kondisi; putuskan token yang di-cache.
4. **Kelayakan ComGenVid:** akses, lisensi, rasio kelas, confound sumber, panjang klip, FPS.
5. **Desain CRF dan protokol re-encode:** grid CRF, CRF 0, kondisi unseen, penyamaan FPS, struktur GOP.
6. **Klaim novelty:** batasi pada penelusuran yang benar-benar dilakukan; jangan klaim pertama menguji ReStraV pada kompresi atau pertama memakai consistency learning.
7. **Generalisasi dan interpretasi:** jangan generalisasi ke semua generator/codec/platform.
8. **Trade-off source:** mitigasi bisa merugikan video source atau kelas real; laporkan kedua sisi.

## 13. Urutan kerja berikutnya yang disarankan

1. Bawa Bagian 8A.1–8A.4 (mekanisme, desain, model B0–P3, hipotesis) ke bimbingan; minta persetujuan arah dan hipotesis.
2. Cocokkan paper ReStraV, repo resmi, dimensi embedding, fitur 21-D, parameter sampling, dan biaya runtime.
3. **Pilot H1 lebih dulu:** misalnya ±100 video × CRF 0/20/40; ukur gap mean curvature real–AI dan AUROC B0. Jika gap menyempit, ini bukti awal untuk mendukung P3.
4. Audit dataset dan lisensi; buat manifest; periksa FPS, panjang klip, duplikasi, dan confound.
5. Tetapkan split berbasis identitas sebelum membuat versi kompresi; tentukan metrik, threshold, seed, aturan tuning.
6. Ekstrak dan cache embedding semua kondisi; jalankan B0, B1, P1, P2, P3 dan ablasi distance/curvature.
7. Analisis berpasangan (source vs CRF, CRF unseen, per kelas/generator/gerak, interval ketidakpastian, biaya).
8. Perbarui pencarian literatur dan rumusan kontribusi, lalu susun laporan sesuai hasil yang sebenarnya.

## 14. Contoh penjelasan singkat kepada dosen

> “Penelitian saya berfokus pada deteksi video hasil generative AI ketika video mengalami kompresi H.264. Video real dan video AI saya uji pada tingkat kompresi yang sama.
>
> Saya menduga kompresi menambah noise pada embedding DINOv2 setiap frame, sehingga curvature lintasan video real naik mendekati video AI dan ReStraV jadi lebih sering salah. Itu hipotesis pertama saya, dan bisa terbukti salah.
>
> Berdasarkan dugaan itu, saya merancang projector kecil di antara DINOv2 dan perhitungan geometri, yang dilatih agar geometri lintasan video asli dan versi terkompresinya konsisten. Saya membandingkannya dengan ReStraV standar, ReStraV dengan augmentasi kompresi biasa, dan projector tanpa loss konsistensi, supaya terlihat apakah komponen baru itu benar-benar memberi manfaat, termasuk pada tingkat kompresi yang tidak dipakai saat training.
>
> Ini masih rancangan yang ingin saya konsultasikan. Saya belum mengasumsikan kompresi pasti menurunkan performa atau metode perbaikan pasti berhasil.”

## 15. Panduan untuk Claude/asisten lanjutan

1. Perlakukan mahasiswa sebagai orang yang ingin **memahami proses dan alasannya**. Jelaskan alur awal–akhir secara runtut, beri contoh sederhana, lalu jelaskan teknis jika diperlukan.
2. Hormati prioritas **ringan dan realistis untuk skripsi**, tetapi jelaskan biaya nyata ekstraksi frame/encoder dan storage.
3. Pertahankan fokus pada video AI, H.264, fitur geometri ReStraV, dan mitigasi (usulan terbaru: Bagian 8A). Jangan kembali menjadikan MISLnet atau perbandingan banyak model sebagai fokus utama tanpa arahan baru.
4. Bedakan secara tegas **hasil yang sudah dilakukan**, **fakta dari literatur**, **usulan asisten**, **arahan dosen yang diceritakan mahasiswa**, serta **hal yang belum diketahui**.
5. Penggunaan DINOv2 bukan novelty tersendiri. Projector konsisten-kompresi masih usulan yang memerlukan keputusan.
6. Jangan membuat angka eksperimen, persetujuan pembimbing, metadata dataset, klaim novelty pertama, atau hasil yang tidak tersedia. Angka simulasi 8A.1 bukan hasil eksperimen.
7. Periksa paper primer dan kode saat ada perbedaan penjelasan, terutama **CLS + patch vs CLS-only**, jumlah frame, dan cara membentuk 21 fitur.
8. Jika diminta menyusun proposal, jadikan bagian-bagian yang belum disepakati sebagai **rancangan** dan jangan mengubahnya diam-diam menjadi keputusan final.

### Pesan pembuka yang bisa ditempelkan bersama dokumen ini

> Saya sedang mengerjakan skripsi tentang deteksi video hasil generative AI pada kondisi degradasi kompresi H.264. Dokumen terlampir berisi riwayat diskusi, arahan dosen yang saya ceritakan, referensi, dan usulan terakhir. Tolong baca seluruhnya sebelum memberi saran. Bedakan keputusan yang benar-benar tercatat dari usulan yang belum saya atau dosen setujui. Fokus utamanya ReStraV/DINOv2, analisis temporal distance dan curvature, serta pengembangan strategi ketahanan kompresi yang ringan. Jangan mengarang hasil eksperimen atau mengasumsikan implementasi sudah dilakukan.

## 16. Jejak percakapan untuk penelusuran ulang

| Judul percakapan | ID | Pokok bahasan |
|---|---|---|
| Sesi Claude, Project “Skripsi”, 13 Nov 2026 | — | Pembacaan penuh paper ReStraV, mekanisme degradasi, projector konsisten-kompresi, hipotesis H1–H3, referensi baru. |
| Studi Literatur 5 | `6ae2b975-59c8-83ec-ace7-3753e136df85` | Kritik dosen, ReStraV vs MISLnet, encoder selain DINOv2, dan rekomendasi mitigasi 29 Okt. |
| Cari Paper Deteksi Video AI Kompresi | `6abdb176-0ce4-83ec-bf2d-6175c6e831f6` | ReStraV, MISLnet, analisis kompresi, dan diskusi gap sebelum pembaruan MTOR. |
| Arah Riset Deteksi Video AI | `6abdb634-a5c4-83ec-8baf-ea35ca950b64` | Analisis vs peningkatan, real dan AI diberi kompresi setara. |
| Ringkasan Studi Literatur | `6ab4b0b7-03c8-83ec-8f4a-1bc1526c0fba` | Empat referensi, pemilihan metode ringan, dan rancangan lebih awal. |
| Ringkasan Referensi Teknis | `6ab4ba68-0930-83ec-ab1c-d17e4b660393` | Ringkasan teknis empat referensi. |
| Studi literatur 1 | `6aa18dd8-ee40-83ec-aae3-62be68f11fe5` | Detail embedding, CLS, patch, dan alur ReStraV. |
| Studi literatur 2 | `6ab49e71-6c50-83ec-a47a-8092c8243eb9` | AIGVDBench dan hubungannya dengan topik. |
| Studi literatur 3 | `6ab4a265-bba0-83ec-8d4b-9aed2bc180ab` | Review deepfake. |
| Studi Literatur 4 | `6ab4a669-81f0-83ec-9173-5d028ce4a7e8` | VideoVeritas/Qwen dan beban komputasi. |
| Kata Kunci Paper AI Video Detection | `6ab49c6e-aab0-83ec-af16-4f0133287193` | Arahan dosen mencari paper/review dan metode ringan. |
| Menulis Judul Seminar Skripsi | `6aa8b59c-7ec0-83ec-a1a6-ff1257baf6b4` | Usulan judul dan abstrak awal. |
| Persiapan Skripsi | `6a98e19e-a08c-83ec-a2e1-b7076bca9744` | ReStraV, repository dan kategori metode. |
| Persiapan Bimbingan Skripsi | `6a8fce3c-1224-83ec-8bca-8985db6cd9d6` | Latar belakang awal dan ringkasan handoff lama. |

---

**Status terakhir yang paling aman untuk diteruskan (13 November 2026):** fokus pada ReStraV + DINOv2 di bawah kompresi H.264, ditambah peningkatan metode. Usulan terbaru adalah **projector konsisten-kompresi sebelum geometri (P3)** dengan pembanding B0, B1, P1, P2 serta hipotesis H1–H3. Semuanya **masih rancangan untuk dibahas dengan dosen**, belum disetujui, belum diimplementasikan, dan belum ada hasil eksperimen. Langkah praktis terdekat: pilot kecil untuk menguji H1.
