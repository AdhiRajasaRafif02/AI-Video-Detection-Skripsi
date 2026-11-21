# Riset Projector untuk Menjawab Dosen (21 November 2026)

> Disalin dari Claude Doc "Riset Projector untuk Menjawab Dosen". Dokumen ini menjawab enam pertanyaan dosen pada bimbingan 21 Nov 2026 tentang projector. Setiap klaim diberi lokasinya di referensi (bagian, persamaan, tabel, halaman). Nomor halaman mengikuti PDF arXiv; kalau dosen memakai versi prosiding, nomor halamannya bisa berbeda, tetapi nomor bagian dan persamaannya tetap sama. Angka simulasi di Q3 berasal dari **data sintetis**, bukan hasil skripsi.

## Ringkasan jawaban

| Pertanyaan dosen | Jawaban singkat |
|---|---|
| Kenapa dinamakan projector padahal neuron biasa? | Memang neuron biasa: satu lapisan linear. Nama "projector" menyebut **fungsinya**, mengikuti SimCLR, BYOL, dan CLIP. Secara matematis, lapisan ini **mempelajari metrik** (cara mengukur jarak dan sudut) tempat geometri ReStraV dihitung. |
| Bentuknya berapa node? | Usulan utama: 384 masukan → 384 keluaran, tanpa bias, diinisialisasi matriks identitas, sehingga titik awalnya persis ReStraV (147.456 bobot). Varian 128/256 keluaran dipilih lewat validation set. |
| Bagaimana efek sebelum dan sesudahnya? | Sebelum: geometri diukur dengan jarak Euclid biasa di ruang DINOv2. Sesudah: geometri diukur dengan metrik yang dipelajari. Efeknya diukur lewat 5 metrik (drift fitur, gap curvature real–AI, curvature video asli, spektrum bobot, AUROC per CRF). Simulasi sintetis menunjukkan drift turun ±40%. |
| Apakah works tanpa contrastive loss? | Secara teori bisa, karena fungsi "negatif" pada contrastive loss adalah mencegah *collapse*, dan di desain ini fungsi itu dipegang oleh loss klasifikasi. Ada preseden kuat (Stability Training, Π-model, AugMix). Tetapi di simulasi, varian kontrastif lebih tahan pada kompresi berat, jadi **varian kontrastif (P4) diusulkan ditambahkan sebagai ablasi**. |
| Apakah ini contrastive learning? | Bukan. Ini *supervised learning* dengan *consistency regularization* berpasangan (arsitektur Siamese, bobot dibagi). Contrastive learning mensyaratkan pasangan negatif. |
| Referensi matematisnya? | Tabel Q6 memetakan setiap rumus ke sumbernya, lengkap dengan bagian dan persamaannya. |

## Q1. Kenapa dinamakan projector padahal neuron biasa?

### Jawaban inti

Dosen benar: projector **memang neuron biasa**, yaitu satu lapisan linear (*fully connected*) berisi k neuron. Setiap neuron menerima 384 angka dari DINOv2 dan mengeluarkan 1 angka. Nama "projector" tidak merujuk ke jenis neuron baru, tetapi ke **peran** lapisan itu, yaitu memetakan (memproyeksikan) representasi encoder ke ruang lain tempat suatu loss atau perhitungan dilakukan.

### Bukti bahwa ini konvensi penamaan di literatur

| Paper | Apa yang dikatakan | Lokasi |
|---|---|---|
| SimCLR (Chen dkk., 2020) | *Projection head* g(·) adalah jaringan saraf kecil yang memetakan representasi ke ruang tempat contrastive loss diterapkan, dengan rumus z = W⁽²⁾σ(W⁽¹⁾h) | Bagian 2.1, hlm. 2 |
| BYOL (Grill dkk., 2020) | Menyebut lapisan ini **projector** dan keluarannya "projection" z = g(y) | Bagian 3.1, hlm. 4; arsitektur di Bagian 3.3, hlm. 5 |
| BYOL (Grill dkk., 2020) | Projector dengan kedalaman 1 berarti **satu lapisan linear** | Lampiran F.1, Tabel 14, hlm. 25 |
| CLIP (Radford dkk., 2021) | Hanya memakai **proyeksi linear** untuk memetakan representasi encoder ke ruang embedding bersama. Matriksnya ditulis W_i dan W_t, "learned proj" | Bagian 2.3, hlm. 4; Gambar 3, hlm. 5 |
| UnivFD (Ojha dkk., 2023), deteksi gambar AI | Satu lapisan linear + sigmoid di atas fitur CLIP yang **dibekukan**, dilatih dengan binary cross-entropy | Bagian 4, hlm. 5 |

Jadi menyebut "satu lapisan linear yang memetakan fitur encoder beku ke ruang baru" sebagai *projector* atau *projection* adalah penamaan yang lazim. Polanya juga sudah dipakai di deteksi konten AI (Ojha dkk., 2023).

### Argumen yang lebih kuat: projector = mempelajari metrik geometri

Ini bagian yang membuat projector **bukan sekadar neuron tambahan**. Ia mengubah cara ReStraV mengukur jarak dan sudut.

1. Projector: uᵢ = W zᵢ + b.
2. ReStraV hanya memakai **selisih** antar-frame: Δuᵢ = uᵢ₊₁ − uᵢ = W(zᵢ₊₁ − zᵢ) = W Δzᵢ. **Bias b hilang**, jadi bias tidak berpengaruh sama sekali dan sebaiknya tidak dipakai.
3. Distance setelah projector: dᵢ = ‖WΔzᵢ‖ = √(Δzᵢᵀ M Δzᵢ), dengan **M = WᵀW**.
4. Curvature setelah projector: cos θᵢ = (Δzᵢᵀ M Δzᵢ₊₁) / (√(ΔzᵢᵀMΔzᵢ) · √(Δzᵢ₊₁ᵀMΔzᵢ₊₁)).

Artinya, distance dan curvature ReStraV kini dihitung memakai **metrik Mahalanobis** M yang dipelajari, bukan jarak Euclid biasa. Bentuk d(x, y) = √((x−y)ᵀM(x−y)) dengan M = WᵀW adalah inti *metric learning* linear: belajar jarak sama dengan belajar transformasi linear W lalu memakai jarak Euclid (Kulis, 2013, Bagian 2 "Distance Learning via Linear Transformations", mulai hlm. 292).

Akibat penting:

- **Kalau W hanya memutar ruang** (matriks ortogonal, WᵀW = I), maka M = I dan geometri tidak berubah sama sekali. Jadi efek projector hanya datang dari **meregangkan atau menyusutkan arah-arah tertentu** di ruang embedding.
- Itulah mekanisme yang diinginkan: arah yang paling diganggu kompresi disusutkan, arah yang membedakan real/AI dipertahankan atau diregangkan.
- Rumus distance dan curvature ReStraV (Internò dkk., 2025, Persamaan 1, Bagian 3, hlm. 4) **tidak diubah**. Yang dipelajari hanya metriknya.

### Catatan kejujuran istilah

Dalam aljabar linear yang ketat, "projector" adalah matriks persegi P dengan P² = P (idempoten), seperti bayangan dari bayangan yang sama dengan bayangan itu sendiri (Trefethen & Bau, 1997, Lecture 6). Projector yang dilatih tidak dijamin memenuhi P² = P. Jadi istilah "projector" di sini mengikuti **konvensi deep learning** (SimCLR, BYOL, CLIP), bukan definisi aljabar yang ketat.

**Saran penamaan di skripsi:** "lapisan proyeksi linear terlatih (*projector*), yang secara ekuivalen mempelajari metrik Mahalanobis M = WᵀW untuk mengukur geometri lintasan".

**Kalimat siap pakai:** *"Secara bentuk, projector adalah satu lapisan neuron linear. Namanya mengikuti istilah projection head di SimCLR, projector di BYOL, dan linear projection di CLIP. Yang membuatnya bermakna bagi ReStraV: karena geometri hanya memakai selisih antar-frame, lapisan ini ekuivalen dengan mempelajari metrik M = WᵀW yang dipakai untuk mengukur distance dan curvature."*

## Q2. Bentuknya berapa node, dan dasar pemilihannya

### Usulan ukuran

| Bagian | Ukuran | Jumlah bobot | Dasar |
|---|---|---|---|
| Input projector | 384 (CLS token DINOv2 ViT-S/14) | — | DINOv2 ViT-S/14 menghasilkan token CLS 384 dimensi (Internò dkk., 2025, Bagian 4, hlm. 5) |
| **Projector utama** | 384 → 384, tanpa bias, inisialisasi W = I | 384 × 384 = **147.456** | Lihat alasan di bawah |
| Varian projector | 384 → k, k ∈ {64, 128, 256}, tanpa bias | 384 × k (k = 128 → 49.152) | Ukuran keluaran 128 di SimCLR (Bagian 2.3, hlm. 3) dan 256 di BYOL (Bagian 3.3, hlm. 5) |
| Classifier | MLP 21 → 64 → 32 → 1 | ± 3.500 | Sama dengan ReStraV (Internò dkk., 2025, Bagian 6, hlm. 6) |
| DINOv2 ViT-S/14 | dibekukan | ± 21 juta, tidak dilatih | — |

**Koreksi dari penjelasan sebelumnya:** sebelumnya tertulis 49.280 bobot (384 × 128 ditambah 128 bias). Karena bias hilang saat menghitung selisih antar-frame (lihat Q1), bias tidak berguna. Jumlah yang benar untuk versi 128 adalah **49.152**.

### Alasan memilih 384 → 384 dengan inisialisasi identitas sebagai versi utama

1. **Titik awalnya persis ReStraV.** Dengan W = I, uᵢ = zᵢ, sehingga semua distance dan curvature sama dengan ReStraV asli. Model P2/P3 mulai dari baseline, dan setiap perubahan performa berasal dari apa yang dipelajari. Pendekatan "mulai tepat dari fungsi pretrained" juga dipakai pada adapter untuk backbone beku: AdapterTune memakai inisialisasi nol agar jaringan yang diadaptasi dimulai tepat di fungsi pretrained (arXiv 2603.14706, Abstrak), dan CLIP-Adapter memakai koneksi residual untuk mencampur fitur asli dengan fitur hasil adaptasi (Gao dkk., 2021, Bagian 3.2).
2. **Tidak membuang informasi sejak awal.** Mengecilkan dimensi ke 128 berarti rank M ≤ 128, sehingga ada arah yang dipaksa hilang sebelum training. Dengan 384 → 384, keputusan membuang arah diserahkan ke proses training.
3. **Tetap ringan.** 147 ribu bobot masih sekitar 0,7% dari DINOv2, dan dilatih dari embedding yang sudah disimpan.

### Bagaimana menentukan k secara adil

Ukuran k adalah hiperparameter. Jangan ditentukan dari hasil test. Coba k ∈ {64, 128, 256, 384}, pilih berdasarkan AUROC validation, lalu laporkan semuanya sebagai ablasi. Kalau hasilnya hampir sama, pilih yang paling kecil.

## Q3. Efek sebelum dan sesudah projector

### Secara konsep

| | Sebelum projector (ReStraV asli) | Sesudah projector |
|---|---|---|
| Ruang tempat geometri dihitung | Embedding DINOv2 apa adanya | Embedding setelah dikali W |
| Cara mengukur jarak dan sudut | Jarak Euclid: semua arah dianggap sama penting | Metrik M = WᵀW: arah tertentu diregangkan atau disusutkan |
| Arah yang diganggu kompresi | Ikut terhitung penuh di distance dan curvature | Disusutkan bobotnya, jika training berhasil |
| Rumus distance dan curvature | Persamaan 1 ReStraV | Sama persis, inputnya u, bukan z |
| Bobot yang dilatih sebelum geometri | Tidak ada, sehingga fitur 21-D tetap dan tidak bisa diperbaiki | W, sehingga fitur 21-D bisa "dibentuk" oleh training |

Baris terakhir adalah **alasan logis kenapa projector harus ada**. Tanpa lapisan yang bisa dilatih sebelum geometri, 21 fitur ReStraV adalah angka tetap. Akibatnya, loss konsistensi yang dihitung dari fitur itu tidak bisa menurunkan apa pun, karena gradiennya ke parameter mana pun bernilai nol.

### Lima ukuran untuk menunjukkan efeknya di skripsi

1. **Drift fitur:** ‖f(asli) − f(CRF)‖ yang sudah distandardisasi, untuk video yang sama, dihitung tanpa dan dengan projector. Turun berarti geometri lebih stabil.
2. **Gap curvature real–AI** per tingkat CRF: rata-rata mean curvature video AI dikurangi video asli. Lebih besar berarti kelasnya lebih mudah dipisah.
3. **Mean curvature video asli per CRF.** Ini menguji H1, yaitu apakah kompresi membuat video asli terlihat "berbelok seperti AI" dan apakah projector menahannya.
4. **Spektrum M = WᵀW** (eigenvalue atau panjang kolom W): menunjukkan arah mana yang disusutkan dan arah mana yang diregangkan.
5. **AUROC per tingkat CRF** untuk B0, B1, P2, P3, P4.

### Simulasi sintetis (ilustrasi, bukan bukti)

Kode: `eksperimen/simulasi_projector/`. Aturan simulasi:

- Lintasan buatan berdimensi 64 dengan 24 frame. Video "asli" berbelok ±22° per langkah, video "AI" ±38°. Panjang langkahnya acak, untuk meniru video dengan gerak kecil dan gerak besar.
- "Kompresi" ditiru dengan noise per frame yang **kuat di dimensi 32–63 dan lemah (30%) di dimensi 0–31**. Frame non-I diberi noise 1,3× lebih besar untuk meniru frame B. Ada 5 tingkat kompresi. Training memakai L1–L3, sedangkan L4–L5 tidak pernah dilihat.
- Model, fitur 21-D, MLP 21→64→32→1, Adam, dan early stopping mengikuti rancangan skripsi. Projector 64→64 tanpa bias dengan inisialisasi identitas. λ dipilih dari {0,1; 1; 10} lewat validation set.

AUROC rata-rata 3 seed (data test; L4–L5 tidak dipakai saat training):

| Model | Asli | L1 | L2 | L3 | L4 | L5 |
|---|---|---|---|---|---|---|
| B0 ReStraV asli | 0,957 | 0,831 | 0,667 | 0,593 | 0,551 | 0,535 |
| B1 + augmentasi | 0,926 | 0,933 | 0,863 | 0,727 | 0,619 | 0,599 |
| P2 projector tanpa konsistensi | 0,923 | 0,923 | 0,849 | 0,730 | 0,628 | 0,594 |
| P3 projector + konsistensi | 0,920 | 0,933 | 0,884 | 0,753 | 0,632 | 0,595 |
| P4 projector + kontrastif | 0,921 | 0,925 | 0,896 | 0,811 | 0,680 | 0,616 |

Simpangan baku antar-seed sekitar 0,01–0,02, kecuali P4 di L3–L5 sekitar 0,03–0,05.

Efek projector P3 (rata-rata 3 seed, data test):

| Ukuran | Tanpa projector | Dengan projector P3 |
|---|---|---|
| Panjang kolom W, dimensi "bersih" 0–31 | 1,000 (awal) | 1,172 (diregangkan) |
| Panjang kolom W, dimensi "bising" 32–63 | 1,000 (awal) | 0,845 (disusutkan) |
| Drift fitur di L3 | 3,81 | 2,23 (turun 41%) |
| Mean curvature video asli di L3 (aslinya 22,0°) | 62,1° | 54,4° |
| Gap curvature AI−real di L3 (di kondisi asli 16,4°) | 7,0° | 8,2° |

### Cara membaca hasil ini dengan jujur

- **Mekanismenya terbukti di simulasi.** Projector belajar sendiri meregangkan arah yang bersih dan menyusutkan arah yang bising. Hasilnya, drift turun, kenaikan curvature video asli tertahan, dan gap real–AI lebih terjaga. Ini sesuai dengan penjelasan Q1 (metrik M).
- **Tetapi efek tambahan dari loss konsistensi kecil.** P2 (projector tanpa loss konsistensi) sudah belajar pola yang mirip: panjang kolom 1,18 vs 0,86 dan drift L3 sekitar 2,15. Keunggulan P3 atas P2 dan B1 hanya sekitar 0,02–0,03 AUROC di L2–L3, dan hampir nol di L5.
- **Varian kontrastif (P4) lebih tahan di kompresi berat**, tetapi variasinya antar-seed lebih besar (±0,04). Temuan ini relevan langsung dengan pertanyaan Q4.
- **Ada harga yang dibayar.** Semua model yang dilatih dengan data terkompresi turun sekitar 0,03 AUROC pada kondisi asli dibanding B0. Trade-off ini wajib dilaporkan.
- Simulasi ini memakai asumsi noise buatan. Hasil pada DINOv2 dan video asli bisa lebih besar, lebih kecil, atau nol. **Jangan dikutip sebagai hasil skripsi.** Fungsinya hanya menunjukkan bahwa desainnya masuk akal dan bisa diukur.

## Q4. Apakah works tanpa contrastive loss?

### Langkah 1: apa sebenarnya fungsi bagian "negatif" di contrastive loss?

Contrastive loss punya dua bagian: menarik pasangan yang sama agar dekat, dan mendorong pasangan berbeda (negatif) agar jauh.

- **Hadsell, Chopra & LeCun (2006)**, paper yang memperkenalkan contrastive loss, menulis tepat setelah Persamaan 4 (Bagian 2.1, ±hlm. 3) bahwa meminimalkan jarak pasangan yang mirip saja biasanya menghasilkan *collapsed solution*, karena loss bisa dibuat nol dengan membuat fungsi pemetaannya konstan. **Bagian negatif ada untuk mencegah collapse.**
- **Wang & Isola (2020)** membuktikan bahwa contrastive loss mengoptimalkan dua hal: *alignment* (pasangan positif dekat, Bagian 4.1.1) dan *uniformity* (fitur tersebar, Bagian 4.1.2). Buktinya ada di Teorema 1 (Bagian 4.2, Persamaan 2). Di Bagian 5 mereka menunjukkan bahwa kalau alignment terlalu dominan, semua input dipetakan ke vektor yang sama (degenerate).

**Kesimpulan langkah 1:** pertanyaan "apakah works tanpa contrastive loss" sama dengan pertanyaan **"apakah ada mekanisme lain yang mencegah collapse?"**

### Langkah 2: di desain ini, apa yang mencegah collapse?

**Loss klasifikasi (binary cross-entropy) dengan label real/AI.** Kalau projector collapse (semua video dipetakan ke geometri yang sama), classifier tidak bisa membedakan real dan AI, sehingga loss klasifikasi naik. Jadi solusi collapse **bukan** minimum dari loss total. Label menjalankan peran yang di contrastive learning dipegang oleh pasangan negatif.

Perbedaan pentingnya: SimCLR dan BYOL bekerja **tanpa label**, sehingga butuh mekanisme anti-collapse khusus. Penelitian ini **punya label**, sehingga mekanisme itu sudah tersedia.

### Langkah 3: preseden bahwa "klasifikasi + konsistensi tanpa negatif" bekerja

| Paper | Bentuk loss | Bukti bekerja | Lokasi |
|---|---|---|---|
| Stability Training (Zheng dkk., 2016) | L₀ + α·L_stability, dengan L_stability = jarak fitur versi asli dan versi terdistorsi. **Tanpa negatif.** | Distorsi yang diuji termasuk **kompresi JPEG**. Performa data asli dipertahankan atau naik (ranking 7.312 → 7.368) | Persamaan 2–3 di Bagian 3.1; L2 untuk embedding di Persamaan 8, Bagian 3.3; jenis distorsi di Bagian 4.2; hasil di Tabel 3–4 |
| Π-model (Laine & Aila, 2017) | Cross-entropy + w(t)·selisih kuadrat dua keluaran dari input yang sama. **Tanpa negatif.** | Menurunkan error **bahkan saat semua data berlabel**: CIFAR-10 7,33% → 6,90% | Bagian 2.1 dan Algoritma 1; Bagian 3.4, Tabel 1–2 |
| AugMix (Hendrycks dkk., 2020) | Cross-entropy + λ·Jensen-Shannon antar-versi augmentasi. **Tanpa negatif.** | Menambah loss JS menurunkan error CIFAR-10-C 13,1% → 11,2% | Persamaan 1, Bagian 3, hlm. 4; Tabel 4, Bagian 4.3, hlm. 8 |
| BYOL dan SimSiam (tanpa label) | Hanya menarik pasangan positif. **Tanpa negatif.** | Bekerja, asalkan ada mekanisme anti-collapse: momentum encoder untuk BYOL, stop-gradient untuk SimSiam | BYOL Abstrak, hlm. 1; SimSiam Bagian 4.1 dan Gambar 2, hlm. 3: tanpa stop-gradient akurasinya 0,1% (collapse) |
| VICReg (Bardes dkk., 2022) | Invariance (MSE) + variance + covariance. **Tanpa negatif.** | Invariance saja collapse, tetapi dengan suku variance tidak | Persamaan 1 dan 5, Bagian 4.1, hlm. 4–5; Tabel 7, Lampiran D.4, hlm. 18 |

Desain ini paling mirip dengan **Stability Training** dan **Π-model**: loss supervised ditambah suku konsistensi berupa jarak kuadrat antara versi bersih dan versi terganggu.

### Langkah 4: jawaban jujur berdasarkan simulasi

- Di simulasi (Q3), P3 tanpa contrastive loss **bekerja**: tidak collapse, dan sedikit lebih baik dari B1 dan P2 di tingkat kompresi yang dilatih.
- Tetapi **P4 (dengan contrastive loss) lebih tahan di kompresi berat** pada 2 dari 3 seed. Penjelasannya konsisten dengan Wang & Isola (2020): P3 hanya mengoptimalkan alignment, sedangkan P4 menambahkan uniformity yang menjaga agar geometri antar-video tetap tersebar dan informatif.
- Jadi jawaban yang paling kuat untuk dosen bukan "pasti works tanpa contrastive", melainkan: **"secara teori bisa dan ada presedennya, dan saya akan membuktikannya dengan membandingkan P3 (tanpa contrastive) dan P4 (dengan contrastive) secara langsung."**

### Pengaman tambahan dalam desain

1. **Konsistensi dihitung pada fitur yang distandardisasi** memakai rata-rata dan simpangan baku fitur versi asli di batch itu, dan statistik itu tidak ikut diturunkan gradiennya (*detach*). Alasannya: distance ikut membesar atau mengecil kalau W diskalakan, sedangkan curvature tidak (sudut tidak berubah oleh skala). Tanpa standardisasi, projector bisa "curang" menurunkan loss konsistensi hanya dengan mengecilkan skala W. Ini penalaran dari rumus. Di simulasi, kecurangan skala tidak muncul dalam 40 epoch, tetapi pengaman ini tetap murah.
2. **Pantau simpangan baku fitur per batch**, mengikuti ide suku variance di VICReg (Bagian 4.1). Kalau mendekati nol, itu tanda collapse.

## Q5. Apakah ini contrastive learning?

**Bukan.** Alasannya ada pada definisi.

### Definisi contrastive learning menurut sumber primernya

- **Hadsell dkk. (2006), Persamaan 4, Bagian 2.1:** loss terdiri dari suku pasangan mirip (ditarik) **dan** suku pasangan tidak mirip (didorong sampai margin m).
- **SimCLR (Chen dkk., 2020), Persamaan 1 (NT-Xent), Bagian 2.1, hlm. 2:** untuk setiap pasangan positif, 2(N−1) contoh lain di minibatch diperlakukan sebagai **negatif**.
- **Supervised Contrastive / SupCon (Khosla dkk., 2020), Bagian 3 dan Persamaan 2:** versi berlabel. Semua sampel sekelas menjadi positif, kelas lain menjadi negatif.

Ciri wajibnya: **ada pasangan negatif yang didorong menjauh.**

### Posisi desain ini

| Metode | Pasangan positif | Pasangan negatif | Label | Termasuk |
|---|---|---|---|---|
| SimCLR | 2 augmentasi gambar yang sama | Ya | Tidak | Contrastive (self-supervised) |
| SupCon | Sampel sekelas | Ya | Ya | Contrastive (supervised) |
| BYOL, SimSiam | 2 augmentasi gambar yang sama | Tidak | Tidak | Non-contrastive (self-supervised) |
| Stability Training, Π-model | Versi asli dan versi terganggu | Tidak | Ya | Supervised + consistency regularization |
| **P3 (usulan)** | Video asli dan versi H.264-nya | **Tidak** | Ya | **Supervised + consistency regularization** |
| P4 (ablasi baru) | Video asli dan versi H.264-nya | Ya, video lain di batch | Ya | Supervised + contrastive |

### Istilah yang tepat untuk skripsi

- *Supervised learning* dengan *paired consistency regularization*.
- Arsitekturnya **Siamese**: dua cabang dengan bobot yang sama memproses versi asli dan versi terkompresi. Istilah Siamese berasal dari Bromley dkk. (1993, NIPS, hlm. 737), yaitu dua sub-jaringan identik yang digabung di keluarannya. Hadsell dkk. (2006, Bagian 3.1, ±hlm. 4) mendefinisikannya sebagai dua salinan fungsi G_W yang memakai parameter W yang sama.
- Dalam kerangka Wang & Isola (2020), P3 hanya memakai suku **alignment** ditambah loss klasifikasi.

### Kenapa wajar kalau dosen mengira ini contrastive

Istilah "projector" memang paling sering muncul di paper contrastive dan self-supervised (SimCLR, BYOL). Tetapi projector hanyalah lapisan, sedangkan contrastive atau tidaknya ditentukan oleh **loss**, bukan oleh adanya projector. CLIP memakai proyeksi linear dengan loss contrastive, sedangkan UnivFD (Ojha dkk., 2023) memakai lapisan linear di atas fitur beku dengan loss binary cross-entropy biasa.

**Kalimat siap pakai:** *"Ini bukan contrastive learning, karena tidak ada pasangan negatif. Yang saya pakai adalah supervised learning dengan regularisasi konsistensi berpasangan, mengikuti Stability Training (Zheng dkk., 2016) dan Π-model (Laine & Aila, 2017). Sebagai pembanding, saya menambahkan varian contrastive (P4) untuk menguji apakah pasangan negatif memberi manfaat tambahan."*

## Q6. Referensi untuk matematisnya, per rumus

Urutan rumus mengikuti alur model, dari frame sampai evaluasi. Untuk rumus turunan, langkah penurunannya ditulis lengkap supaya bisa dicek sendiri.

| # | Rumus | Arti | Jenis dasar | Referensi dan lokasi |
|---|---|---|---|---|
| 1 | zᵢ = CLS DINOv2(frameᵢ) ∈ ℝ³⁸⁴ | Embedding per frame | Diambil | Internò dkk. (2025), Bagian 4, hlm. 5 (DINOv2 ViT-S/14, token CLS 384 + patch, blok 11). Skripsi memakai varian CLS saja, yang perlu ditulis sebagai varian |
| 2 | Δzᵢ = zᵢ₊₁ − zᵢ; dᵢ = ‖Δzᵢ‖₂; θᵢ = arccos(Δzᵢ·Δzᵢ₊₁ / (‖Δzᵢ‖‖Δzᵢ₊₁‖)) | Distance dan curvature lintasan | Diambil | Internò dkk. (2025), **Persamaan 1**, Bagian 3, hlm. 4. Asal konsepnya: Hénaff dkk. (2019), "Estimating perceptual curvature" dan Gambar 1 |
| 3 | uᵢ = W zᵢ, W ∈ ℝ^(k×384) | Projector (lapisan linear) | Diambil | Goodfellow dkk. (2016), Bab 6; proyeksi linear terlatih di CLIP, Gambar 3, hlm. 5; projector linear di BYOL, Lampiran F.1, Tabel 14, hlm. 25 |
| 4 | Δuᵢ = WΔzᵢ (bias hilang) | Kenapa bias tidak dipakai | Diturunkan | (Wzᵢ₊₁ + b) − (Wzᵢ + b) = W(zᵢ₊₁ − zᵢ) |
| 5 | dᵢ = √(Δzᵢᵀ M Δzᵢ), M = WᵀW | Projector = metrik Mahalanobis | Diturunkan dan dirujuk | ‖WΔz‖² = (WΔz)ᵀ(WΔz) = ΔzᵀWᵀWΔz. Kesetaraan "belajar metrik Mahalanobis = belajar transformasi linear": Kulis (2013), Bagian 2, mulai hlm. 292 |
| 6 | W ortogonal ⇒ geometri tidak berubah | Projector harus meregangkan atau menyusutkan arah agar berefek | Diturunkan | Jika WᵀW = I maka M = I, sehingga dᵢ dan θᵢ sama dengan Persamaan 1 |
| 7 | f = [d₁..d₇, θ₁..θ₆, mean/var/min/max(d), mean/var/min/max(θ)] ∈ ℝ²¹ | 21 fitur | Diambil | Internò dkk. (2025), Bagian 6, hlm. 6 |
| 8 | p = σ(MLP(f)), MLP 21→64→32→1 | Classifier | Diambil | Internò dkk. (2025), Bagian 6, hlm. 6 (MLP 64→32); unit sigmoid untuk keluaran biner: Goodfellow dkk. (2016), Subbab 6.2.2.2 |
| 9 | L_BCE = −[y log p + (1−y) log(1−p)] | Loss klasifikasi | Diambil | Goodfellow dkk. (2016), Subbab 6.2.2.2; dipakai untuk real-vs-fake di atas fitur beku oleh Ojha dkk. (2023), Bagian 4, hlm. 5 |
| 10 | L_cons = ‖f̃(asli) − f̃(CRF)‖², f̃ = fitur terstandardisasi | Loss konsistensi | Diambil (bentuk) + desain (standardisasi) | Jarak L2 fitur asli vs terdistorsi: Zheng dkk. (2016), Persamaan 8, Bagian 3.3. Selisih kuadrat sebagai suku konsistensi: Laine & Aila (2017), Bagian 2.1, Algoritma 1, catatan kaki 1. MSE sebagai suku invariance: Bardes dkk. (2022), Persamaan 5. Standardisasi dengan statistik yang di-detach adalah keputusan desain yang diturunkan dari rumus 5–6 (lihat Q4) |
| 11 | L = L_BCE(asli) + L_BCE(CRF) + λ·L_cons | Loss total | Diambil + disesuaikan | Pola "loss tugas + α·loss stabilitas": Zheng dkk. (2016), Persamaan 2, Bagian 3.1. **Perbedaan yang disengaja:** Zheng dkk. tidak menghitung loss tugas pada input terdistorsi (Bagian 3.1, setelah Persamaan 3), sedangkan desain ini juga menghitung L_BCE(CRF), agar P3 dan B1/P2 mendapat paparan data terkompresi yang sama sehingga selisih P3 dan P2 murni berasal dari L_cons. Augmentasi kompresi dirujuk ke Wang dkk. (2020) dan Vahdati dkk. (2024) |
| 12 | Varian P4: L_InfoNCE antar-pasangan dalam batch | Ablasi contrastive | Diambil | NT-Xent, SimCLR Persamaan 1, Bagian 2.1, hlm. 2; dasar teori Wang & Isola (2020), Teorema 1, Bagian 4.2 |
| 13 | ∂L/∂W lewat aturan rantai | Backpropagation | Diambil | Rumelhart dkk. (1986), Nature 323, hlm. 533–536; Goodfellow dkk. (2016), Subbab 6.5 |
| 14 | W, MLP ← Adam(∇L) | Optimizer | Diambil | Kingma & Ba (2015), Algoritma 1 |
| 15 | Berhenti saat AUROC validation tidak naik | Early stopping | Diambil | Goodfellow dkk. (2016), Subbab 7.8 |
| 16 | Bobot projector sama untuk versi asli dan versi CRF | Arsitektur Siamese | Diambil | Bromley dkk. (1993), hlm. 737; Hadsell dkk. (2006), Bagian 3.1; Goodfellow dkk. (2016), Subbab 7.9 |
| 17 | AUROC = P(skor video AI > skor video asli) | Metrik evaluasi | Diambil | Hanley & McNeil (1982), Abstrak; Fawcett (2006) sebagai pengantar ROC |

### Catatan tentang nomor halaman

- Nomor halaman paper arXiv (SimCLR, BYOL, CLIP, AugMix, SimSiam, VICReg, UnivFD) diambil dari PDF arXiv. Sebagian dibaca dari header halaman hasil ekstraksi; cocokkan sekali lagi saat menulis daftar pustaka.
- Untuk Hadsell dkk. (2006), Zheng dkk. (2016), dan Laine & Aila (2017), nomor **bagian dan persamaan** sudah pasti, sedangkan nomor halamannya diberi tanda ± atau tidak dicantumkan.
- Subbab buku Goodfellow dkk. (2016) mengikuti versi daring deeplearningbook.org.

## Usulan perubahan desain setelah masukan dosen

Semuanya masih rancangan untuk dibahas dengan dosen.

| # | Perubahan | Alasan |
|---|---|---|
| 1 | Projector **tanpa bias** | Bias hilang saat menghitung selisih antar-frame (Q6 rumus 4) |
| 2 | Versi utama 384 → 384 dengan **inisialisasi identitas**; varian k ∈ {64, 128, 256} sebagai ablasi | Titik awal persis ReStraV, dan tidak ada arah yang dibuang sebelum training (Q2) |
| 3 | Loss konsistensi pada **fitur terstandardisasi** (statistik versi asli, di-detach) | Mencegah projector menurunkan loss lewat skala (Q4) |
| 4 | **Tambah P4**: projector + loss contrastive (InfoNCE) antar-pasangan | Menjawab pertanyaan dosen secara empiris (Q3, Q4) |
| 5 | **Laporkan 5 ukuran sebelum/sesudah** | Menunjukkan efek projector secara terukur (Q3) |
| 6 | Pantau simpangan baku fitur per batch | Deteksi dini collapse (VICReg, Bagian 4.1) |

### Susunan model yang diperbarui

| Kode | Isi | Yang diuji |
|---|---|---|
| B0 | ReStraV asli, dilatih dengan data asli saja | Seberapa besar dampak kompresi |
| B1 | B0 + augmentasi kompresi | Cara perbaikan standar |
| P2 | Projector + BCE (asli dan CRF) | Efek kapasitas projector saja |
| P3 | P2 + konsistensi berpasangan | **Usulan utama**, tanpa contrastive |
| P4 | P2 + contrastive berpasangan | Apakah pasangan negatif memberi manfaat tambahan |

Perbandingan kunci: **P3 vs P2** (efek konsistensi), **P3 vs B1** (vs cara standar), dan **P3 vs P4** (perlu contrastive atau tidak).

### Risiko yang tetap harus diakui

- Pada data asli, keunggulan P3 bisa kecil atau nol. Di simulasi saja keunggulannya hanya sekitar 0,02–0,03 AUROC.
- Semua model yang dilatih dengan data terkompresi bisa sedikit turun pada video asli.
- Projector hanya bisa menyusutkan arah yang "bising". Kalau noise kompresi berada di arah yang sama dengan sinyal real/AI, informasi itu sudah hilang dan tidak bisa dipulihkan.

## Daftar pustaka

Lihat `referensi/daftar_referensi.md`.
