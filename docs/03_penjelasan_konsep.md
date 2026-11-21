# Penjelasan Konsep (Bahasa Sederhana)

Kumpulan penjelasan atas pertanyaan mahasiswa selama diskusi (13–21 Nov 2026). Dipakai untuk memahami dan menjelaskan ke dosen.

## 1. Alur ReStraV dari frame sampai prediksi

1. **Ambil frame:** 24 frame dari 2 detik video, diubah ukurannya menjadi 224 × 224.
2. **DINOv2 (dibekukan):** gambar dipotong menjadi patch 14 × 14 piksel. Setiap patch menjadi vektor 384 angka, ditambah satu token CLS yang merangkum seluruh gambar. Semuanya melewati 12 lapisan transformer (attention). Keluarannya: CLS 384 angka = "sidik jari" frame = **zᵢ**.
3. **(Usulan) Projector:** uᵢ = W zᵢ.
4. **Langkah:** Δuᵢ = uᵢ₊₁ − uᵢ → 23 vektor langkah.
5. **Distance:** dᵢ = ‖Δuᵢ‖ → 23 angka (seberapa besar perubahan antar-frame).
6. **Curvature:** θᵢ = sudut antara Δuᵢ dan Δuᵢ₊₁ → 22 angka (seberapa tajam lintasan berbelok). Contoh 2-D: (0,0)→(1,0)→(2,0) = 0°; (0,0)→(1,0)→(1,1) = 90°.
7. **21 fitur:** 7 distance + 6 curvature + mean/var/min/max masing-masing → MLP 21→64→32→1 → skor real/AI.

Catatan: paper menulis 196 patch, padahal 224/14 = 16 sehingga seharusnya 256 patch. Cek kode resmi kalau memakai patch.

## 2. Vektor dan geometri

- **Vektor** = deretan angka yang mewakili satu frame, ibarat koordinat titik di peta (di sini 384 angka).
- **Geometri** = bentuk rute yang terbentuk dari titik-titik berurutan: jarak antar-titik (distance) dan belokan (curvature).
- Vektor muncul setelah DINOv2; geometri muncul setelah selisih antar-frame dihitung.

## 3. Projector

- **Bahasa mudah:** lapisan penyaring / pengubah sudut pandang. Analogi: bayangan benda yang disorot senter ke dinding, di mana arah senter menentukan sisi mana yang terlihat.
- **Bentuk:** satu lapisan linear di PyTorch (`nn.Linear(384, 384, bias=False)`), terpisah dari DINOv2. Hasil training disimpan sebagai file bobot.
- **Matematis:** uᵢ = W zᵢ. Setiap angka di u adalah campuran berbobot ("resep") dari 384 angka di z. Karena hanya selisih yang dipakai, ini setara dengan mempelajari metrik M = WᵀW (lihat `02_riset_projector_2026-11-21.md`, Q1).
- **Contoh angka (3 → 2 dimensi):** z₃ = komponen yang bergetar karena kompresi. Video terkompresi (1,0,0,4)→(2,0,−0,4)→(3,0,0,4) punya curvature ±77°. Dengan takaran acak awal (z₃ ikut 0,5), curvature ±50°. Setelah training (takaran z₃ → 0), curvature kembali 0°, sama dengan video aslinya.
- **Kenapa namanya projector:** mengikuti istilah projection head (SimCLR), projector (BYOL), linear projection (CLIP). Isinya tetap neuron biasa.

## 4. Loss dan apa yang terjadi padanya

- **Loss total** = L_BCE(asli) + L_BCE(CRF) + λ·L_konsistensi, berupa satu angka "rapor".
- Loss **tidak** diteruskan ke classifier; ia dipakai untuk menghitung gradien (backpropagation) lalu memperbarui bobot.
- **Yang berubah:** bobot projector dan classifier. **Yang tidak berubah:** DINOv2 (frozen), data, rumus geometri.
- **Jalur gradien:** loss klasifikasi → classifier → fitur → u → projector (memperbarui keduanya). Loss konsistensi → fitur geometri → u → projector (memperbarui projector saja).
- **λ** mengatur seberapa keras projector dipaksa membuat geometri mirip. Terlalu kecil: tidak peduli kompresi. Terlalu besar: membuang informasi. Dipilih lewat validation set.
- Kenapa dua loss: kalau hanya "harus mirip", projector bisa curang dengan membuang semua informasi (collapse). Loss klasifikasi mencegahnya.

## 5. Bobot dan backpropagation

- **Bobot** = angka-angka di dalam modul yang diubah saat training (seperti tombol mixer audio). Projector 384×384 = 147.456 bobot; classifier ±3.500; DINOv2 ±21 juta (tidak diubah).
- **Backpropagation** = menelusuri dari loss ke belakang untuk menghitung "kalau bobot ini diubah sedikit, loss berubah seberapa" (aturan rantai, mengalikan pengaruh tiap mata rantai). Lalu bobot baru = bobot lama − learning rate × gradien.
- Di PyTorch: `loss.backward(); optimizer.step(); optimizer.zero_grad()`. Pastikan geometri ditulis dengan operasi torch dan `arccos` diberi `clamp`.

## 6. Proses training sampai projector "siap"

1. Data dibagi train 70% / validation 15% / test 15%, per identitas video.
2. Per batch (misalnya 64 pasang): forward → loss → backprop → update. Satu putaran seluruh data training = 1 epoch.
3. Setelah setiap epoch: evaluasi di validation (AUROC asli dan terkompresi, drift geometri).
4. **Early stopping:** simpan bobot setiap ada rekor validation baru. Berhenti kalau tidak ada rekor selama ±5 epoch. Pakai bobot epoch terbaik.
5. Pilih λ (misalnya 0,1 / 1 / 10) dan ukuran k lewat validation.
6. Kunci bobot, lalu uji **sekali** di test set pada semua kondisi (asli, CRF 0–40). Test set tidak boleh dipakai untuk tuning.

Proses yang sama berlaku untuk semua model B0–P4 supaya perbandingannya adil.

## 7. Model B dan P

- **B = Baseline** (pembanding), **P = Proposed** (usulan).
- B0: ReStraV asli, dilatih dengan data asli saja. B1: + augmentasi kompresi.
- P1: + perhitungan geometri yang lebih tahan (multi-skala, median), tanpa komponen baru yang dilatih (opsional).
- P2: projector, hanya loss klasifikasi. P3: projector + konsistensi (usulan utama). P4: projector + contrastive (ablasi, ditambahkan 21 Nov).
- Perbandingan kunci: P3 vs P2 (efek konsistensi), P3 vs B1 (vs cara standar), P3 vs P4 (perlu contrastive atau tidak).
- Training dan testing **bukan** eksperimen terpisah. Keduanya tahap yang dilalui setiap model.

## 8. AUROC

- Detektor mengeluarkan skor. AUROC = peluang video AI acak diberi skor "AI" lebih tinggi daripada video asli acak.
- 1,0 = sempurna; 0,5 = setara lempar koin. Tidak bergantung pada pilihan threshold.

## 9. CRF

- Pengaturan kualitas encoder H.264 (libx264). Angka makin besar → kompresi makin kuat → kualitas makin turun. Bukan persen dan bukan bitrate.
- CRF 0 bukan "video sumber". Itu tetap hasil re-encode (bisa lossless terhadap input encoder).
- QP (dipakai MTOR) ≠ CRF.

## 10. Kenapa kompresi diduga merusak ReStraV

- Noise per-frame membuat curvature tertarik ke ±120° untuk kedua kelas (untuk noise acak, cos antar-langkah ≈ −½). Akibatnya gap real–AI menyempit, dan video asli (awalnya lurus) paling terdampak.
- Video dengan gerak kecil paling rentan, karena langkahnya pendek dibanding noise.
- H.264 juga punya fluktuasi kualitas antar jenis frame (I > P > B; ipratio 1,40, pbratio 1,30 secara default), yang diduga menambah jitter. Besarnya perlu diukur.
- Kode simulasi: `eksperimen/simulasi_noise_curvature/`.
