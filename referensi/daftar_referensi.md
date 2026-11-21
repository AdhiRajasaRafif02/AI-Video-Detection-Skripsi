# Daftar Referensi

PDF paper tidak disimpan di repo (hak cipta). Simpan tautan dan catatan di sini. Lokasi rinci (bagian, persamaan, halaman) untuk argumen projector ada di `docs/02_riset_projector_2026-11-21.md`, Q6.

## A. Metode utama dan pengembangan perceptual straightening

| Referensi | Relevansi | Tautan |
|---|---|---|
| Internò dkk. (2025). AI-Generated Video Detection via Perceptual Straightening (ReStraV). NeurIPS 2025 | **Metode dasar.** Pers. 1 (distance/curvature) Bag. 3 hlm. 4; DINOv2 CLS+patch Bag. 4 hlm. 5; 21 fitur + MLP 64→32 Bag. 6 hlm. 6. Tidak menguji kompresi | https://arxiv.org/abs/2507.00583 · kode: https://github.com/ChristianInterno/ReStraV |
| Hénaff, Goris & Simoncelli (2019). Perceptual straightening of natural videos. Nature Neuroscience 22, 984–991 | Asal konsep curvature lintasan | https://www.nature.com/articles/s41593-019-0377-4 |
| Papougiannakis & Tous (2026). Benchmarking and Improving Perceptual Straightening for Robust AI-Generated Video Detection. Electronics 15(15), 3245 | Rich384, GurAI; menyebut uji kompresi FFmpeg multi-CRF sebagai pekerjaan lanjutan (celah skripsi) | https://www.mdpi.com/2079-9292/15/15/3245 |
| Wang dkk. (2026). MTOR | ReStraV sebagai baseline di 12 gangguan (termasuk H.264 QP 23/30/38); rata-rata AP 87,91%, rentang 16,38 pp | https://arxiv.org/html/2610.06378 |
| Cakiroglu dkk. (2026). Where the Real Clips Came From | Audit kebocoran panjang klip/codec; peringatan protokol re-encode dan FPS. Angka ReStraV 0,85→0,60 hanya dari cuplikan pencarian, **verifikasi di PDF** | https://arxiv.org/html/2606.31004 |
| Cao dkk. (2026). TRACE | "Consistency" antar-frame via DiT, bukan pasangan bersih–terkompresi (pembeda) | https://arxiv.org/pdf/2609.25775 |
| Niu, Savin & Simoncelli (2024). Learning predictable and robust neural representations by straightening image sequences. NeurIPS | Straightening ↔ ketahanan noise | https://proceedings.neurips.cc/paper_files/paper/2024/hash/473c578564ed9fe8041abfef772d37db-Abstract.html |
| Harrington dkk. (2022/2023). Exploring perceptual straightness in learned visual representations | Model robust → lintasan lebih lurus | https://nips.cc/virtual/2022/62203 |

## B. Deteksi video/gambar AI dan kompresi

| Referensi | Relevansi | Tautan |
|---|---|---|
| Vahdati dkk. (2024). Beyond Deepfake Images: Detecting AI-Generated Videos (MISLnet). CVPRW, 4397–4408 | H.264 recompression CRF 0–40; pembanding augmentasi | https://arxiv.org/abs/2404.15955 |
| Ma dkk. (2026). AIGVDBench | Benchmark, generalisasi lintas generator | https://arxiv.org/abs/2601.11035 |
| Tan dkk. (2026). VideoVeritas | Pendekatan multimodal berat (pembanding arah) | https://arxiv.org/abs/2602.08828 |
| Raza dkk. (2026). Review deepfake detection | Review metode | https://www.mdpi.com/2673-2688/7/2/68 |
| Ojha, Li & Lee (2023). Towards Universal Fake Image Detectors (UnivFD). CVPR | Linear layer + BCE di atas CLIP beku (Bag. 4 hlm. 5) | https://arxiv.org/abs/2302.10174 |
| Wang dkk. (2020). CNN-generated images are surprisingly easy to spot… for now. CVPR | Augmentasi blur/JPEG | https://arxiv.org/abs/1912.11035 |
| Le & Woo (2022). ADD. AAAI | Distillation untuk deepfake terkompresi | https://ojs.aaai.org/index.php/AAAI/article/view/19886 |
| Le & Woo (2023). QAD. ICCV | Quality-agnostic deepfake detection | https://arxiv.org/abs/2309.05911 |
| Tao dkk. (2025). ODDN. AAAI | Data berpasangan/tidak berpasangan, kompresi OSN | https://ojs.aaai.org/index.php/AAAI/article/view/32063 |
| ComGenVid (dataset kandidat) | 1.700 MSVD real, 1.700 Sora, 1.700 Veo3 | https://huggingface.co/datasets/OmerXYZ/comgenvid · paper: https://arxiv.org/abs/2603.15026 |

## C. Projector, konsistensi, contrastive

| Referensi | Relevansi | Tautan |
|---|---|---|
| Chen dkk. (2020). SimCLR. ICML | Projection head (Bag. 2.1 hlm. 2), output 128 (Bag. 2.3), NT-Xent (Pers. 1) | https://arxiv.org/abs/2002.05709 |
| Grill dkk. (2020). BYOL. NeurIPS | Istilah "projector"; kedalaman 1 = linear (Lamp. F.1 Tabel 14 hlm. 25) | https://arxiv.org/abs/2006.07733 |
| Radford dkk. (2021). CLIP. ICML | Proyeksi linear saja (Bag. 2.3 hlm. 4; Gbr. 3 hlm. 5) | https://arxiv.org/abs/2103.00020 |
| Chen & He (2021). SimSiam. CVPR | Collapse tanpa stop-gradient (Bag. 4.1, Gbr. 2) | https://arxiv.org/abs/2011.10566 |
| Bardes, Ponce & LeCun (2022). VICReg. ICLR | Variance term anti-collapse (Pers. 1); invariance saja collapse (Tabel 7) | https://arxiv.org/abs/2105.04906 |
| Hadsell, Chopra & LeCun (2006). DrLIM. CVPR | Contrastive loss asli (Pers. 4); alasan negatif = anti-collapse; Siamese (Bag. 3.1) | https://doi.org/10.1109/CVPR.2006.100 |
| Bromley dkk. (1993). Siamese TDNN. NIPS 6, 737–744 | Asal arsitektur Siamese | https://proceedings.neurips.cc/paper/1993/hash/288cc0ff022877bd3df94bc9360b9c5d-Abstract.html |
| Khosla dkk. (2020). Supervised Contrastive Learning. NeurIPS | Contrastive berlabel (Bag. 3, Pers. 2) | https://arxiv.org/abs/2004.11362 |
| Wang & Isola (2020). Alignment and Uniformity. ICML | Contrastive = alignment + uniformity (Teorema 1) | https://arxiv.org/abs/2005.10242 |
| Zheng dkk. (2016). Stability Training. CVPR | L₀ + α·L_stability (Pers. 2), L2 (Pers. 8), JPEG (Bag. 4.2) | https://arxiv.org/abs/1604.04326 |
| Laine & Aila (2017). Temporal Ensembling (Π-model). ICLR | CE + selisih kuadrat (Bag. 2.1, Alg. 1); membantu juga full-label (Tabel 1–2) | https://arxiv.org/abs/1610.02242 |
| Hendrycks dkk. (2020). AugMix. ICLR | CE + JS consistency (Pers. 1); ablasi Tabel 4 | https://arxiv.org/abs/1912.02781 |
| Gao dkk. (2021/2024). CLIP-Adapter. IJCV | Adapter ringan di atas backbone beku, residual | https://arxiv.org/abs/2110.04544 |
| AdapterTune (2026) | Inisialisasi agar mulai tepat di fungsi pretrained | https://arxiv.org/html/2603.14706 |
| Kulis (2013). Metric Learning: A Survey. FnTML 5(4), 287–364 | Metrik Mahalanobis = transformasi linear (Bag. 2, hlm. 292) | https://doi.org/10.1561/2200000019 |

## D. Dasar umum

| Referensi | Relevansi |
|---|---|
| Goodfellow, Bengio & Courville (2016). Deep Learning. MIT Press | Sigmoid/BCE (6.2.2.2), backprop (6.5), early stopping (7.8), parameter sharing (7.9) — https://www.deeplearningbook.org |
| Rumelhart, Hinton & Williams (1986). Nature 323, 533–536 | Backpropagation — https://doi.org/10.1038/323533a0 |
| Kingma & Ba (2015). Adam. ICLR | Optimizer — https://arxiv.org/abs/1412.6980 |
| Hanley & McNeil (1982). Radiology 143, 29–36 | Makna AUROC (peluang peringkat benar) |
| Fawcett (2006). An introduction to ROC analysis. Pattern Recognition Letters 27, 861–874 | Pengantar ROC |
| Trefethen & Bau (1997). Numerical Linear Algebra. SIAM, Lecture 6 | Definisi projector P² = P |
| Oquab dkk. (2023). DINOv2 | Encoder visual — https://arxiv.org/abs/2304.07193 |

## E. Belum dikerjakan: studi literatur metode berbasis vektor (tugas dosen, untuk Bab 1)
