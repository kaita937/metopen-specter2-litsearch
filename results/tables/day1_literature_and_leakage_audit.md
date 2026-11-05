# Catatan Literatur Hari 1 & Audit Kebocoran Data (*Data Leakage Audit*)

Dokumen ini merangkum **3 literatur inti Hari 1** serta **verifikasi kronologis kebocoran data** antara model `SPECTER2` dan benchmark `LitSearch` sebagai bahan penulisan **Section I**, **Section II**, **Section III.B–C**, dan **Section IV.C (*Limitations*)** pada naskah IEEE Conference Paper.

---

## 1. Ringkasan 3 Literatur Hari 1

### 1.1. Ajith dkk. (2024) — *LitSearch: A Retrieval Benchmark for Scientific Literature Search* (EMNLP 2024)
- **Sitasi IEEE (`>= 6` penulis -> `et al.`):**
  > A. Ajith *et al.*, "LitSearch: A retrieval benchmark for scientific literature search," in *Proc. Conf. Empirical Methods Natural Lang. Process. (EMNLP)*, Nov. 2024, pp. 15068–15083.
- **Poin Metodologis Kunci untuk Section I & III.B:**
  - Menyediakan korpus `64.183` artikel ilmiah Ilmu Komputer, *Machine Learning*, dan *Natural Language Processing* (`corpus_clean`, diekstrak dari S2ORC) serta `597` kueri pencarian literatur realistis.
  - Kueri terbagi atas dua tingkat kekhususan (`specificity=0` untuk kueri konseptual luas/*broad* dan `specificity=1` untuk kueri spesifik) serta bersumber dari kutipan dalam teks (*inline-citation*) maupun kueri yang ditulis langsung oleh peneliti ahli (*author-written*).
  - Menunjukkan bahwa kueri pencarian literatur memiliki kalimat panjang yang fleksibel dan minim kesamaan kata permukaan (*lexical overlap*) terhadap judul paper target, sehingga metode leksikal seperti BM25 tertinggal signifikan dibanding *dense retrieval*.

### 1.2. Singh dkk. (2023) — *SciRepEval: A Multi-Format Benchmark for Scientific Document Representations* (EMNLP 2023)
- **Sitasi IEEE (`5` penulis -> ditulis lengkap):**
  > A. Singh, M. D'Arcy, A. Cohan, D. Downey, and S. Feldman, "SciRepEval: A multi-format benchmark for scientific document representations," in *Proc. Conf. Empirical Methods Natural Lang. Process. (EMNLP)*, Dec. 2023, pp. 5548–5566.
- **Poin Metodologis Kunci untuk Section II & III.C:**
  - Memperkenalkan keluarga model **SPECTER2** (`allenai/specter2_base`, berdimensi `768` di atas *backbone* SciBERT) dengan modul **task-specific adapters** ringan berbasis Bottleneck Adapter.
  - Menetapkan skema **enkoding asimetris (*asymmetric encoding*)** untuk temu kembali literatur:
    1. Kueri pendek/deskriptif (`Q_pos` maupun `Q_neg`) wajib dipetakan menggunakan adapter **`allenai/specter2_adhoc_query`**.
    2. Dokumen artikel (`Title + [SEP] + Abstract`, batas panjang `512` token) wajib dipetakan menggunakan adapter **`allenai/specter2` (`proximity`)**, diikuti *pooling* token `[CLS]` lapisan terakhir dan normalisasi L2.

### 1.3. Wang, Macdonald, & Ounis (2022) — *Improving Zero-Shot Retrieval Using Dense External Expansion* (*IPM*)
- **Sitasi IEEE (`3` penulis -> ditulis lengkap):**
  > X. Wang, C. Macdonald, and I. Ounis, "Improving zero-shot retrieval using dense external expansion," *Inf. Process. Manage.*, vol. 59, no. 5, p. 103026, Sep. 2022.
- **Poin Metodologis Kunci untuk Section I & II:**
  - Menjelaskan fenomena *vocabulary mismatch* dan pergeseran distribusi representasi ketika sistem *dense retrieval* diterapkan secara *zero-shot* tanpa pelatihan ulang (*fine-tuning*) pada kueri pencarian yang kompleks.
  - Memperkuat motivasi mengapa pengendalian arah vektor pada saat inferensi (*test-time vector steering* seperti metode usulan `M3`) sangat bernilai praktis tanpa memerlukan biaya pelatihan ulang model.

---

## 2. Hasil Audit Kebocoran Data (*Data Leakage Audit* — Wajib Dilaporkan di Section IV.C)

| Aspek Pemeriksaan | Fakta Kronologis & Teknis Terverifikasi | Status Kebocoran |
| :--- | :--- | :--- |
| **Pelatihan Base Model & Adapter `SPECTER2`** | Dilatih pada koleksi tugas *SciRepEval* (sitasi S2ORC & log pencarian pra-2023) dan dipublikasikan pada **EMNLP Desember 2023**. | — |
| **Kurasi Kueri Evaluasi `LitSearch` (`597` kueri)** | Dikurasi dari makalah konferensi ML/NLP mutakhir (2023–2024) serta kueri peneliti baru, lalu dipublikasikan pada **EMNLP November 2024** (`arXiv:2407.18940`). | **Bebas Kebocoran Supervisi Kueri (`0%` *Supervised Query Leakage*)** |
| **Teks Abstrak Korpus Pra-2023 (`corpus_clean`)** | Sebagian artikel dasar di dalam `64.183` dokumen merupakan literatur ML/NLP pra-2023 di S2ORC yang graf sitasinya mungkin pernah terlihat saat *pre-training* umum SPECTER2, namun **tidak satu pun pasangan `(kueri LitSearch, gold paper)` maupun `(Q_pos, Q_neg)` pernah terlihat selama pelatihan adapter**. | **Transparan & Sah untuk Evaluasi *Zero-Shot*** |
| **Kalibrasi Hyperparameter `β*` pada `M3`** | Pemilihan `β*` dikunci murni pada **Validation Split (`20%`)**, terpisah ketat dari **Held-Out Test Split (`80%`)** tempat metrik akhir dilaporkan. | **Bebas *Test-Set Hyperparameter Leakage* (`0%`)** |
