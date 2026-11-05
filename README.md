# Controllable Scientific Literature Retrieval with Negative Query Preferences Using Asymmetric SPECTER2 Embeddings on LitSearch

Repositori ini berisi implementasi eksperimental kuantitatif komparatif dan purwarupa sistem pencarian artikel ilmiah berpreferensi negatif pada dataset **LitSearch** (`princeton-nlp/LitSearch`, EMNLP 2024) menggunakan representasi padat asimetris **SPECTER2** (`allenai/specter2_base`).

## Desain Eksperimen Utama
- **1 Variabel Bebas Kategoris (Strategi Retrieval):**
  1. **`M1` — *Standard Positive-Only* (`β = 0`)**: Pencarian vektor standar yang hanya membaca kueri positif (`Q_pos`).
  2. **`M2` — *Naive Prompt Negation***: Penggabungan kata negasi langsung ke satu kalimat kueri (`"<Q_pos>, excluding <Q_neg>"`) untuk menguji fenomena *negation blindness*.
  3. **`M3` — *Proposed Dual-Input Penalized Cosine Scoring***: Pemisahan enkoding `Q_pos` dan `Q_neg` melalui skor $S_{\text{final}}(d) = S_{\text{pos}}(d) - \beta \cdot S_{\text{neg}}(d)$.
- **1 Hyperparameter (`M3`):** Koefisien penalti negatif `β ∈ {0.0, 0.1, ..., 1.0}` (`11` nilai uji), dikalibrasi secara terpisah khusus pada **Validation Split (`20%`)** sebelum pengujian akhir satu kali pada **Held-Out Test Split (`80%`)**.
- **3 Metrik Evaluasi Utama:** `nDCG@10`, `Recall@10` (tervalidasi silang terhadap `pytrec_eval`, selisih `<= 1e-6`), dan `VR@10` (*Violation Rate @ 10*).

## Spesifikasi Lingkungan Lokal (Terkunci)
- **OS:** Windows (`PowerShell`)
- **Python:** `3.12.8`
- **Paket Inti:** `torch==2.15.0+cpu`, `transformers==4.57.6`, `adapters==1.3.0`, `datasets==5.0.1`, `pytrec-eval-terrier==0.5.11`
- **GPU Encoding (Hari 2):** Google Colab NVIDIA Tesla T4 (`FP16` + *length-sorted dynamic batching*)

## Cara Menjalankan Ulang (Reproducibility)

### 1. Persiapan Lingkungan Virtual
```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### 2. Hari 1 — Unduh, Validasi Dataset LitSearch, & Eksplorasi Token (`TABLE I`)
```powershell
.\.venv\Scripts\python.exe -m src.data_loader
```
Selanjutnya jalankan seluruh sel pada `notebooks/01_eksplorasi_litsearch.ipynb` untuk menghasilkan:
- `results/tables/table1_dataset_stats.csv` (bahan utama **TABLE I** IEEE)
- `results/tables/litsearch_query_breakdown.csv`
- `results/tables/dataset_overview_summary.json`
- `results/figures/eda_token_length_dist.png` & `.pdf` (`300 DPI`)

### 3. Hari 2 — Asymmetric GPU Encoding SPECTER2 (Colab T4) & Validasi `LitSearch-Neg`
1. Unggah `notebooks/02_encoding_specter2_colab.ipynb` ke Google Colab (`Runtime -> Change runtime type -> T4 GPU`), lalu klik **Runtime -> Run all**.
2. Unduh arsip `day2_colab_outputs.zip` dari Google Drive (`MyDrive/metopen_specter2_litsearch_day2/day2_colab_outputs.zip`) dan ekstrak langsung di akar folder proyek ini.
3. Jalankan verifikasi integritas akhir Hari 2 di lokal:
```powershell
.\.venv\Scripts\python.exe -m src.negative_benchmark --verify-only
```

### 4. Hari 3 — Kalibrasi `β*` (Val 20%), Komparasi `M1–M3` & Uji Wilcoxon (Test 80%)
Jalankan seluruh sel pada `notebooks/03_evaluasi_dual_input.ipynb` untuk menghasilkan:
- `embeddings/calibrated_config.json` (nilai terkunci `β*` dari Validation Split `N = 96`)
- `results/tables/beta_sweep_val.csv` & `results/figures/fig2_beta_calibration.png`/`.pdf` (**Fig. 2** `300 DPI`)
- `results/tables/metrics_test.csv` (**TABLE II** pada Held-Out Test Split `N = 384`)
- `results/tables/wilcoxon.csv` (**TABLE III** uji *Paired Wilcoxon* terkoreksi Bonferroni `α_bonf = 0.025`)

### 5. Hari 4 — Error Analysis Kualitatif (`TABLE IV`), Diagram `Fig. 1` & Produk Aplikasi Pencarian
```powershell
.\.venv\Scripts\python.exe -m src.error_analysis
.\.venv\Scripts\python.exe -m src.plot_architecture_fig1
.\.venv\Scripts\streamlit.exe run demo/app.py
```
Luaran yang dihasilkan:
- `results/error_analysis/category_summary.csv`, `selected_cases.csv`, dan `table4_ieee_draft.md` (**TABLE IV**)
- `results/figures/fig1_dual_input_architecture.png` & `.pdf` (**Fig. 1** vektor *double-column* `7.16 in`, `300 DPI`)
- Antarmuka produk pencarian artikel ilmiah dua-masukan pada `demo/app.py` (`β* = 0.2` otomatis tanpa slider, respons `< 10 ms` di CPU)
