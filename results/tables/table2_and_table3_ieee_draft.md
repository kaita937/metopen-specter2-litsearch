# Draft Siap Pakai `TABLE II` & `TABLE III` — Standar IEEE Conference Paper (Section IV.B)

> **Catatan Aturan Template IEEE:** Judul tabel diletakkan **di atas tabel** dengan huruf kapital (`TABLE II`, `TABLE III`), **tanpa garis vertikal**, dan seluruh desimal diawali angka nol (`0.xxxx`).

---

**TABLE II**
**RETRIEVAL PERFORMANCE COMPARISON ACROSS THREE STRATEGIES ON THE HELD-OUT LITSEARCH-NEG TEST SPLIT (N = 384)**

| Strategy | Method Description | nDCG@10 (↑) | Recall@10 (↑) | VR@10 (↓) |
| :---: | :--- | :---: | :---: | :---: |
| **M1** | Standard Positive-Only (`β = 0.0`) | 0.2361 ± 0.3537 | **0.3591** ± 0.4736 | 0.1987 ± 0.1459 |
| **M2** | Naive Prompt Negation (`"<Q_pos>, excluding <Q_neg>"`) | 0.1563 ± 0.2868 | 0.2661 ± 0.4376 | 0.3625 ± 0.2510 |
| **M3** | **Proposed Dual-Input Penalized Cosine (`β* = 0.2`)** | **0.2378** ± 0.3555 | 0.3565 ± 0.4728 | **0.0828** ± 0.0986 |

*Catatan: Nilai disajikan dalam format Mean ± SD pada `N = 384` kueri Held-Out Test Split (`80%`), sementara `β* = 0.2` dikunci murni dari Validation Split terpisah (`20%`, `N = 96`). Seluruh skor `nDCG@10` dan `Recall@10` tervalidasi silang terhadap `pytrec_eval` dengan selisih absolut maksimum `0.000000` (`<= 1e-6`).*

---

**TABLE III**
**PAIRED WILCOXON SIGNED-RANK SIGNIFICANCE TESTS WITH BONFERRONI CORRECTION ON THE HELD-OUT TEST SPLIT (N = 384, α_bonf = 0.025)**

| Pairwise Comparison | Metric | Mean Δ | Rel. Change | Improved / Tied / Degraded | Wilcoxon W | Raw p-value | Bonferroni p-value | Effect Size (r) | Sig. (p_bonf < 0.025) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **M3 vs. M1** | nDCG@10 (↑) | +0.0016 | +0.69% | 56 / 289 / 39 | 2,096.5 | 4.956e-01 | 9.911e-01 | 0.0348 | No (Preserved) |
| **M3 vs. M1** | Recall@10 (↑) | −0.0026 | −0.73% | 10 / 363 / 11 | 110.0 | 8.273e-01 | 1.000e+00 | 0.0111 | No (Preserved) |
| **M3 vs. M1** | **VR@10 (↓)** | **−0.1159** | **−58.32%** | **265 / 119 / 0** | **0.0** | **5.092e-47** | **1.018e-46** | **0.7349** | **Yes** |
| **M3 vs. M2** | **nDCG@10 (↑)** | **+0.0815** | **+52.16%** | **96 / 264 / 24** | **975.5** | **3.552e-12** | **7.103e-12** | **0.3549** | **Yes** |
| **M3 vs. M2** | **Recall@10 (↑)** | **+0.0904** | **+33.95%** | **46 / 329 / 9** | **270.0** | **3.069e-06** | **6.137e-06** | **0.2381** | **Yes** |
| **M3 vs. M2** | **VR@10 (↓)** | **−0.2797** | **−77.16%** | **320 / 64 / 0** | **0.0** | **1.962e-54** | **3.924e-54** | **0.7928** | **Yes** |
