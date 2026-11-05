# Catatan Literatur Hari 3 — *Negation Blindness* (`M2`), *Penalized Cosine Scoring* (`M3`), & Kestabilan Ruang Embedding

Dokumen ini merangkum **3 literatur inti Hari 3** sebagai landasan teoretis dan metodologis untuk **Section I**, **Section II (*Related Work*)**, **Section III.C–D (*Retrieval Strategies & Statistical Protocol*)**, serta **Section IV.A–B (*Calibration of `β*` & Main Statistical Comparison*)** pada naskah IEEE Conference Paper.

---

## 1. Weller, Lawrie, & Van Durme (2024) — *NevIR: Negation in Neural Information Retrieval* (EACL 2024)
- **Sitasi Siap Pakai IEEE (`3` penulis -> ditulis lengkap):**
  > O. Weller, D. Lawrie, and B. Van Durme, "NevIR: Negation in neural information retrieval," in *Proc. 18th Conf. Eur. Chapter Assoc. Comput. Linguistics (EACL)*, Mar. 2024, pp. 2274–2287.
- **Poin Metodologis Kunci untuk Section I, II, & IV.B:**
  - Membuktikan secara komprehensif bahwa arsitektur *bi-encoder*/*dense retrieval* berbasis Transformer gagal memahami operator negasi leksikal (`"not"`, `"excluding"`, `"without"`) ketika disisipkan langsung ke dalam satu kalimat kueri (`M2`).
  - Alih-alih menjauhkan vektor kueri dari konsep yang dinegasikan, kemunculan kata benda topik setelah kata `"excluding"` justru meningkatkan kemiripan kosinus terhadap dokumen pelanggar (*negation blindness*), sehingga membenarkan pemisahan eksplisit kanal `Q_pos` dan `Q_neg` pada metode usulan `M3`.

---

## 2. Rocchio (1971) — *Relevance Feedback in Information Retrieval* (*SMART Retrieval System*)
- **Sitasi Siap Pakai IEEE (`1` penulis):**
  > J. J. Rocchio, "Relevance feedback in information retrieval," in *The SMART Retrieval System—Experiments in Automatic Document Processing*, G. Salton, Ed. Englewood Cliffs, NJ, USA: Prentice-Hall, 1971, pp. 313–323.
- **Poin Metodologis Kunci untuk Section II & III.C:**
  - Memberikan fondasi matematis klasik penggeseran vektor kueri menjauhi sub-ruang dokumen tidak relevan melalui operasi selisih linier $\vec{q}_{\text{opt}} = \alpha \vec{q}_0 - \beta \vec{q}_{\text{neg}}$.
  - Karena sifat kelinearan hasil kali titik (*dot product*) terhadap vektor dokumen ter-normalisasi L2 $\hat{\mathbf{d}}$, pemeringkatan menggunakan vektor selisih $\hat{\mathbf{q}}_{\text{pos}} - \beta \hat{\mathbf{q}}_{\text{neg}}$ secara aljabar ekuivalen dengan fungsi *Penalized Cosine Scoring* `M3`:
    $$S_{M3}(d) = \hat{\mathbf{q}}_{\text{pos}} \cdot \hat{\mathbf{d}} - \beta \left(\hat{\mathbf{q}}_{\text{neg}} \cdot \hat{\mathbf{d}}\right)$$
    yang beroperasi secara *zero-shot* tanpa perlu melatih ulang bobot encoder `SPECTER2`.

---

## 3. Ballester & Penner (2022) — *Robustness, Replicability and Scalability in Topic Modelling* (*J. Informetrics*)
- **Sitasi Siap Pakai IEEE (`2` penulis -> ditulis lengkap):**
  > O. Ballester and O. Penner, "Robustness, replicability and scalability in topic modelling," *J. Informetrics*, vol. 16, no. 1, p. 101224, Feb. 2022.
- **Poin Metodologis Kunci untuk Section III.D, IV.A, & IV.B:**
  - Menekankan pentingnya protokol replikabilitas, pemisahan data kalibrasi parameter dari data evaluasi akhir, serta pengujian signifikansi non-parametrik pada representasi semantik korpus ilmiah.
  - Menopang desain metodologis penelitian ini dalam:
    1. Mengkalibrasi hyperparameter tunggal `β*` murni pada **Validation Split (`20%`, `N = 96`)** sebelum diuji satu kali pada **Held-Out Test Split (`80%`, `N = 384`)**.
    2. Menggunakan **Paired Wilcoxon signed-rank test** dengan koreksi **Bonferroni (`α_bonf = 0.05 / 2 = 0.025`)** karena distribusi metrik peringkat per kueri (`nDCG@10`, `Recall@10`, `VR@10`) bersifat terbatas pada interval `[0, 1]` dan tidak berdistribusi normal.
