# Draft Siap Pakai `TABLE IV` — Standar IEEE Conference Paper (Section IV.C, Double-Column Span)

> **Catatan Aturan Template IEEE:** Tabel studi kasus kualitatif ini menggunakan tata letak melebar 2 kolom (*double-column `table*`*), judul kapital di atas tabel, **tanpa garis vertikal**, dan desimal diawali angka nol (`0.xxxx`).

**TABLE IV**
**QUALITATIVE CASE STUDIES ACROSS THREE DIAGNOSTIC CATEGORIES ON THE HELD-OUT LITSEARCH-NEG TEST SPLIT (β* = 0.2)**

| Category & Query ID | Positive Query (`Q_pos`) & Excluded Concept (`Q_neg`) | Target Gold Paper (Jaccard Overlap) | Gold Rank (`M1 → M2 → M3`) | `VR@10` (`M1 → M2 → M3`) | Key Geometric & Retrieval Observation |
| :--- | :--- | :--- | :---: | :---: | :--- |
| **Category 1**<br>*M2 Negation Blindness Rescued by M3*<br>(`q_586`) | **`Q_pos`:** Which paper found that using common character encodings and ciphers, or even just convincing the model that it is not communicating in natural language, can bypass the safety guardrails of large models?<br>**`Q_neg`:** `machine translation` | *GPT-4 IS TOO SMART TO BE SAFE: STEALTHY CHAT WITH LLMS VIA CIPHER* (`Jaccard = 0.0452`) | `#2 → #4 →` **`#1`** | `0.20 → 0.40 →` **`0.00`** | Appending `'excluding machine translation'` in `M2` triggers negation blindness, pulling `machine translation` violators into Rank #1 and dropping the gold paper from #2 to #4; Dual-Input `M3` clears violations (`VR@10 = 0.00`) and promotes the gold paper to **#1**. |
| **Category 2**<br>*Flexible Multi-Clause Low Lexical Overlap*<br>(`q_317`) | **`Q_pos`:** What research has been conducted on applying contrastive techniques to distinguish normal from abnormal imagery for the creation of radiology reports?<br>**`Q_neg`:** `table-to-text` | *Contrastive Attention for Automatic Chest X-ray Report Generation* (`Jaccard = 0.0309`) | `#4 → #21 →` **`#2`** | `0.10 → 0.10 →` **`0.00`** | Despite minimal surface lexical overlap (`Jaccard = 0.0309`, `21` words), asymmetric `SPECTER2` + `M3` maps the multi-clause description to **#2** while suppressing `table-to-text` distractors (`VR@10`: `0.10 → 0.00`). |
| **Category 3**<br>*Boundary Case / Latent Over-Penalization*<br>(`q_580`) | **`Q_pos`:** Which paper first showed that task-specific knowledge embedded in parameters can be extracted from one LLM using seed samples and transferred to another via parameter-efficient fine-tuning?<br>**`Q_neg`:** `meta-learning` | *SEEKING NEURAL NUGGETS: KNOWLEDGE TRANSFER IN LARGE LANGUAGE MODELS FROM A PARAMETRIC PERSPECTIVE* (`Jaccard = 0.0312`) | `#7 → #8 →` **`#10`** | `0.20 → 0.20 →` **`0.10`** | Although `meta-learning` is lexically absent from the gold paper, high latent dense similarity (`S_neg = 0.8150` vs. `S_pos = 0.7747`) causes fixed `β* = 0.2` to nudge the gold paper from #7 to #10 while still reducing `VR@10` (`0.20 → 0.10`). |

### Distribusi Populasi Kategori pada Held-Out Test Split (`N = 384`)
- **Category 1** (Negation Blindness on M2 Rescued by M3 (VR_M2 >= VR_M1, VR_M3 < VR_M2, nDCG_M3 > nDCG_M2)): **`78` kueri (`20.31%`)** — kasus terpilih: `q_586`.
- **Category 2** (Long Multi-Clause Query (>=20 words, Jaccard <= 0.14) Retrieved in Top-3 by M3): **`49` kueri (`12.76%`)** — kasus terpilih: `q_317`.
- **Category 3** (Latent Semantic Over-Penalization Boundary (nDCG_M3 < nDCG_M1)): **`39` kueri (`10.16%`)** — kasus terpilih: `q_580`.
