# Catatan Literatur Hari 4 — Evaluasi *Zero-Shot* (`BEIR`) & Diagnostik Psikolinguistik Negasi (`BERT`)

Dokumen ini merangkum **2 literatur inti terakhir (Hari 4)** sehingga seluruh **12 referensi primer** penelitian telah lengkap terpetakan untuk penulisan **Section I–V** pada **Hari 5**.

---

## 1. Thakur dkk. (2021) — *BEIR: A Heterogeneous Benchmark for Zero-Shot Evaluation of Information Retrieval Models* (NeurIPS Datasets & Benchmarks 2021)
- **Sitasi Siap Pakai IEEE (`5` penulis -> ditulis lengkap):**
  > N. Thakur, N. Reimers, A. Rücklé, A. Srivastava, and I. Gurevych, "BEIR: A heterogeneous benchmark for zero-shot evaluation of information retrieval models," in *Proc. Neural Inf. Process. Syst. Track Datasets Benchmarks (NeurIPS)*, vol. 1, Dec. 2021, pp. 1–16.
- **Poin Metodologis Kunci untuk Section II, III.B, & III.D:**
  - Menetapkan standar de facto evaluasi *zero-shot information retrieval* lintas domain menggunakan format korpus JSONL (`_id`, `title`, `text`), `queries.jsonl`, dan tabel relevansi `qrels` yang diadopsi langsung dalam repositori ini.
  - Menjustifikasi pemilihan **`nDCG@10`** sebagai metrik kualitas peringkat utama pada daftar pendek rekomendasi literatur ilmiah (didampingi **`Recall@10`** untuk cakupan penemuan *gold paper* dan **`VR@10`** untuk kepatuhan preferensi negatif).

---

## 2. Ettinger (2020) — *What BERT Is Not: Lessons from a New Suite of Psycholinguistic Diagnostics for Language Models* (*TACL*)
- **Sitasi Siap Pakai IEEE (`1` penulis):**
  > A. Ettinger, "What BERT is not: Lessons from a new suite of psycholinguistic diagnostics for language models," *Trans. Assoc. Comput. Linguistics*, vol. 8, pp. 34–48, Jan. 2020.
- **Poin Metodologis Kunci untuk Section I, II, & IV.C (Error Analysis Kategori 1):**
  - Membuktikan melalui uji psikolinguistik terkontrol bahwa representasi kontekstual keluarga BERT (yang menjadi fondasi `SciBERT` dan `SPECTER2`) sangat tidak peka terhadap operator negasi (`not`, `never`, `excluding`): distribusi prediksi dan kemiripan semantiknya tetap didominasi oleh asosiasi kata benda argumen yang dinegasikan.
  - Menjelaskan secara mekanistik temuan empiris **TABLE II** dan **Studi Kasus Kategori 1 (`TABLE IV`)**: ketika pengguna menyisipkan `"..., excluding <Q_neg>"` pada metode `M2`, lapisan *self-attention* justru memusatkan bobot pada token topik `<Q_neg>` ke dalam vektor `[CLS]`, sehingga dokumen pelanggar melonjak ke puncak peringkat (`VR@10` naik dari `0.1987` ke `0.3625`). Sebaliknya, pemisahan kanal enkoding pada `M3` membebaskan model dari kegagalan komposisi negasi internal Transformer.
