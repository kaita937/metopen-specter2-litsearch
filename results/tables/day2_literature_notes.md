# Catatan Literatur Hari 2 — Evolusi Representasi Semantik Dokumen Ilmiah (`SciBERT` → `SPECTER` → `SPECTER2`)

Dokumen ini merangkum **4 literatur inti Hari 2** sebagai bahan langsung penulisan **Section I**, **Section II (*Related Work*)**, **Section III.C (*Asymmetric Encoding & Retrieval Strategies*)**, dan **Section IV.C (*System Efficiency & Cold-Start Practicality*)** pada naskah IEEE Conference Paper.

---

## 1. Cohan dkk. (2020) — *SPECTER: Document-Level Representation Learning Using Citation-Informed Transformers* (ACL 2020)
- **Sitasi Siap Pakai IEEE (`5` penulis -> ditulis lengkap):**
  > A. Cohan, S. Feldman, I. Beltagy, D. Downey, and D. S. Weld, "SPECTER: Document-level representation learning using citation-informed transformers," in *Proc. 58th Annu. Meeting Assoc. Comput. Linguistics (ACL)*, Jul. 2020, pp. 2270–2282.
- **Poin Metodologis Kunci untuk Section II & III.C:**
  - Memperkenalkan pelatihan kontrastif berbasis triplet graf sitasi ilmiah $(d_q, d^+, d^-)$ di atas *backbone* `SciBERT`.
  - Menetapkan konvensi masukan baku **`Title + [SEP] + Abstract`** dan representasi tingkat dokumen dari **vektor token `[CLS]` lapisan terakhir (`768` dimensi)**—bukan *mean pooling*—karena loss kontrastif dilatih langsung pada representasi `[CLS]`.
  - Pada tahap inferensi (*test-time*), model menghasilkan embedding semantik murni dari teks judul dan abstrak tanpa memerlukan informasi tautan sitasi.

---

## 2. Dinh dkk. (2024) — *Enhancing Local Citation Recommendation with Recurrent Highway Networks and SciBERT-Based Embedding* (*ESWA*)
- **Sitasi Siap Pakai IEEE (`4` penulis -> ditulis lengkap):**
  > T. N. Dinh, P. Pham, G. L. Nguyen, and B. Vo, "Enhancing local citation recommendation with recurrent highway networks and SciBERT-based embedding," *Expert Syst. Appl.*, vol. 243, p. 122911, Jun. 2024.
- **Poin Metodologis Kunci untuk Section II & III.C:**
  - Membuktikan bahwa kosakata khusus ilmiah (*SciVocab* WordPiece pada `SciBERT`, yang menjadi fondasi `SPECTER2`) mampu merepresentasikan terminologi teknis Ilmu Komputer/AI secara presisi.
  - Membenarkan keputusan metodologis pada [`build_specter2_doc_text`](../../src/data_loader.py) untuk **tidak melakukan *lowercasing*, *stemming*, maupun *stopword removal* manual**, sebab *subword tokenizer* ilmiah justru memanfaatkan struktur morfologi asli istilah teknis.

---

## 3. Wang dkk. (2024) — *Content-Based Quality Evaluation of Scientific Papers Using Coarse Feature and Knowledge Entity Network* (*JKSUCI*)
- **Sitasi Siap Pakai IEEE (`5` penulis -> ditulis lengkap):**
  > Z. Wang, H. Zhang, H. Chen, Y. Feng, and J. Ding, "Content-based quality evaluation of scientific papers using coarse feature and knowledge entity network," *J. King Saud Univ. - Comput. Inf. Sci.*, vol. 36, no. 6, p. 102119, Jul. 2024.
- **Poin Metodologis Kunci untuk Section II:**
  - Menunjukkan bahwa embedding keluarga `SPECTER` efektif mengelompokkan kedekatan entitas pengetahuan ilmiah (*methods, tasks, datasets*) di ruang vektor padat `768` dimensi.
  - Mendukung asumsi dasar metode `M3` bahwa konsep teknis negatif (`Q_neg`) menempati arah sub-ruang semantik yang terdefinisi jelas sehingga dapat dikurangi melalui *Penalized Cosine Scoring* tanpa merusak kemiripan topik utama (`Q_pos`).

---

## 4. Jiang, Lin, & Huang (2023) — *Deep Representation Learning of Scientific Paper Reveals Its Potential Scholarly Impact* (*J. Informetrics*)
- **Sitasi Siap Pakai IEEE (`3` penulis -> ditulis lengkap):**
  > Z. Jiang, T. Lin, and C. Huang, "Deep representation learning of scientific paper reveals its potential scholarly impact," *J. Informetrics*, vol. 17, no. 1, p. 101376, Feb. 2023.
- **Poin Metodologis Kunci untuk Section I & IV.C:**
  - Menegaskan keunggulan pendekatan berbasis konten teks murni (`title + abstract`) dalam menangani skenario ***cold-start***—yakni makalah atau *preprint* ArXiv CS/AI baru yang belum memiliki riwayat kutipan masuk.
  - Memperkuat relevansi praktis arsitektur *Offline Document Indexing* (`64.183 × 768`) + *Online Dual-Input Retrieval* (`< 15 ms` di CPU) bagi pencarian literatur mutakhir.
