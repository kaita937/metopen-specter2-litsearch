# Draft Siap Pakai `TABLE I` — Standar IEEE Conference Paper (Section III.B)

> **Catatan Aturan Template IEEE:** Judul tabel diletakkan **di atas tabel** dengan huruf kapital (`TABLE I`), **tanpa garis vertikal**, dan seluruh desimal diawali angka nol (`0.xx`).

**TABLE I**
**SUMMARY STATISTICS AND SPECTER2 WORDPIECE TOKEN DISTRIBUTIONS OF THE LITSEARCH BENCHMARK**

| Text Component / Subset | Count (N) | Mean ± SD | Median | 95th Pct. | Min / Max | > 512 Tokens (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Literature Search Queries (`Q_pos`, All) | 597 | 26.11 ± 7.64 | 25.00 | 40.00 | 11 / 62 | 0 (0.00%) |
| — Broad Queries (`specificity = 0`) | 155 | 25.57 ± 8.23 | 25.00 | 37.00 | 12 / 62 | 0 (0.00%) |
| — Specific Queries (`specificity = 1`) | 442 | 26.30 ± 7.42 | 25.00 | 41.00 | 11 / 53 | 0 (0.00%) |
| Corpus Titles Only | 64,183 | 14.28 ± 30.45 | 13.00 | 25.00 | 2 / 3,921 | 42 (0.07%) |
| Corpus Abstracts Only | 64,183 | 162.67 ± 130.79 | 153.00 | 327.00 | 2 / 4,931 | 1,239 (1.93%) |
| **Corpus Full Input (`Title + [SEP] + Abstract`)** | **64,183** | **175.95 ± 135.69** | **166.00** | **346.00** | **3 / 4,934** | **1,382 (2.15%)** |
| Non-Empty Corpus Subset (Diagnostic) | 57,986 | 194.43 ± 129.78 | 176.00 | 361.00 | 3 / 4,934 | 1,382 (2.38%) |

*Catatan tambahan untuk narasi teks **Section III.B**: Dari `597` kueri resmi (`351` bersumber inline-citation dan `246` author-written), terdapat total `639` pasangan relevan (`574` gold paper unik; rata-rata `1.07` gold paper/kueri, rentang `1`–`5`, kecocokan ID di korpus `100.00%`). Pada ekstraksi S2ORC `corpus_clean` (`64,183` artikel), sebanyak `55,631` (`86.68%`) memiliki judul dan abstrak lengkap, `1,792` (`2.79%`) hanya judul, `563` (`0.88%`) hanya abstrak, dan `6,197` (`9.66%`) tanpa teks metadata terurai yang tetap dipertahankan utuh di dalam indeks sesuai protokol resmi LitSearch.*
