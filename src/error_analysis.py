"""Pengelompokan otomatis Error Analysis kualitatif 3 kategori pada Held-Out Test Split (TABLE IV IEEE)."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any

import numpy as np

from src.data_loader import DEFAULT_DATA_DIR, load_litsearch_local
from src.dual_retrieval import (
    DEFAULT_EMBEDDINGS_DIR,
    EmbeddingStore,
    score_m1_positive_only,
    score_m2_naive_negation,
    score_m3_penalized_cosine,
)
from src.negative_benchmark import load_jsonl_records

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ERROR_DIR = PROJECT_ROOT / "results" / "error_analysis"
DEFAULT_TABLES_DIR = PROJECT_ROOT / "results" / "tables"

_STOPWORDS: frozenset[str] = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "could", "direct",
    "do", "does", "for", "from", "find", "give", "have", "has", "how", "i",
    "if", "in", "into", "is", "it", "its", "like", "list", "looking", "me",
    "might", "my", "of", "on", "one", "or", "paper", "papers", "please",
    "point", "publications", "recommend", "research", "some", "studies",
    "study", "suggest", "that", "the", "their", "them", "there", "these",
    "this", "to", "towards", "use", "used", "using", "what", "where", "which",
    "with", "work", "works", "would", "you", "any", "about",
})

_WORD_RE = re.compile(r"[a-z0-9]+")


def content_word_set(text: str) -> set[str]:
    """Ekstrak himpunan kata konten alfanumerik (tanpa stopword kueri umum) untuk perhitungan Jaccard."""
    words = _WORD_RE.findall(text.lower())
    return {w for w in words if len(w) >= 3 and w not in _STOPWORDS}


def compute_jaccard(a_text: str, b_text: str) -> float:
    """Hitung kemiripan leksikal Jaccard berbasis himpunan kata konten."""
    sa = content_word_set(a_text)
    sb = content_word_set(b_text)
    union = sa | sb
    if not union:
        return 0.0
    return float(len(sa & sb) / len(union))


def exact_best_gold_rank(scores_1d: np.ndarray, gold_indices: list[int]) -> tuple[int, int]:
    """Hitung peringkat 1-based terbaik (terkecil) dari gold paper di seluruh 64.183 dokumen korpus."""
    best_rank = len(scores_1d) + 1
    best_gidx = gold_indices[0]
    for gidx in gold_indices:
        g_score = scores_1d[gidx]
        # 1 + jumlah dokumen dengan skor tegas lebih besar
        rank = 1 + int(np.sum(scores_1d > g_score))
        if rank < best_rank:
            best_rank = rank
            best_gidx = gidx
    return best_rank, best_gidx


def run_error_analysis(
    data_dir: Path = DEFAULT_DATA_DIR,
    embeddings_dir: Path = DEFAULT_EMBEDDINGS_DIR,
    tables_dir: Path = DEFAULT_TABLES_DIR,
    out_dir: Path = DEFAULT_ERROR_DIR,
) -> dict[str, Any]:
    """Jalankan pengelompokan otomatis 3 kategori Error Analysis pada Held-Out Test Split (N = 384)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    corpus, _, _ = load_litsearch_local(data_dir)
    store = EmbeddingStore.load(embeddings_dir)
    calibrated = json.loads((embeddings_dir / "calibrated_config.json").read_text(encoding="utf-8"))
    beta_star = float(calibrated["beta_star"])

    test_records = load_jsonl_records(data_dir / "negative_queries_test.jsonl")
    per_query_res = {
        r["query_id"]: r
        for r in load_jsonl_records(tables_dir / "per_query_test_results.jsonl")
    }

    doc_id_to_idx = {did: i for i, did in enumerate(store.doc_ids)}
    test_qids = [r["query_id"] for r in test_records]
    q_pos_mat, q_neg_mat, q_naive_mat = store.subset_vectors(test_qids)

    scores_m1 = score_m1_positive_only(q_pos_mat, store.doc_hat)
    scores_m2 = score_m2_naive_negation(q_naive_mat, store.doc_hat)
    scores_m3 = score_m3_penalized_cosine(q_pos_mat, store.doc_hat, q_neg_mat, beta=beta_star)
    scores_neg_raw = q_neg_mat @ store.doc_hat.T

    diagnostics: list[dict[str, Any]] = []
    for i, rec in enumerate(test_records):
        qid = rec["query_id"]
        pq = per_query_res[qid]
        gold_ids = rec["gold_ids"]
        gold_indices = [doc_id_to_idx[gid] for gid in gold_ids]
        violator_set = set(rec["violator_doc_ids"])

        rank_m1, best_gidx_m1 = exact_best_gold_rank(scores_m1[i], gold_indices)
        rank_m2, _ = exact_best_gold_rank(scores_m2[i], gold_indices)
        rank_m3, best_gidx_m3 = exact_best_gold_rank(scores_m3[i], gold_indices)

        primary_gidx = best_gidx_m3 if rank_m3 <= rank_m1 else best_gidx_m1
        primary_gid = store.doc_ids[primary_gidx]
        gold_doc = corpus[primary_gid]
        gold_full_text = f"{gold_doc.get('title', '')} {gold_doc.get('text', '')}"

        jaccard_full = compute_jaccard(rec["q_pos"], gold_full_text)
        jaccard_title = compute_jaccard(rec["q_pos"], gold_doc.get("title", ""))
        word_count = len(rec["q_pos"].split())

        # Cari contoh dokumen pelanggar tertinggi yang masuk di Top-10 M2
        top_m2_violator_id = ""
        top_m2_violator_rank = -1
        for r_pos, did in enumerate(pq["m2_top10"], start=1):
            if did in violator_set:
                top_m2_violator_id = did
                top_m2_violator_rank = r_pos
                break

        diagnostics.append({
            "query_id": qid,
            "specificity": rec["specificity"],
            "query_set": rec["query_set"],
            "q_pos": rec["q_pos"],
            "q_neg": rec["q_neg"],
            "q_naive_not": rec["q_naive_not"],
            "q_pos_words": word_count,
            "jaccard_gold_doc": round(jaccard_full, 4),
            "jaccard_gold_title": round(jaccard_title, 4),
            "primary_gold_id": primary_gid,
            "primary_gold_title": (gold_doc.get("title") or "").strip(),
            "rank_m1": rank_m1,
            "rank_m2": rank_m2,
            "rank_m3": rank_m3,
            "m1_ndcg10": pq["m1_ndcg10"],
            "m2_ndcg10": pq["m2_ndcg10"],
            "m3_ndcg10": pq["m3_ndcg10"],
            "m1_recall10": pq["m1_recall10"],
            "m2_recall10": pq["m2_recall10"],
            "m3_recall10": pq["m3_recall10"],
            "m1_vr10": pq["m1_vr10"],
            "m2_vr10": pq["m2_vr10"],
            "m3_vr10": pq["m3_vr10"],
            "s_pos_gold": round(float(scores_m1[i, primary_gidx]), 4),
            "s_neg_gold": round(float(scores_neg_raw[i, primary_gidx]), 4),
            "s_m3_gold": round(float(scores_m3[i, primary_gidx]), 4),
            "top_m2_violator_id": top_m2_violator_id,
            "top_m2_violator_rank": top_m2_violator_rank,
            "top_m2_violator_title": (
                (corpus[top_m2_violator_id].get("title") or "").strip()
                if top_m2_violator_id else ""
            ),
        })

    # Hitung populasi masing-masing pola perilaku pada Held-Out Test Split (N = 384)
    cat1_pool = [
        d for d in diagnostics
        if d["m2_vr10"] >= d["m1_vr10"]
        and d["m3_vr10"] < d["m2_vr10"]
        and d["m3_ndcg10"] > d["m2_ndcg10"]
    ]
    cat2_pool = [
        d for d in diagnostics
        if d["q_pos_words"] >= 20
        and d["jaccard_gold_doc"] <= 0.14
        and d["rank_m3"] <= 3
    ]
    cat3_pool = [
        d for d in diagnostics
        if d["m3_ndcg10"] < d["m1_ndcg10"]
    ]

    # Pilih 1 studi kasus paling representatif & jernih untuk tiap kategori di TABLE IV
    cat1_strict = [
        d for d in cat1_pool
        if d["m2_vr10"] > d["m1_vr10"]
        and d["m3_ndcg10"] > d["m1_ndcg10"] > d["m2_ndcg10"]
        and d["rank_m3"] <= 3
        and d["m3_vr10"] == 0.0
    ]
    if not cat1_strict:
        cat1_strict = [
            d for d in cat1_pool
            if d["m3_ndcg10"] > d["m1_ndcg10"] and d["rank_m3"] <= 5
        ] or cat1_pool

    case_1 = max(
        cat1_strict,
        key=lambda d: (
            (d["m3_ndcg10"] - d["m2_ndcg10"])
            + (d["m2_vr10"] - d["m3_vr10"])
            + (d["m3_ndcg10"] - d["m1_ndcg10"]),
            -d["rank_m3"],
        ),
    )

    cat2_candidates = [
        d for d in cat2_pool
        if d["query_id"] != case_1["query_id"]
        and d["m3_vr10"] <= d["m1_vr10"]
        and d["primary_gold_title"] != ""
    ]
    case_2 = min(
        cat2_candidates,
        key=lambda d: (
            d["jaccard_gold_doc"],
            d["rank_m3"],
            -(d["m1_vr10"] - d["m3_vr10"]),
        ),
    )

    cat3_candidates = [
        d for d in cat3_pool
        if d["query_id"] not in {case_1["query_id"], case_2["query_id"]}
        and d["rank_m1"] <= 8
        and d["rank_m3"] <= 20
        and d["primary_gold_title"] != ""
    ]
    if not cat3_candidates:
        cat3_candidates = [
            d for d in cat3_pool
            if d["query_id"] not in {case_1["query_id"], case_2["query_id"]}
        ]
    case_3 = max(
        cat3_candidates,
        key=lambda d: (d["s_neg_gold"], d["m1_ndcg10"] - d["m3_ndcg10"]),
    )

    selected_cases = [
        {"category_id": "Category 1", "category_name": "M2 Negation Blindness Rescued by M3", **case_1},
        {"category_id": "Category 2", "category_name": "Flexible Multi-Clause Low Lexical Overlap", **case_2},
        {"category_id": "Category 3", "category_name": "Boundary Case / Latent Over-Penalization", **case_3},
    ]

    # Simpan ringkasan populasi kategori
    summary_rows = [
        {
            "category_id": "Category 1",
            "phenomenon": "Negation Blindness on M2 Rescued by M3 (VR_M2 >= VR_M1, VR_M3 < VR_M2, nDCG_M3 > nDCG_M2)",
            "matching_test_queries": len(cat1_pool),
            "pct_of_test_384": f"{len(cat1_pool) / len(diagnostics) * 100:.2f}%",
            "representative_query_id": case_1["query_id"],
        },
        {
            "category_id": "Category 2",
            "phenomenon": "Long Multi-Clause Query (>=20 words, Jaccard <= 0.14) Retrieved in Top-3 by M3",
            "matching_test_queries": len(cat2_pool),
            "pct_of_test_384": f"{len(cat2_pool) / len(diagnostics) * 100:.2f}%",
            "representative_query_id": case_2["query_id"],
        },
        {
            "category_id": "Category 3",
            "phenomenon": "Latent Semantic Over-Penalization Boundary (nDCG_M3 < nDCG_M1)",
            "matching_test_queries": len(cat3_pool),
            "pct_of_test_384": f"{len(cat3_pool) / len(diagnostics) * 100:.2f}%",
            "representative_query_id": case_3["query_id"],
        },
    ]

    def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    _write_csv(out_dir / "category_summary.csv", summary_rows)
    _write_csv(out_dir / "selected_cases.csv", selected_cases)
    _write_csv(out_dir / "all_test_diagnostics.csv", diagnostics)

    # Bangun draft TABLE IV siap pakai standar IEEE (double-column span)
    md_lines = [
        "# Draft Siap Pakai `TABLE IV` — Standar IEEE Conference Paper (Section IV.C, Double-Column Span)",
        "",
        "> **Catatan Aturan Template IEEE:** Tabel studi kasus kualitatif ini menggunakan tata letak melebar 2 kolom (*double-column `table*`*), judul kapital di atas tabel, **tanpa garis vertikal**, dan desimal diawali angka nol (`0.xxxx`).",
        "",
        "**TABLE IV**",
        "**QUALITATIVE CASE STUDIES ACROSS THREE DIAGNOSTIC CATEGORIES ON THE HELD-OUT LITSEARCH-NEG TEST SPLIT (β* = 0.2)**",
        "",
        "| Category & Query ID | Positive Query (`Q_pos`) & Excluded Concept (`Q_neg`) | Target Gold Paper (Jaccard Overlap) | Gold Rank (`M1 → M2 → M3`) | `VR@10` (`M1 → M2 → M3`) | Key Geometric & Retrieval Observation |",
        "| :--- | :--- | :--- | :---: | :---: | :--- |",
    ]

    observations = {
        "Category 1": (
            f"Appending `'excluding {case_1['q_neg']}'` in `M2` triggers negation blindness, pulling `{case_1['q_neg']}` violators into Rank #{case_1['top_m2_violator_rank']} and dropping the gold paper from #{case_1['rank_m1']} to #{case_1['rank_m2']}; Dual-Input `M3` clears violations (`VR@10 = {case_1['m3_vr10']:.2f}`) and promotes the gold paper to **#{case_1['rank_m3']}**."
        ),
        "Category 2": (
            f"Despite minimal surface lexical overlap (`Jaccard = {case_2['jaccard_gold_doc']:.4f}`, `{case_2['q_pos_words']}` words), asymmetric `SPECTER2` + `M3` maps the multi-clause description to **#{case_2['rank_m3']}** while suppressing `{case_2['q_neg']}` distractors (`VR@10`: `{case_2['m1_vr10']:.2f} → {case_2['m3_vr10']:.2f}`)."
        ),
        "Category 3": (
            f"Although `{case_3['q_neg']}` is lexically absent from the gold paper, high latent dense similarity (`S_neg = {case_3['s_neg_gold']:.4f}` vs. `S_pos = {case_3['s_pos_gold']:.4f}`) causes fixed `β* = 0.2` to nudge the gold paper from #{case_3['rank_m1']} to #{case_3['rank_m3']} while still reducing `VR@10` (`{case_3['m1_vr10']:.2f} → {case_3['m3_vr10']:.2f}`)."
        ),
    }

    for sc in selected_cases:
        cid = sc["category_id"]
        q_cell = f"**`Q_pos`:** {sc['q_pos']}<br>**`Q_neg`:** `{sc['q_neg']}`"
        gold_cell = f"*{sc['primary_gold_title']}* (`Jaccard = {sc['jaccard_gold_doc']:.4f}`)"
        rank_cell = f"`#{sc['rank_m1']} → #{sc['rank_m2']} →` **`#{sc['rank_m3']}`**"
        vr_cell = f"`{sc['m1_vr10']:.2f} → {sc['m2_vr10']:.2f} →` **`{sc['m3_vr10']:.2f}`**"
        md_lines.append(
            f"| **{cid}**<br>*{sc['category_name']}*<br>(`{sc['query_id']}`) | {q_cell} | {gold_cell} | {rank_cell} | {vr_cell} | {observations[cid]} |"
        )

    md_lines.extend([
        "",
        "### Distribusi Populasi Kategori pada Held-Out Test Split (`N = 384`)",
    ])
    for sr in summary_rows:
        md_lines.append(
            f"- **{sr['category_id']}** ({sr['phenomenon']}): **`{sr['matching_test_queries']}` kueri (`{sr['pct_of_test_384']}`)** — kasus terpilih: `{sr['representative_query_id']}`."
        )

    (out_dir / "table4_ieee_draft.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    return {
        "summary_rows": summary_rows,
        "selected_cases": selected_cases,
    }


if __name__ == "__main__":
    res = run_error_analysis()
    print("=== RINGKASAN ERROR ANALYSIS 3 KATEGORI (TEST N=384) ===")
    for row in res["summary_rows"]:
        print(f"  {row['category_id']}: {row['matching_test_queries']} kueri ({row['pct_of_test_384']}) -> contoh {row['representative_query_id']}")
    print("\n=== DETAIL 3 STUDI KASUS TERPILIH (TABLE IV) ===")
    for sc in res["selected_cases"]:
        print(
            f"  [{sc['category_id']} | {sc['query_id']}] Q_neg={sc['q_neg']!r} | "
            f"Gold Rank M1->M2->M3: #{sc['rank_m1']} -> #{sc['rank_m2']} -> #{sc['rank_m3']} | "
            f"VR@10: {sc['m1_vr10']:.2f} -> {sc['m2_vr10']:.2f} -> {sc['m3_vr10']:.2f} | "
            f"Jaccard={sc['jaccard_gold_doc']:.4f} | Gold={sc['primary_gold_title'][:70]}"
        )
