"""Implementasi 3 metrik evaluasi utama (nDCG@10, Recall@10, VR@10), validasi pytrec_eval, dan uji Wilcoxon."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytrec_eval
from scipy import stats


def ndcg_at_k(ranked_doc_ids: list[str], gold_doc_ids: set[str], k: int = 10) -> float:
    """Hitung nDCG@k relevansi biner per kueri (identik standar trec_eval ndcg_cut_k)."""
    if not gold_doc_ids or k <= 0:
        return 0.0

    dcg = 0.0
    for rank_idx, doc_id in enumerate(ranked_doc_ids[:k], start=1):
        if doc_id in gold_doc_ids:
            dcg += 1.0 / math.log2(rank_idx + 1.0)

    ideal_hits = min(len(gold_doc_ids), k)
    idcg = sum(1.0 / math.log2(rank_idx + 1.0) for rank_idx in range(1, ideal_hits + 1))
    if idcg <= 0.0:
        return 0.0
    return float(dcg / idcg)


def recall_at_k(ranked_doc_ids: list[str], gold_doc_ids: set[str], k: int = 10) -> float:
    """Hitung Recall@k per kueri (identik standar trec_eval recall_k)."""
    if not gold_doc_ids or k <= 0:
        return 0.0
    hits = sum(1 for doc_id in ranked_doc_ids[:k] if doc_id in gold_doc_ids)
    return float(hits / len(gold_doc_ids))


def violation_rate_at_k(
    ranked_doc_ids: list[str],
    violator_doc_ids: set[str],
    k: int = 10,
) -> float:
    """Hitung VR@k (Violation Rate @ k): proporsi slot pada Top-k yang memuat dokumen pelanggar Q_neg."""
    if k <= 0 or not ranked_doc_ids:
        return 0.0
    top_slice = ranked_doc_ids[:k]
    violator_hits = sum(1 for doc_id in top_slice if doc_id in violator_doc_ids)
    return float(violator_hits / k)


def evaluate_rankings_per_query(
    records: list[dict[str, Any]],
    ranked_lists: list[list[str]],
    k: int = 10,
) -> dict[str, np.ndarray]:
    """Evaluasi nDCG@k, Recall@k, dan VR@k untuk setiap kueri pada daftar records."""
    assert len(records) == len(ranked_lists), "Panjang records dan ranked_lists tidak sama!"
    n_q = len(records)
    ndcg_arr = np.empty(n_q, dtype=np.float64)
    recall_arr = np.empty(n_q, dtype=np.float64)
    vr_arr = np.empty(n_q, dtype=np.float64)

    for i, (rec, ranked_docs) in enumerate(zip(records, ranked_lists)):
        gold_set = set(rec["gold_ids"])
        violator_set = set(rec["violator_doc_ids"])
        ndcg_arr[i] = ndcg_at_k(ranked_docs, gold_set, k=k)
        recall_arr[i] = recall_at_k(ranked_docs, gold_set, k=k)
        vr_arr[i] = violation_rate_at_k(ranked_docs, violator_set, k=k)

    return {
        f"ndcg@{k}": ndcg_arr,
        f"recall@{k}": recall_arr,
        f"vr@{k}": vr_arr,
    }


def cross_validate_with_pytrec_eval(
    records: list[dict[str, Any]],
    ranked_lists: list[list[str]],
    top_scores: np.ndarray,
    k: int = 10,
    tol: float = 1e-6,
) -> dict[str, float]:
    """Validasi silang nDCG@k dan Recall@k mandiri terhadap pytrec_eval.RelevanceEvaluator.

    Wajib memenuhi selisih absolut maksimum <= 1e-6 di seluruh kueri uji.
    """
    manual = evaluate_rankings_per_query(records, ranked_lists, k=k)

    qrels_dict: dict[str, dict[str, int]] = {}
    run_dict: dict[str, dict[str, float]] = {}

    for idx, (rec, ranked_docs) in enumerate(zip(records, ranked_lists)):
        qid = rec["query_id"]
        qrels_dict[qid] = {gid: 1 for gid in rec["gold_ids"]}
        # Berikan skor menurun ketat sesuai urutan peringkat eksak + skor asli agar bebas ambiguitas tie string di C
        run_dict[qid] = {
            doc_id: float(top_scores[idx, r_pos]) - (r_pos * 1e-9)
            for r_pos, doc_id in enumerate(ranked_docs[: max(k, len(ranked_docs))])
        }

    ndcg_key = f"ndcg_cut_{k}"
    recall_key = f"recall_{k}"
    evaluator = pytrec_eval.RelevanceEvaluator(qrels_dict, {ndcg_key, recall_key})
    pytrec_res = evaluator.evaluate(run_dict)

    pytrec_ndcg = np.array([pytrec_res[r["query_id"]][ndcg_key] for r in records], dtype=np.float64)
    pytrec_recall = np.array([pytrec_res[r["query_id"]][recall_key] for r in records], dtype=np.float64)

    max_diff_ndcg = float(np.max(np.abs(manual[f"ndcg@{k}"] - pytrec_ndcg)))
    max_diff_recall = float(np.max(np.abs(manual[f"recall@{k}"] - pytrec_recall)))

    assert max_diff_ndcg <= tol, f"Selisih nDCG@{k} vs pytrec_eval ({max_diff_ndcg:.2e}) > {tol}!"
    assert max_diff_recall <= tol, f"Selisih Recall@{k} vs pytrec_eval ({max_diff_recall:.2e}) > {tol}!"

    return {
        f"max_abs_diff_ndcg@{k}": max_diff_ndcg,
        f"max_abs_diff_recall@{k}": max_diff_recall,
    }


def paired_wilcoxon_bonferroni(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    num_comparisons: int = 2,
    alpha: float = 0.05,
    higher_is_better: bool = True,
) -> dict[str, Any]:
    """Uji statistik Paired Wilcoxon signed-rank dua sisi dengan koreksi Bonferroni."""
    a = np.asarray(scores_a, dtype=np.float64)
    b = np.asarray(scores_b, dtype=np.float64)
    assert a.shape == b.shape and a.ndim == 1

    diffs = a - b
    n_total = int(len(diffs))
    n_pos = int(np.sum(diffs > 1e-12))
    n_neg = int(np.sum(diffs < -1e-12))
    n_ties = n_total - n_pos - n_neg
    mean_diff = float(np.mean(diffs))

    improved = n_pos if higher_is_better else n_neg
    degraded = n_neg if higher_is_better else n_pos

    if n_pos + n_neg == 0:
        stat_val = 0.0
        p_raw = 1.0
    else:
        res = stats.wilcoxon(a, b, zero_method="wilcox", alternative="two-sided", mode="approx")
        stat_val = float(res.statistic)
        p_raw = float(res.pvalue)

    p_bonf = min(p_raw * num_comparisons, 1.0)
    alpha_bonf = alpha / num_comparisons

    # Estimasi ukuran efek r = |Z| / sqrt(N) dari p-value dua sisi
    if p_raw < 1.0 and p_raw > 0.0:
        z_abs = float(stats.norm.isf(p_raw / 2.0))
    elif p_raw == 0.0:
        z_abs = 8.5
    else:
        z_abs = 0.0
    effect_r = z_abs / math.sqrt(max(n_total, 1))

    return {
        "n_queries": n_total,
        "mean_a": float(np.mean(a)),
        "mean_b": float(np.mean(b)),
        "mean_diff": mean_diff,
        "improved": improved,
        "ties": n_ties,
        "degraded": degraded,
        "wilcoxon_stat": stat_val,
        "p_raw": p_raw,
        "p_bonf": p_bonf,
        "alpha_bonf": alpha_bonf,
        "significant_bonf": bool(p_bonf < alpha_bonf),
        "effect_size_r": effect_r,
    }
