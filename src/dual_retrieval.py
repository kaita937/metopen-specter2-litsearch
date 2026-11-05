"""Implementasi seragam 3 strategi temu kembali (M1, M2, M3) pada ruang vektor asimetris SPECTER2."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_EMBEDDINGS_DIR = PROJECT_ROOT / "embeddings"


def l2_normalize(arr: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Normalisasi L2 baris/vektor 1D atau matriks 2D dengan pelindung pembagian nol."""
    arr_f32 = np.asarray(arr, dtype=np.float32)
    if arr_f32.ndim == 1:
        norm = max(float(np.linalg.norm(arr_f32)), eps)
        return (arr_f32 / norm).astype(np.float32, copy=False)
    norms = np.maximum(np.linalg.norm(arr_f32, axis=1, keepdims=True), eps)
    return (arr_f32 / norms).astype(np.float32, copy=False)


def score_m1_positive_only(q_pos_hat: np.ndarray, doc_hat: np.ndarray) -> np.ndarray:
    """Strategi M1 (Standard Positive-Only, beta = 0): S_M1(d) = q_pos_hat . d_hat."""
    if q_pos_hat.ndim == 2 and q_pos_hat.shape[0] == 1:
        return (doc_hat @ q_pos_hat[0])[np.newaxis, :]
    return q_pos_hat @ doc_hat.T


def score_m2_naive_negation(q_naive_hat: np.ndarray, doc_hat: np.ndarray) -> np.ndarray:
    """Strategi M2 (Naive Prompt Negation '<Q_pos>, excluding <Q_neg>'): S_M2(d) = q_naive_hat . d_hat."""
    if q_naive_hat.ndim == 2 and q_naive_hat.shape[0] == 1:
        return (doc_hat @ q_naive_hat[0])[np.newaxis, :]
    return q_naive_hat @ doc_hat.T


def score_m3_penalized_cosine(
    q_pos_hat: np.ndarray,
    doc_hat: np.ndarray,
    q_neg_hat: np.ndarray | None = None,
    beta: float = 0.0,
) -> np.ndarray:
    """Strategi M3 (Proposed Dual-Input Penalized Cosine Scoring):
    S_M3(d) = q_pos_hat . d_hat - beta * (q_neg_hat . d_hat).

    Otomatis tereduksi menjadi S_pos(d) apabila q_neg_hat bernilai None atau beta == 0.0.
    """
    if q_pos_hat.ndim == 2 and q_pos_hat.shape[0] == 1:
        if q_neg_hat is None or float(beta) == 0.0:
            return (doc_hat @ q_pos_hat[0])[np.newaxis, :]
        # Karena kelinearan dot product: d_hat . (q_pos - beta * q_neg) hanya butuh 1 operasi SGEMV kontigu!
        q_comb = q_pos_hat[0] - np.float32(beta) * q_neg_hat[0]
        return (doc_hat @ q_comb)[np.newaxis, :]

    s_pos = q_pos_hat @ doc_hat.T
    if q_neg_hat is None or float(beta) == 0.0:
        return s_pos
    s_neg = q_neg_hat @ doc_hat.T
    return (s_pos - np.float32(beta) * s_neg).astype(np.float32, copy=False)


def extract_topk_from_scores(
    score_matrix: np.ndarray,
    doc_ids: list[str],
    top_k: int = 10,
) -> tuple[list[list[str]], np.ndarray]:
    """Ambil daftar ID dokumen dan skor Top-K terurut menurun secara deterministik untuk matriks skor (Q, D)."""
    scores_2d = np.atleast_2d(score_matrix)
    num_q, num_d = scores_2d.shape
    k = min(top_k, num_d)

    # Gunakan argpartition O(D) lalu urutkan eksak di dalam K teratas (tie-break stabil berdasarkan indeks awal)
    part_idx = np.argpartition(-scores_2d, kth=k - 1, axis=1)[:, :k]
    ranked_ids_all: list[list[str]] = []
    top_scores_all = np.empty((num_q, k), dtype=np.float32)

    for i in range(num_q):
        cand_idx = part_idx[i]
        cand_scores = scores_2d[i, cand_idx]
        # Tie-breaker deterministik: skor tertinggi lebih dulu, indeks dokumen lebih kecil lebih dulu
        order = np.lexsort((cand_idx, -cand_scores))
        sorted_idx = cand_idx[order]
        ranked_ids_all.append([doc_ids[int(j)] for j in sorted_idx])
        top_scores_all[i] = scores_2d[i, sorted_idx]

    return ranked_ids_all, top_scores_all


@dataclass
class EmbeddingStore:
    """Wadah matriks embedding ter-normalisasi L2 untuk eksperimen M1, M2, dan M3."""

    doc_ids: list[str]
    query_ids: list[str]
    neg_query_ids: list[str]
    doc_hat: np.ndarray
    q_pos_hat: np.ndarray
    q_neg_hat: np.ndarray
    q_naive_hat: np.ndarray
    pos_idx_map: dict[str, int]
    neg_idx_map: dict[str, int]

    @classmethod
    def load(cls, embeddings_dir: Path = DEFAULT_EMBEDDINGS_DIR) -> "EmbeddingStore":
        """Muat seluruh matriks embedding dari disk dan terapkan normalisasi L2 satu kali."""
        doc_ids = json.loads((embeddings_dir / "doc_ids.json").read_text(encoding="utf-8"))
        query_ids = json.loads((embeddings_dir / "query_ids.json").read_text(encoding="utf-8"))
        neg_query_ids = json.loads((embeddings_dir / "neg_query_ids.json").read_text(encoding="utf-8"))

        doc_hat = l2_normalize(np.load(embeddings_dir / "doc_emb.npy"))
        q_pos_hat = l2_normalize(np.load(embeddings_dir / "q_pos_emb.npy"))
        q_neg_hat = l2_normalize(np.load(embeddings_dir / "q_neg_emb.npy"))
        q_naive_hat = l2_normalize(np.load(embeddings_dir / "q_naive_not_emb.npy"))

        return cls(
            doc_ids=doc_ids,
            query_ids=query_ids,
            neg_query_ids=neg_query_ids,
            doc_hat=doc_hat,
            q_pos_hat=q_pos_hat,
            q_neg_hat=q_neg_hat,
            q_naive_hat=q_naive_hat,
            pos_idx_map={qid: i for i, qid in enumerate(query_ids)},
            neg_idx_map={qid: i for i, qid in enumerate(neg_query_ids)},
        )

    def subset_vectors(
        self, query_id_list: list[str]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Ambil irisan baris (q_pos_hat, q_neg_hat, q_naive_hat) sesuai urutan query_id_list."""
        pos_rows = [self.pos_idx_map[qid] for qid in query_id_list]
        neg_rows = [self.neg_idx_map[qid] for qid in query_id_list]
        return (
            self.q_pos_hat[pos_rows],
            self.q_neg_hat[neg_rows],
            self.q_naive_hat[neg_rows],
        )


def measure_cpu_query_latency_ms(
    store: EmbeddingStore,
    sample_query_ids: list[str],
    beta: float = 0.4,
    top_k: int = 10,
    n_repeats: int = 3,
) -> dict[str, float]:
    """Ukur rata-rata dan median latensi pencarian 1 kueri tunggal (M3) terhadap 64.183 dokumen di CPU."""
    q_pos_sub, q_neg_sub, _ = store.subset_vectors(sample_query_ids)

    # Warm-up cache memori CPU
    _ = score_m3_penalized_cosine(q_pos_sub[:1], store.doc_hat, q_neg_sub[:1], beta=beta)

    latencies_ms: list[float] = []
    for _ in range(n_repeats):
        for i in range(len(sample_query_ids)):
            qp = q_pos_sub[i : i + 1]
            qn = q_neg_sub[i : i + 1]
            t0 = time.perf_counter()
            sc = score_m3_penalized_cosine(qp, store.doc_hat, qn, beta=beta)
            _ = extract_topk_from_scores(sc, store.doc_ids, top_k=top_k)
            latencies_ms.append((time.perf_counter() - t0) * 1000.0)

    arr = np.array(latencies_ms, dtype=np.float64)
    return {
        "mean_ms": round(float(arr.mean()), 3),
        "std_ms": round(float(arr.std(ddof=1)), 3),
        "median_ms": round(float(np.median(arr)), 3),
        "p95_ms": round(float(np.percentile(arr, 95)), 3),
    }
