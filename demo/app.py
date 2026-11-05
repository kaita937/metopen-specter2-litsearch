"""Aplikasi Produk Pencarian Artikel Ilmiah Dua-Masukan (Dual-Input SPECTER2) pada Korpus LitSearch."""

from __future__ import annotations

import csv
import html
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import DEFAULT_DATA_DIR, load_litsearch_local
from src.dual_retrieval import (
    DEFAULT_EMBEDDINGS_DIR,
    EmbeddingStore,
    extract_topk_from_scores,
    l2_normalize,
    score_m1_positive_only,
    score_m2_naive_negation,
    score_m3_penalized_cosine,
)
from src.metrics import ndcg_at_k, recall_at_k, violation_rate_at_k
from src.negative_benchmark import build_naive_negation_query, load_jsonl_records

SELECTED_CASES_CSV = PROJECT_ROOT / "results" / "error_analysis" / "selected_cases.csv"


@st.cache_resource(show_spinner="Memuat indeks 64.183 artikel ilmiah CS/AI ke memori...")
def load_search_index() -> dict[str, Any]:
    """Muat matriks korpus ter-normalisasi L2, metadata LitSearch, nilai terkunci beta*, dan benchmark uji."""
    corpus, queries, qrels = load_litsearch_local(DEFAULT_DATA_DIR)
    store = EmbeddingStore.load(DEFAULT_EMBEDDINGS_DIR)
    calibrated = json.loads(
        (DEFAULT_EMBEDDINGS_DIR / "calibrated_config.json").read_text(encoding="utf-8")
    )
    beta_star = float(calibrated["beta_star"])

    test_records = load_jsonl_records(DEFAULT_DATA_DIR / "negative_queries_test.jsonl")
    val_records = load_jsonl_records(DEFAULT_DATA_DIR / "negative_queries_val.jsonl")
    bench_by_qid = {r["query_id"]: r for r in (test_records + val_records)}

    # Peta teks -> vektor precomputed untuk respons instan (< 10 ms) pada kueri benchmark
    pos_text_to_vec: dict[str, np.ndarray] = {}
    for qid, idx in store.pos_idx_map.items():
        pos_text_to_vec[queries[qid]["text"].strip()] = store.q_pos_hat[idx]

    neg_text_to_vec: dict[str, np.ndarray] = {}
    naive_text_to_vec: dict[str, np.ndarray] = {}
    for qid, idx in store.neg_idx_map.items():
        rec = bench_by_qid[qid]
        neg_text_to_vec[rec["q_neg"].strip().lower()] = store.q_neg_hat[idx]
        naive_text_to_vec[rec["q_naive_not"].strip()] = store.q_naive_hat[idx]

    selected_cases: list[dict[str, str]] = []
    if SELECTED_CASES_CSV.exists():
        with SELECTED_CASES_CSV.open("r", encoding="utf-8") as f:
            selected_cases = list(csv.DictReader(f))

    doc_id_to_idx = {did: i for i, did in enumerate(store.doc_ids)}

    return {
        "corpus": corpus,
        "queries": queries,
        "qrels": qrels,
        "store": store,
        "beta_star": beta_star,
        "test_records": test_records,
        "bench_by_qid": bench_by_qid,
        "pos_text_to_vec": pos_text_to_vec,
        "neg_text_to_vec": neg_text_to_vec,
        "naive_text_to_vec": naive_text_to_vec,
        "selected_cases": selected_cases,
        "doc_id_to_idx": doc_id_to_idx,
    }


@st.cache_resource(show_spinner="Memuat encoder kueri asimetris SPECTER2 (adhoc_query) di CPU...")
def load_live_query_encoder() -> tuple[Any, Any]:
    """Muat model SPECTER2 + adapter adhoc_query secara malas (lazy) hanya saat pengguna mengetik kueri baru."""
    import torch
    from adapters import AutoAdapterModel
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("allenai/specter2_base")
    model = AutoAdapterModel.from_pretrained("allenai/specter2_base")
    model.load_adapter(
        "allenai/specter2_adhoc_query",
        source="hf",
        load_as="adhoc_query",
        set_active=True,
    )
    active_str = str(model.active_adapters)
    assert "adhoc_query" in active_str and "proximity" not in active_str, (
        f"Adapter salah pada demo/app.py! Aktif: {active_str}"
    )
    model.to("cpu")
    model.eval()
    return tokenizer, model


def encode_query_vector(text: str, cache_map: dict[str, np.ndarray], lowercase_key: bool = False) -> tuple[np.ndarray, float, bool]:
    """Ambil vektor L2 dari cache precomputed jika tersedia, atau enkode langsung di CPU."""
    import torch

    key = text.strip().lower() if lowercase_key else text.strip()
    if key in cache_map:
        return cache_map[key][np.newaxis, :], 0.0, True

    tokenizer, model = load_live_query_encoder()
    t0 = time.perf_counter()
    with torch.no_grad():
        enc = tokenizer(
            [text.strip()],
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        cls_vec = model(**enc).last_hidden_state[:, 0, :].float().cpu().numpy()
    vec_hat = l2_normalize(cls_vec)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return vec_hat, elapsed_ms, False


def check_concept_in_doc(doc: dict[str, Any], q_neg_text: str) -> bool:
    """Deteksi apakah frasa preferensi negatif muncul secara leksikal di judul/abstrak dokumen."""
    neg_clean = q_neg_text.strip().lower()
    if not neg_clean:
        return False
    haystack = f"{doc.get('title', '')} {doc.get('text', '')}".lower().replace("-", " ")
    needle = re.escape(neg_clean.replace("-", " "))
    return bool(re.search(rf"\b{needle}(?:s|es)?\b", haystack))


def execute_dual_search(
    idx_bundle: dict[str, Any],
    q_pos_text: str,
    q_neg_text: str,
    top_k: int = 10,
    matched_bench_rec: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Jalankan pencarian M3 (serta M1 dan M2 sebagai pembanding transparan) di atas 64.183 artikel."""
    store: EmbeddingStore = idx_bundle["store"]
    beta_star: float = idx_bundle["beta_star"]

    q_pos_hat, enc_pos_ms, cached_pos = encode_query_vector(
        q_pos_text, idx_bundle["pos_text_to_vec"], lowercase_key=False
    )

    has_neg = bool(q_neg_text.strip())
    q_neg_hat: np.ndarray | None = None
    q_naive_hat: np.ndarray | None = None
    q_naive_str = ""
    enc_neg_ms = 0.0
    cached_neg = True

    if has_neg:
        q_neg_hat, ms1, c1 = encode_query_vector(
            q_neg_text, idx_bundle["neg_text_to_vec"], lowercase_key=True
        )
        q_naive_str = build_naive_negation_query(q_pos_text, q_neg_text)
        q_naive_hat, ms2, c2 = encode_query_vector(
            q_naive_str, idx_bundle["naive_text_to_vec"], lowercase_key=False
        )
        enc_neg_ms = ms1 + ms2
        cached_neg = c1 and c2

    # Ukur waktu pencarian matriks murni M3 pada 64.183 dokumen
    t_s0 = time.perf_counter()
    scores_m3 = score_m3_penalized_cosine(
        q_pos_hat, store.doc_hat, q_neg_hat=q_neg_hat, beta=beta_star if has_neg else 0.0
    )
    ranked_m3, top_scores_m3 = extract_topk_from_scores(scores_m3, store.doc_ids, top_k=top_k)
    search_m3_ms = (time.perf_counter() - t_s0) * 1000.0

    # Hitung pula M1 (tanpa penalti) dan M2 (kalimat negasi naif) untuk lencana perubahan peringkat & tab pembanding
    scores_m1 = score_m1_positive_only(q_pos_hat, store.doc_hat)
    ranked_m1, top_scores_m1 = extract_topk_from_scores(scores_m1, store.doc_ids, top_k=top_k)

    if has_neg and q_naive_hat is not None:
        scores_m2 = score_m2_naive_negation(q_naive_hat, store.doc_hat)
        ranked_m2, top_scores_m2 = extract_topk_from_scores(scores_m2, store.doc_ids, top_k=top_k)
        scores_neg_only = score_m1_positive_only(q_neg_hat, store.doc_hat)
    else:
        scores_m2 = scores_m1
        ranked_m2, top_scores_m2 = ranked_m1, top_scores_m1
        scores_neg_only = np.zeros_like(scores_m1)

    # Peringkat eksak setiap dokumen Top-K M3 pada M1 agar perubahan posisi terlihat presisi
    m1_1d = scores_m1[0]
    doc_id_to_idx = idx_bundle["doc_id_to_idx"]
    m1_ranks_for_m3_top: dict[str, int] = {}
    for did in ranked_m3[0]:
        didx = doc_id_to_idx[did]
        m1_ranks_for_m3_top[did] = 1 + int(np.sum(m1_1d > m1_1d[didx]))

    return {
        "has_neg": has_neg,
        "beta_star": beta_star,
        "q_naive_str": q_naive_str,
        "search_m3_ms": round(search_m3_ms, 2),
        "encode_cpu_ms": round(enc_pos_ms + enc_neg_ms, 2),
        "used_precomputed": cached_pos and cached_neg,
        "ranked_m3": ranked_m3[0],
        "top_scores_m3": top_scores_m3[0],
        "ranked_m1": ranked_m1[0],
        "top_scores_m1": top_scores_m1[0],
        "ranked_m2": ranked_m2[0],
        "top_scores_m2": top_scores_m2[0],
        "scores_pos_1d": m1_1d,
        "scores_neg_1d": scores_neg_only[0],
        "m1_ranks_for_m3_top": m1_ranks_for_m3_top,
        "matched_bench_rec": matched_bench_rec,
    }


def render_app() -> None:
    st.set_page_config(
        page_title="LitSearch Dual-Input Paper Finder (SPECTER2)",
        page_icon="🔬",
        layout="wide",
    )

    st.markdown(
        """
        <style>
        .paper-card {
            border: 1px solid #dfe6ed;
            border-radius: 10px;
            padding: 14px 16px;
            margin-bottom: 10px;
            background-color: #ffffff;
        }
        .badge-gold {
            background-color: #d4efdf;
            color: #145a32;
            padding: 2px 8px;
            border-radius: 6px;
            font-size: 0.78rem;
            font-weight: 600;
            margin-right: 6px;
        }
        .badge-up {
            background-color: #ebf5fb;
            color: #1b4f72;
            padding: 2px 8px;
            border-radius: 6px;
            font-size: 0.78rem;
            font-weight: 600;
            margin-right: 6px;
        }
        .badge-violator {
            background-color: #fadbd8;
            color: #78281f;
            padding: 2px 8px;
            border-radius: 6px;
            font-size: 0.78rem;
            font-weight: 600;
            margin-right: 6px;
        }
        .score-pill {
            color: #566573;
            font-size: 0.82rem;
            font-family: monospace;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    idx_bundle = load_search_index()
    beta_star = idx_bundle["beta_star"]
    selected_cases = idx_bundle["selected_cases"]
    test_records = idx_bundle["test_records"]
    corpus = idx_bundle["corpus"]
    doc_id_to_idx = idx_bundle["doc_id_to_idx"]

    if "q_pos_input" not in st.session_state:
        if selected_cases:
            st.session_state["q_pos_input"] = selected_cases[0]["q_pos"]
            st.session_state["q_neg_input"] = selected_cases[0]["q_neg"]
            st.session_state["active_qid"] = selected_cases[0]["query_id"]
        else:
            st.session_state["q_pos_input"] = test_records[0]["q_pos"]
            st.session_state["q_neg_input"] = test_records[0]["q_neg"]
            st.session_state["active_qid"] = test_records[0]["query_id"]

    st.title("🔬 Pencarian Artikel Ilmiah Berpreferensi Negatif (SPECTER2 — LitSearch)")
    st.caption(
        f"Indeks Aktif: **64.183 Artikel Ilmiah CS/ML/NLP (`princeton-nlp/LitSearch`)** · "
        f"Encoder Asimetris: `specter2_adhoc_query` & `specter2 (proximity)` · "
        f"Koefisien Terkalibrasi Otomatis: **`β* = {beta_star:.1f}`**"
    )

    # Pintasan 1 klik untuk 3 studi kasus TABLE IV + pemilih kueri benchmark
    if selected_cases:
        st.markdown("**Pintasan Studi Kasus Kualitatif (`TABLE IV` — Held-Out Test Split):**")
        c_cols = st.columns(3)
        short_labels = [
            "📌 Kasus 1: Penyelamatan Negation Blindness",
            "📌 Kasus 2: Kueri Fleksibel Minim Irisan Kata",
            "📌 Kasus 3: Batas Metode (Over-Penalization)",
        ]
        for col, sc, btn_lbl in zip(c_cols, selected_cases, short_labels):
            with col:
                if st.button(f"{btn_lbl} ({sc['query_id']})", width="stretch"):
                    st.session_state["q_pos_input"] = sc["q_pos"]
                    st.session_state["q_neg_input"] = sc["q_neg"]
                    st.session_state["active_qid"] = sc["query_id"]
                    st.rerun()

    with st.expander("📂 Telusuri dari 384 Kueri Benchmark Held-Out Test Split (Opsional)", expanded=False):
        options_map = {
            f"{r['query_id']} | [- {r['q_neg']}] {r['q_pos'][:95]}...": r
            for r in test_records
        }
        chosen_label = st.selectbox("Pilih kueri uji LitSearch-Neg:", list(options_map.keys()))
        if st.button("Muat Kueri Benchmark Ini", width="stretch"):
            chosen_rec = options_map[chosen_label]
            st.session_state["q_pos_input"] = chosen_rec["q_pos"]
            st.session_state["q_neg_input"] = chosen_rec["q_neg"]
            st.session_state["active_qid"] = chosen_rec["query_id"]
            st.rerun()

    # === DUA KOTAK INPUT BERSIH (TANPA SLIDER BETA) ===
    col_pos, col_neg = st.columns([2, 1])
    with col_pos:
        q_pos_val = st.text_area(
            "1. Positive Query — Ide / Topik Artikel Ilmiah yang Dicari (Fleksibel)",
            key="q_pos_input",
            height=95,
            placeholder="Contoh: Are there methods to compress large language models via knowledge distillation?",
        )
    with col_neg:
        q_neg_val = st.text_input(
            "2. Negative Query (Opsional) — Konsep / Metode / Domain yang Dihindari",
            key="q_neg_input",
            placeholder="Contoh: machine translation (kosongkan jika tanpa penalti)",
        )
        st.caption(
            f"Bobot penalti terkunci otomatis **`β* = {beta_star:.1f}`** "
            "(tanpa slider manual; otomatis menjadi pencarian standar `M1` bila dikosongkan)."
        )

    if not q_pos_val.strip():
        st.warning("Silakan isi kotak **Positive Query** terlebih dahulu.")
        return

    # Cek apakah input saat ini persis cocok dengan salah satu record benchmark untuk menampilkan metrik evaluasi resmi
    matched_rec: dict[str, Any] | None = None
    cand_qid = st.session_state.get("active_qid", "")
    if cand_qid in idx_bundle["bench_by_qid"]:
        rec_cand = idx_bundle["bench_by_qid"][cand_qid]
        if (
            rec_cand["q_pos"].strip() == q_pos_val.strip()
            and rec_cand["q_neg"].strip().lower() == q_neg_val.strip().lower()
        ):
            matched_rec = rec_cand

    res = execute_dual_search(
        idx_bundle=idx_bundle,
        q_pos_text=q_pos_val,
        q_neg_text=q_neg_val,
        top_k=10,
        matched_bench_rec=matched_rec,
    )

    # Baris indikator sistem & metrik
    m_cols = st.columns(4)
    m_cols[0].metric(
        "Strategi Aktif",
        f"Dual-Input M3 (β*={beta_star:.1f})" if res["has_neg"] else "Positive-Only M1 (β=0.0)",
    )
    m_cols[1].metric(
        "Latensi Pencarian Vektor (64k Paper)",
        f"{res['search_m3_ms']:.2f} ms",
        delta="CPU Real-Time" if res["used_precomputed"] else f"+{res['encode_cpu_ms']:.0f} ms encode CPU",
    )

    gold_set: set[str] = set(matched_rec["gold_ids"]) if matched_rec else set()
    violator_set: set[str] = set(matched_rec["violator_doc_ids"]) if matched_rec else set()

    if matched_rec:
        nd_m1 = ndcg_at_k(res["ranked_m1"], gold_set, k=10)
        nd_m2 = ndcg_at_k(res["ranked_m2"], gold_set, k=10)
        nd_m3 = ndcg_at_k(res["ranked_m3"], gold_set, k=10)
        vr_m1 = violation_rate_at_k(res["ranked_m1"], violator_set, k=10)
        vr_m2 = violation_rate_at_k(res["ranked_m2"], violator_set, k=10)
        vr_m3 = violation_rate_at_k(res["ranked_m3"], violator_set, k=10)

        m_cols[2].metric(
            f"nDCG@10 ({matched_rec['query_id']})",
            f"{nd_m3:.4f}",
            delta=f"{nd_m3 - nd_m2:+.4f} vs M2 ('excluding')",
        )
        m_cols[3].metric(
            "Violation Rate VR@10 (↓)",
            f"{vr_m3:.2f}",
            delta=f"{vr_m3 - vr_m2:+.2f} vs M2 | {vr_m3 - vr_m1:+.2f} vs M1",
            delta_color="inverse",
        )
    else:
        lex_v_m3 = sum(
            1 for did in res["ranked_m3"] if check_concept_in_doc(corpus[did], q_neg_val)
        ) / 10.0
        lex_v_m2 = sum(
            1 for did in res["ranked_m2"] if check_concept_in_doc(corpus[did], q_neg_val)
        ) / 10.0
        m_cols[2].metric("Mode Kueri", "Kueri Bebas (Live CPU)")
        m_cols[3].metric(
            "Kemunculan Konsep Dihindari (Top-10)",
            f"{lex_v_m3:.2f}",
            delta=f"{lex_v_m3 - lex_v_m2:+.2f} vs M2" if res["has_neg"] else "Tanpa Q_neg",
            delta_color="inverse",
        )

    tab_main, tab_compare = st.tabs([
        "📄 Rekomendasi Utama — Dual-Input M3",
        "⚖️ Komparasi Berdampingan: M3 (Dual-Input) vs M2 (Kalimat 'excluding') vs M1",
    ])

    with tab_main:
        for rank_idx, (did, s_final) in enumerate(zip(res["ranked_m3"], res["top_scores_m3"]), start=1):
            doc = corpus[did]
            didx = doc_id_to_idx[did]
            s_pos_val = float(res["scores_pos_1d"][didx])
            s_neg_val = float(res["scores_neg_1d"][didx])
            prev_m1_rank = res["m1_ranks_for_m3_top"][did]
            rank_shift = prev_m1_rank - rank_idx

            badges_html = ""
            if did in gold_set:
                badges_html += '<span class="badge-gold">✓ Gold Paper (LitSearch)</span>'
            if res["has_neg"]:
                if rank_shift > 0:
                    badges_html += f'<span class="badge-up">⬆️ Naik +{rank_shift} (dari #{prev_m1_rank} di M1)</span>'
                elif rank_shift < 0:
                    badges_html += f'<span class="badge-up">⬇️ Turun {rank_shift} (dari #{prev_m1_rank} di M1)</span>'
                else:
                    badges_html += '<span class="badge-up">➖ Peringkat Tetap vs M1</span>'

            is_violator = (did in violator_set) if matched_rec else (
                res["has_neg"] and check_concept_in_doc(doc, q_neg_val)
            )
            if is_violator:
                badges_html += f'<span class="badge-violator">⚠️ Memuat Konsep: {html.escape(q_neg_val)}</span>'

            title_safe = html.escape((doc.get("title") or "(Tanpa Judul)").strip())
            score_detail = (
                f"S_final = {s_final:.4f}  (S_pos = {s_pos_val:.4f} − {beta_star:.1f} × S_neg[{s_neg_val:.4f}])"
                if res["has_neg"]
                else f"S_final = {s_final:.4f}  (Positive-Only M1)"
            )

            st.markdown(
                f"""
                <div class="paper-card">
                    <div style="margin-bottom: 4px;">{badges_html}</div>
                    <div style="font-size: 1.02rem; font-weight: 600; color: #1a252f;">
                        #{rank_idx}. {title_safe}
                    </div>
                    <div class="score-pill" style="margin-top: 4px;">
                        Corpus ID: {did} &nbsp;|&nbsp; {score_detail}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            with st.expander(f"Baca Abstrak #{rank_idx} — Corpus ID {did}", expanded=False):
                st.write((doc.get("text") or "_Abstrak tidak tersedia pada ekstraksi S2ORC._").strip())

    with tab_compare:
        if res["has_neg"]:
            st.info(
                f"**Kalimat Uji `M2` (Naive Prompt Negation):** `{res['q_naive_str']}`\n\n"
                "Perhatikan bagaimana penyisipan kata `excluding` langsung ke dalam satu kueri Transformer (`M2`) "
                "sering kali justru menarik artikel bertopik terlarang ke peringkat atas (*negation blindness*), "
                "sedangkan pemisahan vektor **`M3` (`Dual-Input`)** menurunkannya secara konsisten."
            )

        c_m3, c_m2, c_m1 = st.columns(3)
        columns_spec = [
            (c_m3, f"✅ M3 — Proposed Dual-Input (β*={beta_star:.1f})", res["ranked_m3"], res["top_scores_m3"]),
            (c_m2, "⚠️ M2 — Naive 'excluding' Satu Kalimat", res["ranked_m2"], res["top_scores_m2"]),
            (c_m1, "🔹 M1 — Positive-Only (Tanpa Q_neg)", res["ranked_m1"], res["top_scores_m1"]),
        ]
        for col_obj, header_txt, r_ids, r_scores in columns_spec:
            with col_obj:
                st.subheader(header_txt)
                for pos, (did, sc_val) in enumerate(zip(r_ids, r_scores), start=1):
                    d_obj = corpus[did]
                    is_g = did in gold_set
                    is_v = (did in violator_set) if matched_rec else (
                        res["has_neg"] and check_concept_in_doc(d_obj, q_neg_val)
                    )
                    prefix = "🟢 **[GOLD]** " if is_g else ("🔴 **[PELANGGAR]** " if is_v else "")
                    t_short = (d_obj.get("title") or "(Tanpa Judul)").strip()
                    st.markdown(f"**#{pos}** (`{sc_val:.4f}`) {prefix}{t_short}")


if __name__ == "__main__":
    render_app()
