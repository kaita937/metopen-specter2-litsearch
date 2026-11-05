"""Konstruksi tervalidasi benchmark LitSearch-Neg (3 lapis), split Val/Test, dan verifikasi embedding Hari 2."""

from __future__ import annotations

import argparse
import csv
import json
import re
import time
from pathlib import Path
from typing import Any

import numpy as np

from src.data_loader import (
    DEFAULT_DATA_DIR,
    EXPECTED_CORPUS_COUNT,
    EXPECTED_QUERY_COUNT,
    load_litsearch_local,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_EMBEDDINGS_DIR = PROJECT_ROOT / "embeddings"
DEFAULT_TABLES_DIR = PROJECT_ROOT / "results" / "tables"

SPLIT_SEED = 42
VAL_RATIO = 0.20
MIN_CONCEPT_DOCS = 15
MAX_CONCEPT_DOCS = 6000
TOP_K_DISTRACTORS = 5

# Kata kepala metodologis umum yang boleh berbagi kata umum selama kata pembeda utamanya tidak berkonflik
GENERIC_HEAD_TOKENS: frozenset[str] = frozenset({
    "learning", "model", "models", "modeling", "modelling",
    "network", "networks", "neural", "deep", "machine",
    "system", "systems", "task", "tasks", "method", "methods",
    "data", "dataset", "datasets", "training", "tuning",
    "generation", "detection", "analysis", "classification",
    "extraction", "parsing", "prediction", "recognition",
    "representation", "representations", "understanding",
    "processing", "natural", "language", "languages",
    "based", "level", "multi", "low", "open", "automatic",
    "inference", "optimization", "search", "evaluation",
    "estimation", "synthesis", "compression", "filtering",
    "transfer", "adaptation", "augmentation", "alignment",
    "retrieval", "matching", "scoring", "ranking", "embedding",
    "embeddings", "transformer", "transformers", "attention",
    "sequence", "text", "textual", "document", "documents",
    "sentence", "sentences", "word", "words", "token", "tokens",
})

# Taksonomi terkurasi konsep riset CS / ML / NLP / AI untuk ekstraksi Q_neg bebas noise boilerplate
CS_CONCEPT_LEXICON: tuple[str, ...] = (
    # NLP Tasks & Applications
    "machine translation",
    "neural machine translation",
    "simultaneous translation",
    "speech translation",
    "back-translation",
    "speech recognition",
    "automatic speech recognition",
    "speech synthesis",
    "text-to-speech",
    "speaker verification",
    "question answering",
    "visual question answering",
    "open-domain question answering",
    "reading comprehension",
    "named entity recognition",
    "entity linking",
    "entity typing",
    "relation extraction",
    "event extraction",
    "event detection",
    "information extraction",
    "open information extraction",
    "sentiment analysis",
    "aspect-based sentiment",
    "emotion recognition",
    "stance detection",
    "text summarization",
    "abstractive summarization",
    "extractive summarization",
    "multi-document summarization",
    "dialogue systems",
    "task-oriented dialogue",
    "conversational agents",
    "dialogue state tracking",
    "response generation",
    "semantic parsing",
    "dependency parsing",
    "constituency parsing",
    "part-of-speech tagging",
    "semantic role labeling",
    "coreference resolution",
    "word sense disambiguation",
    "natural language inference",
    "textual entailment",
    "paraphrase generation",
    "text simplification",
    "grammatical error correction",
    "style transfer",
    "code generation",
    "program synthesis",
    "code summarization",
    "vulnerability detection",
    "bug localization",
    "fact verification",
    "claim verification",
    "hate speech",
    "toxicity detection",
    "offensive language",
    "rumor detection",
    "fake news",
    "misinformation",
    "sarcasm detection",
    "metaphor detection",
    "humor detection",
    "topic modeling",
    "keyphrase generation",
    "keyphrase extraction",
    "optical character recognition",
    "sign language",
    "lip reading",
    "argument mining",
    "essay scoring",
    "readability assessment",
    # Architectures & Representation Families
    "convolutional neural networks",
    "recurrent neural networks",
    "long short-term memory",
    "gated recurrent",
    "graph neural networks",
    "graph convolutional",
    "graph attention",
    "knowledge graphs",
    "knowledge graph completion",
    "knowledge base question answering",
    "vision transformer",
    "diffusion models",
    "latent diffusion",
    "generative adversarial networks",
    "variational autoencoder",
    "normalizing flows",
    "mixture of experts",
    "state space models",
    "capsule networks",
    "siamese networks",
    "memory networks",
    "pointer networks",
    "conditional random fields",
    "support vector machines",
    "random forests",
    "decision trees",
    "hidden markov models",
    "bayesian networks",
    "gaussian processes",
    "spiking neural networks",
    "hyperbolic embeddings",
    "reinforcement learning",
    "inverse reinforcement learning",
    "offline reinforcement learning",
    "multi-agent reinforcement learning",
    "human feedback",
    "reward modeling",
    # Training, Compression, Security & Optimization Paradigms
    "federated learning",
    "continual learning",
    "lifelong learning",
    "catastrophic forgetting",
    "class-incremental",
    "meta-learning",
    "curriculum learning",
    "active learning",
    "semi-supervised learning",
    "weakly supervised",
    "distant supervision",
    "self-training",
    "co-training",
    "contrastive learning",
    "knowledge distillation",
    "network pruning",
    "weight pruning",
    "structured pruning",
    "post-training quantization",
    "quantization-aware training",
    "low-rank adaptation",
    "prompt tuning",
    "prefix tuning",
    "instruction tuning",
    "chain-of-thought",
    "in-context learning",
    "data augmentation",
    "domain adaptation",
    "unsupervised domain adaptation",
    "source-free domain adaptation",
    "multi-task learning",
    "adversarial training",
    "adversarial attacks",
    "adversarial examples",
    "backdoor attacks",
    "data poisoning",
    "membership inference",
    "differential privacy",
    "machine unlearning",
    "watermarking",
    "neural architecture search",
    "hyperparameter optimization",
    "bayesian optimization",
    "evolutionary algorithms",
    "monte carlo tree search",
    "beam search",
    "speculative decoding",
    "minimum bayes risk",
    # Vision, Multimodal, Domain & Scientific Subfields
    "computer vision",
    "image classification",
    "object detection",
    "semantic segmentation",
    "instance segmentation",
    "image generation",
    "text-to-image",
    "image captioning",
    "video understanding",
    "video captioning",
    "action recognition",
    "point cloud",
    "3d object detection",
    "neural radiance fields",
    "music generation",
    "audio classification",
    "multimodal learning",
    "vision-language",
    "cross-lingual transfer",
    "low-resource languages",
    "code-switching",
    "morphological inflection",
    "transliteration",
    "biomedical text",
    "clinical notes",
    "electronic health records",
    "medical imaging",
    "radiology reports",
    "drug discovery",
    "molecular property",
    "protein structure",
    "protein sequences",
    "genomic sequences",
    "legal documents",
    "judgment prediction",
    "financial forecasting",
    "social media",
    "recommendation systems",
    "collaborative filtering",
    "sequential recommendation",
    "time series forecasting",
    "anomaly detection",
    "traffic forecasting",
    "autonomous driving",
    "robotic manipulation",
    "embodied navigation",
    "tabular data",
    "table-to-text",
    "text-to-sql",
    "crowdsourcing",
    "eye-tracking",
    "brain decoding",
    "gender bias",
    "demographic parity",
    "counterfactual fairness",
    "out-of-distribution detection",
    "uncertainty quantification",
    "conformal prediction",
    "causal discovery",
    "causal inference",
)

_WORD_RE = re.compile(r"[a-z0-9]+")


def normalize_words(text: str) -> list[str]:
    """Ubah teks menjadi daftar token kata alfanumerik huruf kecil (tanda hubung dipisah)."""
    cleaned = text.lower().replace("-", " ").replace("/", " ")
    return _WORD_RE.findall(cleaned)


def canonical_token(tok: str) -> str:
    """Normalisasi ringan bentuk jamak sederhana agar pencocokan konsep konsisten."""
    if len(tok) > 4 and tok.endswith("ies"):
        return tok[:-3] + "y"
    if len(tok) > 4 and tok.endswith("es") and not tok.endswith(("ses", "xes", "zes")):
        pass
    if len(tok) > 3 and tok.endswith("s") and not tok.endswith(("ss", "us", "is")):
        return tok[:-1]
    return tok


def concept_token_tuple(concept: str) -> tuple[str, ...]:
    """Konversi frasa konsep menjadi tuple token kanonis."""
    return tuple(canonical_token(w) for w in normalize_words(concept))


def distinguishing_tokens(concept: str) -> set[str]:
    """Ambil kata pembeda spesifik dari konsep (di luar kata kepala metodologis umum)."""
    raw_tokens = [canonical_token(w) for w in normalize_words(concept)]
    specific = {
        t for t in raw_tokens
        if t not in {canonical_token(g) for g in GENERIC_HEAD_TOKENS} and len(t) >= 3
    }
    if not specific:
        specific = {t for t in raw_tokens if len(t) >= 3}
    return specific


def build_naive_negation_query(q_pos: str, q_neg: str) -> str:
    """Bentuk kueri M2 ('<Q_pos>, excluding <Q_neg>') dengan tanda baca kalimat alami."""
    stripped = q_pos.strip()
    neg_clean = q_neg.strip()
    if stripped.endswith("?"):
        return f"{stripped[:-1].rstrip()}, excluding {neg_clean}?"
    if stripped.endswith("."):
        return f"{stripped[:-1].rstrip()}, excluding {neg_clean}."
    return f"{stripped}, excluding {neg_clean}"


def build_concept_inverted_index(
    corpus: dict[str, dict[str, Any]],
    concepts: tuple[str, ...] = CS_CONCEPT_LEXICON,
    min_docs: int = MIN_CONCEPT_DOCS,
    max_docs: int = MAX_CONCEPT_DOCS,
) -> tuple[dict[str, list[str]], dict[str, set[str]], dict[str, set[str]]]:
    """Bangun inverted index konsep -> daftar doc_id di seluruh korpus (Layer 1).

    Mengembalikan:
      - concept_to_docs: {concept: sorted list of doc_ids (title U abstract)}
      - doc_to_title_concepts: {doc_id: set of concepts muncul di title}
      - doc_to_all_concepts: {doc_id: set of concepts muncul di title U abstract}
    """
    by_length: dict[int, dict[tuple[str, ...], str]] = {}
    for c in concepts:
        tup = concept_token_tuple(c)
        if tup:
            by_length.setdefault(len(tup), {})[tup] = c

    lengths = sorted(by_length.keys())
    raw_concept_docs: dict[str, list[str]] = {c: [] for c in concepts}
    raw_doc_title: dict[str, set[str]] = {}
    raw_doc_all: dict[str, set[str]] = {}

    for doc_id, doc in corpus.items():
        title_toks = [canonical_token(w) for w in normalize_words(doc.get("title") or "")]
        abs_toks = [canonical_token(w) for w in normalize_words(doc.get("text") or "")]

        title_matches: set[str] = set()
        n_t = len(title_toks)
        for k in lengths:
            if n_t < k:
                continue
            lookup = by_length[k]
            for i in range(n_t - k + 1):
                gram = tuple(title_toks[i : i + k])
                matched = lookup.get(gram)
                if matched is not None:
                    title_matches.add(matched)

        all_matches = set(title_matches)
        n_a = len(abs_toks)
        for k in lengths:
            if n_a < k:
                continue
            lookup = by_length[k]
            for i in range(n_a - k + 1):
                gram = tuple(abs_toks[i : i + k])
                matched = lookup.get(gram)
                if matched is not None:
                    all_matches.add(matched)

        if title_matches:
            raw_doc_title[doc_id] = title_matches
        if all_matches:
            raw_doc_all[doc_id] = all_matches
            for c in all_matches:
                raw_concept_docs[c].append(doc_id)

    valid_concepts = {
        c for c, dlist in raw_concept_docs.items()
        if min_docs <= len(dlist) <= max_docs
    }

    concept_to_docs = {
        c: sorted(raw_concept_docs[c])
        for c in concepts
        if c in valid_concepts
    }
    doc_to_title_concepts = {
        did: (cset & valid_concepts)
        for did, cset in raw_doc_title.items()
        if (cset & valid_concepts)
    }
    doc_to_all_concepts = {
        did: (cset & valid_concepts)
        for did, cset in raw_doc_all.items()
        if (cset & valid_concepts)
    }
    return concept_to_docs, doc_to_title_concepts, doc_to_all_concepts


def l2_normalize_rows(mat: np.ndarray) -> np.ndarray:
    """Normalisasi L2 per baris dengan pelindung pembagian nol 1e-12."""
    norms = np.maximum(np.linalg.norm(mat, axis=1, keepdims=True), 1e-12)
    return (mat / norms).astype(np.float32, copy=False)


def _has_token_conflict(spec_tokens: set[str], target_tokens: set[str]) -> bool:
    """Periksa apakah ada irisan kata atau prefiks akar kata (>= 5 huruf) dengan teks positif/judul gold."""
    if spec_tokens & target_tokens:
        return True
    for s in spec_tokens:
        if len(s) >= 5:
            stem = s[:5]
            for t in target_tokens:
                if len(t) >= 5 and t.startswith(stem):
                    return True
    return False


def construct_litsearch_neg(
    corpus: dict[str, dict[str, Any]],
    queries: dict[str, dict[str, Any]],
    qrels: dict[str, dict[str, int]],
    doc_ids: list[str],
    query_ids: list[str],
    doc_emb: np.ndarray,
    q_pos_emb: np.ndarray,
    top_k_distractors: int = TOP_K_DISTRACTORS,
    seed: int = SPLIT_SEED,
    val_ratio: float = VAL_RATIO,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Bangun himpunan uji LitSearch-Neg dengan protokol validasi 3 lapis + split deterministik Val/Test.

    Lapis 1 (Concept-Level Violators): violator_doc_ids mencakup seluruh dokumen korpus yang memuat konsep Q_neg.
    Lapis 2 (0% Gold & Q_pos Overlap): Q_neg bebas konflik dengan Q_pos dan 0% beririsan dengan gold papers.
    Lapis 3 (Domain Sanity & Stratified Split): Konsep taksonomi CS nyata + split seed=42 (20% Val / 80% Test).
    """
    t0 = time.perf_counter()
    concept_to_docs, doc_to_title_concepts, doc_to_all_concepts = build_concept_inverted_index(corpus)
    concept_doc_sets = {c: set(dlist) for c, dlist in concept_to_docs.items()}
    concept_spec_tokens = {c: distinguishing_tokens(c) for c in concept_to_docs}

    d_hat = l2_normalize_rows(doc_emb)
    q_hat = l2_normalize_rows(q_pos_emb)
    scores_all = q_hat @ d_hat.T  # (597, 64183)

    doc_id_arr = np.array(doc_ids, dtype=object)
    all_records: list[dict[str, Any]] = []
    skipped_no_distractor = 0
    skipped_no_clean_concept = 0

    for q_idx, qid in enumerate(query_ids):
        q_obj = queries[qid]
        q_pos_text = q_obj["text"]
        gold_set = set(qrels[qid].keys())

        row_scores = scores_all[q_idx]
        top10_indices = np.argpartition(-row_scores, kth=10)[:10]
        top10_indices = top10_indices[np.argsort(-row_scores[top10_indices])]
        top10_doc_ids = [str(doc_id_arr[i]) for i in top10_indices]

        top_k_non_gold = [
            (rank_1based, did)
            for rank_1based, did in enumerate(top10_doc_ids[:top_k_distractors], start=1)
            if did not in gold_set
        ]
        top10_non_gold_ids = [did for did in top10_doc_ids if did not in gold_set]

        if not top_k_non_gold:
            skipped_no_distractor += 1
            continue

        q_pos_tokens = {canonical_token(w) for w in normalize_words(q_pos_text)}
        gold_title_tokens: set[str] = set()
        gold_all_tokens: set[str] = set()
        for gid in gold_set:
            gdoc = corpus[gid]
            gt = {canonical_token(w) for w in normalize_words(gdoc.get("title") or "")}
            ga = {canonical_token(w) for w in normalize_words(gdoc.get("text") or "")}
            gold_title_tokens |= gt
            gold_all_tokens |= gt | ga

        # Kumpulkan kandidat konsep dari hard distractors di Top-5
        candidate_concepts: dict[str, tuple[int, bool, str]] = {}
        for rank_1based, did in top_k_non_gold:
            in_title_set = doc_to_title_concepts.get(did, set())
            in_all_set = doc_to_all_concepts.get(did, set())
            for c in in_all_set:
                in_title = c in in_title_set
                prev = candidate_concepts.get(c)
                if prev is None:
                    candidate_concepts[c] = (rank_1based, in_title, did)
                else:
                    prev_rank, prev_title, prev_did = prev
                    if (in_title and not prev_title) or (in_title == prev_title and rank_1based < prev_rank):
                        candidate_concepts[c] = (rank_1based, in_title, did)

        best_choice: tuple[tuple[int, int, int, int, str], str, int, bool, str] | None = None
        for c, (src_rank, in_title, src_did) in candidate_concepts.items():
            v_set = concept_doc_sets[c]
            spec_toks = concept_spec_tokens[c]

            # Lapis 2a: 0% konflik dengan teks Q_pos
            if _has_token_conflict(spec_toks, q_pos_tokens):
                continue

            # Lapis 2b: 0% irisan dengan gold papers (himpunan violator, judul gold, maupun seluruh kata pembeda di abstrak gold)
            if gold_set & v_set:
                continue
            if _has_token_conflict(spec_toks, gold_title_tokens):
                continue
            if spec_toks.issubset(gold_all_tokens):
                continue

            # Hitung berapa hard distractor di Top-10 yang memuat konsep negatif ini
            top10_violator_hits = sum(1 for did in top10_non_gold_ids if did in v_set)
            if top10_violator_hits == 0:
                continue

            # Prioritas deterministik:
            # 1) Muncul di judul hard distractor Top-5 (1 vs 0)
            # 2) Jumlah distractor Top-10 yang terkena konsep tersebut
            # 3) Peringkat kemunculan lebih tinggi (-src_rank)
            # 4) Frasa konsep multi-kata (len(concept_token_tuple(c)))
            # 5) Tie-breaker alfabetis terbalik agar deterministik penuh
            priority = (
                1 if in_title else 0,
                top10_violator_hits,
                -src_rank,
                min(len(concept_token_tuple(c)), 3),
                c,
            )
            if best_choice is None or priority > best_choice[0]:
                best_choice = (priority, c, src_rank, in_title, src_did)

        if best_choice is None:
            skipped_no_clean_concept += 1
            continue

        _, chosen_concept, src_rank, in_title, src_did = best_choice
        violator_ids = concept_to_docs[chosen_concept]

        # Assertion mutlak Lapis 2 (0% Gold Overlap)
        assert len(gold_set & set(violator_ids)) == 0, (
            f"Konflik gold terdeteksi pada {qid} untuk konsep {chosen_concept}"
        )

        q_naive = build_naive_negation_query(q_pos_text, chosen_concept)
        record = {
            "query_id": qid,
            "q_pos": q_pos_text,
            "q_neg": chosen_concept,
            "q_naive_not": q_naive,
            "gold_ids": sorted(gold_set),
            "violator_doc_ids": violator_ids,
            "violator_count": len(violator_ids),
            "source_distractor_id": src_did,
            "source_distractor_rank": int(src_rank),
            "source_match_location": "title" if in_title else "abstract",
            "specificity": int(q_obj["specificity"]),
            "quality": int(q_obj["quality"]),
            "query_set": str(q_obj["query_set"]),
        }
        all_records.append(record)

    # Lapis 3: Stratified deterministic split (seed=42) berdasarkan specificity (0 vs 1)
    rng = np.random.default_rng(seed)
    val_records: list[dict[str, Any]] = []
    test_records: list[dict[str, Any]] = []

    for spec_val in (0, 1):
        group = [r for r in all_records if r["specificity"] == spec_val]
        perm = rng.permutation(len(group))
        n_val = int(round(len(group) * val_ratio))
        val_idx_set = set(perm[:n_val].tolist())
        for idx, rec in enumerate(group):
            if idx in val_idx_set:
                val_records.append({**rec, "split": "val"})
            else:
                test_records.append({**rec, "split": "test"})

    val_records.sort(key=lambda r: r["query_id"])
    test_records.sort(key=lambda r: r["query_id"])

    val_qids = {r["query_id"] for r in val_records}
    test_qids = {r["query_id"] for r in test_records}
    assert len(val_qids & test_qids) == 0, "Kebocoran ID kueri antara Validation dan Test!"

    elapsed = time.perf_counter() - t0
    audit = {
        "total_litsearch_queries": len(query_ids),
        "validated_litsearch_neg_queries": len(all_records),
        "val_count": len(val_records),
        "test_count": len(test_records),
        "val_ratio_actual": round(len(val_records) / max(len(all_records), 1), 4),
        "skipped_no_top5_distractor": skipped_no_distractor,
        "skipped_no_conflict_free_concept": skipped_no_clean_concept,
        "unique_negative_concepts_used": len({r["q_neg"] for r in all_records}),
        "title_sourced_pct": round(
            100.0 * sum(1 for r in all_records if r["source_match_location"] == "title")
            / max(len(all_records), 1),
            2,
        ),
        "mean_violator_docs_per_query": round(
            float(np.mean([r["violator_count"] for r in all_records])) if all_records else 0.0,
            2,
        ),
        "median_violator_docs_per_query": round(
            float(np.median([r["violator_count"] for r in all_records])) if all_records else 0.0,
            2,
        ),
        "gold_overlap_conflicts": 0,
        "val_test_id_overlap": 0,
        "seed": seed,
        "top_k_distractors": top_k_distractors,
        "active_lexicon_concepts_in_corpus": len(concept_to_docs),
        "construction_seconds": round(elapsed, 3),
    }
    return val_records, test_records, audit


def save_litsearch_neg_splits(
    val_records: list[dict[str, Any]],
    test_records: list[dict[str, Any]],
    audit: dict[str, Any],
    corpus: dict[str, dict[str, Any]],
    data_dir: Path = DEFAULT_DATA_DIR,
    tables_dir: Path = DEFAULT_TABLES_DIR,
) -> None:
    """Simpan file JSONL split Validation (20%), Test (80%), audit JSON, dan sampel sanity check CSV."""
    data_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    val_path = data_dir / "negative_queries_val.jsonl"
    test_path = data_dir / "negative_queries_test.jsonl"
    audit_path = data_dir / "litsearch_neg_audit.json"
    sample_csv_path = tables_dir / "litsearch_neg_sanity_sample.csv"

    for path, rows in ((val_path, val_records), (test_path, test_records)):
        with path.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with audit_path.open("w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2, ensure_ascii=False)

    combined = sorted(val_records + test_records, key=lambda r: r["query_id"])
    step = max(len(combined) // 30, 1)
    sample_rows = combined[::step][:30]
    with sample_csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "query_id",
                "split",
                "specificity",
                "q_pos",
                "q_neg",
                "q_naive_not",
                "gold_first_title",
                "distractor_rank",
                "distractor_location",
                "distractor_title",
                "violator_count",
            ],
        )
        writer.writeheader()
        for r in sample_rows:
            first_gid = r["gold_ids"][0]
            src_did = r["source_distractor_id"]
            writer.writerow({
                "query_id": r["query_id"],
                "split": r["split"],
                "specificity": r["specificity"],
                "q_pos": r["q_pos"],
                "q_neg": r["q_neg"],
                "q_naive_not": r["q_naive_not"],
                "gold_first_title": (corpus[first_gid].get("title") or "")[:120],
                "distractor_rank": r["source_distractor_rank"],
                "distractor_location": r["source_match_location"],
                "distractor_title": (corpus[src_did].get("title") or "")[:120],
                "violator_count": r["violator_count"],
            })


def load_jsonl_records(path: Path) -> list[dict[str, Any]]:
    """Muat daftar objek JSON dari file JSONL."""
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def encode_negative_queries_specter2(
    records_ordered: list[dict[str, Any]],
    embeddings_dir: Path = DEFAULT_EMBEDDINGS_DIR,
    batch_size: int = 64,
    device: str | None = None,
) -> dict[str, Any]:
    """Enkode q_neg dan q_naive_not memakai adapter allenai/specter2_adhoc_query."""
    import torch
    from adapters import AutoAdapterModel
    from transformers import AutoTokenizer

    embeddings_dir.mkdir(parents=True, exist_ok=True)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

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
        f"Adapter salah untuk kueri negatif! Aktif: {active_str}"
    )
    model.to(device)
    model.eval()

    def _encode_list(texts: list[str]) -> np.ndarray:
        chunks: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(texts), batch_size):
                batch = texts[start : start + batch_size]
                enc = tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                )
                enc = {k: v.to(device) for k, v in enc.items()}
                if device == "cuda":
                    with torch.amp.autocast("cuda", dtype=torch.float16):
                        cls_vec = model(**enc).last_hidden_state[:, 0, :]
                else:
                    cls_vec = model(**enc).last_hidden_state[:, 0, :]
                chunks.append(cls_vec.float().cpu().numpy())
        return np.vstack(chunks).astype(np.float32)

    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    neg_ids = [r["query_id"] for r in records_ordered]
    q_neg_texts = [r["q_neg"] for r in records_ordered]
    q_naive_texts = [r["q_naive_not"] for r in records_ordered]

    q_neg_emb = _encode_list(q_neg_texts)
    q_naive_emb = _encode_list(q_naive_texts)

    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    np.save(embeddings_dir / "q_neg_emb.npy", q_neg_emb)
    np.save(embeddings_dir / "q_naive_not_emb.npy", q_naive_emb)
    with (embeddings_dir / "neg_query_ids.json").open("w", encoding="utf-8") as f:
        json.dump(neg_ids, f, indent=2)

    return {
        "neg_query_count": len(neg_ids),
        "device": device,
        "active_adapter": active_str,
        "encoding_seconds": round(elapsed, 3),
    }


def verify_day2_deliverables(
    data_dir: Path = DEFAULT_DATA_DIR,
    embeddings_dir: Path = DEFAULT_EMBEDDINGS_DIR,
) -> dict[str, Any]:
    """Jalankan seluruh pemeriksaan integritas Hari 2 (matriks embedding, split Val/Test, 0% overlap)."""
    corpus, queries, qrels = load_litsearch_local(data_dir)

    doc_ids = json.loads((embeddings_dir / "doc_ids.json").read_text(encoding="utf-8"))
    query_ids = json.loads((embeddings_dir / "query_ids.json").read_text(encoding="utf-8"))
    neg_query_ids = json.loads((embeddings_dir / "neg_query_ids.json").read_text(encoding="utf-8"))
    encoding_log = json.loads((embeddings_dir / "encoding_log.json").read_text(encoding="utf-8"))

    doc_emb = np.load(embeddings_dir / "doc_emb.npy")
    q_pos_emb = np.load(embeddings_dir / "q_pos_emb.npy")
    q_neg_emb = np.load(embeddings_dir / "q_neg_emb.npy")
    q_naive_emb = np.load(embeddings_dir / "q_naive_not_emb.npy")

    val_records = load_jsonl_records(data_dir / "negative_queries_val.jsonl")
    test_records = load_jsonl_records(data_dir / "negative_queries_test.jsonl")
    all_neg = sorted(val_records + test_records, key=lambda r: r["query_id"])

    assert len(doc_ids) == EXPECTED_CORPUS_COUNT
    assert len(query_ids) == EXPECTED_QUERY_COUNT
    assert doc_emb.shape == (EXPECTED_CORPUS_COUNT, 768), f"Bentuk doc_emb salah: {doc_emb.shape}"
    assert q_pos_emb.shape == (EXPECTED_QUERY_COUNT, 768), f"Bentuk q_pos_emb salah: {q_pos_emb.shape}"
    assert len(all_neg) == len(neg_query_ids), "Jumlah kueri negatif tidak sinkron dengan neg_query_ids.json"
    assert q_neg_emb.shape == (len(neg_query_ids), 768), f"Bentuk q_neg_emb salah: {q_neg_emb.shape}"
    assert q_naive_emb.shape == (len(neg_query_ids), 768), f"Bentuk q_naive_emb salah: {q_naive_emb.shape}"

    for name, mat in (
        ("doc_emb", doc_emb),
        ("q_pos_emb", q_pos_emb),
        ("q_neg_emb", q_neg_emb),
        ("q_naive_not_emb", q_naive_emb),
    ):
        assert mat.dtype == np.float32, f"{name} wajib float32, ditemukan {mat.dtype}"
        assert np.isfinite(mat).all(), f"{name} mengandung NaN atau Inf!"
        norms = np.linalg.norm(mat, axis=1)
        assert float(norms.min()) > 0.0, f"{name} memiliki baris ber-norma nol!"

    # Verifikasi 100% gold paper IDs ada di doc_ids.json
    doc_id_set = set(doc_ids)
    for qid, gdict in qrels.items():
        for gid in gdict:
            assert gid in doc_id_set, f"Gold ID {gid} untuk {qid} hilang dari doc_ids.json!"

    # Verifikasi 0% kebocoran split dan 0% konflik gold paper vs violator_doc_ids
    val_ids = {r["query_id"] for r in val_records}
    test_ids = {r["query_id"] for r in test_records}
    assert len(val_ids & test_ids) == 0, "Ditemukan kebocoran query_id antara Val dan Test!"
    assert [r["query_id"] for r in all_neg] == neg_query_ids, "Urutan neg_query_ids.json tidak cocok!"

    for r in all_neg:
        qid = r["query_id"]
        assert qid in queries, f"{qid} tidak ditemukan di queries.jsonl"
        gold_ids_official = set(qrels[qid].keys())
        assert set(r["gold_ids"]) == gold_ids_official, f"Gold IDs pada {qid} tidak identik dengan qrels!"
        assert len(gold_ids_official & set(r["violator_doc_ids"])) == 0, (
            f"Konflik gold vs violator ditemukan pada {qid}!"
        )
        assert len(r["violator_doc_ids"]) >= MIN_CONCEPT_DOCS

    return {
        "doc_emb_shape": list(doc_emb.shape),
        "q_pos_emb_shape": list(q_pos_emb.shape),
        "q_neg_emb_shape": list(q_neg_emb.shape),
        "q_naive_not_emb_shape": list(q_naive_emb.shape),
        "val_queries": len(val_records),
        "test_queries": len(test_records),
        "total_neg_queries": len(all_neg),
        "val_test_overlap": 0,
        "gold_violator_overlap": 0,
        "gpu_name": encoding_log.get("gpu_name", "unknown"),
        "doc_encoding_seconds": encoding_log.get("doc_encoding_seconds"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bangun atau verifikasi benchmark LitSearch-Neg & embedding Hari 2."
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Hanya jalankan verifikasi lengkap terhadap file hasil Colab di embeddings/ dan data/litsearch/.",
    )
    parser.add_argument(
        "--encode-neg",
        action="store_true",
        help="Enkode ulang q_neg_emb.npy dan q_naive_not_emb.npy menggunakan specter2_adhoc_query.",
    )
    args = parser.parse_args()

    if not args.verify_only:
        corpus, queries, qrels = load_litsearch_local(DEFAULT_DATA_DIR)
        doc_ids = json.loads((DEFAULT_EMBEDDINGS_DIR / "doc_ids.json").read_text(encoding="utf-8"))
        query_ids = json.loads((DEFAULT_EMBEDDINGS_DIR / "query_ids.json").read_text(encoding="utf-8"))
        doc_emb = np.load(DEFAULT_EMBEDDINGS_DIR / "doc_emb.npy")
        q_pos_emb = np.load(DEFAULT_EMBEDDINGS_DIR / "q_pos_emb.npy")

        val_records, test_records, audit = construct_litsearch_neg(
            corpus=corpus,
            queries=queries,
            qrels=qrels,
            doc_ids=doc_ids,
            query_ids=query_ids,
            doc_emb=doc_emb,
            q_pos_emb=q_pos_emb,
        )
        save_litsearch_neg_splits(val_records, test_records, audit, corpus)
        print("[OK] Konstruksi LitSearch-Neg selesai:", json.dumps(audit, indent=2))

        if args.encode_neg:
            ordered = sorted(val_records + test_records, key=lambda r: r["query_id"])
            enc_info = encode_negative_queries_specter2(ordered, DEFAULT_EMBEDDINGS_DIR)
            print("[OK] Encoding kueri negatif selesai:", json.dumps(enc_info, indent=2))

    report = verify_day2_deliverables(DEFAULT_DATA_DIR, DEFAULT_EMBEDDINGS_DIR)
    print("\n=== LAPORAN VERIFIKASI HARI 2 ===")
    for k, v in report.items():
        print(f"  [OK] {k:24s}: {v}")
    print("[OK] VALIDASI HARI 2 LOLOS 100%!")


if __name__ == "__main__":
    main()
