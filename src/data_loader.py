"""Modul pemuat, pengonversi, dan validator dataset princeton-nlp/LitSearch."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from datasets import load_dataset

EXPECTED_CORPUS_COUNT = 64183
EXPECTED_QUERY_COUNT = 597

DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "litsearch"


def build_specter2_doc_text(doc: dict[str, Any], sep_token: str) -> str:
    """Bentuk teks dokumen standar resmi SPECTER2: title + [SEP] + abstract.

    Tanpa lowercasing, stemming, atau stopword removal manual.
    """
    title = (doc.get("title") or "").strip()
    abstract = (doc.get("text") or "").strip()
    return f"{title}{sep_token}{abstract}"


def download_and_prepare_litsearch(
    data_dir: Path = DEFAULT_DATA_DIR,
    force: bool = False,
) -> None:
    """Unduh LitSearch dari Hugging Face lalu simpan ke JSONL & TSV lokal."""
    corpus_path = data_dir / "corpus.jsonl"
    queries_path = data_dir / "queries.jsonl"
    qrels_dir = data_dir / "qrels"
    qrels_path = qrels_dir / "test.tsv"

    if (
        not force
        and corpus_path.exists()
        and queries_path.exists()
        and qrels_path.exists()
    ):
        print(f"[INFO] File lokal sudah tersedia di: {data_dir}")
        return

    data_dir.mkdir(parents=True, exist_ok=True)
    qrels_dir.mkdir(parents=True, exist_ok=True)

    print("[1/3] Mengunduh konfigurasi 'corpus_clean' dari princeton-nlp/LitSearch...")
    corpus_ds = load_dataset("princeton-nlp/LitSearch", "corpus_clean", split="full")

    print("[2/3] Mengunduh konfigurasi 'query' dari princeton-nlp/LitSearch...")
    query_ds = load_dataset("princeton-nlp/LitSearch", "query", split="full")

    print("[3/3] Menyimpan corpus.jsonl, queries.jsonl, dan qrels/test.tsv...")
    with corpus_path.open("w", encoding="utf-8") as f_corpus:
        for row in corpus_ds:
            record = {
                "_id": str(row["corpusid"]),
                "title": (row.get("title") or ""),
                "text": (row.get("abstract") or ""),
            }
            f_corpus.write(json.dumps(record, ensure_ascii=False) + "\n")

    with (
        queries_path.open("w", encoding="utf-8") as f_queries,
        qrels_path.open("w", encoding="utf-8", newline="") as f_qrels,
    ):
        writer = csv.writer(f_qrels, delimiter="\t")
        writer.writerow(["query-id", "corpus-id", "score"])

        for idx, row in enumerate(query_ds):
            qid = f"q_{idx:03d}"
            gold_ids = [str(cid) for cid in row["corpusids"]]
            q_record = {
                "_id": qid,
                "text": (row.get("query") or "").strip(),
                "specificity": int(row.get("specificity", 0)),
                "quality": int(row.get("quality", 0)),
                "query_set": str(row.get("query_set", "")),
                "gold_ids": gold_ids,
            }
            f_queries.write(json.dumps(q_record, ensure_ascii=False) + "\n")

            seen_cids: set[str] = set()
            for cid in gold_ids:
                if cid not in seen_cids:
                    writer.writerow([qid, cid, 1])
                    seen_cids.add(cid)


def load_litsearch_local(
    data_dir: Path = DEFAULT_DATA_DIR,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], dict[str, dict[str, int]]]:
    """Muat corpus.jsonl, queries.jsonl, dan qrels/test.tsv dari disk lokal."""
    corpus: dict[str, dict[str, Any]] = {}
    with (data_dir / "corpus.jsonl").open("r", encoding="utf-8") as f:
        for line in f:
            doc = json.loads(line)
            corpus[doc["_id"]] = doc

    queries: dict[str, dict[str, Any]] = {}
    with (data_dir / "queries.jsonl").open("r", encoding="utf-8") as f:
        for line in f:
            q = json.loads(line)
            queries[q["_id"]] = q

    qrels: dict[str, dict[str, int]] = {}
    with (data_dir / "qrels" / "test.tsv").open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            qid = row["query-id"]
            cid = row["corpus-id"]
            score = int(row["score"])
            qrels.setdefault(qid, {})[cid] = score

    return corpus, queries, qrels


def validate_litsearch_integrity(
    corpus: dict[str, dict[str, Any]],
    queries: dict[str, dict[str, Any]],
    qrels: dict[str, dict[str, int]],
) -> dict[str, Any]:
    """Jalankan assertion integritas ketat pada korpus, kueri, dan qrels."""
    assert len(corpus) == EXPECTED_CORPUS_COUNT, (
        f"Jumlah dokumen ({len(corpus)}) != {EXPECTED_CORPUS_COUNT}"
    )
    assert len(queries) == EXPECTED_QUERY_COUNT, (
        f"Jumlah kueri ({len(queries)}) != {EXPECTED_QUERY_COUNT}"
    )
    assert len(qrels) == EXPECTED_QUERY_COUNT, (
        f"Jumlah kueri di qrels ({len(qrels)}) != {EXPECTED_QUERY_COUNT}"
    )

    total_qrel_pairs = 0
    missing_gold_ids: list[tuple[str, str]] = []
    for qid, doc_scores in qrels.items():
        assert qid in queries, f"Query ID {qid} di qrels tidak ada di queries.jsonl"
        for cid in doc_scores:
            total_qrel_pairs += 1
            if cid not in corpus:
                missing_gold_ids.append((qid, cid))

    assert len(missing_gold_ids) == 0, (
        f"Ditemukan {len(missing_gold_ids)} gold paper ID yang tidak ada di korpus!"
    )

    empty_titles = sum(1 for d in corpus.values() if not d["title"].strip())
    empty_abstracts = sum(1 for d in corpus.values() if not d["text"].strip())

    return {
        "corpus_count": len(corpus),
        "query_count": len(queries),
        "total_qrel_pairs": total_qrel_pairs,
        "missing_gold_count": len(missing_gold_ids),
        "empty_titles": empty_titles,
        "empty_abstracts": empty_abstracts,
    }


if __name__ == "__main__":
    download_and_prepare_litsearch()
    corpus_data, queries_data, qrels_data = load_litsearch_local()
    summary = validate_litsearch_integrity(corpus_data, queries_data, qrels_data)

    sample_doc = next(iter(corpus_data.values()))
    sample_formatted = build_specter2_doc_text(sample_doc, sep_token=" [SEP] ")

    print("\n=== LAPORAN VALIDASI DATASET LITSEARCH ===")
    print(f"[OK] Total dokumen korpus   : {summary['corpus_count']}")
    print(f"[OK] Total kueri pencarian  : {summary['query_count']}")
    print(f"[OK] Total pasangan relevan : {summary['total_qrel_pairs']}")
    print(f"[OK] Gold ID hilang (wajib 0): {summary['missing_gold_count']}")
    print(f"[INFO] Judul kosong         : {summary['empty_titles']}")
    print(f"[INFO] Abstrak kosong       : {summary['empty_abstracts']}")
    print(f"[INFO] Contoh teks SPECTER2 : {sample_formatted[:120]}...")
    print("[OK] VALIDASI LANGKAH 2 LOLOS 100%!")