"""
Universal metric evaluator for IR back-ends.
Calculates Precision@k, Recall@k, MRR once per query and
aggregates the results for any number of difficulty groups
(easy / medium / hard / …) plus an automatic “all” bucket.

Usage
-----
tfidf_backend = TfidfResearch(max_features=30_000)
tfidf_backend.index(texts, ids)

metrics_df = evaluate_groups(
    backend      = tfidf_backend,
    q_groups     = q_groups,       # {'easy': {...}, 'medium': {...}, ...}
    top_k        = 10,
    max_workers  = 20,
)
print(metrics_df)
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
RESULTS_CSV = Path("eval_results.csv")

# ─────────── low-level helpers ────────────────────────────────────────────────
def _hit_rank(ranked: List[str], gt: List[str]) -> int:
    """Return position (1-based) of the first relevant doc or 0 if missing."""
    for rank, doc_id in enumerate(ranked, 1):
        if doc_id in gt:
            return rank
    return 0


def _metrics_for_query(
    query: str,
    gt: List[str],
    backend,
    top_k: int,
) -> Tuple[str, float, float, float]:
    ranked_ids = [doc_id for doc_id, _ in backend.search(query, top_k)]
    p = len(set(ranked_ids[:top_k]) & set(gt)) / top_k if gt else 0.0
    r = len(set(ranked_ids[:top_k]) & set(gt)) / len(gt) if gt else 0.0
    rank = _hit_rank(ranked_ids, gt)
    mrr  = 1.0 / rank if rank else 0.0
    return query, p, r, mrr


# ─────────── public API ───────────────────────────────────────────────────────
def evaluate_groups(
    backend,
    q_groups: Dict[str, Dict[str, List[str]]],
    *,
    top_k: int = 10,
    max_workers: int | None = None,
) -> pd.DataFrame:
    """
    Parameters
    ----------
    backend      : any object with .search(q, top_k) -> List[(id, score)]
    q_groups     : {'easy': {'q1': [gt_ids], ...}, 'medium': {...}, ...}
    top_k        : cut-off for P@k / R@k
    max_workers  : threads for concurrent execution

    Returns
    -------
    pd.DataFrame  (index = level, columns = Precision, Recall, MRR)
    """

    # 1) Flatten all queries to compute each only once
    flat_qgt: Dict[str, List[str]] = {
        q: gt for group in q_groups.values() for q, gt in group.items()
    }

    # 2) Parallel evaluation per query
    records: List[Tuple[str, float, float, float]] = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        fut2q = {
            pool.submit(_metrics_for_query, q, gt, backend, top_k): q
            for q, gt in flat_qgt.items()
        }
        for fut in as_completed(fut2q):
            try:
                records.append(fut.result())
            except Exception as e:
                bad_q = fut2q[fut]
                print(f"[Error] {bad_q!r}: {type(e).__name__}: {e}")
                records.append((bad_q, 0.0, 0.0, 0.0))

    per_query = (
        pd.DataFrame(records, columns=["query", "P", "R", "MRR"])
        .set_index("query")
    )

    # 3) Aggregate for each difficulty level
    rows = []
    for level, qgt in q_groups.items():
        subset = per_query.loc[qgt.keys()]
        rows.append(
            {
                "level": level,
                f"Precision@{top_k}": subset["P"].mean(),
                f"Recall@{top_k}": subset["R"].mean(),
                "MRR": subset["MRR"].mean(),
            }
        )

    # 4) “all” bucket (union of every query)
    rows.append(
        {
            "level": "all",
            f"Precision@{top_k}": per_query["P"].mean(),
            f"Recall@{top_k}": per_query["R"].mean(),
            "MRR": per_query["MRR"].mean(),
        }
    )

    return pd.DataFrame(rows).set_index("level")

def run_and_log(
    *,
    backend,
    q_groups: dict[str, dict[str, list[str]]],
    backend_name: str,          # e.g. "tfidf"
    test_name: str,             # e.g. "tfidf_30k_vocab"
    top_k: int = 10,
    max_workers: int | None = 20,
    dest: Path = RESULTS_CSV,
) -> pd.DataFrame:
    """
    • runs evaluation once
    • adds metadata columns
    • appends/creates CSV with all previous runs
    """
    metrics = evaluate_groups(
        backend,
        q_groups,
        top_k=top_k,
        max_workers=max_workers,
    ).reset_index()                       # 'level' column appears

    # enrich with metadata
    metrics["backend"]   = backend_name
    metrics["test_name"] = test_name

    # re-order for readability
    ordered = [
        "test_name", "backend", "level",
        f"Precision@{top_k}", f"Recall@{top_k}", "MRR"
    ]
    metrics = metrics[ordered]

    # append (write header only once)
    header_needed = not dest.exists()
    metrics.to_csv(dest, mode="a", header=header_needed, index=False)

    return metrics