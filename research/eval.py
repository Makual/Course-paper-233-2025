# researches/eval.py
"""
Оценка поисковых бэкендов: Precision@k, Recall@k, MRR.
Ground-truth: {query: [doc_id1, doc_id2, …]}.
"""
from typing import Dict, List
import numpy as np

def _hit_rank(ranked_ids: List[str], gt_ids: List[str]) -> int:
    for rank, doc_id in enumerate(ranked_ids, 1):
        if doc_id in gt_ids:
            return rank
    return 0

def precision_at_k(ranked_ids: List[str], gt_ids: List[str], k: int = 10) -> float:
    if not gt_ids:
        return 0.0
    return len(set(ranked_ids[:k]) & set(gt_ids)) / k

def recall_at_k(ranked_ids: List[str], gt_ids: List[str], k: int = 10) -> float:
    if not gt_ids:
        return 0.0
    return len(set(ranked_ids[:k]) & set(gt_ids)) / len(gt_ids)

def mrr(ranked_ids: List[str], gt_ids: List[str]) -> float:
    rank = _hit_rank(ranked_ids, gt_ids)
    return 1.0 / rank if rank else 0.0

def evaluate(
    backend,
    queries_gt: Dict[str, List[str]],
    top_k: int = 10,
) -> Dict[str, float]:
    """
    Выполняет оценку Precision@k, Recall@k и MRR для каждого query → gt списков.
    backend.search(q: str, top_k: int) -> List[(id, score)]
    """
    p_list, r_list, m_list = [], [], []

    for q, gt in queries_gt.items():
        results = backend.search(q, top_k)
        ranked_ids = [doc_id for doc_id, _ in results]
        p_list.append(precision_at_k(ranked_ids, gt, top_k))
        r_list.append(recall_at_k(ranked_ids, gt, top_k))
        m_list.append(mrr(ranked_ids, gt))
    return {
        f"Precision@{top_k}": float(np.mean(p_list)),
        f"Recall@{top_k}":    float(np.mean(r_list)),
        "MRR":                float(np.mean(m_list)),
    }
