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

import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List

def _eval_single(
    q: str,
    gt: List[str],
    backend,
    top_k: int,
) -> tuple[float, float, float]:
    """
    Для одного запроса q вычисляет Precision@k, Recall@k и MRR.
    """
    results = backend.search(q, top_k)
    ranked_ids = [doc_id for doc_id, _ in results]
    p = precision_at_k(ranked_ids, gt, top_k)
    r = recall_at_k(ranked_ids, gt, top_k)
    m = mrr(ranked_ids, gt)
    return p, r, m

def evaluate(
    backend,
    queries_gt: Dict[str, List[str]],
    top_k: int = 10,
    max_workers: int | None = None,
) -> Dict[str, float]:
    """
    Выполняет оценку Precision@k, Recall@k и MRR для каждого query → gt списка.
    Параметры:
      • backend.search(q: str, top_k: int) -> List[(id, score)]
      • queries_gt: {query: [relevant_doc_id, ...], ...}
      • top_k: глубина ранжирования
      • max_workers: число потоков (None → os.cpu_count()*5 по умолчанию)
    
    Возвращает средние метрики по всем запросам.
    """
    p_vals = []
    r_vals = []
    m_vals = []

    # Если max_workers=None, ThreadPoolExecutor сам выберет разумное число
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_eval_single, q, gt, backend, top_k): q
            for q, gt in queries_gt.items()
        }

        for future in as_completed(futures):
            try:
                p, r, m = future.result()
            except Exception as e:
                # Если поиск или вычисление метрик упало,
                # логируем ошибку и считаем метрики как нули
                q_failed = futures[future]
                print(f"[Error] query={q_failed!r} → {type(e).__name__}: {e}")
                p, r, m = 0.0, 0.0, 0.0

            p_vals.append(p)
            r_vals.append(r)
            m_vals.append(m)

    return {
        f"Precision@{top_k}": float(np.mean(p_vals)),
        f"Recall@{top_k}":    float(np.mean(r_vals)),
        "MRR":                float(np.mean(m_vals)),
    }