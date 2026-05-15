"""NDCG@k computation.

Relevance grades follow the assignment definition:
    booking_bool == 1  →  5
    click_bool   == 1  →  1
    otherwise          →  0

Gain uses the standard `2**rel - 1` form, which gives 31 / 1 / 0 — the same
shape LightGBM's `lambdarank` objective optimises.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def relevance_grade(click_bool: np.ndarray, booking_bool: np.ndarray) -> np.ndarray:
    """Per-row relevance grade. Booking dominates click."""
    rel = np.where(booking_bool == 1, 5, np.where(click_bool == 1, 1, 0))
    return rel.astype(np.int8)


def _dcg_at_k(rel_sorted: np.ndarray, k: int) -> float:
    rel_sorted = rel_sorted[:k]
    if rel_sorted.size == 0:
        return 0.0
    gains = (2.0 ** rel_sorted) - 1.0
    discounts = np.log2(np.arange(rel_sorted.size) + 2.0)
    return float((gains / discounts).sum())


def ndcg_at_k_per_query(scores: np.ndarray, relevances: np.ndarray, k: int = 5) -> float:
    """NDCG@k for a single query. Returns 0 if the query has no positive label."""
    order = np.argsort(-scores, kind="stable")
    dcg = _dcg_at_k(relevances[order], k)
    ideal_order = np.argsort(-relevances, kind="stable")
    idcg = _dcg_at_k(relevances[ideal_order], k)
    if idcg == 0.0:
        return 0.0
    return dcg / idcg


def ndcg_at_k(
    qids: np.ndarray,
    scores: np.ndarray,
    relevances: np.ndarray,
    k: int = 5,
) -> float:
    """Mean NDCG@k across queries (queries with no positive label contribute 0).

    Parameters
    ----------
    qids : array of query / search ids, same length as scores.
    scores : predicted scores (higher = more relevant).
    relevances : 0/1/5 relevance grades.
    """
    df = pd.DataFrame({"q": qids, "s": scores, "r": relevances})
    out = []
    for _, g in df.groupby("q", sort=False):
        out.append(ndcg_at_k_per_query(g["s"].to_numpy(), g["r"].to_numpy(), k))
    return float(np.mean(out)) if out else 0.0
