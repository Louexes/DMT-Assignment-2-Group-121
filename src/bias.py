"""Bias detection and mitigation for the hotel ranker.

Following Lecture 8's recommendation for ranking tasks: pick a sensitive
group, measure NDCG@5 disparity, mitigate via re-weighting.

Sensitive attribute: ``prop_brand_bool`` (1 = chain hotel, 0 = independent).
Independent properties are typically disadvantaged by algorithmic ranking
relative to chains; we measure the gap and shrink it via per-row training
weights computed from booking prevalence within each group.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config as C
from .ndcg import ndcg_at_k_per_query


def per_group_ndcg(
    df: pd.DataFrame,
    score_col: str = "score",
    group_col: str = "prop_brand_bool",
    k: int = 5,
) -> dict:
    """Per-query NDCG@k attributed to each group, plus overall.

    A query contributes to group g's mean if any of its booked items are in
    group g. The overall mean is over all queries with at least one positive.
    """
    out: dict[str, float] = {}
    overall = []
    by_group: dict[int, list[float]] = {0: [], 1: []}
    for _, sub in df.groupby("srch_id", sort=False):
        rel = sub["relevance"].to_numpy()
        if rel.max() == 0:
            continue
        scores = sub[score_col].to_numpy()
        ndcg = ndcg_at_k_per_query(scores, rel, k)
        overall.append(ndcg)
        # Attribute the query to whichever group its booked item belongs to.
        booked = sub.loc[sub["booking_bool"] == 1, group_col]
        if len(booked):
            g = int(booked.iloc[0])
            by_group[g].append(ndcg)
    out["overall"] = float(np.mean(overall)) if overall else 0.0
    out["chain"] = float(np.mean(by_group[1])) if by_group[1] else 0.0
    out["independent"] = float(np.mean(by_group[0])) if by_group[0] else 0.0
    out["chain_n"] = int(len(by_group[1]))
    out["independent_n"] = int(len(by_group[0]))
    out["gap"] = out["chain"] - out["independent"]
    return out


def exposure_ratio(df: pd.DataFrame, score_col: str = "score") -> dict:
    """Average predicted rank position of chain vs independent items.

    Computed only over queries that contain at least one of each group.
    """
    eligible = (
        df.groupby("srch_id")["prop_brand_bool"].agg(lambda x: x.nunique() == 2)
    )
    keep_qids = set(eligible[eligible].index.tolist())
    df = df[df["srch_id"].isin(keep_qids)].copy()
    df["pred_rank"] = df.groupby("srch_id", sort=False)[score_col].rank(
        ascending=False, method="first"
    )
    g = df.groupby("prop_brand_bool")["pred_rank"].mean()
    return {
        "mean_rank_chain": float(g.loc[1]),
        "mean_rank_independent": float(g.loc[0]),
        "rank_gap": float(g.loc[0] - g.loc[1]),  # positive = independent ranked worse
        "n_eligible_queries": int(len(keep_qids)),
    }


def topk_share(df: pd.DataFrame, score_col: str = "score", k: int = 5) -> dict:
    """Among predicted top-k items, share that are chain vs independent."""
    df = df.copy()
    df["pred_rank"] = df.groupby("srch_id", sort=False)[score_col].rank(
        ascending=False, method="first"
    )
    topk = df[df["pred_rank"] <= k]
    base = df["prop_brand_bool"].mean()
    pred = topk["prop_brand_bool"].mean()
    return {
        "chain_share_in_topk": float(pred),
        "chain_share_baseline": float(base),
        "delta": float(pred - base),
    }


def compute_reweighing_weights(
    df: pd.DataFrame,
    group_col: str = "prop_brand_bool",
    click_boost: float = 0.5,
) -> np.ndarray:
    """Per-row training weights using the Lecture 8 formula:

        w_i = 1 / prevalence(label=positive | group(i))     for booked
        w_i = click_boost / prevalence(...)                  for clicked-only
        w_i = 1                                              otherwise

    With binary group, this lifts the rarer-group bookings relative to the
    more common one.
    """
    grp = df[group_col].to_numpy()
    booked = df["booking_bool"].to_numpy().astype(bool)
    clicked_only = (df["click_bool"].to_numpy().astype(bool)) & (~booked)

    n = len(df)
    weights = np.ones(n, dtype=np.float32)
    for g in np.unique(grp):
        mask_g = grp == g
        n_g = mask_g.sum()
        if n_g == 0:
            continue
        # Prevalence of bookings within this group.
        p_book = max(float(booked[mask_g].mean()), 1e-6)
        weights[mask_g & booked] = 1.0 / p_book
        weights[mask_g & clicked_only] = click_boost / p_book
    return weights


def write_bias_plot(pre: dict, post: dict, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    labels = ["chain", "independent", "overall"]
    pre_vals = [pre["chain"], pre["independent"], pre["overall"]]
    post_vals = [post["chain"], post["independent"], post["overall"]]
    x = np.arange(len(labels))
    width = 0.35
    ax.bar(x - width / 2, pre_vals, width, label="pre-mitigation", color="steelblue")
    ax.bar(x + width / 2, post_vals, width, label="post-mitigation", color="darkorange")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("NDCG@5")
    ax.set_title("NDCG@5 by hotel group, before vs after reweighing")
    ax.legend()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
