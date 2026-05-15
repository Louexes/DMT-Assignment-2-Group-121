"""LightGBM LambdaRank training and inference helpers."""
from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from . import config as C
from .ndcg import ndcg_at_k


def load_features() -> tuple[pd.DataFrame, list[str], list[str]]:
    """Load labeled features, return (df sorted by srch_id, feature_cols, categorical_cols)."""
    with open(C.FEATURE_LIST) as f:
        meta = json.load(f)
    feat_cols = meta["features"]
    cat_cols = meta["categorical"]

    cols = list({"srch_id", "is_val", "click_bool", "booking_bool", "relevance", *feat_cols})
    df = pd.read_parquet(C.LABELED_FEAT, columns=cols)
    df = df.sort_values("srch_id", kind="stable").reset_index(drop=True)
    return df, feat_cols, cat_cols


def _group_sizes(srch_ids: np.ndarray) -> np.ndarray:
    """Return contiguous group sizes (count of rows per query, in order)."""
    _, counts = np.unique(srch_ids, return_counts=True)
    # `np.unique` sorts; we already sort the dataframe by srch_id, so counts align.
    return counts


def split_train_val(df: pd.DataFrame, feat_cols: list[str]) -> dict:
    is_val = df["is_val"].to_numpy().astype(bool)
    tr = df.loc[~is_val]
    va = df.loc[is_val]
    return {
        "X_tr": tr[feat_cols],
        "y_tr": tr["relevance"].to_numpy(),
        "q_tr": tr["srch_id"].to_numpy(),
        "g_tr": _group_sizes(tr["srch_id"].to_numpy()),
        "X_va": va[feat_cols],
        "y_va": va["relevance"].to_numpy(),
        "q_va": va["srch_id"].to_numpy(),
        "g_va": _group_sizes(va["srch_id"].to_numpy()),
        "click_va": va["click_bool"].to_numpy(),
        "book_va": va["booking_bool"].to_numpy(),
    }


def train(
    parts: dict,
    cat_cols: list[str],
    sample_weight: np.ndarray | None = None,
    params: dict | None = None,
    save_path: Path | None = None,
) -> tuple[lgb.Booster, dict]:
    p = dict(C.LGBM_PARAMS if params is None else params)
    train_set = lgb.Dataset(
        parts["X_tr"], label=parts["y_tr"], group=parts["g_tr"],
        categorical_feature=cat_cols, weight=sample_weight, free_raw_data=False,
    )
    val_set = lgb.Dataset(
        parts["X_va"], label=parts["y_va"], group=parts["g_va"],
        categorical_feature=cat_cols, reference=train_set, free_raw_data=False,
    )
    eval_log: dict = {}
    booster = lgb.train(
        params=p,
        train_set=train_set,
        num_boost_round=C.LGBM_NUM_BOOST,
        valid_sets=[val_set],
        valid_names=["val"],
        callbacks=[
            lgb.early_stopping(C.LGBM_EARLY_STOP, verbose=False),
            lgb.log_evaluation(period=50),
            lgb.record_evaluation(eval_log),
        ],
    )
    if save_path is not None:
        booster.save_model(str(save_path))
    return booster, eval_log


def evaluate(booster: lgb.Booster, parts: dict) -> dict:
    preds = booster.predict(parts["X_va"], num_iteration=booster.best_iteration)
    rel = parts["y_va"]
    qids = parts["q_va"]
    ndcg5 = ndcg_at_k(qids, preds, rel, k=5)
    ndcg10 = ndcg_at_k(qids, preds, rel, k=10)
    return {
        "best_iteration": int(booster.best_iteration),
        "ndcg@5": float(ndcg5),
        "ndcg@10": float(ndcg10),
        "preds": preds,
    }


def predict_submit(booster: lgb.Booster, feat_cols: list[str]) -> pd.DataFrame:
    sub = pd.read_parquet(C.SUBMIT_FEAT, columns=["srch_id", "prop_id", *feat_cols])
    sub = sub.sort_values("srch_id", kind="stable").reset_index(drop=True)
    sub["score"] = booster.predict(sub[feat_cols], num_iteration=booster.best_iteration)
    return sub[["srch_id", "prop_id", "score"]]
