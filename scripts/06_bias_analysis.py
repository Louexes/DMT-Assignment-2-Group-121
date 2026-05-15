"""Phase 6 — bias detection on the unweighted LGBM val predictions, then
re-train LGBM with reweighing, then re-evaluate.

Outputs:
    outputs/bias_metrics.json
    outputs/figures/bias_pre_vs_post.png
    artifacts/lightgbm_debiased.txt
    artifacts/lightgbm_debiased_val_preds.parquet
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import bias as B
from src import config as C
from src import lightgbm_train as L


def attach_by_position(val_preds_path: Path) -> pd.DataFrame:
    """val_preds were saved row-aligned with the LGBM val split. Attach
    prop_brand_bool by re-loading the labeled feature set in identical order.
    """
    cols = ["srch_id", "prop_id", "prop_brand_bool", "click_bool", "booking_bool", "is_val"]
    aux = pd.read_parquet(C.LABELED_FEAT, columns=cols)
    aux = aux.sort_values("srch_id", kind="stable").reset_index(drop=True)
    aux = aux[aux["is_val"] == 1].reset_index(drop=True)
    val_preds = pd.read_parquet(val_preds_path).reset_index(drop=True)
    assert len(aux) == len(val_preds), f"row mismatch {len(aux)} vs {len(val_preds)}"
    assert (aux["srch_id"].to_numpy() == val_preds["srch_id"].to_numpy()).all(), (
        "srch_id alignment broken"
    )
    aux["score"] = val_preds["score"].to_numpy()
    if "relevance" not in aux.columns:
        aux["relevance"] = (
            aux["booking_bool"].astype("int8") * 5
            + ((aux["click_bool"].astype("int8") == 1) & (aux["booking_bool"].astype("int8") == 0)).astype("int8")
        )
    return aux


def main() -> None:
    t0 = time.time()
    # ----- Pre-mitigation metrics from the unweighted LGBM val preds -----
    pre_df = attach_by_position(C.LGBM_VAL_PREDS)
    pre_per_group = B.per_group_ndcg(pre_df, score_col="score")
    pre_exposure = B.exposure_ratio(pre_df, score_col="score")
    pre_topk = B.topk_share(pre_df, score_col="score", k=5)
    print(f"[pre] overall NDCG@5={pre_per_group['overall']:.4f}  "
          f"chain={pre_per_group['chain']:.4f}  indep={pre_per_group['independent']:.4f}  "
          f"gap={pre_per_group['gap']:+.4f}")
    print(f"[pre] mean predicted rank — chain={pre_exposure['mean_rank_chain']:.2f} "
          f"indep={pre_exposure['mean_rank_independent']:.2f}")
    print(f"[pre] chain share in top-5={pre_topk['chain_share_in_topk']:.3f} "
          f"(baseline={pre_topk['chain_share_baseline']:.3f})")

    # ----- Train debiased LGBM with reweighing -----
    df, feat_cols, cat_cols = L.load_features()
    parts = L.split_train_val(df, feat_cols)
    train_mask = df["is_val"].to_numpy() == 0
    train_brand = df.loc[train_mask, "prop_brand_bool"].to_numpy()
    train_book = df.loc[train_mask, "booking_bool"].to_numpy()
    train_click = df.loc[train_mask, "click_bool"].to_numpy()
    weight_df = pd.DataFrame({
        "prop_brand_bool": train_brand,
        "booking_bool": train_book,
        "click_bool": train_click,
    })
    weights = B.compute_reweighing_weights(weight_df)
    print(f"[weights] mean={weights.mean():.3f}  max={weights.max():.1f}  "
          f"booked_chain mean={weights[(train_brand==1)&(train_book==1)].mean():.1f}  "
          f"booked_indep mean={weights[(train_brand==0)&(train_book==1)].mean():.1f}")

    booster, _ = L.train(parts, cat_cols, sample_weight=weights, save_path=C.LGBM_DEBIAS_MODEL)
    metrics = L.evaluate(booster, parts)
    print(f"[debias] overall NDCG@5={metrics['ndcg@5']:.4f}  best_iter={metrics['best_iteration']}")

    # Save debiased val preds (row-aligned with parts['X_va'] which is the same
    # order as L.load_features → sort_values("srch_id") + filter is_val==1).
    debias_val = pd.DataFrame({
        "srch_id": parts["q_va"],
        "score": metrics["preds"],
    })
    debias_val_path = C.ART_DIR / "lightgbm_debiased_val_preds.parquet"
    debias_val.to_parquet(debias_val_path, compression="zstd", index=False)

    # ----- Post-mitigation metrics -----
    post_df = attach_by_position(debias_val_path)
    post_per_group = B.per_group_ndcg(post_df, score_col="score")
    post_exposure = B.exposure_ratio(post_df, score_col="score")
    post_topk = B.topk_share(post_df, score_col="score", k=5)
    print(f"[post] overall NDCG@5={post_per_group['overall']:.4f}  "
          f"chain={post_per_group['chain']:.4f}  indep={post_per_group['independent']:.4f}  "
          f"gap={post_per_group['gap']:+.4f}")
    print(f"[post] mean predicted rank — chain={post_exposure['mean_rank_chain']:.2f} "
          f"indep={post_exposure['mean_rank_independent']:.2f}")
    print(f"[post] chain share in top-5={post_topk['chain_share_in_topk']:.3f} "
          f"(baseline={post_topk['chain_share_baseline']:.3f})")

    out = {
        "sensitive_attribute": "prop_brand_bool",
        "pre": {
            "ndcg_by_group": pre_per_group,
            "exposure": pre_exposure,
            "topk_share": pre_topk,
        },
        "post": {
            "ndcg_by_group": post_per_group,
            "exposure": post_exposure,
            "topk_share": post_topk,
        },
        "delta": {
            "overall_ndcg": post_per_group["overall"] - pre_per_group["overall"],
            "gap_chain_minus_indep": post_per_group["gap"] - pre_per_group["gap"],
            "rank_gap": post_exposure["rank_gap"] - pre_exposure["rank_gap"],
        },
        "training_time_sec": round(time.time() - t0, 1),
    }
    with open(C.OUT_DIR / "bias_metrics.json", "w") as f:
        json.dump(out, f, indent=2)
    B.write_bias_plot(pre_per_group, post_per_group, C.FIG_DIR / "bias_pre_vs_post.png")
    print(f"[ok] phase 6 done in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
