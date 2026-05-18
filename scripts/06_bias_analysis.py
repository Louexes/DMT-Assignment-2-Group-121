"""Phase 6 — bias detection and mitigation.

Pipeline:
    1. Pre-mitigation metrics on the unweighted LGBM v1 val predictions
       (per-group NDCG, exposure, top-5 share).
    2. Re-train LGBM v1 with prevalence-based instance weights and save
       both the booster and its val predictions.
    3. Post-mitigation metrics on the debiased booster.
    4. Bonus: pre/post bias metrics for the deployed three-way ensemble
       (LGBM v1 + LGBM v3 + RankFormer at the weights in src/config.py).
       Pre uses the unweighted v1 leg; post swaps in the debiased v1 leg.
       v3 and RankFormer are unchanged in both halves.

Outputs:
    outputs/bias_metrics.json                — LGBM-only pre/post metrics
    outputs/bias_metrics_ensemble.json       — ensemble pre/post metrics
    outputs/figures/bias_pre_vs_post.png     — LGBM-only NDCG bar chart
    outputs/figures/bias_pre_vs_post_ensemble.png
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


def _norm_within_srch(df: pd.DataFrame, col: str) -> np.ndarray:
    g = df.groupby("srch_id", sort=False)[col]
    eps = 1e-9
    return ((df[col] - g.transform("min")) / (g.transform("max") - g.transform("min") + eps)).to_numpy()


def _assemble_ensemble_val(v1_path: Path) -> pd.DataFrame:
    """Return a val-fold dataframe with the deployed three-way ensemble score
    in the ``score`` column. ``v1_path`` chooses which LGBM v1 val preds to use
    (unweighted for pre-mitigation, debiased for post-mitigation)."""
    cols = ["srch_id", "prop_id", "prop_brand_bool", "click_bool", "booking_bool", "is_val"]
    aux = pd.read_parquet(C.LABELED_FEAT, columns=cols)
    aux = aux.sort_values("srch_id", kind="stable").reset_index(drop=True)
    aux = aux[aux["is_val"] == 1].reset_index(drop=True)

    v1 = pd.read_parquet(v1_path).reset_index(drop=True)
    v3 = pd.read_parquet(C.LGBM_V3_VAL_PREDS).reset_index(drop=True)
    rf = pd.read_parquet(C.RF_VAL_PREDS).reset_index(drop=True)
    assert len(aux) == len(v1) == len(v3) == len(rf)
    assert (aux["srch_id"].to_numpy() == v1["srch_id"].to_numpy()).all()
    assert (aux["srch_id"].to_numpy() == v3["srch_id"].to_numpy()).all()
    assert (aux["srch_id"].to_numpy() == rf["srch_id"].to_numpy()).all()

    aux["score_v1"] = v1["score"].to_numpy()
    aux["score_v3"] = v3["score"].to_numpy()
    aux["score_rf"] = rf["score"].to_numpy()

    v1_n = _norm_within_srch(aux, "score_v1")
    v3_n = _norm_within_srch(aux, "score_v3")
    rf_n = _norm_within_srch(aux, "score_rf")
    aux["score"] = (
        C.ENS_W_LGBM_V1 * v1_n
        + C.ENS_W_LGBM_V3 * v3_n
        + C.ENS_W_RF * rf_n
    )
    aux["relevance"] = (
        aux["booking_bool"].astype("int8") * 5
        + ((aux["click_bool"].astype("int8") == 1) & (aux["booking_bool"].astype("int8") == 0)).astype("int8")
    )
    return aux


def _report(df: pd.DataFrame, tag: str) -> dict:
    per_group = B.per_group_ndcg(df, score_col="score")
    exposure = B.exposure_ratio(df, score_col="score")
    topk = B.topk_share(df, score_col="score", k=5)
    print(f"[{tag}] overall NDCG@5={per_group['overall']:.4f}  "
          f"chain={per_group['chain']:.4f}  indep={per_group['independent']:.4f}  "
          f"gap={per_group['gap']:+.4f}")
    print(f"[{tag}] mean predicted rank — chain={exposure['mean_rank_chain']:.2f} "
          f"indep={exposure['mean_rank_independent']:.2f}  rank_gap={exposure['rank_gap']:+.3f}")
    print(f"[{tag}] chain share in top-5={topk['chain_share_in_topk']:.3f} "
          f"(baseline={topk['chain_share_baseline']:.3f}  delta={topk['delta']:+.3f})")
    return {"ndcg_by_group": per_group, "exposure": exposure, "topk_share": topk}


def main() -> None:
    t0 = time.time()

    # ----- Pre-mitigation (LGBM v1 alone) -----
    pre_df = attach_by_position(C.LGBM_VAL_PREDS)
    pre = _report(pre_df, "pre-lgbm")

    # ----- Train debiased LGBM v1 -----
    df, feat_cols, cat_cols = L.load_features()
    parts = L.split_train_val(df, feat_cols)
    train_mask = df["is_val"].to_numpy() == 0
    weight_df = pd.DataFrame({
        "prop_brand_bool": df.loc[train_mask, "prop_brand_bool"].to_numpy(),
        "booking_bool": df.loc[train_mask, "booking_bool"].to_numpy(),
        "click_bool": df.loc[train_mask, "click_bool"].to_numpy(),
    })
    weights = B.compute_reweighing_weights(weight_df)
    print(f"[weights] mean={weights.mean():.3f}  max={weights.max():.1f}  "
          f"booked_chain mean={weights[(weight_df['prop_brand_bool']==1)&(weight_df['booking_bool']==1)].mean():.1f}  "
          f"booked_indep mean={weights[(weight_df['prop_brand_bool']==0)&(weight_df['booking_bool']==1)].mean():.1f}")
    booster, _ = L.train(parts, cat_cols, sample_weight=weights, save_path=C.LGBM_DEBIAS_MODEL)
    metrics = L.evaluate(booster, parts)
    print(f"[debias] overall NDCG@5={metrics['ndcg@5']:.4f}  best_iter={metrics['best_iteration']}")

    debias_val_path = C.ART_DIR / "lightgbm_debiased_val_preds.parquet"
    pd.DataFrame({"srch_id": parts["q_va"], "score": metrics["preds"]}).to_parquet(
        debias_val_path, compression="zstd", index=False
    )

    # ----- Post-mitigation (LGBM v1 alone, debiased) -----
    post_df = attach_by_position(debias_val_path)
    post = _report(post_df, "post-lgbm")

    out_lgbm = {
        "scope": "lightgbm_v1_alone",
        "sensitive_attribute": "prop_brand_bool",
        "pre": pre,
        "post": post,
        "delta": {
            "overall_ndcg": post["ndcg_by_group"]["overall"] - pre["ndcg_by_group"]["overall"],
            "gap_chain_minus_indep": post["ndcg_by_group"]["gap"] - pre["ndcg_by_group"]["gap"],
            "rank_gap": post["exposure"]["rank_gap"] - pre["exposure"]["rank_gap"],
        },
        "training_time_sec": round(time.time() - t0, 1),
    }
    with open(C.OUT_DIR / "bias_metrics.json", "w") as f:
        json.dump(out_lgbm, f, indent=2)
    B.write_bias_plot(pre["ndcg_by_group"], post["ndcg_by_group"], C.FIG_DIR / "bias_pre_vs_post.png")

    # ----- Pre/post on the deployed three-way ensemble -----
    if not (C.LGBM_V3_VAL_PREDS.exists() and C.RF_VAL_PREDS.exists()):
        print("[ensemble] skip: v3 / RF val preds not yet generated "
              "(run phases 4b and 5 first).")
        print(f"[ok] phase 6 done in {time.time()-t0:.1f}s")
        return

    print(f"\n[ensemble] weights w_v1={C.ENS_W_LGBM_V1:.2f}  "
          f"w_v3={C.ENS_W_LGBM_V3:.2f}  w_rf={C.ENS_W_RF:.2f}")
    pre_ens = _assemble_ensemble_val(C.LGBM_VAL_PREDS)
    pre_e = _report(pre_ens, "pre-ens")
    post_ens = _assemble_ensemble_val(debias_val_path)
    post_e = _report(post_ens, "post-ens")

    out_ens = {
        "scope": "three_way_ensemble_v1+v3+rf",
        "weights": {
            "w_v1": C.ENS_W_LGBM_V1,
            "w_v3": C.ENS_W_LGBM_V3,
            "w_rf": C.ENS_W_RF,
        },
        "sensitive_attribute": "prop_brand_bool",
        "pre": pre_e,
        "post": post_e,
        "delta": {
            "overall_ndcg": post_e["ndcg_by_group"]["overall"] - pre_e["ndcg_by_group"]["overall"],
            "gap_chain_minus_indep": post_e["ndcg_by_group"]["gap"] - pre_e["ndcg_by_group"]["gap"],
            "rank_gap": post_e["exposure"]["rank_gap"] - pre_e["exposure"]["rank_gap"],
            "topk_chain_delta": post_e["topk_share"]["delta"] - pre_e["topk_share"]["delta"],
        },
    }
    with open(C.OUT_DIR / "bias_metrics_ensemble.json", "w") as f:
        json.dump(out_ens, f, indent=2)
    B.write_bias_plot(
        pre_e["ndcg_by_group"],
        post_e["ndcg_by_group"],
        C.FIG_DIR / "bias_pre_vs_post_ensemble.png",
    )

    print(f"[ok] phase 6 done in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
