"""Phase 7 — three-way weighted ensemble Kaggle submission.

Per-srch_id min-max normalises three score columns into a comparable
[0, 1] scale, then linearly blends them with the weights in
``src/config.py`` (``ENS_W_LGBM_V1``, ``ENS_W_LGBM_V3``, ``ENS_W_RF``).
Re-running with cached submit predictions in place is essentially free.

Outputs:
    artifacts/lightgbm_submit_preds.parquet
    artifacts/lightgbm_v3_submit_preds.parquet
    artifacts/rankformer_submit_preds.parquet
    outputs/submission_ensemble_v1{ww}_v3{ww}_rf{ww}.csv
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config as C
from src import rankformer_train as RT
from src import submission
from src.rankformer import RankFormer, RankFormerConfig


def _load_rf_stats(npz_path: Path, num_cols: list[str], cat_cols: list[str]) -> RT.FeatureStats:
    z = np.load(npz_path, allow_pickle=True)
    means = z["num_means"].astype(np.float32)
    stds = z["num_stds"].astype(np.float32)
    cat_maps: dict[str, dict] = {}
    cat_cards: dict[str, int] = {}
    for c in cat_cols:
        keys = z[f"map__{c}__keys"]
        vals = z[f"map__{c}__vals"]
        cat_maps[c] = {int(k): int(v) for k, v in zip(keys, vals)}
        cat_cards[c] = int(vals.max()) + 1
    return RT.FeatureStats(
        num_cols=num_cols,
        cat_cols=cat_cols,
        num_means=means,
        num_stds=stds,
        cat_maps=cat_maps,
        cat_cardinalities=cat_cards,
    )


def _predict_lgbm(feat_cols: list[str], model_path: Path, score_col: str) -> pd.DataFrame:
    booster = lgb.Booster(model_file=str(model_path))
    cols = list({"srch_id", "prop_id", *feat_cols})
    sub = pd.read_parquet(C.SUBMIT_FEAT, columns=cols)
    sub = sub.sort_values("srch_id", kind="stable").reset_index(drop=True)
    # ``num_threads=8`` overrides any global OMP_NUM_THREADS=1 the user may
    # have set to keep PyTorch single-threaded (avoids a known macOS libomp
    # deadlock in tensor.clone()). LGBM itself does not hit that deadlock.
    sub[score_col] = booster.predict(sub[feat_cols], num_threads=8)
    return sub[["srch_id", "prop_id", score_col]]


def _predict_rankformer(
    feat_cols: list[str], cat_cols: list[str], num_cols: list[str], model_path: Path
) -> pd.DataFrame:
    stats = _load_rf_stats(C.ART_DIR / "rankformer_stats.npz", num_cols, cat_cols)
    cols = list({"srch_id", "prop_id", *feat_cols})
    df = pd.read_parquet(C.SUBMIT_FEAT, columns=cols)
    df = df.sort_values("srch_id", kind="stable").reset_index(drop=True)

    num, cats = RT.transform(df, stats)
    offsets, qids = RT.build_session_arrays(df, num, cats)

    dummy_rel = np.zeros(len(df), dtype=np.int8)
    ds = RT.SessionDataset(offsets, qids, num, cats, dummy_rel)
    collate = RT.make_collate(C.RF_TRAIN["max_list_len"], len(num_cols), list(cat_cols))
    loader = DataLoader(
        ds, batch_size=C.RF_TRAIN["batch_sessions"], shuffle=False, collate_fn=collate
    )

    cat_dims = {
        c: 32 if c == "srch_destination_id"
        else 8 if c.endswith("country_id")
        else 4
        for c in cat_cols
    }
    cfg_model = RankFormerConfig(
        n_num=len(num_cols),
        cat_cardinalities=stats.cat_cardinalities,
        cat_dims=cat_dims,
        d_model=C.RF_PARAMS["d_model"],
        n_heads=C.RF_PARAMS["n_heads"],
        n_layers=C.RF_PARAMS["n_layers"],
        ffn_dim=C.RF_PARAMS["ffn_dim"],
        dropout=C.RF_PARAMS["dropout"],
        max_list_len=C.RF_TRAIN["max_list_len"],
        numeric_mlp=C.RF_PARAMS.get("numeric_mlp", False),
        fuse_mlp=C.RF_PARAMS.get("fuse_mlp", False),
        use_pos_emb=C.RF_PARAMS.get("use_pos_emb", True),
    )
    model = RankFormer(cfg_model).to(RT.DEVICE)
    ckpt = torch.load(model_path, map_location=RT.DEVICE, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])

    qids_out, scores_out, _ = RT.predict_loader(model, loader)
    assert len(scores_out) == len(df), f"{len(scores_out)} vs {len(df)}"
    assert (qids_out == df["srch_id"].to_numpy()).all(), "srch_id alignment broke"
    df["score_rf"] = scores_out
    return df[["srch_id", "prop_id", "score_rf"]]


def _norm_within_srch(df: pd.DataFrame, col: str) -> pd.Series:
    g = df.groupby("srch_id", sort=False)[col]
    eps = 1e-9
    return (df[col] - g.transform("min")) / (g.transform("max") - g.transform("min") + eps)


def main() -> None:
    t0 = time.time()
    with open(C.FEATURE_LIST) as f:
        meta = json.load(f)
    feat_cols = meta["features"]
    cat_cols = meta["categorical"]
    num_cols = [c for c in feat_cols if c not in cat_cols]

    w_v1 = C.ENS_W_LGBM_V1
    w_v3 = C.ENS_W_LGBM_V3
    w_rf = C.ENS_W_RF
    assert abs(w_v1 + w_v3 + w_rf - 1.0) < 1e-6, "ensemble weights must sum to 1"
    print(f"[blend] w_v1={w_v1:.2f}  w_v3={w_v3:.2f}  w_rf={w_rf:.2f}")

    # ----- LGBM v1 submit preds -----
    if C.LGBM_SUBMIT_PREDS.exists():
        v1_df = pd.read_parquet(C.LGBM_SUBMIT_PREDS)
        print(f"[lgbm-v1] cached ({len(v1_df):,} rows, {time.time()-t0:.1f}s)")
    else:
        print("[lgbm-v1] predicting…")
        v1_df = _predict_lgbm(feat_cols, C.LGBM_MODEL, "score_lgbm")
        v1_df.to_parquet(C.LGBM_SUBMIT_PREDS, compression="zstd", index=False)
        print(f"[lgbm-v1] {len(v1_df):,} rows  ({time.time()-t0:.1f}s)")

    # ----- LGBM v3 submit preds -----
    if C.LGBM_V3_SUBMIT_PREDS.exists():
        v3_df = pd.read_parquet(C.LGBM_V3_SUBMIT_PREDS)
        print(f"[lgbm-v3] cached ({len(v3_df):,} rows, {time.time()-t0:.1f}s)")
    else:
        print("[lgbm-v3] predicting…")
        v3_df = _predict_lgbm(feat_cols, C.LGBM_V3_MODEL, "score_v3")
        v3_df.to_parquet(C.LGBM_V3_SUBMIT_PREDS, compression="zstd", index=False)
        print(f"[lgbm-v3] {len(v3_df):,} rows  ({time.time()-t0:.1f}s)")

    # ----- RankFormer submit preds -----
    if C.RF_SUBMIT_PREDS.exists():
        rf_df = pd.read_parquet(C.RF_SUBMIT_PREDS)
        print(f"[rf] cached ({len(rf_df):,} rows, {time.time()-t0:.1f}s)")
    else:
        print("[rf] predicting…")
        rf_df = _predict_rankformer(feat_cols, cat_cols, num_cols, C.RF_MODEL)
        rf_df.to_parquet(C.RF_SUBMIT_PREDS, compression="zstd", index=False)
        print(f"[rf] {len(rf_df):,} rows  ({time.time()-t0:.1f}s)")

    merged = v1_df.merge(v3_df, on=["srch_id", "prop_id"], how="left")
    merged = merged.merge(rf_df, on=["srch_id", "prop_id"], how="left")
    assert merged["score_v3"].notna().all(), "v3 scores missing for some rows"
    assert merged["score_rf"].notna().all(), "rf scores missing for some rows"
    merged = merged.sort_values("srch_id", kind="stable").reset_index(drop=True)

    merged["v1_n"] = _norm_within_srch(merged, "score_lgbm")
    merged["v3_n"] = _norm_within_srch(merged, "score_v3")
    merged["rf_n"] = _norm_within_srch(merged, "score_rf")
    merged["score_ens"] = w_v1 * merged["v1_n"] + w_v3 * merged["v3_n"] + w_rf * merged["rf_n"]
    merged["rank_ens"] = (-merged["score_ens"]).groupby(merged["srch_id"], sort=False).rank(
        method="first"
    )

    out_path = C.OUT_DIR / (
        f"submission_ensemble_v1{int(round(w_v1*100)):03d}"
        f"_v3{int(round(w_v3*100)):03d}_rf{int(round(w_rf*100)):03d}.csv"
    )
    submission.write_submission(merged, "rank_ens", out_path)
    print(f"[write] {out_path} ({out_path.stat().st_size/1024**2:.1f} MB)")
    chk = submission.verify_against_sample(out_path)
    print(f"[verify] {chk}")
    if not (chk["queries_match"] and chk["rows_match"] and chk["per_query_counts_match"]):
        raise RuntimeError(f"submission failed verification: {chk}")
    print(f"[ok] phase 7 done in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
