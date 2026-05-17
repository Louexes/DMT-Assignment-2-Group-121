"""Phase 7 — weighted ensemble Kaggle submission.

Per-srch_id min-max normalises LightGBM and RankFormer scores into a
comparable scale, then linearly blends them with weight ``w_lgbm``
(default 0.72). Re-running with the same artefacts in place reuses the
cached submit predictions, so changing the blend weight is essentially
free.

Outputs:
    artifacts/lightgbm_submit_preds.parquet
    artifacts/rankformer_submit_preds.parquet
    outputs/submission_ensemble_lgbm{w_lgbm*100:03d}_rf{w_rf*100:03d}.csv
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


def _predict_lgbm(feat_cols: list[str], model_path: Path) -> pd.DataFrame:
    booster = lgb.Booster(model_file=str(model_path))
    cols = list({"srch_id", "prop_id", *feat_cols})
    sub = pd.read_parquet(C.SUBMIT_FEAT, columns=cols)
    sub = sub.sort_values("srch_id", kind="stable").reset_index(drop=True)
    # ``num_threads=8`` overrides any global OMP_NUM_THREADS=1 the user may
    # have set to keep PyTorch single-threaded (avoids a known macOS libomp
    # deadlock in tensor.clone()). LGBM itself does not hit that deadlock.
    sub["score_lgbm"] = booster.predict(sub[feat_cols], num_threads=8)
    return sub[["srch_id", "prop_id", "score_lgbm"]]


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


def main() -> None:
    t0 = time.time()
    with open(C.FEATURE_LIST) as f:
        meta = json.load(f)
    feat_cols = meta["features"]
    cat_cols = meta["categorical"]
    num_cols = [c for c in feat_cols if c not in cat_cols]

    w_lgbm = 0.72
    w_rf = 1.0 - w_lgbm

    if C.LGBM_SUBMIT_PREDS.exists():
        lgb_df = pd.read_parquet(C.LGBM_SUBMIT_PREDS)
        print(f"[lgbm] cached: {C.LGBM_SUBMIT_PREDS.name} ({len(lgb_df):,} rows, {time.time()-t0:.1f}s)")
    else:
        print(f"[lgbm] predicting…")
        lgb_df = _predict_lgbm(feat_cols, C.LGBM_MODEL)
        lgb_df.to_parquet(C.LGBM_SUBMIT_PREDS, compression="zstd", index=False)
        print(f"[lgbm] {len(lgb_df):,} rows  ({time.time()-t0:.1f}s) — cached")

    if C.RF_SUBMIT_PREDS.exists():
        rf_df = pd.read_parquet(C.RF_SUBMIT_PREDS)
        print(f"[rf] cached: {C.RF_SUBMIT_PREDS.name} ({len(rf_df):,} rows, {time.time()-t0:.1f}s)")
    else:
        print(f"[rf] predicting…")
        rf_df = _predict_rankformer(feat_cols, cat_cols, num_cols, C.RF_MODEL)
        rf_df.to_parquet(C.RF_SUBMIT_PREDS, compression="zstd", index=False)
        print(f"[rf] {len(rf_df):,} rows  ({time.time()-t0:.1f}s) — cached")

    merged = lgb_df.merge(rf_df, on=["srch_id", "prop_id"], how="left")
    assert merged["score_rf"].notna().all(), "rf scores missing for some rows"

    # Per-srch_id min-max normalisation puts the two heterogeneous score
    # distributions on a [0, 1] scale within each query so the blend weight
    # behaves the same way for every query.
    g = merged.groupby("srch_id", sort=False)
    lgb_min = g["score_lgbm"].transform("min")
    lgb_max = g["score_lgbm"].transform("max")
    rf_min = g["score_rf"].transform("min")
    rf_max = g["score_rf"].transform("max")
    eps = 1e-9
    merged["lgbm_n"] = (merged["score_lgbm"] - lgb_min) / (lgb_max - lgb_min + eps)
    merged["rf_n"] = (merged["score_rf"] - rf_min) / (rf_max - rf_min + eps)
    merged["score_ens"] = w_lgbm * merged["lgbm_n"] + w_rf * merged["rf_n"]
    merged["rank_ens"] = (-merged["score_ens"]).groupby(merged["srch_id"], sort=False).rank(
        method="first"
    )

    out_path = C.OUT_DIR / (
        f"submission_ensemble_lgbm{int(round(w_lgbm*100)):03d}"
        f"_rf{int(round(w_rf*100)):03d}.csv"
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
