"""Phase 3 — build engineered feature parquets for both labeled and submit sets.

Side effects:
    artifacts/labeled_feat.parquet
    artifacts/submit_feat.parquet
    artifacts/prop_history.parquet
    artifacts/split.json
    artifacts/feature_list.json
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

from src import config as C
from src import features as F


def main() -> None:
    t0 = time.time()
    labeled = pd.read_parquet(C.LABELED_PARQUET)
    submit = pd.read_parquet(C.SUBMIT_PARQUET)
    print(f"[load] labeled={len(labeled):,}  submit={len(submit):,}  ({time.time()-t0:.1f}s)")

    train_qids, val_qids = F.make_split(labeled, val_frac=0.2)
    train_qids_set = set(int(x) for x in train_qids)
    val_qids_set = set(int(x) for x in val_qids)
    assert not (train_qids_set & val_qids_set), "split overlap"
    print(f"[split] train queries={len(train_qids):,}  val queries={len(val_qids):,}")

    # Compute history tables on the train fold only.
    train_mask = labeled["srch_id"].isin(train_qids)
    train_df = labeled.loc[train_mask].copy()
    train_df = F.fix_zero_as_missing(train_df)  # use repaired values for stats

    prop_history = F.compute_prop_history(train_df)
    prop_history.to_parquet(C.PROP_HIST, compression="zstd", index=False)
    print(f"[prop_history] {len(prop_history):,} unique prop_ids  ({time.time()-t0:.1f}s)")

    dest_history = F.compute_dest_history(train_df)
    dest_history.to_parquet(C.ART_DIR / "dest_history.parquet", compression="zstd", index=False)
    print(f"[dest_history] {len(dest_history):,} unique destinations")

    dest_month_pos = F.compute_dest_month_pos(train_df)
    dest_month_pos.to_parquet(C.ART_DIR / "dest_month_pos.parquet", compression="zstd", index=False)
    print(f"[dest_month_pos] {len(dest_month_pos):,} (destination, month) cells")

    country_loc2 = F.compute_country_loc2(train_df)
    country_loc2.to_parquet(C.ART_DIR / "country_loc2.parquet", compression="zstd", index=False)
    print(f"[country_loc2] {len(country_loc2):,} prop_country_ids")

    # K-fold target encoding — one of the most important features by gain.
    prop_te_oof, prop_te_full = F.compute_target_encoding(
        train_df, group_col="prop_id", n_folds=5, smooth=50.0,
    )
    dest_te_oof, dest_te_full = F.compute_target_encoding(
        train_df, group_col="srch_destination_id", n_folds=5, smooth=200.0,
    )
    prop_te_full.to_parquet(C.ART_DIR / "prop_te_full.parquet", compression="zstd", index=False)
    dest_te_full.to_parquet(C.ART_DIR / "dest_te_full.parquet", compression="zstd", index=False)
    print(f"[target_enc] prop_te_full={len(prop_te_full):,}  dest_te_full={len(dest_te_full):,}")

    # Global log-price clipping bounds from train fold (0.1% / 99.9%).
    log_price_train = np.log1p(train_df["price_usd"].astype("float32"))
    lo = float(log_price_train.quantile(0.001))
    hi = float(log_price_train.quantile(0.999))
    print(f"[log_price_clip] (q001={lo:.3f}, q999={hi:.3f})")

    # Engineer features.
    t1 = time.time()
    labeled_feat = F.engineer(
        labeled.copy(), prop_history, dest_history, dest_month_pos, country_loc2,
        is_labeled=True, log_price_clip=(lo, hi),
        prop_te_oof=prop_te_oof, prop_te_full=prop_te_full,
        dest_te_oof=dest_te_oof, dest_te_full=dest_te_full,
        train_qids=train_qids,
    )
    print(f"[engineer:labeled] {labeled_feat.shape}  ({time.time()-t1:.1f}s)")
    t1 = time.time()
    submit_feat = F.engineer(
        submit.copy(), prop_history, dest_history, dest_month_pos, country_loc2,
        is_labeled=False, log_price_clip=(lo, hi),
        prop_te_oof=None, prop_te_full=prop_te_full,
        dest_te_oof=None, dest_te_full=dest_te_full,
        train_qids=None,
    )
    print(f"[engineer:submit] {submit_feat.shape}  ({time.time()-t1:.1f}s)")

    # Save a fold indicator on the labeled set (1 = val, 0 = train) so downstream
    # scripts share the same split without rerunning GroupShuffleSplit.
    labeled_feat["is_val"] = labeled_feat["srch_id"].isin(val_qids).astype("int8")

    labeled_feat.to_parquet(C.LABELED_FEAT, compression="zstd", index=False)
    submit_feat.to_parquet(C.SUBMIT_FEAT, compression="zstd", index=False)
    print(
        f"[write] labeled_feat={C.LABELED_FEAT.stat().st_size/1024**2:.1f} MB  "
        f"submit_feat={C.SUBMIT_FEAT.stat().st_size/1024**2:.1f} MB"
    )

    # Persist the split (lists of srch_ids) for any downstream needs.
    with open(C.ART_DIR / "split.json", "w") as f:
        json.dump({"train_qids": [int(x) for x in train_qids],
                   "val_qids": [int(x) for x in val_qids]}, f)

    # Persist the model-input column list.
    feat_cols = F.model_input_columns(labeled_feat)
    # is_val is a metadata column, not a feature.
    feat_cols = [c for c in feat_cols if c != "is_val"]
    with open(C.FEATURE_LIST, "w") as f:
        json.dump({"features": feat_cols, "categorical": C.CATEGORICAL_COLS}, f, indent=2)
    print(f"[features] {len(feat_cols)} model inputs (categorical: {len(C.CATEGORICAL_COLS)})")
    print(f"[ok] phase 3 done in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
