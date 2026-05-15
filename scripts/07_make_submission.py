"""Phase 7 — generate the Kaggle submission (LGBM-only).

Predicts on artifacts/submit_feat.parquet with the trained LightGBM model,
sorts within each `srch_id` by score descending, and writes the CSV.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config as C
from src import submission


def predict_lightgbm(feat_cols: list[str], model_path: Path) -> pd.DataFrame:
    booster = lgb.Booster(model_file=str(model_path))
    submit_cols = list({"srch_id", "prop_id", *feat_cols})
    sub = pd.read_parquet(C.SUBMIT_FEAT, columns=submit_cols)
    sub = sub.sort_values("srch_id", kind="stable").reset_index(drop=True)
    sub["score_lgbm"] = booster.predict(sub[feat_cols])
    return sub[["srch_id", "prop_id", "score_lgbm"]]


def main() -> None:
    t0 = time.time()
    with open(C.FEATURE_LIST) as f:
        meta = json.load(f)
    feat_cols = meta["features"]
    cat_cols = meta["categorical"]

    # Sample paper §6 dropped RankFormer because GBDT outperformed it; we
    # confirmed the same on our val set (LGBM 0.381 vs RF 0.367 vs ensemble
    # 0.378). Submit LGBM-only scores.
    lgb_path = C.LGBM_MODEL
    print(f"[lgbm] using {lgb_path.name}")
    lgb_df = predict_lightgbm(feat_cols, lgb_path)
    print(f"[lgbm] {len(lgb_df):,} rows  ({time.time()-t0:.1f}s)")

    merged = lgb_df
    merged["rank_ens"] = merged.groupby("srch_id", sort=False)["score_lgbm"].rank(
        ascending=False, method="average"
    )
    out_path = C.OUT_DIR / "submission_v4_lgbm_only.csv"
    submission.write_submission(merged, "rank_ens", out_path)
    print(f"[write] {out_path} ({out_path.stat().st_size/1024**2:.1f} MB)")
    chk = submission.verify_against_sample(out_path)
    print(f"[verify] {chk}")
    if not (chk["queries_match"] and chk["rows_match"] and chk["per_query_counts_match"]):
        raise RuntimeError(f"submission failed verification: {chk}")
    print(f"[ok] phase 7 done in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
