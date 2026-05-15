"""Generate a Kaggle submission using a feature-subset LightGBM model.

Usage: python scripts/07b_submit_subset.py <subset_name>
Example: python scripts/07b_submit_subset.py top_40
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


def main() -> None:
    subset = sys.argv[1] if len(sys.argv) > 1 else "top_40"

    model_path = C.ART_DIR / f"lightgbm_{subset}.txt"
    metrics_path = C.OUT_DIR / f"lightgbm_metrics_{subset}.json"
    if not model_path.exists():
        raise SystemExit(f"Missing model: {model_path}")
    if not metrics_path.exists():
        raise SystemExit(f"Missing metrics file (needed for feature list): {metrics_path}")

    with open(metrics_path) as f:
        meta = json.load(f)
    feat_cols = meta["features"]
    print(f"[subset] {subset}: {len(feat_cols)} features (val NDCG@5 = {meta['ndcg@5']:.4f})")

    t0 = time.time()
    booster = lgb.Booster(model_file=str(model_path))
    sub = pd.read_parquet(C.SUBMIT_FEAT, columns=["srch_id", "prop_id", *feat_cols])
    sub = sub.sort_values("srch_id", kind="stable").reset_index(drop=True)
    print(f"[load] {len(sub):,} rows  ({time.time()-t0:.1f}s)")

    sub["score"] = booster.predict(sub[feat_cols])
    sub["rank"] = sub.groupby("srch_id", sort=False)["score"].rank(
        ascending=False, method="average"
    )

    out_path = C.OUT_DIR / f"submission_{subset}.csv"
    submission.write_submission(sub, "rank", out_path)
    print(f"[write] {out_path} ({out_path.stat().st_size/1024**2:.1f} MB)")

    chk = submission.verify_against_sample(out_path)
    print(f"[verify] {chk}")
    if not (chk["queries_match"] and chk["rows_match"] and chk["per_query_counts_match"]):
        raise RuntimeError(f"submission failed verification: {chk}")
    print(f"[ok] subset {subset} done in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
